import argparse
import json
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
import warnings

# Suppress warnings for cleaner console output during the 5-minute run
warnings.filterwarnings('ignore')

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return datetime(2026, 6, 1) # Hackathon present day context

def calculate_behavioral_multiplier(row):
    multiplier = 1.0

    # Logistics
    loc = str(row.get('location', '')).lower()
    willing = row.get('willing_to_relocate', False)
    if 'pune' in loc or 'noida' in loc:
        multiplier *= 1.15  
    elif willing:
        multiplier *= 1.0   
    else:
        multiplier *= 0.4   

    notice_days = row.get('notice_period_days', 90)
    if notice_days <= 30:
        multiplier *= 1.10  
    elif notice_days > 60:
        multiplier *= 0.85  

    # Engagement
    resp_rate = row.get('recruiter_response_rate', 0.0)
    multiplier *= (0.2 + (0.8 * resp_rate))

    # Credibility
    github = row.get('github_activity_score', -1)
    if github > 50:
        multiplier *= 1.10  
    elif github == -1:
        multiplier *= 0.95  

    if row.get('verified_email') and row.get('verified_phone') and row.get('linkedin_connected'):
        multiplier *= 1.05

    # Market Signal
    if row.get('saved_by_recruiters_30d', 0) > 10:
        multiplier *= 1.05
    if row.get('interview_completion_rate', 0.0) > 0.8:
        multiplier *= 1.05

    return multiplier

# ==========================================
# 2. CORE PIPELINE CLASSES
# ==========================================

class CandidateProcessor:
    def __init__(self):
        self.consulting_firms = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'}

    def process(self, file_path):
        print(f"Loading and processing {file_path}...")
        data = []
        
        with open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                try:
                    cand = json.loads(line)
                    cid = cand.get('candidate_id')
                    if not cid: continue
                    
                    prof = cand.get('profile', {})
                    signals = cand.get('redrob_signals', {})
                    
                    # Pre-compute checks
                    companies = [job.get('company', '').lower() for job in cand.get('career_history', [])]
                    is_only_consulting = (len(companies) > 0) and all(c in self.consulting_firms for c in companies)
                    
                    has_date_paradox = False
                    total_jobs = len(cand.get('career_history', []))
                    for job in cand.get('career_history', []):
                        start = parse_date(job.get('start_date', '2026-06-01'))
                        end = parse_date(job.get('end_date', '2026-06-01')) if job.get('end_date') else datetime(2026, 6, 1)
                        actual_months = (end.year - start.year) * 12 + (end.month - start.month)
                        claimed_months = job.get('duration_months', 0)
                        if claimed_months > actual_months + 2: 
                            has_date_paradox = True
                            break
                            
                    grad_year = max([edu.get('end_year', 0) for edu in cand.get('education', [])] + [0])
                    fake_expert = sum(1 for s in cand.get('skills', []) if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0)
                    
                    history_text = " | ".join([f"Role: {j.get('title', '')}. Desc: {j.get('description', '')}" for j in cand.get('career_history', [])])
                    skills_text = " ".join([s.get('name', '') for s in cand.get('skills', [])])

                    data.append({
                        'candidate_id': cid,
                        'current_title': prof.get('current_title', '').lower(),
                        'years_of_experience': prof.get('years_of_experience', 0),
                        'location': prof.get('location', '').lower(),
                        'total_jobs': total_jobs,
                        'grad_year': grad_year,
                        'fake_expert_count': fake_expert,
                        'has_date_paradox': has_date_paradox,
                        'is_only_consulting': is_only_consulting,
                        'history_text': history_text,
                        'skills_text': skills_text,
                        'recruiter_response_rate': signals.get('recruiter_response_rate', 0.0),
                        'last_active_date': signals.get('last_active_date', ''),
                        'github_activity_score': signals.get('github_activity_score', -1),
                        'willing_to_relocate': signals.get('willing_to_relocate', False),
                        'notice_period_days': signals.get('notice_period_days', 90),
                        'saved_by_recruiters_30d': signals.get('saved_by_recruiters_30d', 0),
                        'interview_completion_rate': signals.get('interview_completion_rate', 0.0),
                        'verified_email': signals.get('verified_email', False),
                        'verified_phone': signals.get('verified_phone', False),
                        'linkedin_connected': signals.get('linkedin_connected', False)
                    })
                except json.JSONDecodeError:
                    continue
                    
        df = pd.DataFrame(data)
        return self._sweep_honeypots(df)

    def _sweep_honeypots(self, df):
        print("Sweeping Honeypots & Fast-Failing Mismatches...")
        df['years_since_grad'] = 2026 - df['grad_year']
        
        bad_engineering = ['mechanical', 'civil', 'chemical', 'electrical', 'qa ', 'quality', 'test']
        bad_business = ['marketing', 'hr ', 'human resources', 'accountant', 'support', 'sales', 'graphic', 'operations', 'analyst', 'project manager']
        
        def is_bad_title(title):
            t = str(title).lower()
            return any(b in t for b in bad_business) or any(b in t for b in bad_engineering)

        # 1. The Logistics Fast-Fail
        loc_str = df['location'].fillna('').str.lower()
        logistics_mask = (~df['willing_to_relocate']) & (~loc_str.str.contains('pune|noida'))

        # 2. The Lexical Fast-Fail
        core_terms = r'\b(ai|ml|machine learning|llm|rag|embedding|vector|pinecone|faiss|weaviate|milvus|recommendation|retrieval|python|nlp)\b'
        combined_text = df['history_text'].fillna('').str.lower() + " " + df['skills_text'].fillna('').str.lower()
        lexical_mask = ~combined_text.str.contains(core_terms, regex=True)

        trap_mask = (
            ((df['grad_year'] > 0) & (df['years_of_experience'] > df['years_since_grad'] + 2)) |
            (df['fake_expert_count'] > 0) |
            (df['current_title'].apply(is_bad_title)) |
            ((df['recruiter_response_rate'] < 0.05) & (~df['last_active_date'].str.startswith('2026', na=False))) |
            (df['is_only_consulting'] == True) |
            ((df['years_of_experience'] > 4.0) & ((df['years_of_experience'] * 12) / df['total_jobs'] < 18)) |
            (df['has_date_paradox'] == True) |
            logistics_mask |
            lexical_mask
        )
        
        clean_df = df[~trap_mask].copy()
        print(f"Clean Candidates Remaining for CPU Encoding: {len(clean_df)} / {len(df)}")
        return clean_df

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Redrob AI Production Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to output submission.csv")
    args = parser.parse_args()

    # 1. Load and Clean
    processor = CandidateProcessor()
    df_clean = processor.process(args.candidates)

    if df_clean.empty:
        print("Warning: No candidates survived the sweep filters. Exiting.")
        return

    # 2. Semantic Search (Loaded locally from artifacts)
    print("Loading Local Embedding Model...")
    model = SentenceTransformer('./artifacts/all-MiniLM-L6-v2')
    
    jd_target = (
        "Senior AI Engineer with production experience in embeddings-based retrieval systems. "
        "Expertise in vector databases like Pinecone, FAISS, Weaviate, or Milvus. "
        "Strong Python programming. Built and evaluated recommendation systems or search ranking models. "
        "Familiar with NDCG, MAP evaluation metrics, LLM fine-tuning, and RAG architectures."
    )
    jd_embedding = model.encode(jd_target, convert_to_tensor=True)
    
    print("Encoding Candidate Histories...")
    candidate_embeddings = model.encode(df_clean['history_text'].tolist(), convert_to_tensor=True)
    df_clean['semantic_score'] = util.cos_sim(jd_embedding, candidate_embeddings)[0].cpu().numpy()

    # 3. Behavioral Multiplier & Composite Score
    print("Applying Behavioral Modifiers...")
    df_clean['behavior_multiplier'] = df_clean.apply(calculate_behavioral_multiplier, axis=1)
    df_clean['score'] = df_clean['semantic_score'] * df_clean['behavior_multiplier']

    # 4. Tie-Breaking and Sorting (Descending Score, Ascending ID)
    top_100 = df_clean.sort_values(by=['score', 'candidate_id'], ascending=[False, True]).head(100).copy()
    top_100['rank'] = range(1, len(top_100) + 1)

    # 5. Local SLM Reasoning Generation
    print("Loading Local SLM for Reasoning Generation...")
    llm = Llama(
        model_path="./artifacts/tinyllama-1.1b-chat.Q4_K_M.gguf",  
        n_ctx=512, 
        verbose=False
    )

    def generate_slm_reasoning(row):
        prompt = f"""<|system|>
You are an expert technical recruiter. Write exactly 2 sentences explaining why this candidate is a good fit for a Senior AI Engineer role. 
Rules: Use specific facts provided. Acknowledge any obvious gaps. Do NOT invent skills.
Candidate Data:
- Score: {row['score']:.2f}
- Title: {row['current_title']}
- Experience: {row['years_of_experience']} years
- Response Rate: {row['recruiter_response_rate'] * 100}%
- Notice Period: {row['notice_period_days']} days
- History: {str(row['history_text'])[:200]}
</s>
<|user|>
Write the 2-sentence reasoning.</s>
<|assistant|>"""
        
        output = llm(prompt, max_tokens=60, stop=["</s>"], echo=False)
        return output['choices'][0]['text'].strip().replace('\n', ' ')

    print("Generating Dynamic Reasoning...")
    top_100['reasoning'] = top_100.apply(generate_slm_reasoning, axis=1)

    # 6. Format and Export
    final_submission = top_100[['candidate_id', 'rank', 'score', 'reasoning']]
    final_submission.to_csv(args.out, index=False)
    print(f"Success! Top 100 written to {args.out}")

if __name__ == "__main__":
    main()