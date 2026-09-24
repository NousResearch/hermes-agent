# Theoretical Foundations — SOTA Personality Simulation & Prediction

Compiled from 30+ papers and frameworks. This is the scientific backbone
of Hermes Simulator.

## Core Architecture: What The Research Says

### The HumanLLM Approach (Microsoft, KDD 2026, arxiv 2601.15793)
**Most directly applicable to our use case.**

Based on Lewin's Equation: **B = f(P, E)** — behavior is a function of person + environment.

4-level user profiling hierarchy:
1. **Persona** — brief identity (role, affiliation, public image)
2. **Profile** — detailed background (career, education, beliefs, social graph)
3. **Stories** — key life events, formative experiences, narrative arcs
4. **Writing Style** — linguistic fingerprint (syntax, vocabulary, tone, quirks)

Trained on "Cognitive Genome Dataset": 5.5M+ user logs from Reddit, Twitter,
Blogger, Amazon (282K users, 886K scenarios, 1.27M social QA pairs).

6 training tasks: profile generation, scenario generation, social QA,
writing style transfer, action prediction, mental state inference.

**Key insight for us**: The 4-level hierarchy maps perfectly to our dossier
template. OSINT research fills each level with real data.

### Generative Agent Simulations of 1,000 People (Stanford/Google, arxiv 2411.10109)
**The accuracy benchmark.**

- Simulated 1,052 REAL individuals from 2-hour qualitative interviews
- **85% accuracy** replicating survey responses
- As accurate as humans replicating their OWN answers 2 weeks later
- Interview-based agent creation >> demographic-profile-based agents
- Reduces racial/ideological bias vs stereotype-based approaches

**Key insight**: Real data about a person (interviews, posts, etc.) massively
outperforms demographic inference. Our OSINT approach is correct.

### The Memory Accumulation Paradox (ACL 2025, FineRob Dataset)
**Critical finding for memory management.**

- Created 78.6K QA records from 1,866 real users across Twitter, Reddit, Zhihu
- **Performance PEAKS at 30-50 memory entries, then DECLINES**
- More data ≠ better predictions past the sweet spot
- Two reasoning patterns:
  - Role Stereotype-based (static profile) — less accurate
  - Observation & Memory-based (dynamic history analysis) — much more accurate
- OM-CoT framework: Oracle-guided chain-of-thought improves prediction ~4.5% F1

**Key insight**: Don't dump everything into the prompt. Curate the 30-50 most
representative/distinctive data points about a person. Quality >> quantity.

### LLM Personality Limitations (arxiv 2602.07414, Feb 2026)
**What we're fighting against.**

- LLMs show polarized/rigid strategies vs human adaptive flexibility
- Humans: neuroticism is strongest behavioral predictor
- LLMs: agreeableness/extraversion dominate (wrong weighting)
- Claude closest to human behavior; GPT-4 tends to escalate
- LLMs are "sycophantic" and overly agreeable by default
- Neuroticism is hardest trait to simulate (F1=0.63 vs 0.87 for Openness)

**Key insight**: We need to actively fight LLM defaults. Push against
agreeableness. Inject friction. Real people are messy and contradictory.

### BehaviorChain Benchmark (ACL 2025, Peking University)
**Realistic accuracy expectations.**

- 15,846 behaviors across 1,001 personas
- Even GPT-4o achieves only ~56% accuracy on behavior prediction
- Errors compound: wrong at step N makes step N+1 harder
- Models worse at predicting mundane/non-key behaviors
- Best model: Llama-3.1-70B at 57.4%

**Key insight**: Be honest about uncertainty. Don't oversell accuracy.
Flag predictions as high/medium/low confidence.

## Personality Modeling Techniques

### Big Five (OCEAN) — The Standard
- **Openness**: curiosity, creativity, preference for novelty
- **Conscientiousness**: organization, dependability, self-discipline
- **Extraversion**: sociability, assertiveness, positive emotions
- **Agreeableness**: cooperation, trust, empathy
- **Neuroticism**: anxiety, emotional instability, moodiness

### Inferring Big Five from Social Media (Azucar et al. 2018 meta-analysis)
Features that predict personality from posts:
- **LIWC** (Linguistic Inquiry Word Count): 74 features — function words,
  pronouns, emotion words, cognitive process words
- **Semantic embeddings**: BERT 768-dim vectors from post text
- **Social metadata**: follower count, friend count, post frequency
- **Sentiment**: VADER positive/negative scores
- Best achievable AUC: ~0.67 (modest but meaningful)
- E/I (Extraversion) most predictable; N/S least predictable

### Personality Conditioning Methods (ranked by effectiveness)
1. **Training-based** (SFT/DPO on personality-grounded data) — STRONGEST
   - BIG5-CHAT: 100K dialogues, trait correlations match human data
2. **Persona Vectors** (Anthropic 2025) — monitor/control traits at activation level
3. **Adjective-based prompting** — 70 bipolar adjective pairs, 3 per trait
   with intensity modifiers ("very" for high, "a bit" for low)
4. **Prompt-based** (describe traits in system prompt) — WEAKEST

For our simulator, we use method 3+4 combined (adjective-based + rich prompt),
since we can't fine-tune per-person.

## Social Simulation Frameworks

### OASIS (CAMEL-AI, GitHub 4.1K stars, arxiv 2411.11581)
- Simulates up to 1 MILLION agents on Twitter/Reddit clones
- 23 action types (follow, comment, repost, like, mute, etc.)
- Built-in recommendation systems (interest-based, hot-score)
- Per-agent model customization
- **Relevant for**: understanding platform dynamics, realistic engagement patterns

### AgentSociety (Tsinghua, arxiv 2502.08691)
- 10,000+ agents, ~5 million interactions
- Validated against real-world experimental results
- Supports interventions and scenario injection

### Generative Agents Architecture (Park et al. 2023, THE foundational paper)
Three components:
1. **Observation**: perceive environment, store in memory stream
2. **Planning**: generate action plans based on goals and context
3. **Reflection**: synthesize observations into higher-level insights

Memory stream with importance scoring + recency + relevance weighting.
Emergent behaviors: autonomous party planning, coordinated social events.

### Y Social (arxiv 2408.00818)
- Social media digital twin platform
- Each agent: Big Five traits, age, political leaning, topics, education
- Agents autonomously decide actions (post, comment, like, follow)
- Multiple LLM backends supported

## Role-Playing & Character Simulation

### Key Frameworks
- **CoSER** (ICML 2025): Trains on ALL characters simultaneously, handles major + minor roles
- **RoleLLM** (ACL 2024): Benchmark + elicit + enhance pipeline
- **Character-LLM** (EMNLP 2023): Trainable agent for role-playing
- **ChatHaruhi** (2023): Reviving characters via LLMs with dialogue grounding
- **OpenCharacter** (2025): Training with large-scale synthetic personas
- **Neeko** (2024): Dynamic LoRA for multi-character role-playing
- **Test-Time-Matching** (2025): Decouples personality, memory, and linguistic style at inference

## Curated GitHub Resources

### Awesome Lists (essential reading)
- `Persdre/awesome-llm-human-simulation` (109★, ICLR 2025) — ALL human simulation papers
- `Neph0s/awesome-llm-role-playing-with-persona` (1K★) — All role-playing/persona papers
- `Arstanley/Awesome-LLM-Conversation-Simulation` — Conversation simulation papers
- `FudanDISC/SocialAgent` — Social simulation survey resources

### Frameworks
- `camel-ai/oasis` (4.1K★) — Social media sim, up to 1M agents
- `tsinghua-fib-lab/agentsociety` — Large-scale societal simulation
- `YSocialTwin` — Social media digital twin platform
- `microsoft/autogen` — Multi-agent conversation framework

### Personality Research
- `mary-silence/simulating_personality` — Big Five LLM testing code
- `hjian42/PersonaLLM` — Persona experiment code
- `cambridgeltl/persona_effect` — Quantifying persona effects
- `OL1RU1/BehaviorChain` — Behavior chain benchmark

## Key Numbers to Remember

| Metric | Value | Source |
|--------|-------|--------|
| Interview-grounded agent accuracy | 85% | Park et al. 2024 |
| GPT-4o behavior prediction | ~56% | BehaviorChain 2025 |
| Optimal memory entries | 30-50 | FineRob/ACL 2025 |
| MBTI prediction AUC | 0.67 | Watt et al. 2024 |
| Personality questionnaire reliability | α > 0.85 | Molchanova 2025 |
| Neuroticism simulation F1 | 0.63 | Molchanova 2025 |
| Openness simulation F1 | 0.87 | Molchanova 2025 |
| LLM forecasting Brier score | 0.135-0.159 | Various 2025 |
| Human superforecaster Brier | ~0.02 | Tetlock |
