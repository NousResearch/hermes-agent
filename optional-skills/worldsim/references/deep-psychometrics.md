# Deep Psychometrics — Beyond Big Five

Multi-layer psychological profiling from public posts. Each layer adds
a dimension to the personality model, making simulations more nuanced
and predictions more accurate.

## The Profiling Stack

| Layer | What It Measures | Tool/Method | Accuracy | Min Posts |
|-------|-----------------|-------------|----------|-----------|
| Big Five (OCEAN) | Core personality traits | RoBERTa embeddings + BiLSTM | AUROC 0.78-0.82 | 30-50 |
| Moral Foundations | Ethical intuitions | eMFDscore (pip) | Validated dictionary | 20+ |
| Schwartz Values | Core value priorities | DeBERTa on ValueEval | F1 0.56 (macro) | 20+ |
| Cognitive Style | Thinking patterns | AutoIC + LIWC features | r=0.70-0.82 doc-level | 20+ |
| Narrative Framing | How they frame issues | GPT-4 few-shot | F1 ~70% | 10+ |
| Behavioral Metadata | Non-text patterns | Feature extraction | r=0.29-0.40 per trait | 20+ |

## Layer 1: Big Five Personality (Foundation)

### Accuracy Bounds (peer-reviewed)
- AUROC 0.78-0.82 with RoBERTa embeddings + BiLSTM (JMIR 2025)
- Per-trait binary accuracy: O=0.637, C=0.602, E=0.620, A=0.590, N=0.620
- Meta-analytic correlations (Azucar 2018, 16 studies):
  Extraversion r=0.40, Openness r=0.39, Conscientiousness r=0.35,
  Neuroticism r=0.33, Agreeableness r=0.29
- These hit the "personality coefficient" ceiling of r=0.30-0.40 —
  digital footprints are as predictive as any behavioral measure

### What Actually Works
- Fine-tuned embeddings >> zero-shot LLMs. GPT-4o zero-shot is UNRELIABLE.
- RoBERTa embeddings are free and nearly as good as OpenAI embeddings
- Aggregation across posts is essential — single posts are noise
- 30-50 posts of ~90 words each = practical minimum
- Training data: PANDORA Reddit corpus (1568 users, ~935K posts)

### For The Simulator (without running models)
Since we can't fine-tune per-simulation, use LLM-as-rater with caveats:
- Provide 10-20 actual posts as evidence
- Ask for trait estimation with reasoning, not just scores
- Anchor with the adjective-based method (see prediction-engine.md)
- Frame estimates as ranges, not points: "Openness: HIGH (0.7-0.9)"
- Known bias: LLMs overestimate agreeableness and underestimate neuroticism

### Key Insight: LLMs Already Know Public Figures
Nature Scientific Reports 2024: GPT-3's semantic space already encodes
perceived personality of public figures from their names alone. For
famous people, the LLM's latent knowledge is a STARTING POINT that
OSINT data confirms or corrects.

## Layer 2: Moral Foundations (Ethical Compass)

Jonathan Haidt's Moral Foundations Theory. Six foundations:

| Foundation | Liberal emphasis | Conservative emphasis |
|-----------|-----------------|---------------------|
| Care/Harm | ★★★ HIGH | ★★ MODERATE |
| Fairness/Cheating | ★★★ HIGH | ★★ MODERATE |
| Loyalty/Betrayal | ★ LOW | ★★★ HIGH |
| Authority/Subversion | ★ LOW | ★★★ HIGH |
| Sanctity/Degradation | ★ LOW | ★★★ HIGH |
| Liberty/Oppression | ★★ MODERATE | ★★ MODERATE |

### Tool: eMFDscore
```
pip install emfdscore
# GitHub: github.com/medianeuroscience/emfdscore
# Built on spaCy, GPL-3.0
```

Output per post: scores for each foundation (virtue + vice dimensions)
Aggregate across 20+ posts → 10-dimensional moral profile

### Application to Simulation
Moral foundations predict:
- What topics trigger emotional responses
- What arguments they find persuasive vs repulsive
- How they frame political/social issues
- Who they instinctively ally with vs oppose
- What kind of content they share/amplify

Example: High Loyalty/Authority person will defend their tribe even when
wrong. High Care/Fairness person will break from their tribe on justice
issues. This shapes conversation dynamics.

### For The Simulator (without running eMFDscore)
Infer moral foundations from:
- Political positions and framing in their posts
- What they get angry about vs what they celebrate
- Who they defend and who they attack
- Key moral vocabulary: "protect", "fair", "loyal", "respect", "pure", "free"

## Layer 3: Schwartz Values (Core Motivations)

19 values in circular continuum (adjacent values are compatible,
opposite values are in tension):

**Self-Transcendence** ↔ **Self-Enhancement**
- Universalism, Benevolence ↔ Power, Achievement

**Openness to Change** ↔ **Conservation**
- Self-Direction, Stimulation, Hedonism ↔ Tradition, Conformity, Security

### SemEval-2023 Task 4 Results
- Best macro-F1: 0.56 (ensemble of 12 DeBERTa/RoBERTa models)
- Most reliable: universalism (nature), security, power
- Least reliable: stimulation, hedonism, humility
- Dataset: 9,324 annotated arguments, available via Touché

### Key Finding: Value Perception Is Subjective
Epstein et al. (2026): human inter-rater agreement on values is only r=0.201.
Fine-tuned GPT-4o reaches r=0.294 — BETTER than human-human agreement.
Personalized models reach r=0.334.

### For The Simulator
Values predict MOTIVATION — why someone holds positions, not just what
positions they hold. Two people with the same political stance may have
completely different underlying values:
- "I support open source because FREEDOM" (Self-Direction)
- "I support open source because FAIRNESS" (Universalism)
- "I support open source because it WORKS BETTER" (Achievement)
Same position, different framing, different behavioral predictions.

## Layer 4: Cognitive Style (How They Think)

### Integrative Complexity (AutoIC)
Measures differentiation (seeing multiple perspectives) and integration
(synthesizing perspectives into coherent frameworks).

- Low IC: black-and-white thinking, strong convictions, simple language
- High IC: nuanced, sees multiple sides, hedging, complex sentences

AutoIC (Conway et al.): 3,500+ complexity-relevant root words/phrases,
13 dictionary categories, validated r=0.70-0.82 at document level.

**WARNING**: LIWC's "analytic thinking" correlates only r=0.14 with actual
integrative complexity. Don't use LIWC's score as a proxy.

### Computational Indicators of Cognitive Style
Extractable from 20-50 posts without specialized tools:

| Indicator | High Cognition | Low Cognition |
|-----------|---------------|---------------|
| Vocabulary diversity (TTR) | HIGH | LOW |
| Avg sentence length | LONGER | SHORTER |
| Causal connectives ("because", "therefore") | MORE | FEWER |
| Hedging ("perhaps", "it seems") | MORE | FEWER |
| Abstract vs concrete language | MORE ABSTRACT | MORE CONCRETE |
| Question-asking | MORE | FEWER |
| Binary framing ("always/never") | LESS | MORE |

### For The Simulator
Cognitive style directly shapes VOICE:
- High IC person: longer posts, more caveats, "on the other hand"
- Low IC person: punchy takes, strong assertions, no hedging
- This is one of the strongest differentiators between similar-sounding people

## Layer 5: Narrative Framing (Their Lens on Reality)

How someone frames an issue reveals deep cognitive and value patterns.

### Common Frames (Semetko & Valkenburg)
- **Conflict**: issue as battle between opposing sides
- **Human interest**: personal stories, emotional impact
- **Economic**: costs, benefits, financial impact
- **Morality**: right vs wrong, ethical principles
- **Attribution of responsibility**: who's to blame / who should fix it

### Detection
GPT-4 few-shot with frame definitions achieves F1=70.4%
Best for diverse topics where fine-tuned models are too narrow

### For The Simulator
Framing predicts:
- How they'll react to news (through which lens)
- What aspects they'll emphasize in conversation
- What arguments they'll find compelling
- Whether they personalize or systematize issues

Example: Same AI safety event, different frames:
- Conflict framer: "The open vs closed battle heats up"
- Economic framer: "This will cost the industry billions"
- Moral framer: "This is irresponsible and dangerous"
- Attribution framer: "The regulators need to step in"

## Layer 6: Behavioral Metadata (Non-Text Signals)

Extractable from X API / Bluesky AT Protocol without NLP:

| Feature | What It Reveals |
|---------|----------------|
| Posting time distribution | Timezone, sleep patterns, work schedule |
| Reply vs original ratio | Conversational vs broadcast personality |
| Emoji frequency & types | Emotional expression style |
| Hashtag usage | Community identification, signal boosting |
| Media attachment rate | Visual vs text orientation |
| Thread length | Depth of engagement preference |
| Retweet/repost ratio | Amplifier vs creator |
| Average post length | Conciseness vs verbosity |
| Response latency | Impulsiveness vs deliberation |

### Trait Correlations (meta-analytic)
- **Extraversion**: more posts, more friends, more photos, more group activity
- **Neuroticism**: more self-disclosure, more passive consumption, more late-night posting
- **Agreeableness**: fewer swear words, more positive emotion, more supportive replies
- **Conscientiousness**: more regular posting patterns, more task-oriented content
- **Openness**: more diverse topics, more original content, larger networks

## Putting It All Together: The Deep Dossier

At high fidelity, compile a multi-layer profile:

```
PSYCHOMETRIC PROFILE: @handle
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Big Five: O[HIGH] C[MED] E[HIGH] A[LOW] N[LOW]
  Evidence: {real quotes showing each trait}

Moral Foundations: Care★★ Fair★★★ Loyal★ Auth★ Sanct★ Liberty★★★
  Evidence: {what they get angry/excited about}

Values: Self-Direction dominant, Achievement secondary
  Evidence: {how they justify their positions}

Cognitive Style: HIGH integrative complexity
  Evidence: {hedging patterns, nuanced takes, sentence complexity}

Dominant Frame: Attribution of Responsibility
  Evidence: {they consistently focus on who's to blame}

Behavioral: Night owl, reply-heavy, low emoji, threads > one-shots
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

This multi-layer profile makes predictions much more nuanced than
Big Five alone. It tells you not just WHAT someone will say but
WHY they'll say it and HOW they'll frame it.
