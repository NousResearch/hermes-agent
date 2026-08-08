# GEPA Evolution — Automated Self-Improvement via hermes-agent-self-evolution

## What This Is

The hermes-agent-self-evolution repo (NousResearch/hermes-agent-self-evolution)
uses DSPy + GEPA (Genetic-Pareto Prompt Evolution) to automatically evolve
Hermes Agent skills. GEPA is an ICLR 2026 Oral paper — it reads EXECUTION
TRACES to understand WHY things fail, then proposes targeted mutations.

This means: we can point GEPA at the worldsim skill and automatically evolve
every component — simulation instructions, anti-slop rules, star thread
methodology, mechanical verification checklist, dossier templates — using
our own simulation outputs scored against real data as the eval signal.

The recursive self-improvement pipeline we built manually (log failures →
promote patterns → update rules) can be AUTOMATED via GEPA.

## How It Applies to WorldSim

### What GEPA Evolves (text, not weights)
GEPA evolves the TEXT of prompts and instructions. For worldsim, that means:

| Target | What Gets Evolved | Eval Signal |
|--------|------------------|-------------|
| SKILL.md | Immersion protocol, pipeline instructions | Simulation quality scores |
| star-thread.md | Methodology for finding star threads | Thread-to-voice accuracy |
| anti-slop.md | Slop word lists, structural patterns | Slop detection recall/precision |
| simulation-engine.md | Platform formats, conversation dynamics | Voice fidelity scores |
| adversarial-refinement.md | Mechanical check thresholds, GAN loop | Pre vs post refinement delta |
| prediction-engine.md | Forecasting methodology | Prediction Brier scores |
| dossier template | Profile structure and fields | Profile quality scores |

### The Eval Dataset
Built from worldsim's own outputs + real data:

1. **Voice fidelity pairs**: (simulated post, real post from same person) →
   LLM-as-judge scores similarity 0-1
2. **Mechanical check logs**: what did the checks catch? what slipped through?
3. **Prediction accuracy**: tracked predictions scored against reality
4. **Held-out tests**: predicted tweets vs actual tweets
5. **Turing test results**: could the discriminator tell real from fake?
6. **User corrections**: any time the user catches something the system missed
   (like the emoji fabrication incident — that's the richest signal)

### The GEPA Loop for WorldSim

```
1. RUN worldsim simulation (creates execution traces)
2. SCORE outputs against real data (voice, position, mechanical)
3. LOG traces + scores + user feedback to eval dataset
4. GEPA EVOLVES the skill component that had lowest scores
   - Reads traces to understand WHY it scored low
   - Proposes mutation to that specific reference file
   - Tests mutation against held-out eval data
   - If improved: create PR, human reviews
5. REPEAT — each cycle makes the skill better
```

### Concrete Example

GEPA discovers from traces that simulated conversations always have
symmetric turn-taking (4/4/4). It reads the mechanical check log that
caught this in 3 of the last 5 simulations. It reads the current
simulation-engine.md and sees the conversation architecture section.
It proposes a mutation:

OLD: "Opening Moves (1-3 posts) → Development (4-8 posts) → Peak → Resolution"
NEW: "Opening: most impulsive person posts. Others join ASYMMETRICALLY — one person
gets 40-50% of turns, one gets 15-20%, others fill the rest. The ratio should
match their real reply-to-original ratios from the dossier."

This mutation gets tested against the next 5 simulations. If symmetry
violations drop and voice scores don't decrease, it gets merged.

## Setup

```bash
# Clone the evolution repo
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes

# Evolve the worldsim skill specifically
python -m evolution.skills.evolve_skill \
    --skill hermes-simulator \
    --iterations 10 \
    --eval-source sessiondb
```

## What Makes This Different From Manual Self-Improvement

The manual pipeline (references/recursive-self-improvement.md) requires the
agent to notice its own failures and write rules. This has two problems:

1. The agent shares weights with the generator — it's biased toward
   approving its own output (the emoji incident proved this)
2. Promoting patterns to rules is slow and requires 3+ occurrences

GEPA solves both:
1. The eval signal comes from EXTERNAL data (real posts, user corrections,
   mechanical checks) — not the agent's self-assessment
2. Evolution happens per-iteration, not per-3-failures
3. Mutations are tested against held-out data before merging
4. The Pareto frontier maintains diversity — different strategies for
   different types of people/conversations

## Integration Points

### Eval Dataset Builder
Mine rehoboam DB for training data:
- simulation_logs table → execution traces
- prediction_scores table → accuracy data
- audit_log table → mechanical check results
- user correction events → highest-value signal

### Fitness Function for WorldSim
```python
def worldsim_fitness(simulation_output, real_data):
    scores = {}
    # Voice fidelity: embed real + simulated, cosine similarity
    scores["voice"] = embed_and_compare(simulation_output, real_data.tweets)
    # Mechanical pass rate: what % of checks passed without fixes
    scores["mechanical"] = mechanical_check_pass_rate(simulation_output)
    # Slop score: count of slop words/patterns detected
    scores["anti_slop"] = 1.0 - (slop_count / total_words)
    # Structure: turn asymmetry, conversation naturalness
    scores["structure"] = naturalness_score(simulation_output)
    # Textual feedback for GEPA's reflective mutation
    feedback = generate_textual_feedback(scores, simulation_output, real_data)
    return aggregate_score(scores), feedback
```

### The Key Insight: Textual Feedback
GEPA's superpower is that it doesn't just get a scalar score — it gets
TEXTUAL FEEDBACK explaining what went wrong. Our mechanical verification
system already produces this:

"@nosilverv avg 33.2 words vs real 15.6 (113% deviation) — SHORTEN"
"Parallel antithesis detected: 'The most X... The most Y...' — STRIP"
"Emoji rate 0% simulated but 10% real — OK (within tolerance)"

This text goes directly into GEPA's reflective mutation pipeline. It reads
these messages and proposes changes to the skill instructions that would
prevent these specific failures in future simulations.

## Evolution Targets by Priority

1. **simulation-engine.md** — highest impact on output quality
2. **anti-slop.md** — directly measurable, highest precision eval
3. **star-thread.md** — hardest to evaluate but most impactful on voice
4. **adversarial-refinement.md** — meta: improving the improvement system
5. **SKILL.md pipeline instructions** — orchestration optimization
6. **dossier template** — structure optimization
7. **prediction-engine.md** — measurable via Brier scores

## The Virtuous Cycle

```
More simulations → more eval data → better GEPA mutations
→ better skill instructions → better simulations → more eval data → ...
```

This is the endgame: the worldsim skill evolves itself through use.
Every simulation makes the next one better, not just through logged
rules, but through automated evolutionary optimization of the
instructions themselves. The system doesn't just learn WHAT went wrong —
it rewrites its own code to prevent it.
