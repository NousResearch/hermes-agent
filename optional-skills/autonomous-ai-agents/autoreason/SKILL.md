---
name: autoreason
description: Autoreason: Self-Refinement That Knows When to Stop — iterative LLM output improvement via 3-version competition (A/B/AB) with blind Borda judging and automatic convergence detection. Based on NousResearch/autoreason by SHL0MS.
version: 1.0.0
author: SHL0MS, enhanced by Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [self-refinement, iterative-improvement, llm-optimization, borda-count, convergence, autoreason]
    category: autonomous-ai-agents
    homepage: https://github.com/NousResearch/autoreason
---

# Autoreason: Self-Refinement That Knows When to Stop

Iterative self-refinement fails for three structural reasons: *prompt bias* (models hallucinate flaws when asked to critique), *scope creep* (outputs expand unchecked each pass), and *lack of restraint* (models never say "no changes needed"). Autoreason fixes all three.

Each iteration produces three competing versions — the **unchanged incumbent (A)**, an **adversarial revision (B)**, and a **synthesis (AB)** — judged by fresh agents with no shared context via blind Borda count. "Do nothing" is always a first-class option.

## Flow

```
Task Prompt → Incumbent A (initial generation)
                   ↓
         Critic (fresh agent)   →   Critique (problems only, no fixes)
         Author B (fresh)       →   Revision (B): address each critique
         Synthesizer (fresh)    →   Synthesis (AB): best of A+B per dimension
                   ↓
         Judge Panel (3+ fresh agents, blind Borda count)
                   ↓
              Winner → new A   (if A wins 2 consecutive → converge)
```

## When to load this skill

Load `/skill autoreason` when the user asks you to:

- "Run autoreason on this task: [task]"
- "Apply the self-refinement pipeline to improve this output"
- "Use the three-version tournament to refine this proposal"
- "Do autoreason self-refinement on this"

## Usage instructions

When loaded, apply the autoreason pipeline by calling `scripts/run_autoreason.py`:

```bash
python path/to/autoreason/experiments/v2/run_overnight.py  # if inside the autoreason repo
```

Or run the pipeline manually by following the flow above with these prompt templates:

### Author System Prompt
```
You are a senior consultant producing professional deliverables.
Be specific, concrete, and practical. Avoid generic advice.
Tailor everything to the constraints stated in the task.
```

### Critic System Prompt
```
You are a critical reviewer. Your only job is to find real problems.
Be specific and concrete. Do not suggest fixes.
```

### Author B (Revision) System Prompt
```
You are a senior consultant revising a proposal based on specific criticisms.
Address each valid criticism directly. Do not make changes that aren't
motivated by an identified problem.
```

### Synthesizer System Prompt
```
You are a senior consultant. You are given two versions as equal inputs.
Take the strongest elements from each and produce a coherent synthesis.
This is not a compromise — pick the best answer per dimension.
```

### Judge System Prompt
```
You are an independent evaluator. You have no authorship stake in any
version. Evaluate which version best accomplishes the original task.
```

### Judge Ranking Prompt (3 versions)
```
ORIGINAL TASK:
---
{task_prompt}
---

Three proposals have been produced independently. Evaluate how well each accomplishes the stated task.

{judge_proposals}

For each proposal, state what it gets right and what it gets wrong.
Then rank all three from best to worst:

RANKING: [best], [second], [worst]

Where each slot is 1, 2, or 3.
```

## Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Author temperature | 0.8 | Higher for creative generation |
| Judge temperature | 0.3 | Lower for consistent evaluation |
| Number of judges | 3 | 7 converges 3× faster but costs more |
| Convergence | 2 consecutive A wins | A = "do nothing" wins twice → done |
| Max passes | 30 | Safety limit |

## Key design decisions

1. **Tiebreak: A (incumbent) wins ties** — deliberate. Without this, "tinker unnecessarily" beats "keep the known good."
2. **Critic must not suggest fixes** — doing so contaminates B's independence.
3. **Synthesizer input order is randomized** — avoids position bias. Shuffle which is vx vs vy.
4. **Each judge is a fresh agent** — zero shared context between judges and authors.
5. **Labels are shuffled per judge** — no judge sees A/B/AB in the same order.

## Known results

| Finding | Detail |
|---------|--------|
| **42/42 perfect sweep** | Haiku 3.5 + autoreason scored perfect Borda across 5 tasks; all baselines degraded below single-pass |
| **77% vs 73%** | Sonnet 4.6 on 150 CodeContests problems (private-test), autoreason vs single-pass |
| **40% vs 31%** | Haiku 3.5 autoreason vs best-of-6 at matched compute |
| **Haiku 4.5: transition point** | At 60% private accuracy, autoreason's held-out gains vanish — generation-evaluation gap closed |
| **7 judges → 3× faster convergence** | Than 3 judges; 1 judge is noisy and slow |
| **Both B and AB necessary** | Removing either collapses the tournament (converges in 2–3 passes vs 24) |
| **Length-controlled: 21/28 wins** | Autoreason beats 3 of 4 baselines even at matched word count |
| **Refinement destroys weak models** | Critique-and-revise reduced Haiku 3.5 outputs by 59–70% over 15 passes |

## Pitfalls

- **Token cost**: Each pass = 6–10 LLM calls. Accumulates fast on long texts.
- **Weak judge model = noisy results**: Use the same or stronger model as the author.
- **Not for simple Q&A**: Single pass suffices for trivial tasks.
- **Code domain uses a different flow**: For code tasks, use test-feedback analysis on failure, not 3-version tournaments.

## References

- Paper: [NousResearch/autoreason](https://github.com/NousResearch/autoreason)
- Paper PDF: [autoreason.pdf](https://github.com/NousResearch/autoreason/blob/main/paper/autoreason.pdf)
- Experiment code: `experiments/v2/run_overnight.py` (writing tasks), `experiments/v2/run_code_overnight.py` (code tasks)
- Ablations: `experiments/v2/run_ablations.py` (judge count, aggregation, component)
- Statistics: `experiments/v2/compute_stats.py` (bootstrap CIs, McNemar tests)
