---
name: autoreason
description: Self-refine LLM outputs via a 3-version tournament.
version: 1.1.0
author: Xinyu Du (@Starfie1d1272), SHL0MS
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [self-refinement, iterative-improvement, llm-optimization, borda-count, convergence, autoreason]
    category: autonomous-ai-agents
    homepage: https://github.com/NousResearch/autoreason
---

# Autoreason Skill

Autoreason is a self-refinement pipeline that iteratively improves LLM outputs via a 3-version competition: the unchanged incumbent (A), an adversarial revision (B), and a synthesis (AB), judged by fresh agents with no shared context via blind Borda count. "Do nothing" is always a first-class option — the pipeline stops when the incumbent wins twice in a row.

It fixes the three structural failures of naive critique-and-revise: *prompt bias* (models hallucinate flaws when asked to critique), *scope creep* (outputs expand unchecked each pass), and *lack of restraint* (models never say "no changes needed").

## When to Use

Load `/skill autoreason` when the user asks you to:

- "Run autoreason on this task: [task]"
- "Apply the self-refinement pipeline to improve this output"
- "Use the three-version tournament to refine this proposal"

Do **not** use for: simple Q&A (a single pass suffices), high-frequency batch work, or models already near their capability ceiling.

## Prerequisites

- Python 3.11+
- `litellm` installed (`pip install litellm` or `uv pip install litellm`)
- An API key for the target model (e.g. `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`)

## How to Run

Run the bundled CLI via the `terminal` tool:

```
terminal(command="python scripts/run_autoreason.py --task 'Your task prompt here'")
```

Options: `--model` (default `anthropic/claude-sonnet-4-20250514`), `--judges` (default 3; 7 converges ~3x faster but costs more), `--max-passes` (default 30), `--output`. Results land in `autoreason_output/` with per-pass logs, `final_output.md`, and `history.json`.

## Quick Reference

- Flow: Generate A → Critic (fresh agent, problems only) → Author B (revision) → Synthesizer (AB) → 3+ fresh judges via blind Borda count → winner becomes new A
- Convergence: A wins 2 consecutive passes
- Tiebreak: A wins ties (conservative — prefer the known-good incumbent)
- Defaults: author temp 0.8, judge temp 0.3, 3 judges, max 30 passes

## Procedure

1. Get the task prompt and choose a model.
2. Run the pipeline with `terminal`, e.g. `python scripts/run_autoreason.py --task '...' --model anthropic/claude-sonnet-4-20250514`.
3. Inspect `autoreason_output/history.json` for the winner trajectory and word count per pass.
4. Read `autoreason_output/final_output.md` and deliver it to the user.

If the CLI is unavailable (no `litellm` or no API key), run the pipeline manually with native tools: spawn fresh Critic, Author B, Synthesizer, and Judge agents via `delegate_task` (each with no shared context), then aggregate rankings by Borda count. Prompt templates are below.

## Prompt Templates

### Author System
```
You are a senior consultant producing professional deliverables.
Be specific, concrete, and practical. Avoid generic advice.
Tailor everything to the constraints stated in the task.
```

### Critic System
```
You are a critical reviewer. Your only job is to find real problems.
Be specific and concrete. Do not suggest fixes.
```

### Author B (Revision) System
```
You are a senior consultant revising a proposal based on specific criticisms.
Address each valid criticism directly. Do not make changes that aren't
motivated by an identified problem.
```

### Synthesizer System
```
You are a senior consultant. You are given two versions as equal inputs.
Take the strongest elements from each and produce a coherent synthesis.
This is not a compromise — pick the best answer per dimension.
```

### Judge System
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

## Pitfalls

- **Token cost**: Each pass = 6–10 LLM calls. Accumulates fast on long texts.
- **Weak judge model = noisy results**: Use the same or stronger model as the author.
- **Judge temperature**: Keep at 0.3. Higher temps add ranking noise.
- **Critic must not suggest fixes**: Fixes contaminate B's independence.
- **Synthesizer input order is randomized**: Avoids position bias; the CLI shuffles which input is vx vs vy.
- **Not for simple Q&A**: Single pass suffices for trivial tasks.
- **Code domain uses a different flow**: For code tasks, use test-feedback analysis on failure, not 3-version tournaments.

## Verification

- A run converges: stdout prints "Converged at pass N (A won 2x consecutively)".
- `autoreason_output/final_output.md` exists and differs from `initial_a.md` only when B/AB won.
- `history.json` has one entry per pass with winner, scores, and word count.

## References

- Paper: [NousResearch/autoreason](https://github.com/NousResearch/autoreason)
- Experimental results: see `references/results.md` in this skill
