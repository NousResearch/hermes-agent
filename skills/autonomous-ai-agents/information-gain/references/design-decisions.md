# Design decisions — the "explore vs disregard" model

This note records the conceptual model the skill implements and the decisions behind it, so the
rationale lives with the code. See `methodology.md` for the academic grounding (EVSI/EIG).

## The decision the skill supports

Answering questions is **expensive**, so the real per-question decision is a meta-decision:
**explore it** (spend effort to answer it) or **disregard it** (skip it, proceed on your
assumption). The skill ranks questions by how much it's worth paying to answer them, top-down, so
you explore within budget and disregard the rest (carrying their default assumptions forward).

A question has a **variety of possible answers** (and "no answer / indeterminate" is just one of
those outcomes) — it is *not* binary true/false. We evaluate one layer deep: enumerate the answers,
score the question, done. We do **not** build a 2–3 step projected chain (question→answer→question…)
— that explodes combinatorially and compounds the model's projection error. The multi-step depth
comes from the **evidence loop** instead (below), grounded on real answers, not hypotheticals.

## The one quantity that matters: value of answering = cost of disregarding

These are the same number from opposite sides:

```
value of answering a question  =  cost of disregarding it
                               =  Σ over the variety of answers:  P(answer) × regret(default plan, answer)
```

i.e., for each way the answer could come out, how much you'd regret having acted on your default —
weighted by how likely that outcome is. This is the **EVSI** (Expected Value of Sample Information).

## Vocabulary (one name per quantity)

| term | meaning | range |
|---|---|---|
| **uncertainty** (`U`) | is the answer unknown *and* reducible? `entropy(answers) × (1 − derivable_prob)` | 0–1 |
| **value of answering** (`EVSI`) | regret you'd avoid, summed over the variety of answers (`Σ P·Δplan·stakes`) | 0–1 |
| **answerability** | can it actually be resolved if you explore it? (vs. judgment-call / unknowable) | 0–1 |
| **exploration value** (`value`) | the number you rank by | 0–1 |

## The formula

```
exploration value = answerability × √(uncertainty × value-of-answering)
```

= `P(you can resolve it) × (worth if resolved)`. Properties:
- **answerability defaults to 1.0**, so if it isn't estimated the score is identical to the prior
  `√(U × EVSI)` — no threshold recalibration needed. It only ever *discounts* hard-to-resolve
  questions, sending the expensive answering budget to questions that are *both* high-regret-to-skip
  *and* actually resolvable.
- `uncertainty` and `answerability` are distinct: "I don't know it" vs "it's knowable at all." Both
  gate the value.
- **Risk-neutral** by default (probability-weighted). A risk-averse tilt (flag a catastrophic-but-
  unlikely branch even when improbable) is a deliberate future option, not the default.

## The evidence loop (how multi-step depth happens)

The skill is a **stateless, report-only primitive**. To iterate:
1. Run → get ranked questions.
2. You / the Hermes agent go answer the top ones and bring back **real evidence**.
3. Re-run with that evidence folded into the same problem context → the next-best questions.
4. Repeat until the bucket comes back empty (well-specified).

Mechanically, `--evidence` facts are woven into three stages: **framing** (the baseline plan reflects
what's known), **generation** (don't re-ask the resolved), and **answer-projection** (resolved
questions read as derivable → `U → 0` → they drop out automatically). The convergence is free: the
scoring retires answered questions and promotes the next tier. The answering and the looping live
**outside** the skill, where the caller put them.

## Decided / deferred

- **Decided, keep:** one layer of projected answers (no chain) · within-round semantic consolidation
  only · `--mode focus` default behavior unchanged · report-only (never answers/asks itself).
- **Deferred (not bundled):** making `deepseek` the default judge (the "generous judge" calibration
  fix) · risk-averse tilt · pushing the branch / baking into the image.
