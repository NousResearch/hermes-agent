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
| **exploration value** (`value`) | the number you rank by | 0–1 |

## The formula

```
exploration value = √(uncertainty × value-of-answering)
```

= `√(U × EVSI)`. Properties:
- `value` is 0 if EITHER the uncertainty gate or the EVSI is 0 (the necessary-condition gate).
- The geometric mean keeps it on an interpretable ~0–1 scale, so absolute thresholds (0.40/0.60)
  are meaningful.
- **Risk-neutral** by default (probability-weighted). A risk-averse tilt (flag a catastrophic-but-
  unlikely branch even when improbable) is a deliberate future option, not the default.

> **Tried and removed: answerability.** An `answerability × …` multiplier (P a determinate answer is
> obtainable if explored) was added and then removed after a benchmark showed it inert — pinned at
> ~0.95 in 15/16 cells and reordering the ranking in 0/15 — because clarifying questions are almost
> always answerable. It added a field + prompt complexity for no measured effect.

> **Phase-1 validation (2026-06) — `U` inert, EVSI not-yet-validated.** The realized-vs-projected
> study (`evsi-validation-findings.md`) found: (a) the **Δ component is directionally calibrated**
> (per-answer ρ=0.39, cluster p=0.005); (b) **`U` is inert** — `√(U·EVSI)` ranks identically to
> EVSI-only (0/40 within-prompt reorderings) and `U`-alone is anti-predictive → candidate for removal;
> (c) the **full stakes-weighted EVSI is not-yet-validated**: it is null against the only clean signal
> (realized response-change, ρ=−0.009), and its apparent +0.605 "validation" is a **stakes-reuse
> confound** (the realized-EVSI target recycles projected stakes; partial-ρ\|stakes = −0.13);
> (d) **max-Δ** is the best clean-signal predictor but marginal (p=0.064). *Caveat:* n=17 / 3 prompts,
> and `U`'s range is compressed (0.725–0.984), so its inertness is unproven beyond this sample.
> **`U` has two roles** — it's inert as a *ranking* factor but **load-bearing as the gate**
> (`is_gated_out`: `derivable_prob`→1 → `U`→0 retires answered questions across rounds, §"evidence
> loop"); the ablation only tested the ranking role. So a future "drop U" means removing it from the
> `value` number **only**, never from the derivability gate.
> **Domain update — `U` is NOT inert in the target domain.** A 34-prompt/17-category scan
> (`evsi-validation-findings.md` §Domain sensitivity) shows the "U inert" result was a **life-domain
> artifact**: on agentic/tool/coding tasks U's spread is **0.26 (vs 0.07)** and it is the
> **ask-the-user vs go-find-out discriminator** (via `derivable_prob` — high derivable → U→0 → route to
> research, the Phase-2 trigger). So **`U` stays.** The domain also breaks the absolute thresholds
> (61% of agentic candidates fall below the life-tuned 0.40) → rank-relative selection is now required.
> **Decision (2026-06): the formula is FROZEN** — no changes on n=17. A de-confounded, **agentic**,
> per-regime re-run (#21) that measures realized *stakes* and registers max-Δ as a competitor decides
> every formula question; the wrapper build is gated on it.

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
