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
> research, the Phase-2 trigger). So **`U` stays.** The domain looked like it broke the absolute thresholds
> (61% of agentic candidates fall below the life-tuned 0.40) — but the later realized-improvement scan
> (`evsi-validation-findings.md` §Stop + breadth calibration) showed most of those 61% are *genuinely*
> low-value (realized improvement ~0.15 below value ~0.30), so the fix was **calibrating the absolute
> floor (0.40 → 0.30)**, not going relative. The rank-relative mechanism (`rel_keep_frac`) is built but
> stays **off**.
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

## Comparative elicitation (#24) — built, off by default, A/B-gated

The one *measured* weakness is **within-task ranking** (per-prompt Spearman ρ≈0.34): given one task's
candidate questions, the top-ranked isn't reliably the most valuable. The likely cause is the same
fragility that collapsed the realized-stakes instrument — **absolute 0-1 Δ/stakes elicitation**, which
models do poorly. The fix is **comparative elicitation**: ask forced-choice comparisons ("which answer
changes the response more?", which models do well) and aggregate them, instead of scoring each answer
in isolation.

This is built as an **off-by-default, A/B-gated experiment** so it can only ever *help*, never regress
the live skill:

- **`scripts/pairwise.py`** (pure, tested) — Bradley-Terry MLE (phantom-regularized) + win-count
  fallback + anchored [0,1] mapping. The subtle part is preserving **between-task** scale (the
  validated ρ≈0.66): two virtual ANCHOR items — `FLOOR` ("no change") → 0 and `CEILING` ("completely
  different") → 1 — sit in *every* question's comparison set, so a question whose answers merely tie
  FLOOR lands near 0 (low EVSI) while a high-impact question's answers land high. Pairwise fixes the
  within-question ordering without flattening cross-question magnitude.
- **`pipeline.judge_plan_change_pairwise[_batch]`** — same contract as the absolute judge (writes the
  same per-answer `delta_plan`/`stakes` that `voi.evsi`/`score_record` read), so it's a drop-in. Two
  model calls/question (change, stakes); safe-zeroes on any parse failure.
- **Selector** — `value_judge_mode` ("absolute" | "pairwise", default **"absolute"**), special-cased
  like `--mode`; absent key → "absolute", so every cfg built from DEFAULTS is byte-identical. One call
  site branches (`infogain.run`).
- **The gate** — `validate_evsi --ab` scores BOTH methods on the SAME question/answer set (realized
  measured once, shared); `analyze_evsi` prints each method's within-task mean ρ. **Pairwise becomes
  the default ONLY if it measurably beats absolute** (Δρ > 0.02 on realized_change / realized_regret);
  otherwise absolute (validated) is untouched.
- **A/B verdict (2026-06): #24 CLOSED — KEEP ABSOLUTE (powered null).** Powered 12-prompt A/B on the
  corrected metric (gate ranks on **`realized_regret`** = realized EVSI, with a per-prompt paired-Δρ
  broad-win guard): pairwise is **slightly worse** on every realized target (regret abs +0.360 vs pw
  +0.204; loses 9/12 prompts) — comparative elicitation does **not** help projected Δ/stakes. Pairwise
  stays built + off as a documented negative result; the de-saturated realized judge is **NOT built**
  (pointless — pairwise doesn't even help on projected). *The n=6 sub-narratives ("realized_change is
  within-task-dead", "pairwise +0.07 edge", "saturation", "stakes is the unique signal") were all
  SMALL-SAMPLE NOISE* — at n=12 realized_change is +0.30 and pairwise is −0.02; the binding limit was
  power, exactly as the adversarial check predicted. **Strong positive:** the `p1c` ablation vs
  realized_regret ranks `√(U·EVSI)` **best (+0.360)** above every component (U-only +0.264, EVSI-only
  +0.202, stakes-only +0.157, max-Δ +0.075) → within-task ranking is modest-but-real (ρ≈0.36) and the
  frozen formula is validated within-task, not just between-regime. See `evsi-validation-findings.md`
  §"Comparative elicitation (#24)".

## Sampled P(a) (#26) — built, off by default, A/B-gated

The 2024-26 literature converges on one critique of rankers like ours: the load-bearing numbers are
**LLM self-reported**, and LLM probabilities are poorly calibrated — BED-LLM (arXiv:2508.21184) shows
Monte-Carlo EIG from *sampled* answer rollouts materially beats LLM-scored/entropy proxies; OPEN
(arXiv:2403.05534) offloads probability estimation from the LLM for the same reason. That maps
directly onto our one measured weakness (within-task ranking, ρ≈0.34-0.36) — so #26 replaces the
projection call's stated `P(a)` with an **empirical forced-choice frequency**, gated exactly like #24:

- **Cheap hybrid, not free-form rollouts.** The projection call is unchanged (it still enumerates the
  answer support + `derivable_prob`). Then N (default 6) tiny forced-choice draws — options shuffled
  per draw to kill position bias, temperature 1.0, ~16 output tokens, "reply with the option number"
  — and Laplace-smoothed (α=0.5) frequencies become `P(a)`. This tests the calibration claim at ~1/7
  the cost of free-form sampling and avoids fragile free-text→option mapping on local models.
- **Stated probs survive** as `stated_prob` (the control arm and the fallback: < ⌈N/2⌉ parseable
  draws → keep stated, tag `prob_mode_used="stated-fallback"`). `voi.py` reads the same `prob` field
  — the frozen formula is untouched; only the input estimate changes.
- **Selector** — `answer_prob_mode` ("stated" | "sampled", default **"stated"**), special-cased like
  `value_judge_mode`; absent key → "stated", so every cfg built from DEFAULTS is byte-identical.
  Knobs `answer_samples` / `answer_sample_temperature` in DEFAULTS. One branch at the projection
  seam (`infogain.run`).
- **The gate** — `validate_evsi --ab-probs`: the run samples, then the SAME records are re-scored
  under stated P (swap `prob`↔`stated_prob` + `voi.score_record` — zero extra model calls); realized
  is measured once over the union of each arm's top-N answers. `analyze_evsi`'s generalized A/B gate
  (control = `stated`) applies the #24 decisive rule: adopt only on a broad, beyond-noise Δρ > 0.02
  win on `realized_regret` at n=12.
- Smoke-verified: the arms genuinely diverge (11/14 pairs shift P by >0.05, 9/14 shift q_value).

## Solution-space Δplan (#27) — built, off by default, A/B-gated

Second frontier critique (Active Task Disambiguation, arXiv:2502.04485; ClarifyGPT): score questions
by how they **split the viable solution set**, not by an abstract "how much would your response
change, 0-1?" judgment. #27 grounds `delta_plan` in a concrete self-consistency set:

- **Stage 0b, once per run** — `pipeline.sample_solutions`: K (default 4) candidate responses;
  solution 1 is the existing `baseline_plan` (free), K−1 sampled at temperature 0.8. Reused across
  ALL questions and rounds → +K−1 = 3 calls per run, the cheapest experiment yet.
- **Stage 3 variant** — `pipeline.judge_plan_change_solution[_batch]`: for each projected answer the
  judge returns which of the K numbered solutions remain viable if it's true; `delta_plan =
  invalidated/K` (plus `viable_solutions` as a diagnostic). Stakes elicitation unchanged. Same
  output fields → drop-in for `voi.score_record`, exactly like the pairwise judge; safe-zeroes on
  parse failure.
- **Selector** — `value_judge_mode="solution"` (a third choice on the #24 seam); knobs
  `solution_samples` / `solution_temperature`. `infogain.run` samples once after framing and binds
  the solution set via `functools.partial`.
- **Accepted caveat** — Δplan quantizes to {0, 1/K, …, 1} and a collapsed solution set (K
  near-identical responses) pushes it toward 0/1. Inherent to the framing; the gate decides whether
  grounding beats granularity.
- **The gate** — `validate_evsi --ab-solution` (re-judge the same records, realized shared;
  `--answer-prob-mode sampled` pins the #26 winner if adopted), same decisive rule, control =
  `absolute`.

## Pre-mortem lens (#25) — a fourth question family, auto-on by design

The three existing lenses cover coverage (scoped), premise (contrarian), and source-divergence
(vantage) — but mapped onto `EVSI = Σ P·Δplan·stakes`, **none systematically hunts the `stakes`
term**: the catastrophic/irreversible tail where getting it wrong is expensive. The **pre-mortem lens**
fills that gap: a family whose questions assume the baseline plan *shipped and failed in production* and
hunt the latent hazard (data loss, security compromise, irreversible/destructive actions, silent wrong
output, runaway cost).

- **It is the generation-side, formula-FROZEN half of the deferred "risk-averse tilt."** The tilt
  (reweighting an improbable-but-catastrophic branch) is a *scoring* change → still deferred. The lens
  only ensures the catastrophic-tail question **enters the candidate set** so it can be scored
  risk-neutrally. No formula change; a lurid-but-improbable question still self-prunes on low P.
- **Auto-on by design** (like vantage was), gated by `pipeline._premortem_relevant` — a *conservative*
  failure-surface keyword gate (writes/deploys/payments/migrations/secrets), so read-only
  summarize/research tasks are untouched. Force with `--premortem on|off|auto` / `INFOGAIN_PREMORTEM`.
- **Mechanically:** one `_LENS_DIRECTIVE["premortem"]` entry + one `families_prompt` branch + the gate.
  Nothing downstream branches on lens (scoring/MMR/dedup are lens-agnostic) — every question still scores
  on its own merit; the lens is pure domain *exposure*. Chosen over success-criteria (overlaps the
  stage-0 `success_criteria` framing field), stakeholder (niche; `audience` is already a question
  `type`), and reversibility (folds into the pre-mortem directive) — see the `[[information-gain-skill]]`
  plan for the graded comparison.
- **Do-no-harm posture:** auto-on, but the eval ladder (`score_scan.py` / `validate_evsi.py` two-arm
  premortem off-vs-on, rows now tagged with `lens`/`family`) *confirms* it earns its place — it should
  add distinct realized-valuable questions on failure-surface tasks and stay quiet on read-only
  controls. **Rollback trigger:** if it adds low-value noise on read-only prompts or drops adjudicator
  `diversity`, downgrade `FAMILIES["premortem"]` to `"off"` (one-line change). The measurability caveat
  (absolute realized-stakes collapses, §Comparative elicitation) means the primary evidence is the
  **failure-surface-vs-read-only differential**, not absolute stakes.
- **Tier-1 verdict (2026-07-01, 14-cell two-arm at shipped defaults, `max_rounds=1`): auto-on
  CONFIRMED; rollback trigger NOT tripped.** Failure-surface arms: on true act-and-break tasks
  (security-audit, deploy-app) the lens's 3 questions all cleared the 0.30 floor (0.42–0.70) and 3
  displaced weaker questions into the capped bucket (rollback strategy, pending schema migrations,
  lockout thresholds) — content with **zero** equivalents in any off arm; on lower-hazard tasks
  (add-auth, query-db) its questions scored 0.03–0.22 and self-pruned. Read-only controls: **no
  pre-mortem question entered any bucket even forced `--premortem on`** (values 0.0–0.12) — the
  risk-neutral scoring is a sufficient second net. Cost of the lens on gated runs: ~+6 calls / +30 s.
  One fix fell out: the gate had bare artifact nouns (`email`/`message`/`database`) that tripped on
  *retrieval* tasks (gmail-triage fired, wasting ~6 calls, though scoring still pruned everything) —
  nouns removed, verbs kept; pinned in `test_premortem_lens_directive_and_gate`. Raw runs:
  `premortem_ab.json` (14 cells; job tmp — regenerate via `score_scan.py --families --premortem on|off`
  or `validate_evsi.py --families` for the realized tier-2 arm.)
- **Tier-2 verdict (2026-07-01, realized two-arm, 6 prompts × off/on, all-fast pinned): the lens
  EARNS its slots at the realized level.** Premortem is the TOP lens by realized_regret in the on
  arm (0.416 vs scoped 0.297 / contrarian 0.240 / vantage 0.253); on failure-surface prompts its
  questions realize **0.602 vs 0.386** for everything else (~1.6×), while forced-on read-only pm
  questions realize ~0.045 and are correctly priced at ~0.06 (pruned). Both ladder tiers now
  confirm auto-on; rollback untripped. Full numbers:
  `evsi-validation-findings.md` §"Pre-mortem lens tier-2 (#25) + selection policies (#23)".
- **Independently replicated same day** (deepseek judge, bucket source, different 6-prompt subset,
  34-prompt bank-wide scan): premortem again TOP lens (realized_change 0.984, regret 0.765); zero
  read-only bucket entries; adjudicator-`diversity` trigger explicitly cleared (0.65→0.70). See
  findings §"Independent replication (#25)".

## Decided / deferred

- **Decided, keep:** one layer of projected answers (no chain) · within-round semantic consolidation
  only · `--mode focus` default behavior unchanged · report-only (never answers/asks itself) · the
  **pre-mortem lens auto-on** (#25) · the **formula stays FROZEN** (pairwise changes only *how Δ/stakes
  are elicited*, never the √(U·EVSI) form).
- **Confirmation tooling (do-no-harm):** `saturation_scan.py --scored` tests whether the HIGH-value
  signal saturates earlier than distinct-target coverage (it shouldn't keep growing) — evidence that
  breadth is bounded by value, so modest breadth + the families layer is the right coverage mechanism.
  The premortem off/on two-arm scan (rows lens-tagged) confirms the fourth lens adds distinct realized
  value on failure-surface tasks, not noise.
- **Deferred (not bundled):** making `deepseek` the default judge (the "generous judge" calibration
  fix) · the risk-averse *scoring* tilt (its generation-side half is now built as the pre-mortem lens) ·
  a success-criteria / stakeholder lens (documented as the sanctioned "add a second lens later" option) ·
  realized-pairwise stakes for the de-confounded clean floor (#21) · pushing the branch / baking into
  the image.
