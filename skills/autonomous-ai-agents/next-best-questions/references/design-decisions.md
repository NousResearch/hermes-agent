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
- **A/B verdict (2026-07-02): #26 CLOSED — KEEP STATED (powered null).** Powered 12-prompt
  REALIZED_SUBSET A/B, all-`fast` pinned (judge/elicit/gen), 488 rows / 244 shared realized pairs:
  on the PRIMARY target `realized_regret`, Δρ mean **+0.010** (sd 0.39, se 0.11), sampled wins only
  4/12 — nowhere near the broad-win gate. Secondary `realized_stakes` +0.041 (7/12) also
  indecisive. The null is REAL, not a no-contrast artifact: the arms disagreed on 79% of pairs
  (|ΔP| > 0.05) and shifted q_value on 76% — a materially different P estimate that produced the
  same within-task discrimination. Conclusion mirrors #24: the binding weakness of within-task
  ranking is NOT input miscalibration of P(a); BED-LLM's calibration gains don't transfer to this
  utility-weighted, coarse-realized-target setting. Sampled stays built + off as a documented
  negative result. **Bonus replication:** the same run's p1c ablation ranks `√(U·EVSI)` best on ALL
  three realized targets (regret +0.356 ≈ the prior +0.360; stakes +0.266; change +0.325) — a
  second independent within-task validation of the frozen formula.
- **Re-confirmed under deepseek (2026-07-03):** same gate with deepseek elicitation + judging —
  Δρ +0.058, wins 5/12 → still keep `stated`. The null is instrument-robust (findings §Deepseek
  re-adjudication).

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
- **A/B verdict (2026-07-02): #27 CLOSED — KEEP ABSOLUTE (powered, decisively worse).** Powered
  12-prompt REALIZED_SUBSET A/B, all-`fast` pinned, stated P (#26 closed first), 432 rows / 216
  shared pairs: solution Δplan within-task ρ on `realized_regret` is **−0.047 vs absolute +0.360**
  (paired Δρ **−0.343**, loses 7/10) and negative on stakes too (−0.169). The accepted-caveat
  failure mode is exactly what happened: the anticipated collapse — **69% of solution deltas are
  exactly 0** ("no solution invalidated") and most of the rest land at 1.0 (support ≈ {0: 149,
  1: 45}) — flattens Δplan to near-binary and destroys within-task discrimination, even though the
  arms diverged on 90% of pairs. Active Task Disambiguation's solution-space scoring does not
  survive contact with a small sampled solution set + a strict viability judge in this domain. The
  absolute 0-1 judge, for all its known fragility, carries strictly more within-task signal.
  Solution stays built + off as a documented negative result; do NOT iterate on K or judge leniency
  without a new hypothesis for the mass-at-zero problem.
- **Re-confirmed under ALL-deepseek (2026-07-03) — the "fast was too weak" rescue is falsified:**
  with deepseek doing solution sampling, viability judging AND realized judging, granularity
  partially recovers (mass-at-0 69%→53%, support fills {0,¼,½,¾,1}) and the method is WORSE:
  Δρ −0.369, wins 1/10. The collapse is inherent to the K-solutions framing, model-robustly
  (findings §Deepseek re-adjudication).

## Derive-or-ask — derivable questions become evidence, tested not trusted (1.3.0, 2026-07-03)

Jim's Bayesian reframe of the gate: the projected answer distribution is the model's posterior.
Peaked + derivable → **not a valid question — it's evidence wearing a question mark**; spread +
underivable → a valid question whose weight is the expected impact of resolving genuine ignorance.
Pre-1.3.0 the skill only *suppressed* the first class (`derivable→U→0→SKIP`): the answer was never
actually produced, never became evidence, and a *wrongly-inflated* derivability claim silently
killed a real question. Now every claim ≥ `derive_threshold` is **tested by an actual derivation
attempt** (`pipeline.attempt_derivation`): success → the answer is **tombstoned into the working
evidence** (the investigator's tombstone pattern; refill rounds re-plan against it and the
question retires naturally) and reported under "Resolved during analysis"; CANNOT_DERIVE → the
claim failed its experiment, `derivable_prob` is capped at `cannot_derive_cap` and the question
re-scored — it re-enters ranking honestly. A final-round derivation grants one bounded extra
refill round so fresh tombstones get exploited.

- **The derive prompt is knowledge-INCLUSIVE ("…or your own general knowledge") — load-bearing.**
  `derivable_prob`'s semantics are "asking adds nothing", which covers prompt facts, established
  facts, and parametric knowledge. Probe (2026-07-03): 22% of bank candidates claim derivable
  ≥0.8, largely knowledge-class (all 9 on research-ratelimit); a strict "from the prompt alone"
  wording makes those FAIL derivation, and the correction branch would re-inflate their U and
  flood buckets with questions the gate retires correctly today. Pinned by a test.
- **Derive model = value_judge_model (deepseek), not `fast`:** same probe — fast is over-strict on
  knowledge questions (CANNOT_DERIVE where deepseek correctly answers). Fabrication risk measured
  LOW: 0/12 fabrications across both models on user-only/tool-only questions (the escape works);
  visible provenance in the report is the remaining guard.
- **Hedges count as failed derivations** (`pipeline._HEDGE_RE`): the do-no-harm check caught fast
  tombstoning "The prompt does not specify whether …" as if it were an answer — a non-answer
  wearing an answer's clothes, i.e. junk evidence. Hedge phrasings (does not specify / not
  specified / insufficient information / …) now route to the CANNOT_DERIVE branch. Pinned in
  tests.
- **Default ON for the CLI; absent-key OFF** (the families/#26 byte-identical convention) so
  harness-built cfgs and the six powered eval datasets stay comparable. Rollback:
  `INFOGAIN_AUTO_DERIVE=off`. Not a #26 re-open: #26 tested *replacing* stated P(a) elicitation
  for ranking; this *reconciles* one stated input against a performed computation.
- **Known residual:** when the bucket early-stops in the same round as a derivation, that round's
  bucket was projected pre-tombstone (no later round re-plans). Bounded: any bucket member with a
  claim ≥ threshold was itself derive-tested pre-selection, so stale *derivable* stragglers can
  only exist below threshold. The report wording reflects this ("refill rounds re-plan against
  it").

## Behavior-Δ judge (#28) — built, off by default, the first OBJECTIVELY-gated experiment

The objective-outcome tier exposed the absolute judge's volume bias (Δplan read as text change:
one-token/output-flipping questions gated, boilerplate promoted — findings §Objective-outcome).
`value_judge_mode="behavior"` re-elicits Δplan as **behavior/outcome change of the delivered
result** ("consequence, not code size"; one-token flip ~1.0, robustness boilerplate ≤0.2) — same
JSON contract, `voi.py` untouched. **Gate verdict (2026-07-03): NO ADOPT** — paired vs absolute
on 28 objective tasks: +0.064 with 6W/5L (broad-win guard requires wins ≥ 2×losses); unanswerable
65% vs the ~60% bar; proxy sanity passed (ρ +0.204). Directionally right — on the agentic tier it
nearly tripled the skill's end-to-end benefit — but zeroshot still out-asked it by +0.157, so the
value model is only part of P4's gap; the remainder is generation altitude (successor hypothesis:
first-order-question exposure at GENERATION, not judging). Stays built for re-testing; the
objective harness is now the standing gate for any elicitation change.

## Reach lens (#29) — a fifth question family: reachable other points of view

Jim's framing: vantage asks "would the answer differ by where you look from?"; reach asks
**"does a DIFFERENT, REACHABLE point of view exist that would turn an unknown into an
observable — possibly via CHAINED hops (machine → machine → service)?"** It hunts the
answerability/derivability term no other lens hunts, and connects to derive-or-ask (derivability
is vantage-relative: CANNOT_DERIVE from here ≠ from inside the container). The ranker only
SURFACES reach questions; the investigator executes any actual hop. Directive requires naming
the hop chain, the access each hop needs, what the final vantage reveals, and the hop's
cost/risk (ASPI, arXiv:2605.17324: every hop widens the injection/trust surface).
- Mechanics mirror premortem exactly (one `_LENS_DIRECTIVE` entry + a gated `families_prompt`
  line + auto knob); the auto gate IS the vantage gate (`_reach_relevant = _vantage_relevant`) —
  same systems/access surface, no new false-positive list. Default `auto`, like vantage.
- **Tier-1 verdict (2026-07-03): PASS.** Two-arm scan over the 11 gate-firing bank prompts:
  reach questions survive buckets exactly on access/systems tasks (debug-slow 0.63,
  security-audit 0.66, fix-test 0.61, setup-ci 0.52, gmail-find 0.47/0.58 — on-mission for
  retrieval-with-access) and prune to ZERO on whatsapp-send / research-* / query-db / deploy-app.
  The gate proposes, the formula disposes.
- **Tier-2 verdict (same day): PASS — auto-on stands.** Realized per-lens attribution (n=6
  systems prompts, deepseek judge, `evsi_reach_t2.json`): reach realized_regret **0.351** vs
  vantage 0.362 (n_q=18 each), realized_change 0.584 vs 0.566 — squarely in vantage's band, the
  pre-registered "adds signal, not noise" bar for an exposure lens. Rollback unchanged: set
  `FAMILIES["reach"]="off"` if read-only pollution or diversity harm ever shows.

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
  Since 1.2.1 both auto-gates match the **raw problem text as well as the framing** — framing is
  model-paraphrased, so a hint verb in the prompt could vanish before the gate saw it (the
  post-rename "migrate prod DB" miss). Hint lists unchanged (verb-only; nouns re-open the
  gmail-triage false positive). Precision re-verified post-change: scored families scan over all
  13 bank+LIFE prompts whose raw text trips a gate — premortem questions survive into buckets
  ONLY on the genuine failure-surface tasks (deploy-app 0.51/0.65, setup-ci 0.60); the doc-writing
  misfire class ("write a brief/plan": write-brief 0.52/0.28/0.0, gtm-plan all 0.0) generated
  candidates but zero survived scoring. No worse than the #25 17/18-pruned baseline.
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

## First-order candidate source (#32) — built, off by default, the first COST-AWARE-gated experiment

The #28 successor hypothesis: the residual P4 gap over the naive `zeroshot` arm is **generation
altitude**, not the value judge — so inject one naive "K best clarifying questions" call's output
as round-1 candidates (family `First-order semantics`, lens `firstorder`), scored by the frozen
pipeline. `pipeline.firstorder_questions` (one `raw_chat` + numbered-parse → tagged dicts) merges
into round 1 when `families["firstorder"] != "off"`; formula untouched. Selector `--firstorder
on|off` / `INFOGAIN_FIRSTORDER`; default `"off"` (absent-key convention — harness cfgs
byte-identical). This is also the first experiment gated on **efficiency alongside Δresult**: every
arm now reports mean wall/tokens/calls, and the adopt rule carries a cost ceiling.

**Gate (2026-07-04, objective outcome harness, n=34 = 20 micro + 14 agentic, K=3, all-deepseek,
`--max-rounds 1`, `--strict-preflight`; raw `~/.hermes/outcome_eval_32.json`):**

| arm | pass | Δ vs baseline | wins/losses | unanswerable | wall | tokens | calls |
|---|---|---|---|---|---|---|---|
| baseline | 0.460 | — | — | — | — | — | — |
| nbq | 0.543 | +0.083 | 9/4 | 74% | 25.0s | 20221 | 38.3 |
| nbq-firstorder | 0.592 | +0.132 | 7/3 | 78% | 29.2s | 23944 | 45.4 |
| nbq-firstorder-behavior | 0.574 | +0.114 | 9/5 | 63% | 29.7s | 24892 | 45.1 |
| zeroshot | 0.734 | +0.274 | 15/1 (p=0.0005) | 31% | 5.9s | 154 | 1 |

**Verdict: NO ADOPT** — the pre-registered rule requires ALL of: paired nbq-firstorder vs nbq
Δpass > 0 with wins ≥ 2×losses; unanswerable ≤ 50%; no lens-payoff regression; and mean added
wall ≤ 10% of an nbq run. Mechanically, on the paired nbq-firstorder-vs-nbq comparison:
- Δpass **+0.049 > 0** ✓ but **6W/6L/22-tie** — the broad-win guard (wins ≥ 2×losses) **fails**;
  the mean lift is a few-task effect, not a broad win.
- **Unanswerable 77% ≫ 50%** ✗ — the naive first-order questions *fish more*, so the strict
  simulator resolves fewer of them (the P4 failure mode itself, now injected earlier).
- **Lens-payoff regression** ✗ — `log-clean` (irreversible/keep-newest class) fell 0.67→0.33 even
  as `json-migrate` rose 0.25→0.75; net wash on the lens tier.
- **Efficiency +16.8% wall** (+4.2s), +18% tokens, +7 calls ✗ — busts the 10% ceiling.

**Diagnosis (the product of this branch):** generation altitude *does* move the mean (nbq-firstorder
+0.132 vs nbq +0.083 vs baseline), confirming first-order exposure has signal — but injecting it
into the EVSI pipeline **does not close the P4 gap**: `zeroshot` still dominates decisively
(+0.274, 15W/1L, p=0.0005) at ~1/5 the wall and ~1/150 the tokens. The pipeline's answerability
handling is the remaining gap, not candidate altitude: first-order candidates enter but raise the
unanswerable rate rather than lowering it. This **triggers #30's re-open condition** (answerability
weighting, previously gated on "post-#32 IF unanswerable > 50%" — now 77%), with the standing caveat
that the mechanism must NOT be self-rated. Stays built, off-by-default, for re-testing; the cost
ceiling now travels with every future elicitation/generation gate.

## Discrimination preflight (#33) — instrument adopted (audit A7)

`validate_evsi.discrimination_preflight`: 8 static forced-choice fixtures (one-token semantic flips
+ cosmetic rewording) with correct answers fixed by construction; score < 6/8 → exit 2. Closes the
gap the old emptiness preflight left open — "a judge that answers but judges randomly still passes."
Opt-in via `--strict-preflight` (8 calls/model). **Live check (2026-07-04): PASS** — `fast` 8/8,
`deepseek` 8/8; both eval-duty models discriminate perfectly. No elicitation change, so no outcome
gate; adopted as a standing (opt-in) instrument.

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
