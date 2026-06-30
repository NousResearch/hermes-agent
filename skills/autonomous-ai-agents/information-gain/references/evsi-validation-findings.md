# EVSI validation findings (2026-06, Phase 1 — P1a calibration + P1c ablations)

The Phase-1 test of the central question: **does a high `value` / EVSI actually predict a question
whose answer improves the response?** Verdict up front: **the Δ component is directionally calibrated,
but the full stakes-weighted EVSI is NOT-yet-validated, and the `U` factor is inert.** Reproduced
independently and stress-tested by adversarial refutation (4 claims) — see "Verification" below.
**Directional, not settled** — 51 answer-rows / 17 questions / **3 prompt clusters**.

## Setup

- **Harness:** `evals/validate_evsi.py` → rows; `evals/analyze_evsi.py` → stats (pure-stdlib,
  no scipy). Run on the host against `localhost:11434`, incremental writes.
- **Design.** For each prompt, run info-gain (focus, 1 round) to get ranked questions with their
  **projected** scores (`projected_delta`/`stakes`/`prob` per answer; `U`/`EVSI`/`value` per question,
  EVSI from the **shipped deepseek judge**). Then for each (question, answer): inject the answer as an
  established fact, **re-derive** the baseline response, and have a strong blind judge rate
  `realized_change` ∈ [0,1] = how much the response actually moved vs the no-evidence baseline.
- **Prompts:** `buy-rent` (6 q), `gtm-plan` (6 q), `remote-hybrid` (5 q). usaw-calendar excluded
  (the benchmark showed it's a niche-domain/model failure, not a rating problem).
- **Targets.** `realized_change` is the only thing **measured**. Per-question aggregates:
  `realized_change_q = Σ P'·realized_change` (P' = prob renormalized over tested answers) and
  `realized_evsi_q = Σ P'·realized_change·stakes` — note this **reuses projected `stakes`** (see the
  measurement gap), so it is **not** a clean ground truth.

## Results

**P1a — calibration (the Δ judge).** Projected Δ tracks realized change, directionally:

| projected_delta bin | n | mean realized_change |
|---|---:|---:|
| [0.0, 0.2) | 3 | 0.43 |
| [0.2, 0.4) | 9 | 0.52 |
| [0.4, 0.6) | 8 | 0.56 |
| [0.6, 0.8) | 17 | 0.75 |
| [0.8, 1.0] | 14 | 0.83 |

- per-answer **Spearman(projected_delta, realized_change) = +0.394** (quartile binning strictly
  monotone 0.45→0.56→0.75→0.83). Cluster-respecting (question-level) permutation **p = 0.005**;
  prompt-cluster bootstrap 95% CI [0.235, 0.662]; survives drop-one-prompt (min 0.243, always positive).
- **realized_change saturates: 71% (36/51) sit at exactly 0.0 or 1.0** — the change judge is coarse,
  so most rank signal lives in the extremes (binarizing at 0.5 drops ρ to 0.21).

**P1c — formula ablations** (mean per-prompt Spearman vs each target):

| formula | vs realized_change (clean) | vs realized_evsi (confounded) |
|---|---:|---:|
| `value = √(U·EVSI)` | +0.153 | **+0.848** |
| EVSI-only | +0.153 | **+0.848** |
| mean-Δ (P-weighted) | +0.195 | +0.795 |
| **max-Δ** (max over answers) | **+0.526** | +0.784 |
| U-only | +0.147 | +0.102 |

- vs the **clean** signal, `value`/EVSI ≈ 0 (per-question ρ = **−0.009**); **max-Δ is the best
  predictor (+0.526)** and the only one positive in all three prompts (0.892/0.239/0.447).
- `value` and EVSI-only are **byte-identical** — `U` never changes within-prompt order.

## The confound (why +0.848/+0.605 is not validation)

`realized_evsi_q = Σ P'·realized_change·**stakes**` recycles the same projected `stakes` already
inside `EVSI = Σ P·Δ·stakes`. `q_evsi` is **0.96-collinear** with mean stakes, so the partial
correlation controlling for stakes **collapses +0.605 → −0.13**, and stakes *alone* predicts
`realized_evsi_q` as well or better. **≈100% of EVSI's apparent "validation" is the stakes factor
correlating with itself.** Against the one unconfounded signal (`realized_change`), EVSI is null.

## Measurement gap (what blocks clean validation)

We measured realized **Δ** (did the response change) but never realized **stakes** (did the change
matter). Since `EVSI = Σ P·Δ·stakes`, any "realized EVSI" must substitute projected stakes for the
missing realized stakes → the target shares a factor with the predictor. **We can validate the Δ
half; we cannot validate the stakes half, hence not the full formula.** (Even the "clean" Δ signal is
mildly stakes-entangled: projected stakes alone predicts realized_change at answer level ρ=0.417,
p=0.002.)

## Verification (independent reproduce + adversarial refute)

`Workflow: verify-evsi-calibration` — 1 reproduction agent + 4 adversarial skeptics (one per claim) +
synthesis. All 5 headline numbers reproduced within rounding; verdicts:

| claim | verdict | confidence |
|---|---|---|
| **A** — Δ-judge directionally calibrated (ρ=0.39) | **supported** | medium (magnitude leans on gtm-plan; sign robust, cluster p=0.005) |
| **B** — `U` is inert → drop it | **supported** | high (0/40 within-prompt reorderings; U-only anti-predictive) |
| **C** — EVSI confounded; clean-signal null; max-Δ best | **supported** | high (partial-ρ\|stakes = −0.13; max-Δ marginal, p=0.064) |
| **D** — n=17/3-cluster too underpowered to rank formulas | **partial** | per-prompt power *is* fatal; pooled n=17 is OK but its winner rides the confound |

## What it means for the rating

1. **`U` (uncertainty) is inert *for ranking*** in this sample (range-compressed 0.725–0.984) and
   anti-predictive on its own. `√(U·EVSI)` ranks identically to EVSI. **But `U` is load-bearing for the
   *gate*** (`is_gated_out`: `derivable_prob`→1 → `U`→0 retires answered questions across rounds) — the
   ablation only tested the ranking role. So a future "drop U" removes it from the `value` number
   **only**, keeping the derivability gate. *Hedge:* inertness unproven beyond this narrow U spread;
   one buy-rent pair came within 0.002 of flipping.
2. **The full EVSI is not-yet-validated.** Don't ship the ranker on this evidence; **gate the Phase-2
   wrapper on a de-confounded #21.** Stop citing +0.605 as validation — it's a stakes-reuse artifact.
   **Decision (2026-06): freeze the formula — no changes on n=17;** #21 decides every formula question.
3. **max-Δ is a live contender** (best clean-signal predictor) but **marginal** (p=0.064) — a
   hypothesis to test in #21, not a switch to flip now.
4. **Floor: defer.** Directionally a floor exists (low-Δ questions realize ~0.43 vs ~0.83 at top), but
   its numeric location is not estimable at n=17 / with a saturating judge. Set it from #21's blind
   improvement-vs-value curve.

## Reshaped next experiment (#21, hard requirements)

Run the grounded validity study (baseline vs top-K vs low-K, blind-judged, pass = top > low ≥ baseline)
**plus**: (a) an **independent blind realized-stakes judgment** (rate the *importance* of the
differences, not just whether they changed) so a realized EVSI can be computed **without** reusing
projected stakes — the only way to break the ρ=0.96 collinearity; (b) **register max-Δ** as a named
competitor against √(U·EVSI) / EVSI-only / U-only on the blind realized-improvement axis;
(c) **pool across many more than 3 prompts** with a prompt-cluster bootstrap CI. The improvement-vs-value
curve also yields `diminishing_floor`.

## Domain sensitivity — the value structure is domain-bound (a 3-regime spectrum)

The Phase-1 numbers above were measured on **generic life questions** — which turn out to be a
degenerate corner. The real target is **agentic / tool-access / coding** tasks. A value-structure
scan across a **34-prompt, 17-category bank** (`evals/testbank.py` + `evals/score_scan.py`, deepseek
judge) shows the life conclusions do **not** transfer:

| | U spread | derivable_prob | value < 0.40 (life-tuned) |
|---|---|---|---|
| **LIFE** | sd **0.07**, [0.72–0.98] | mean 0.01, [0.00–0.10] | **11%** |
| **AGENTIC** | sd **0.26**, [0.02–0.98] | mean 0.15, [0.00–0.95] | **61%** |

In life questions all uncertainty is **homogeneous, non-derivable user-intent**, so U is pinned high
and inert. Agentic tasks span a wide **derivability** axis, and as `derivable_prob` rises, `U` falls
and the bucket empties — sorted by category (mean over all scored candidates):

```
category        buck deriv  U_mean U_sd value evsi  <thr   regime
planning           6  0.00   0.87  0.04  0.71  0.58   0%    ── ASK THE USER (high U, low deriv,
finance            4  0.00   0.83  0.09  0.48  0.32  33%       real decision-changing forks):
life              16  0.01   0.87  0.07  0.60  0.43  11%       behaves like the life set — the
code-review        8  0.02   0.81  0.12  0.46  0.34  33%       skill produces genuine questions
code-feature       7  0.08   0.67  0.13  0.45  0.31  42%
code-debug         5  0.07   0.75  0.23  0.39  0.28  58%
devops             6  0.10   0.71  0.23  0.42  0.28  50%
system-files       4  0.03   0.79  0.15  0.26  0.14  67%   ── JUST DO IT / DEFAULT (low value:
email              5  0.11   0.67  0.20  0.26  0.16  72%       answer wouldn't change the plan;
automation         5  0.11   0.62  0.12  0.32  0.22  58%       assume the modal answer)
data               2  0.13   0.69  0.22  0.16  0.09  83%
comms-send         1  0.17   0.68  0.30  0.24  0.12  92%
docs               6  0.18   0.61  0.32  0.34  0.27  50%
comms-retrieve     7  0.19   0.54  0.26  0.34  0.32  61%   ── GO FIND OUT (high deriv -> U->0 ->
calendar           3  0.22   0.59  0.27  0.28  0.22  75%       gate fires): route to grounded
web-research       3  0.38   0.49  0.33  0.28  0.17  75%       research, not a user question
knowledge          0  0.90   0.05  0.01  0.05  0.08 100%       (explain-oauth: deriv .90, U .04,
                                                               0 questions — correctly silent)
```

**Three usage regimes, mapped onto the skill's three levers:**
1. **Ask-the-user** (spec-heavy: planning, coding features, security audits, finance) — high U, low
   `derivable_prob`, real EVSI. The skill produces genuine clarifying questions, exactly as for life.
2. **Go-find-out** (research / knowledge / retrieval: web-research, `explain-oauth`, calendar sync) —
   high `derivable_prob` → the **U-gate fires** (`U`→0) → few/no user questions. The skill is already
   signalling *"don't ask, resolve this by research"* — which is precisely the **Phase-2 grounded
   answerer's** trigger. `explain-oauth` (deriv 0.90, U 0.04, 0 questions) is the gate working perfectly.
3. **Just-do-it** (data pulls, sends, file ops, email summaries) — low `EVSI`/value: the answer
   wouldn't change the plan, so assume the modal default. The skill correctly discards these.

**What this overturns / sharpens:**
- **"Drop U" is dead in the target domain.** U's spread is 0.26 here (not 0.07) and it is the
  **ask-vs-find-out discriminator** (regime 1 vs 2) via `derivable_prob`. Removing it would erase that
  routing. The freeze decision was correct — the n=17 life-only "U inert" was a domain artifact.
- **Rank-relative selection (#23) is required, not "likely."** The life-tuned 0.40 cutoff discards 61%
  of agentic candidates — and for *different reasons per regime* (regime 2: low U; regime 3: low value;
  and regime 1's legitimate questions are also pushed under as the whole distribution shifts down).
  An absolute threshold cannot serve a domain this heterogeneous; select by rank / round-relative.
- **The skill's derivability gate is already doing Phase-2's job.** The go-find-out regime is exactly
  where the iterate-context wrapper's grounded research (and NOT_FOUND tombstones) earns out; info-gain
  flags it via `U`→0. This is design-validating, not a defect.

**Implication for #21:** validate on the **agentic bank**, not life questions — and analyze **per
regime** (a single pooled number would average three different mechanisms into mush).

### Agentic realized calibration (the reversal)

A realized-change run on the agentic domain (one prompt per regime — `add-auth`/`gmail-triage`/
`research-ratelimit`, `--source all_scored`, n=54 answers / 18 questions) shows the **calibration is
stronger here than on life, and — unlike life — EVSI/value predict the clean realized-change signal:**

| | per-answer ρ(Δ, realized) | per-q EVSI vs realized_**change** | per-q value vs realized_change |
|---|---|---|---|
| LIFE | +0.39 | **−0.009** (null) | +0.11 |
| AGENTIC | **+0.64** | **+0.70** | **+0.66** |

Calibration curve monotone 0.16→0.26→0.48→0.76→0.98. The life-domain null was an artifact of the
**compressed** life value distribution (no variance to predict); the target domain has real spread, so
the formula discriminates. **This partially rehabilitates EVSI for the actual use case** — but with two
honest qualifiers:
- **Mostly between-regime.** The strength comes from correctly separating tasks (value/realized means:
  ask-user 0.50/0.87, just-do-it 0.18/0.18, go-find-out 0.11/0.14 — monotone). **Within** a task the
  ranking is positive but modest (avg per-prompt ρ ≈ 0.34). So the formula is excellent at *"which task
  needs clarification at all"* and decent at *"which question within a task."*
- **Stakes still unmeasured.** value-vs-realized-**change** (+0.66) is clean (no stakes), but the full
  `EVSI = Σ P·Δ·stakes` still can't be validated without realized stakes (realized-EVSI +0.89 remains a
  projected-stakes confound). n=18 / 3 prompts / 72% saturation — directional.

Net verdict shift: **the Δ-half and the cross-task value ranking show real signal in the target domain
(a clear improvement over the life-only read); the stakes-weighting and within-task ranking still ride
on the powered, de-confounded #21.**

### The realized-stakes instrument is the hard part (→ go comparative)

Building #21's de-confounding step surfaced a methodological wall. To break the projected-stakes
confound we must measure realized **stakes** independently (`evals/validate_evsi.py::stakes_judge`,
`analyze_validity.py`). An **absolute** post-hoc stakes judge proved too fragile:
- **Catastrophe anchor** ("how materially worse… serious problems") → collapse: **35/36 rated 0.0**,
  only the *compliance* question (genuinely legal-grade) got 1.0. Zero variance → de-confounded test
  uninformative (value vs realized_regret ρ=+0.26, but realized_regret was ≈0 everywhere).
- **Graded anchor** ("would a knowledgeable user care… full range") → variance returns (mean 0.62,
  sd 0.15) and becomes distinct from realized_change, **but central-tendency clusters** (12/18 snap to
  0.6). Better, still not discriminating.

So the realized judges are fragile in **opposite** ways — change saturates at 0/1, stakes piles on the
middle anchor. Note the *projected* deepseek stakes is sensibly graded (sd 0.26; auth 0.70 / scale 0.10
/ compliance 0.95) — it's the post-hoc *measurement* of stakes that resists absolute rating.

**Conclusion — promote comparative elicitation (1.4 / #24) from conditional to the path forward.**
Models are far better at **relative** judgments than calibrated absolute numbers. The de-confounded
study should measure realized stakes **pairwise** — *"for this prompt, which of these two clarifications
matters more for the outcome?"* — yielding a ranking (Bradley-Terry / Elo) instead of brittle 0–1
ratings. The same likely applies to *eliciting* projected stakes. Until then the **stakes-half of EVSI
remains unvalidated by instrument limitation** (not by a negative result); the **Δ-half stands**
(agentic per-answer ρ 0.64, value-vs-realized-change 0.66).

## Wrapper end-to-end (the honest verdict)

The #21 end-to-end test (`evals/validate_wrapper.py`): for each prompt, produce a baseline response
(answer no clarifying questions) and a wrapper response (research the top-K via grounded `ask`, then
respond), blind-judge which better serves the user. Findings — and the confound that nearly buried them:

- **First pass (default env) → baseline 2-0.** But inspection killed the conclusion: the test runs in
  a synthetic container with **10 projects** under `/opt/data/projects/`, so "add auth to my web app"
  is genuinely ambiguous, and the grounded answerer ran in the install cwd (`/opt/hermes`) — it found
  the Hermes codebase and honestly said "no web app here" while the baseline picked a real project
  (`fastapi-tasks`) and delivered. A **fair, balanced re-judge** (penalizing over-assumption *and*
  punting, ignoring length) still favored baseline — because the baseline **is itself a capable
  investigating agent**, so a redundant k=1 clarification couldn't beat it.
- **De-confounded (both pinned to the real project, responder given file tools) → 1-1 (k=1).**
  `add-auth` → **wrapper wins** (researching the actual stack yields a "complete, production-ready
  implementation"); `fix-test` → **baseline wins** ("correctly identifies the missing test files" — a
  capable agent just investigates the failure directly; the clarification is redundant).

**Verdict:** the wrapper is **not a universal win over a capable baseline agent** — its value is
**task-dependent**. It helps where a clarification *shapes* the work (build/spec tasks: knowing the
stack/constraints changes the implementation) and is redundant where the agent can *self-investigate*
(debug tasks). Its **distinctive, non-redundant value is the genuinely user-only constraints** a
capable agent can't investigate away. Two real levers were found and fixed: the grounded answerer's
**`cwd`** must be the user's project (`answer_cwd`/`responder_cwd`), and the responder's tools.

**Caveats:** n=2 de-confounded, k=1, single project, one judge — directional, not settled. k≥2 and
genuinely user-only-constraint prompts are where the wrapper should show its clearest edge.

**Implication:** ship the wrapper as a working v1 (ranking validated via realized_change; mechanically
correct; de-confounded it holds its own). The strategic open question is emphasis: autonomous research
loop (redundant with capable agents on investigable tasks) vs **surfacing the ranked user-only
clarifications** (the report-only strength — the non-redundant value). Left for the user to steer.

## Caveats

- 3 independent prompt clusters; n=51/n=17 overstate power. The +0.394 leans on gtm-plan (dropping it
  → 0.243). Treat all magnitudes as directional.
- `realized_change` saturates (71% at 0/1) — coarse ground truth; the per-question aggregate is
  tie-free, but row-level rank signal is concentrated at the extremes.
- Projected scores use the shipped deepseek judge; `realized_change` uses a deepseek change-judge —
  not de-confounded from each other by model.
- **Domain scan:** 1 prompt/cell, fast generation + deepseek judge, the value distribution only (no
  realized_change). Some of the agentic downshift could be model-capability (the fast model projecting
  agentic answers less richly, à la usaw) rather than pure domain structure — but the U-spread /
  derivability pattern is structurally sensible (research tasks *are* more derivable), so it most likely
  reflects a real domain effect. The agentic *realized*-change calibration (per-regime) is the follow-up.
