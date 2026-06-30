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

## Caveats

- 3 independent prompt clusters; n=51/n=17 overstate power. The +0.394 leans on gtm-plan (dropping it
  → 0.243). Treat all magnitudes as directional.
- `realized_change` saturates (71% at 0/1) — coarse ground truth; the per-question aggregate is
  tie-free, but row-level rank signal is concentrated at the extremes.
- Projected scores use the shipped deepseek judge; `realized_change` uses a deepseek change-judge —
  not de-confounded from each other by model.
