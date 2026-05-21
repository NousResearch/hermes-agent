# May 22 Spec Review Prep — Three Verdicts Due

**Date:** May 21 (prep) | May 22 (verdict review)  
**Specs:** 072 (vNext D8/D9), Clinical Phase A, Catalyst Phase A  
**Blocker Status:** Alpha freeze active — all decisions require Checklist v2 or evidence of degradation  

---

## Spec 072 vNext D8/D9 — Clinical Quality Candidate

**Status:** Ready for verdict review  
**What's at stake:** First structurally clean non-coinvest signal in entire research thread

### The Finding (Load-bearing numbers)

| Feature | D8 Within-Quint IC | D9 Bin-Residualized IC | Raw t-stat | NW-Adjusted t (est.) |
|---------|------------------|--------------------|-----------|--------------------|
| `clinical_score_v2_z` | +0.173 | +0.202 | **+5.05** | **~+3.0** (after lag correction) |
| `clinical_score` | +0.178 | +0.200 | **+5.00** | **~+3.0** |
| `readout_density_90` | +0.128 | +0.128 | +3.16 | ~+1.9 |

**Key caveat:** Raw t=+5 will shrink to ~+3.0 after Newey-West lag correction (n=600 pool has autocorrelation; effective obs ≈ 245).

### Critical Constraints for Tomorrow's Verdict

**Must verify all 4 conditions:**

1. ✅ **At least 1 distinct signal cluster post-dedup** — clinical_score family is ONE cluster, not 6 features
2. ✅ **Cohort-quarantine window has closed** (2026-05-15) AND signal survives in post-window subset
3. ⏳ **Cross-feature correlations computed** within L3 — choose 1 representative feature per cluster
4. ⏳ **Spec 071 Lane 1 has shipped** (false-catalyst audit at 17.6% error rate) — impact on L3 still unquantified

**Current status on condition 4:** Spec 071 not yet shipped → cannot fully exonerate the clinical IC from false-catalyst artifact.

### Recommendation for May 22

**Path A: CONDITIONAL ADVANCE** (if conditions 1–3 are verified)
- Approve `clinical_score_v2_z` OR `clinical_score` (pick one, not both) for **ranker-only shadow test** post-h20d
- Explicit Checklist v2 requirement (no shortcut)
- Condition: "Defer final promotion until Spec 071 Lane 1 ships and false-catalyst impact on L3 is quantified"
- Lock in: "Post-dedup, only test the single feature chosen; do NOT broaden to endpoint_strength or readout_curve"

**Path B: DEFER** (if condition 2 fails or data quality issues found)
- Remark: "Architecture is sound; signal exists in gated universe. Will retry in next window when n ≥ 30 live days and post-window subset is large enough."
- Keep ranker-shadow code in place; no code deletions

**Path C: REJECT** (if conditions 1 or 3 fail fundamentally)
- Only if correlation audit shows clinical features are NOT distinct from coinvest/financial/inst_delta within L3
- Unlikely given D7 orthogonality tests passed

### Talking Points

- "This is the cleanest signal we've found outside coinvest. The architecture that separates validation → filtering → ranking is working."
- "We have the raw t-stats, but need NW correction before promotion claim. That moves the needle from t=+5 to t≈+3, which is still significant."
- "The trap layer (catalyst + runway + execution + thesis + dilution + governance) is doing real work — we're not seeing spurious correlations."
- "Spec 071 false-catalyst hygiene will tell us whether some of this IC is artifact or signal. Until then, it's ranker-shadow only."

---

## Clinical Phase A Verdict — SELECTOR CLOSED / RANKER SHADOW

**Status:** Frozen (no promotion before 2026-05-22 re-run)  
**Bottom line:** Clinical is valid within coinvest but negatively-correlated unconditionally

### The Audit Results

| Role | Finding | Status |
|------|---------|--------|
| **Selector** | ρ(clinical, selector_score) = −0.16 median (13 snaps negatively correlated) | ❌ **NO_GO** |
| **Ranker (shadow)** | `clinical_design_quality` ρ = +0.084 within top-coinvest tertile (13/17 snaps positive) | 🟡 **SHADOW, deferred** |
| **EV transmission** | 64 drops, 0 gains over 16 snaps; outcome binder unfinished | ⏳ **Non-evaluable** |

### Why Selector is Closed

Adding clinical as a new selector weight would fight the current composite (B6). The +0.103 IC from Spec 057 was CONDITIONAL on coinvest gate; unconditionally, clinical is dead weight for selection. Lane closed.

### Why Ranker Shadow is Open (Carefully)

`clinical_design_quality` is the **cleanest single candidate** within the top-coinvest tertile. But:
- Cannot promote until Checklist v2 is satisfied
- Only test this ONE feature; do NOT broaden
- Requires full Spec 072 Phase A reconfirmation post-13F refresh

### Recommendation for May 22

**VERDICT: REAFFIRM PHASE A AUDIT — NO SELECTOR CHANGES**
- Selector lane closed; do not revisit until evidence of degradation
- Ranker shadow on `clinical_design_quality` only, deferred until Spec 072 vNext verification post-13F (May 26 h20d gate)
- EV transmission stays non-evaluable until outcome binder ships (Spec 077 scoped 2026-05-06)

### Talking Points

- "Phase A confirmed clinical is independent from coinvest, inst_delta, and financial. That's real. But it only works inside the coinvest filter."
- "The selector lane is closed because unconditional correlation is negative. That's not a failure of clinical — it's a finding that clinical plays a different role."
- "We have one clean ranker candidate (`clinical_design_quality`). We're shadowing it and waiting for Spec 072 final approval before running a real test."
- "The EV transmission is blocked on an outcome binder. That's a dependency, not a decision — once the binder ships, we can evaluate."

---

## Catalyst Phase A Verdict — SELECTOR ACTIVE/NO_MORE_WEIGHT / RANKER SHADOW

**Status:** Frozen / shadow deferred  
**Bottom line:** Catalyst is already in production; no new selector weight; ranker candidate exists but requires false-catalyst hygiene

### The Audit Results

| Role | Finding | Status |
|------|---------|--------|
| **Selector** | Already weighted via `selector_catalyst_block` (0.25 weight in module_5, ρ=+0.27) | ✅ **ACTIVE / NO_MORE_WEIGHT** |
| **Ranker (shadow)** | `catalyst_score` ρ = +0.19 within top-coinvest (17/17 snaps positive, more stable than clinical's +0.08) | 🟡 **SHADOW, deferred** |
| **EV predictions** | 43 HIT/MISS resolved; 81% aggregate hit rate; `prediction_composite_score` binder closed but WRONG field; correct field (`event_ev_p_hit`) not yet bound | ⏳ **Non-evaluable until spec_077 ships** |

### Why Selector is "NO_MORE_WEIGHT"

Catalyst already carries its selector weight. Adding new features (has_catalyst, proximity, catalyst_in_window) would double-count or import 18.8% false-catalyst noise. Production design is correct; do not change.

### Why Ranker Shadow is Open (Carefully)

`catalyst_score` shows +0.19 conditional ρ, more stable than clinical. BUT:
- Overlaps mechanically with `selector_catalyst_block` already in production
- 18.8% false-catalyst contamination must be addressed first (Spec 071 Lane 1)
- Only test `catalyst_score` in isolation; do NOT add `catalyst_in_window` separately

### Critical Issue: EV Binder Closed But Wrong Field

**Status:** `prediction_composite_score` binder was completed (spec_073, 2026-05-04) but is the WRONG field.
- `prediction_composite_score` is a stock-quality composite (coinvest + financial + inst_delta), not an event-likelihood score
- Brier score is WORSE than baseline; HIT rate is INVERTED (high bucket = worst performance)
- Correct field is `event_ev_p_hit` from event_ev outcome_model (Bayesian posterior P(HIT) per event)
- **Spec_077 is scoped to bind the correct field forward-only** (backfill 30% match rate, unsafe)
- Calibration cannot run until n ≥ 30 (~2026-07-01)

### Recommendation for May 22

**VERDICT: REAFFIRM PHASE A AUDIT — SELECTOR LOCK, RANKER SHADOW DEFERRED**

1. **Selector:** NO new weight. Existing `selector_catalyst_block` 0.25 is optimal. Lane closed.
2. **Ranker:** Shadow on `catalyst_score` only; require false-catalyst hygiene gate before any promotion test; defer until post-Spec 071 + post-13F refresh (~May 26 h20d gate)
3. **EV:** Acknowledge binder mistake (prediction_composite_score is wrong field). Spec_077 will fix it. Do NOT use prediction_composite_score for any EV calibration or decision.

### Talking Points

- "Catalyst is already in production and working. Phase A confirms no new selector weight is needed — we've already captured the sweet spot."
- "The ranker signal is real (+0.19 conditional ρ, 17/17 snaps positive), but it's harder to separate from what the selector already does."
- "False-catalyst contamination is 18.8% at universe — we need Spec 071 Lane 1 to fix that before we can trust a catalyst-quality ranker test."
- "We made a mistake with the EV binder. `prediction_composite_score` is not an event-likelihood score. Spec_077 will bind the right field going forward."
- "Even with the binder fixed, we won't have enough live trading days for EV calibration until ~July 1. That's expected — we're building for Q3 evidence."

---

## Summary Talking Points (All Three)

**For h20d gate on May 26:**
- vNext architecture is validated (D8/D9 gates pass; no coinvest leakage)
- Clinical and Catalyst roles are clarified (selector closed, ranker candidates identified)
- All three specs are frozen pending two gates:
  1. **Spec 071 Lane 1** (false-catalyst hygiene) — unblocks catalyst & clinical final tests
  2. **13F refresh + h20d decision** (May 26) — unblocks Phase 2 KG live + ranker shadow → real test

**No alpha freeze violations in any of these verdicts** — they're all descriptive audits and shadow candidates, not promotions. All deferrals are conditional on documented evidence, not arbitrary hold-backs.

---

## Files to Reference Tomorrow

- `screener_vnext_d8_d9_first_candidate_2026_05_01.md` (vNext candidate details)
- `clinical_phase_a_verdict_2026_05_04.md` (clinical audit)
- `catalyst_phase_a_verdict_2026_05_04.md` (catalyst audit)
- Spec 072 full spec: `specs/changes/spec_072_screener_vnext_manager_gate_traps_catalyst_rank.md`

---

## What's Next (May 26 h20d gate)

Once these verdicts are recorded:
1. **Spec 071 Lane 1** ships → catalyst/clinical hygiene gates pass → ranker shadow tests can begin
2. **h20d decision** → architecture freeze lift (conditional) → Phase 2 KG live + ranker shadow validation on real snapshot window
3. **Phase 2 Step 5** (KG gating on ranker final_score) becomes unblocked
