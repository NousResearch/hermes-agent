# h20d Decision Memo — DRAFT FRAMEWORK

**Decision Date:** May 26, 2026  
**Clearance Gate:** Phase 2 KG implementation + architecture freeze lift  
**Timeline Impact:** 13F refresh validation (May 23–26) + Spec review verdicts (May 22) + h20d verdict (May 26)  

---

## Executive Summary

Three decision paths available May 26. Path chosen depends on two clearance gates:

1. **13F cohort validation** — As of May 19: **PASSED all 6 gates**. Jaccard 0.875; Quarantine-ready to lift (conditional on freeze).
2. **Spec review verdicts** — May 22: vNext, Clinical, Catalyst Phase A verdicts determine downstream signal confidence.

The h20d decision unlocks or defers **Phase 2 Step 5 (KG gating on ranker final_score)** and **Phase 2.5 (ranker shadow → real production test)**.

---

## Three Decision Paths

### Path A: FREEZE LIFT — Full Implementation (Recommended if 13F & Specs pass)

**Conditions met:**
- 13F Jaccard ≥ 0.70 + all 6 validation gates PASS (✅ May 19 status)
- Spec 072 vNext D8/D9 advances to ranker-shadow (May 22 verdict: CONDITIONAL ADVANCE)
- Spec 071 Lane 1 is shipped OR impact on L3 is quantified
- Architecture freeze can safely lift per policy (no pending security/governance issues)

**Decision:** LIFT architecture freeze May 26  
**Unblocks:**
- Phase 2 KG implementation goes live (68/68 tests PASS; ready for production snapshot)
- Phase 2 Step 5 (KG gating on ranker final_score) — integrate KG contradictions into ranker decision flow
- Ranker shadow validation → real test on Phase 2.5 cohort (post-13F, post-KG)

**Constraints:**
- Ranker 2-feature freeze remains (no new ranker features until Checklist v2 + forward evidence)
- No clinical/catalyst promotion until shadow tests complete (≥30 live days post-Phase 2.5 start)
- KG application is governance/lineage only; no KG-derived scores in alpha (hard boundary per knowledge_graph_strategic_roadmap)

**Timeline:**
- May 26: Announce freeze lift
- May 27–31: Deploy Phase 2 KG to production snapshot pipeline
- June 1: First live KG-gated snapshot; ranker shadow validation begins
- June 15–July 1: ≥30 live trading days for clinical/catalyst shadow signal confidence

**Success criteria:**
- Phase 2 KG snapshot pipeline operates without errors (regression tests PASS)
- Top-30 changes post-KG launch are ≤ marginal vs. pre-KG (governance only, no alpha)
- Clinical/catalyst shadow signals maintain IC over ≥30 live days (confidence threshold for Spec 072 final verdict July 1)

**Risk:** If Phase 2 KG deployment encounters unforeseen bugs, revert to Path B (defer + patch) rather than rolling back to Path C.

---

### Path B: HYBRID DEFER — Partial Implementation (If conditions partially met)

**Conditions met:**
- 13F validation gates mostly pass (Jaccard 0.70–0.85; borderline but acceptable)
- Spec review verdicts have 1+ open issues (e.g., Spec 071 Lane 1 still not shipped; false-catalyst impact unquantified)
- Architecture freeze could lift but downstream risk is elevated

**Decision:** DEFER freeze lift to June 2 or June 9; implement Phase 2 KG in staging only; ranker shadow in monitoring mode

**Unblocks (partial):**
- Phase 2 KG goes live in **staging/sandbox** with full test suite (non-production)
- Ranker shadow validation runs on staging snapshots (no live trading day clock)
- Clinical/catalyst candidates are monitored but NOT promoted

**Delays:**
- Phase 2 Step 5 deferred; KG gating remains offline
- Ranker shadow test timeline slips by 1–2 weeks (June 1 → June 9 / June 15)
- Clinical/catalyst promotion verdicts shift to July 15 (30 live days measurement pushed back)

**Conditions to re-evaluate (June 9):**
- Spec 071 Lane 1 ships (if not by May 26)
- Spec 072 vNext retest post-13F refresh clears (if May 22 verdict was conditional)
- Any 13F validation gate that was borderline stabilizes

**Success criteria:**
- Staging phase 2 KG runs clean over 2 weeks
- No blockers identified that would prevent production lift June 9
- Ranker shadow IC holds in staging; ready for live test June 9

**Risk:** If staging phase 2 KG reveals systemic issues, Path B becomes a 2-week debug cycle (acceptable).

---

### Path C: EXTENDED HOLD — No Implementation (Only if major blockers)

**Conditions requiring Path C:**
- 13F validation fails on critical gate (Jaccard < 0.70, or ≥2 validation gates fail)
- Spec review verdicts block Phase 2 KG (e.g., if vNext D8/D9 retest shows signal disappeared)
- Governance issue emerges that makes freeze-lift unsafe (unlikely but possible)

**Decision:** HOLD architecture freeze through June 20 (post-13F final deadline); no Phase 2 KG deployment; ranker frozen

**Delays:**
- Phase 2 KG stays in code review (not deployed)
- Ranker shadow candidates are NOT tested
- Clinical/catalyst signals remain unvalidated
- All Phase 2.5+ work deferred to Q3

**Conditions to re-evaluate (June 20):**
- 13F cohort stabilizes after June 15 deadline (all 48 managers filed)
- Spec 072 vNext retest with full n ≥ 30 live days (June 1 → June 30)
- New evidence emerges that reverses May 26 blockers

**Success criteria:**
- By June 30, evidence is strong enough to recommend July Phase 2 KG deployment
- Roadmap shifts Phase 2 Step 5 to July 1 start

**Risk:** Path C extends uncertainty into Q3 and delays ranker signal validation 4–6 weeks. Choose only if evidence strongly blocks Paths A/B.

---

## Decision Matrix

| Condition | Path A | Path B | Path C |
|-----------|--------|--------|--------|
| 13F Jaccard ≥ 0.70 | ✅ | ✅ borderline | ❌ |
| 13F all 6 gates PASS | ✅ | 4–5/6 | ≤3/6 |
| Spec 072 vNext advances (May 22) | ✅ | Conditional | ❌ |
| Spec 071 Lane 1 shipped | ✅ | ⏳ (re-eval June 9) | ❌ |
| Architecture freeze safe to lift | ✅ | ⏳ (review June 9) | ❌ |
| Phase 2 KG deployment | Production | Staging only | Deferred |
| Ranker shadow timeline | June 1 start | June 9 start | Deferred |
| Clinical/Catalyst verdicts | July 1 | July 15 | Post-June 30 |

---

## Pre-May-26 Checklist

**By May 22 (spec reviews complete):**
- [ ] Spec 072 vNext verdict recorded (ADVANCE / DEFER / REJECT)
- [ ] Spec 071 Lane 1 status confirmed (shipped / in progress / blocked)
- [ ] Clinical Phase A ranker-shadow candidate confirmed (`clinical_design_quality` only)
- [ ] Catalyst Phase A ranker-shadow candidate confirmed (`catalyst_score` only)

**By May 26 (13F refresh + h20d decision):**
- [ ] 13F validation gates re-run (if ≥34 managers filed by May 23); all 6 gates audit complete
- [ ] Cohort Jaccard ≥ 0.70 confirmed in post-refresh snapshot
- [ ] Top-30 churn audit completed (artifact vs. signal attribution)
- [ ] h20d decision matrix filled in with actual results

---

## Key Talking Points (May 26)

**If Path A chosen:**
- "13F cohort is validated and stable (Jaccard 0.875). Phase 2 KG goes live June 1. Ranker shadow validation starts immediately on live trading days."
- "Specs are clear: vNext architecture is sound, clinical and catalyst have specific roles. This unblocks both Phase 2 Step 5 and Phase 2.5."
- "We maintain the 2-feature ranker freeze and no KG-to-alpha leakage. The governance layer is live; signal validation is still shadow-only."

**If Path B chosen:**
- "13F is borderline; Spec 071 is still in flight. We're deploying Phase 2 KG to staging for a 2-week validation run, then deciding on production June 9."
- "This buys us time for Spec 071 to ship and for the 13F final cohort to stabilize. Ranker shadow validation moves to June 9."

**If Path C chosen:**
- "Spec 072 retest showed signal regression / 13F validation gated on an issue we can't resolve by May 26. We're holding the freeze through June 20 and re-evaluating after the 13F deadline."
- "This is a 4-week slip, but it's the right call given the blockers. Ranker and KG validation timeline moves to Q3."

---

## Appendix: Phase 2 KG Scope (Reminder)

**What Phase 2 KG does (implementation locked 2026-05-19):**
- Provenance graph: 56 nodes, 16 edges, 5 query patterns (spec_110 PoC 22/22 tests PASS)
- Governance KG (spec_089): node types + edge types + contradiction rules for regulatory/governance lineage
- Ranker integration (spec_089.1.5A): KG contradictions flow into ranker decision tree (gating, not alpha)

**What Phase 2 KG does NOT do:**
- No KG-derived scores (hard boundary per memory)
- No graph centrality / PageRank as alpha features
- No K-hop neighborhood aggregation for ranking
- Governance lane only; no commercial signals

**Frozen since 2026-05-19; no changes authorized until Phase 2 completion audit (post-deployment).**

---

## What to Finalize After May 26

1. **Decision memo populated with actual results** (replace [OPEN] placeholders below with numbers)
2. **Downstream owner notified** (13F monitoring cron → Phase 2 KG deployment; Spec 072/071/089 engineers)
3. **Timeline updated in roadmap** (phase 2/2.5/3 start dates)
4. **Risk register updated** (if Path B/C chosen, re-assess post-13F and post-Spec 071)

---

## Decision Memo Template (Fill in May 26)

```
# h20d Decision — FINAL

**Date:** 2026-05-26  
**Chosen Path:** [A / B / C]  

## Clearance Results

### 13F Validation Gates (Run May 23–26)
| Gate | Threshold | Result | Pass? |
|------|-----------|--------|-------|
| 1. Filed Count | ≥34 mgrs | [ACTUAL] | [Y/N] |
| 2. Cohort Jaccard | ≥0.70 | [ACTUAL] | [Y/N] |
| 3. Producer Freshness | cache advanced | [ACTUAL] | [Y/N] |
| 4. Position Completeness | no Q4 stale | [ACTUAL] | [Y/N] |
| 5. Top-30 Stability | KS < 0.20 | [ACTUAL] | [Y/N] |
| 6. Coverage/Diversity | drop < 10pp | [ACTUAL] | [Y/N] |

### Spec Review Verdicts (May 22)
- Spec 072 vNext: [ADVANCE / DEFER / REJECT] — [REASON]
- Spec 071 Lane 1: [SHIPPED / IN PROGRESS / BLOCKED] — [REASON]
- Clinical Phase A ranker: [APPROVED AS SHADOW / DEFERRED / CLOSED] — [REASON]
- Catalyst Phase A ranker: [APPROVED AS SHADOW / DEFERRED / CLOSED] — [REASON]

## Decision Rationale

[1–2 paragraph explaining which path and why, given the above results]

## Unblocked / Deferred

- Phase 2 KG implementation: [UNBLOCKED / DEFERRED] → Target deployment: [DATE]
- Ranker shadow validation: [UNBLOCKED / DEFERRED] → Target start: [DATE]
- Clinical/Catalyst verdict (live): [JULY 1 / JULY 15 / JUNE 30] depending on shadow start

## Constraints / Assumptions

[List any hard constraints maintained or new assumptions from this decision]

## Sign-off

**Decided by:** [NAME]  
**Consensus:** [YES / ADVISORY]  
**Commit hash:** [GIT SHA FOR REPRODUCIBILITY]
```

---

## Files to Reference (May 26)

- `13f_q1_2026_monitoring_live_2026_05_15.md` (live filing status as of May 15 — will be updated May 23–26)
- `13f_refresh_runbook_complete_2026_05_17.md` (the 6 validation gates + commands)
- `spec_110_phase_1_poc_complete_2026_05_21.md` (KG PoC evidence)
- `h20d_decision_memo_framework_2026_05_21.md` (this file)
- Spec 072 full: `specs/changes/spec_072_screener_vnext_manager_gate_traps_catalyst_rank.md`

---

## Owner / Next Steps

**Immediate (May 21–22):**
- Finalize May 22 spec review prep (done — see MAY_22_SPEC_REVIEW_PREP.md)
- Record spec verdicts May 22

**May 23–26:**
- Monitor 13F filing progress (cron runs weekdays 6:22 PM ET)
- Run 13F validation gates when ≥34 managers filed
- Prepare decision matrix by end of May 25

**May 26:**
- Present h20d decision
- Announce Path A/B/C choice
- Update downstream timelines

