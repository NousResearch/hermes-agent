# CLOE Operational Validation Pattern

## When to Use

After a CLOE version achieves **technical certification** (`CLOE_VX_STRATEGIC_DECISION_COPILOT_CERTIFIED`) and you need to validate it through sustained real-world operator use. This pattern builds the **measurement infrastructure** — not more certification benchmarks, but the governed tooling for:

1. Recording real operator decisions alongside CLOE recommendations
2. Capturing operator feedback (accepted/modified/rejected + why)
3. Tracking execution outcomes
4. Measuring recommendation quality, operator adoption, and confidence calibration
5. Detecting recurring failure modes
6. Building a ranked evidence backlog for the next version (VX+1)

## Core Principle

**Operational validation is NOT technical certification.** It measures real operator value and outcomes, not answer-format compliance or benchmark scores. The technical certification is frozen as the baseline — it is never reopened during operational validation.

## 9-Phase Roadmap

| Phase | Purpose | Key Deliverable |
|-------|---------|-----------------|
| 1 — Baseline Freeze | Freeze VX certification evidence; define boundary between tech cert and operational validation | `docs/operator/receipts/cloe-vX-operational-validation-baseline-freeze.md` |
| 2 — Decision Record Contract | Privacy-safe append-only decision record schema + store | `src/decision/cloe-vX-operational-decision-record.js` |
| 3 — Operator Feedback | Governed feedback mechanism (accept/modify/reject + reason + outcome) | `src/decision/cloe-vX-operator-feedback.js` |
| 4 — Outcome & Value Scoring | 9 deterministic scoring functions | `src/decision/cloe-vX-outcome-scoring.js` |
| 5 — Operational Review Workflow | 6 read-only review CLI commands | `tools/cloe/cloe-vX-operational-review-workflow.mjs` |
| 6 — Real Usage Pilot | Governed pilot instructions for the operator | `docs/operator/cloe-vX-real-usage-pilot-instructions.md` |
| 7 — Failure Analysis | 8-category structured root-cause analysis | `src/decision/cloe-vX-failure-analysis.js` |
| 8 — V6 Evidence Backlog | Ranked evidence backlog (no V6 implementation) | `src/decision/cloe-vX-v6-backlog.js` |
| 9 — Final Validation | Reconcile 75+ real decisions; issue verdict | CLI output + report |

## Decision Record Schema (24 fields)

```
decision_id            — UUID
timestamp              — ISO timestamp
category               — One of 11: architecture, roadmap, prioritization, roi,
                         provider_choice, discovery, campaigns, governance,
                         risk, maintenance, constrained_time
operator_question_summary       — Safe summary of what the operator asked
relevant_context_summary        — Context needed to understand the decision
cloe_recommendation             — What CLOE recommended verbatim
rejected_alternatives           — Options CLOE considered and rejected
tradeoff                        — Tradeoff analysis
confidence                      — CLOE's confidence (0-100)
immediate_next_action           — The next step CLOE suggested
operator_decision               — What the operator actually decided
adoption_status                 — accepted | modified | rejected
operator_reason                 — Why the operator chose differently
execution_status                — pending | in_progress | completed | cancelled
expected_result                 — What was expected to happen
observed_result                 — What actually happened
time_saved                      — Minutes saved (or negative if lost)
risk_avoided_or_introduced      — Description
retrospective_usefulness        — 1-5 scale
confidence_calibration          — Was CLOE confidence accurate?
lessons_learned                 — Free text
```

## Final Assessment Thresholds

| Threshold | Target |
|-----------|--------|
| Recommendations rated immediately useful | ≥ 85% |
| Decisions with clear single recommendation | ≥ 80% |
| Accepted or adapted recommendations | ≥ 80% |
| Weak due to avoidable reasoning defects | < 10% |
| Critical leaks | 0 |
| Governance violations | 0 |
| Unauthorized executions | 0 |
| Confidence calibration materially consistent | YES |

## Key Design Decisions

1. **Append-only** — Records are created once; `updateRecord` and `appendOutcome` add data, never overwrite. Individual JSON files per decision.
2. **Privacy-safe** — Privacy filters reject `sk-` patterns, Bearer tokens, credential-like values, base64 secrets, forbidden keys. Safe for operator to use during real work.
3. **No auto-execution** — CLOE recommends; the operator decides. The system never autonomously executes any recommendation.
4. **No V6 implementation** — Phase 8 creates only a ranked evidence backlog. V6 candidates must include `evidence_that_not_workflow_fixable` to prove they can't be fixed with prompts/context/evaluators alone.
5. **No threshold modification** — Certification thresholds are frozen. Operational validation adds new measurements but never changes existing ones.
6. **Evidence-based final verdict** — The final `CLOE_VX_OPERATIONALLY_VALIDATED_STRATEGIC_COPILOT` verdict requires ≥75 real decision records with operator feedback and outcome evidence. Cannot be issued from infrastructure alone.

## File Organization

```
src/decision/
  cloe-vX-operational-decision-record.js   — record store (append-only)
  cloe-vX-operator-feedback.js             — feedback mechanism
  cloe-vX-outcome-scoring.js               — 9 scoring functions
  cloe-vX-failure-analysis.js              — 8-category root cause analysis
  cloe-vX-v6-backlog.js                    — ranked evidence backlog

tools/cloe/
  cloe-vX-operational-validation-cli.mjs   — decision recording CLI
  cloe-vX-operational-review-workflow.mjs  — 6 review commands

docs/operator/
  cloe-vX-real-usage-pilot-instructions.md — pilot instructions
  CLOE_VX_OPERATIONAL_VALIDATION_OPERATOR_PACKET.md
  receipts/
    cloe-vX-operational-validation-baseline-freeze-v1.md
    cloe-vX-operational-validation-continuity.json
    local-ci-verified-cloe-vX-operational-validation-v1.json

test/
  cloe-vX-operational-decision-record.test.js
  cloe-vX-failure-analysis-and-backlog.test.js
```

## Pitfalls

- **Do NOT reopen technical certification** — the baseline freeze is read-only. Operational validation is additive.
- **Do NOT implement VX+1 during this mission** — only create the evidence backlog.
- **Do NOT issue OPERATIONALLY_VALIDATED without real decision volume** — infrastructure alone cannot produce this verdict.
- **Do NOT store secrets** — the privacy filter is a safety net, not a substitute for operator judgment.
- **Do NOT weaken existing governance gates** — V2/V3/V4/V5 gates remain at their original thresholds.
- **Do NOT treat missing CI (GitHub Actions down, billing exhausted) as a blocker** — use LOCAL_CI_VERIFIED_MERGE_POLICY_V1 with local test receipts.

## Reference Implementation

CLOE V5 Operational Validation was implemented at PR #623 (merge commit fb2ac9f) in openclaw-runtime. 14 new files, 79 deterministic tests, 0 regressions. Verdict: `CLOE_V5_OPERATIONAL_VALIDATION_PILOT_READY`.
