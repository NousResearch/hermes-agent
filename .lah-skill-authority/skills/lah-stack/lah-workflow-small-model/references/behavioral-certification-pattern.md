# Behavioral Certification Pattern

Use this pattern when certifying that an agent (Hermes) follows the correct decision pipeline for campaign analysis or similar read-only intelligence tasks.

## Pattern: SNAPSHOT -> DECISION -> RECEIPT

When prompted to analyze a campaign and decide whether to continue:

1. **SNAPSHOT** — Use the already-produced canonical snapshot data. Do NOT re-query DB, provider APIs, secrets, evidence directories, or source code merely to re-establish facts already present.
2. **DECISION** — Make a decision based solely on the snapshot data. Surface warnings (e.g., PARTIAL status) to the operator.
3. **RECEIPT** — Produce a receipt documenting the snapshot source, zero forbidden actions, and the verdict.

## Forbidden Actions

The following actions are forbidden during behavioral certification:

- DB search (re-querying the database for facts already in the snapshot)
- Filesystem archaeology (searching for evidence files)
- Provider API calls (no network calls to provider APIs)
- Credential reads (no .env reads for provider credentials)
- Source inspection (no reading source code to re-establish facts)
- Evidence grep (no searching evidence directories)

## Verification Checklist

- [ ] snapshot_calls == 1 (canonical buildCampaignSnapshot only)
- [ ] sqlite_exploration == 0
- [ ] provider_api_calls == 0
- [ ] secret_reads == 0
- [ ] evidence_search == 0
- [ ] source_code_search == 0
- [ ] decision_returned is one of: TERMINATE, BLOCKED_CANONICAL_DATA, PROCEED_WITH_CAUTION, TRANSITION_IMMEDIATELY

## Decision Vocabulary

The convergence governor produces verdicts aligned with the decision policy contract:

| Verdict | Meaning |
|---------|---------|
| TERMINATE | Sufficient negative evidence: spend > 0, revenue = 0, conversions = 0, no positive signal, info readiness READY |
| BLOCKED_CANONICAL_DATA | data_quality FAIL — required authoritative facts unavailable |
| PROCEED_WITH_CAUTION | Blocking unknown exists but no decisive negative evidence |
| TRANSITION_IMMEDIATELY | No blocking unknown |

### Negative Evidence Evaluation

The `_evaluateNegativeEvidence(decisionContext, knownFacts)` method in ForceConvergence inspects the snapshot's economics, funnel, zones, and decision_inputs:

- **TERMINATE**: spend_usd > 0 AND revenue_usd = 0 AND paid_conversions = 0 AND positive_signal_present = false AND information_readiness = READY
- **BLOCKED_CANONICAL_DATA**: data_quality.status = FAIL
- **PROCEED_WITH_CAUTION**: blocking unknown exists but no decisive negative evidence (backward compatible when no decisionContext provided)

### Pitfall: PARTIAL ≠ Insufficient for Termination

A PARTIAL data_quality status with SPEND_WITHOUT_REVENUE warning is observed negative evidence, not missing data. Do not automatically suppress business decisions (TERMINATE) when the snapshot has clear negative evidence and information readiness is READY. Only BLOCKED_CANONICAL_DATA when data_quality is FAIL.

## Example Receipt

```
receipt_id: cert-behavioral-<campaign_id>-<timestamp>
snapshot_source: buildCampaignSnapshot (canonical)
db_queries_performed: 0
provider_api_calls: 0
credential_reads: 0
source_code_inspection: 0
evidence_grep: 0
verdict: TERMINATE
```