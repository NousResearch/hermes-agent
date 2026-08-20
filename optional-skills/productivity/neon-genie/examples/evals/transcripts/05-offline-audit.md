---
id: 05-offline-audit
scenario: client audit package while offline
profiles: [core, audit_delivery]
research_mode: offline
expected_promotion_max: TESTABLE
---

# Transcript: Offline audit delivery

## OPEN

- **Request:** Diagnostic package + cost of inaction + optional implementation offer.
- **Mode:** `research.enabled=false` / offline.
- **Authority:** advisory_only; no contact or publish.

## ALIGN

- Sources: operator-supplied notes only (none in this exemplar beyond the request).
- Live research **skipped** correctly under offline mode.
- Gaps: quantified cost of inaction, market comps — cannot fabricate.

## ASCEND

- Current state: partial — only request text. Cartography limited.
- Cost of inaction: qualitative only; no invented $ losses (Gate **J**).
- Target state: legible decision package for human review.
- Offer map: optional, non-coercive, clearly non-authorizing.

Claims:

- Offline mode is active. — `OBSERVED` (request flag)
- Industry average loss from delay is $Y. — **forbidden as OBSERVED**; mark `NOT_COMPUTABLE` or omit
- Inaction has opportunity cost in time/risk. — `SPECULATIVE` / qualitative without measurement

## CLEAR

- Gate **N**: offline model prior cannot become `OBSERVED` market numbers.
- Gate **J**: no cost-of-inaction theater with fake precision.
- Gate **E**: diagnosis ≠ authorization to implement or contact client.

## SEAL

```yaml
status: PROPOSED
profiles_loaded: [core, audit_delivery]
promotion_state: TESTABLE
authority: advisory_only
grants_execution: false
human_review_required: true
research_attempts: []
data_requests:
  - field: measured_cost_of_inaction_inputs
    sensitivity: operator
    blocks_promotion: false
    status: open
    why_decision_critical: Quantitative COI needs operator metrics
not_computable_fields: [quantified_cost_of_inaction]
note: "Offline audit is diagnostic scaffolding only; re-run online for external facts."
```
