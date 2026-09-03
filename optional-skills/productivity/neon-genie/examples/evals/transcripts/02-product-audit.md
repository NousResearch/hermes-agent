---
id: 02-product-audit
scenario: product coherence audit with commercial + wayfinder handoff
profiles: [core, product_architecture, commercial, wayfinder_handoff, evidence_intelligence]
research_mode: online
expected_promotion_max: SPEC_COMPLETE
---

# Transcript: Product audit + Wayfinder handoff

## OPEN

- **Request:** Audit product coherence; produce Wayfinder-ready handoff without rewriting intent.
- **Target user:** Operator building a governed multi-skill product.
- **Constraints:** advisory_only; do not modify repositories.
- **Artifacts:** `NeonGenieProductPacket`, `WayfinderExecutionPacket`, receipt.

## ALIGN

- Operator files rank highest; workspace may supply partial intent.
- Gap-detect: critical integration access unknown; buyer map incomplete.
- Research plan: public comparable product boundaries (host web tools) if available.
- Auto-load `evidence_intelligence` for external product-pattern facts.

### Research attempt (public)

- Query: comparable multi-skill product boundaries / handoff patterns.
- Outcome: illustrative public patterns only; cite if fetched. — any concrete competitor claim remains `SPECULATIVE` without URL.

### DataRequest (private)

```yaml
request_id: dr-product-access
field: critical_integration_access
why_decision_critical: Integration feasibility cannot be OBSERVED without declared access
sensitivity: private
suggested_source: Operator environment credentials or access inventory
blocks_promotion: true
status: open
```

## ASCEND

- Product boundary (proposed): in = intent, scorecard, handoff stub; out = spend, publish, eng status.
- Core loop: OPEN→SEAL advisory loop for product intent.
- Commercial roles: beneficiary/user/buyer not fully separated — flag for request.
- Wayfinder handoff: objective + non-goals + acceptance criteria; **no intent rewrite**.

Claims:

- Handoff must not modify product intent. — `OBSERVED` (Wayfinder contract)
- Integration is feasible in production. — `NOT_COMPUTABLE` (open DataRequest)
- Multi-skill handoff is a common pattern. — `SPECULATIVE` (unless cited)

## CLEAR

- Gate **Q**: private access without invent → DataRequest open.
- Gate **H**: handoff cannot rewrite intent; `product_intent_changes_require_neon_genie_review: true`.
- Gate **E**: no execution authority in packets.
- Composite score does not clear open blocking request.

## SEAL

```yaml
status: PROPOSED
profiles_loaded: [core, product_architecture, commercial, wayfinder_handoff, evidence_intelligence]
promotion_state: SPEC_COMPLETE
authority: advisory_only
grants_execution: false
human_review_required: true
product_intent_changes_require_neon_genie_review: true
open_blocking_requests:
  - critical_integration_access
research_attempts:
  - query: "comparable multi-skill product boundaries"
    outcome: partial_or_tooling_dependent
```

**Operator next step:** Satisfy DataRequest (access inventory); then re-score integration feasibility.
