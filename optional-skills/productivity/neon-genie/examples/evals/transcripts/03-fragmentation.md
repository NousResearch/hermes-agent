---
id: 03-fragmentation
scenario: multi-portal status friction scan
profiles: [core, fragmentation, product_architecture]
research_mode: online
expected_promotion_max: MAPPED
---

# Transcript: Fragmentation scan

## OPEN

- **Request:** Map workflow/status friction across portals; propose defrag only if burden < capturable value.
- **Current state:** Many portals, repeated handoffs, incompatible status views.
- **Authority:** advisory_only.

## ALIGN

- Fragmentation classes of interest: workflow, identity, payment, data.
- Non-goals: build a platform “because everything is fragmented.”
- Public research: optional public integration patterns; no private API probing.

### DataRequest

```yaml
field: authority_and_api_access_for_portals
why_decision_critical: Cannot OBSERVE integration burden without access or architecture docs
sensitivity: private
suggested_source: Operator system inventory + access rights
blocks_promotion: true
status: open
```

## ASCEND

- Friction site (proposed): multi-portal status lookup, daily, high severity.
- Coordination gap: systems work alone; handoffs re-enter manual reconciliation.
- Defrag candidate: status unifier archetype — **reject if integration burden exceeds value**.
- Integration feasibility: `NOT_COMPUTABLE` until access request satisfied.

Claims:

- Repeated handoffs exist as operator-described. — `OBSERVED` (request text) or `INFERRED` if only paraphrased
- Status unifier is net positive ROI. — `NOT_COMPUTABLE` / `SPECULATIVE` without measurement

## CLEAR

- Gate **K**-adjacent: unknown critical integration access → do not promote as BUILD_READY.
- Reject integration-negative opportunities when burden > value (once measurable).
- No silent subsystem duplicate without wrapper classification (Gate M).

## SEAL

```yaml
status: PROPOSED
profiles_loaded: [core, fragmentation, product_architecture]
promotion_state: MAPPED
authority: advisory_only
grants_execution: false
human_review_required: true
open_blocking_requests: [authority_and_api_access_for_portals]
not_computable_fields: [integration_feasibility, capturable_value_quantified]
```
