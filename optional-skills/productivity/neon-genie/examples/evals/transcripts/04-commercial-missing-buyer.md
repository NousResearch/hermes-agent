---
id: 04-commercial-missing-buyer
scenario: pricing request with collapsed commercial roles
profiles: [core, commercial, opportunity_mining]
research_mode: online
expected_promotion_max: CONCEPTUAL
---

# Transcript: Commercial model — missing buyer separation

## OPEN

- **Request:** “What should we charge? Revenue model for this product.”
- **Evidence:** No buyer map; user/beneficiary/payer collapsed into one role in the prompt.
- **Authority:** advisory_only; drafting only.

## ALIGN

- Commercial profile requires separation of beneficiary, user, buyer, authorizer, payer, risk bearer.
- Public research may inform **market pricing bands** for similar categories if tools available — labeled carefully.
- Private gap: who actually pays and who authorizes budget.

### DataRequest

```yaml
field: buyer_budget_authority
why_decision_critical: Cannot separate buyer vs beneficiary without budget authority
sensitivity: private
suggested_source: Operator CRM or declared stakeholder interview notes
blocks_promotion: true
status: open
```

### Research (public, if tools available)

- Attempt: public pricing pages for comparable category (if named).
- Without a named category + cite → any $ figure is at most `SPECULATIVE`.

## ASCEND

- Role map: all roles currently same string → Gate **C** risk.
- Pricing model: withhold firm price; offer scenario frames only after roles exist.
- First-customer profile: `NOT_COMPUTABLE` until buyer identified.

Claims:

- Buyer equals end user. — fails CLEAR unless evidenced — treat as unproven (`SPECULATIVE` / reject as fact)
- Market will pay $X. — `NOT_COMPUTABLE` or `SPECULATIVE` without cite + buyer

## CLEAR

- Gate **C**: buyer/beneficiary conflation without evidence of identity.
- Gate **B**: forecasts as fact forbidden.
- Gate **Q**: private budget authority needs DataRequest (emitted).
- Scorecard cannot override Gate C failure.

## SEAL

```yaml
status: GATE_FAIL
profiles_loaded: [core, commercial, opportunity_mining]
promotion_state: CONCEPTUAL
authority: advisory_only
grants_execution: false
human_review_required: true
open_blocking_requests: [buyer_budget_authority]
gates_failed: [C]
not_computable_fields: [pricing, first_customer, payer]
```

**Operator next step:** Provide role map (who pays, who uses, who authorizes); then re-run commercial simulation.
