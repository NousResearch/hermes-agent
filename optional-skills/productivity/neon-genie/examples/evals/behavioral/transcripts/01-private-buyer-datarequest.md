---
id: behavioral-01-private-buyer
scenario: pricing without defined buyer — private gap must request
profiles: [core, commercial, opportunity_mining]
research_mode: online
expected_promotion_max: CONCEPTUAL
---

# Behavioral: private buyer → DataRequest

## OPEN

- **Request:** “We have an audio continuity tool but no defined buyer. What should we charge?”
- **Evidence:** Product category named; buyer/payer/authorizer **not** supplied.
- **Authority:** advisory_only; no spend, publish, or repo mutation.

## ALIGN

- Public research may inform **category pricing bands** if host tools run — labeled carefully.
- Private decision-critical gap: who pays and who authorizes budget.
- Emit DataRequest rather than invent a buyer persona.

### DataRequest

```yaml
field: buyer
why_decision_critical: Pricing and first-offer design require a paying role distinct from end user
sensitivity: private
suggested_source: Operator CRM, sales notes, or declared stakeholder map
blocks_promotion: true
status: open
```

### Research (public)

- Attempt: comparable public list prices for audio/dev tooling (if tools available).
- Without a cited source, any $ figure remains `SPECULATIVE` at most.

## ASCEND

Claims:

- Product category is audio continuity tooling. — `OBSERVED` (operator statement)
- A defined buyer exists. — `NOT_COMPUTABLE` (private gap; DataRequest open)
- Firm price recommendation is ready. — `NOT_COMPUTABLE` (depends on buyer)
- Market will pay $49/mo. — `SPECULATIVE` (no cite + no buyer)

## CLEAR

- Gate **Q**: private decision-critical buyer gap requires DataRequest (emitted, open).
- Gate **C**: do not collapse user and buyer without evidence.
- Gate **B**: do not present invented price as fact.
- Promotion capped while blocking DataRequest open.

## SEAL

```yaml
status: GATE_FAIL
profiles_loaded: [core, commercial, opportunity_mining]
promotion_state: CONCEPTUAL
authority: advisory_only
grants_execution: false
human_review_required: true
data_requests:
  - field: buyer
    sensitivity: private
    blocks_promotion: true
    status: open
open_blocking_requests: [buyer]
gates_failed: [Q]
not_computable_fields: [buyer, pricing, first_customer]
```

**Operator next step:** Identify buyer/payer/authorizer; re-run commercial simulation.
