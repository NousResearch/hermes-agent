# Impact Object Template

Machine-readable campaign card for `capital_sprint`.
Operator fills placeholders in **workspace only**. Do not commit real org identity to shared corpus without explicit publish opt-in.

```yaml
impact_object:
  schema_version: "1.0"
  campaign_id: "ORG-SPRINT-YYYYMMDD"   # opaque id; avoid public brand in shared files
  title: "<short campaign title>"
  goal_amount_usd: null                 # floor the org will stand behind
  stretch_amount_usd: null
  deadline: "YYYY-MM-DD"
  geography: "<city/region or multi>"
  beneficiaries:
    description: "<who benefits if funded>"
    count_estimate: null                # OBSERVED or INFERRED only
  unit_costs:                           # concrete, checkable units
    - label: "one day of operations"
      amount_usd: null
    - label: "one member-month of subsidized access"
      amount_usd: null
  use_of_funds:
    - category: "space | staff | programs | equipment | other"
      share_pct: null
      notes: ""
  verification:
    funds_received: false
    receipt_issued: false
    donor_record_updated: false
  donation_rails:
    - processor: "<Every.org | Give Lively | Stripe | other>"
      url: "<operator workspace only>"
  disclosure:
    tax_status: "501(c)(3) | other | unknown"
    ein: "<operator workspace only>"
    agent_mediated_ask: false           # must be true + disclosed if AI completes gift
  claim_labels:
    goal_amount_usd: "SPECULATIVE"      # until board-confirmed
    unit_costs: "INFERRED"              # until sourced from books
```

## Rules

- Completion proof = funds received + receipt + CRM/donor record — not page views.
- Membership dues ≠ donations; say so in every public frame.
- Shared examples use placeholders only (`ORG`, `DEADLINE`, `FLOOR`).
