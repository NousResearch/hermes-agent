# Hermes AlphaHunt Research Output Contract

This contract defines the standard Hermes research envelope that can be handed
to AlphaHunt P08 `create_project_from_research.py`.

Hermes research output must contain two synchronized artifacts:

1. A human-readable markdown research body.
2. A machine-readable AlphaHunt project research YAML envelope.

The YAML envelope is reference-only. Hermes and AlphaHunt do not execute trades,
place orders, sign transactions, place wagers, or manage funds.

## Envelope

```yaml
subject: "<human-readable name and idempotency anchor>"
kind: <stock|etf|protocol|commodity_theme|macro_event|industry_theme|market>
asset_class: "<known class or DRAFT:new_class>"
tickers: ["..."]
chain: "<chain>"
contract_addresses: ["0x..."]
aliases: ["...", "..."]
market_meta:
  rules: "<settlement rule text>"
  settlement: "<settlement condition>"
  deadline: "<ISO8601>"
research_markdown: |
  <Hermes markdown research body>
note:
  thesis: "<one-sentence core thesis>"
  key_assumptions: ["...", "..."]
  risk_triggers: ["...", "..."]
  invalidation_conditions: ["...", "..."]
  observables:
    - name: "<metric>"
      source: "<source>"
      threshold: "<threshold>"
      direction: "up|down"
  next_check_at: "2026-06-18T00:00:00+00:00"
  bull_case: ["..."]
  bear_case: ["..."]
  confidence: 0.6
  action_suggestion: "<reference-only suggestion>"
  source_references: ["https://..."]
```

## Required Fields

Validation must fail when any of these are missing or empty:

- `subject`
- `kind`
- `asset_class`
- `research_markdown`
- `note.thesis`
- `note.key_assumptions`
- `note.risk_triggers`
- `note.invalidation_conditions`
- `note.observables`
- `note.next_check_at`
- `note.source_references`

`note.observables` must contain at least one item. Each item must include
`name`, `source`, `threshold`, and `direction`; `direction` must be `up` or
`down`.

`note.next_check_at` must be parseable ISO8601.

`source_references` belongs under `note.source_references`. Top-level
`source_references` must fail validation so the envelope stays aligned with the
AlphaHunt `project_research_note` contract.

## Kind

`kind` is a closed enum:

- `stock`
- `etf`
- `protocol`
- `commodity_theme`
- `macro_event`
- `industry_theme`
- `market`

Unknown kinds must fail validation.

## Asset Class

`asset_class` must be one of the known Hermes/AlphaHunt classes accepted by
`gateway.alphahunt.research_yaml`, or it must use the draft form:

```yaml
asset_class: "DRAFT:new_class_name"
```

New asset classes without the `DRAFT:` prefix must fail validation.

## Market Boundary

`market` research is only research. It is not a betting, trading, execution, or
funds-management instruction.

Additional `market` requirements:

- `market_meta.rules`, `market_meta.settlement`, and `market_meta.deadline` are required.
- `market_meta.deadline` must be parseable ISO8601.
- `note.action_suggestion` must be one of `ignore`, `observe`, `research`,
  `manual_review`, or `no_participation`.
- The output must state `reference-only`, `no_participation`, and `no_execution`.
- The output must not contain English execution or wagering trigger words such
  as `bet`, `wager`, `execute`, `stake`, or `kelly`.

Chinese operator-facing language may additionally state: 不自动下注, 不自动交易,
不保证收益.

## Validator

Use the pure Hermes-side validator:

```bash
python -m gateway.alphahunt.research_yaml --validate docs/alphahunt/samples/protocol_ethena.yaml
```

The validator does not read secrets, call AlphaHunt APIs, write databases, call
live endpoints, or access external data sources.
