# Commercial Modeling Profile

## Triggers

pricing, buyer, revenue, costs, market pressure, business model, first customer.

## Runes

- `RUNE.NG.COMMERCIAL.BUYER_MAP`
- `RUNE.NG.COMMERCIAL.MODEL`
- `RUNE.NG.COMMERCIAL.SIMULATE`
- `RUNE.NG.COMMERCIAL.PRESSURE_SCAN`
- `RUNE.NG.COMMERCIAL.PORTFOLIO`

## Role map (required separation — Gate C)

Separate **beneficiary**, **user**, **buyer**, **authorizer**, **payer**, and **risk bearer**. Conflation fails commercial CLEAR.

## Generate

- startup and operating cost ranges (with provenance);
- pricing model and packaging;
- conservative, balanced, and aggressive scenarios;
- first-customer profile;
- acquisition channels;
- integration economics;
- data moat (only if evidenced);
- network effect (only if evidenced);
- regulatory and platform pressure;
- first 90-day validation plan.

## Number discipline

| Situation | Label |
|-----------|--------|
| Cited public price / filing | `OBSERVED` |
| Model from cited inputs | `INFERRED` |
| Unanchored projection | `SPECULATIVE` |
| No basis after research | `NOT_COMPUTABLE` |

Unsupported numerical projections remain `SPECULATIVE` or `NOT_COMPUTABLE` — never `OBSERVED` (Gate B).

## Outputs

- `CommercialSimulationPacket`
- `NeonGenieRunReceipt`

Schema: `schemas/commercial-simulation.schema.json`
