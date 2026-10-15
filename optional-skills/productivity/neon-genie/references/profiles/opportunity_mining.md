# Opportunity Mining Profile

Find valuable state transitions that users cannot complete efficiently.

## Triggers

new venture, unmet need, market opportunity, blocked transition, weak signal, portfolio idea.

## Signal classes

- human friction;
- institutional friction;
- economic friction;
- technology frontier;
- repeated workarounds;
- exception patterns;
- unused rights, benefits, capacity, or capability.

## Runes

- `RUNE.NG.SENSE`
- `RUNE.NG.TRANSITION_MARKET`
- `RUNE.NG.INTERVENTION.SHAPE`
- `RUNE.NG.PORTFOLIO.ROUTE`

## Required fields

| Field | Rule |
|-------|------|
| affected user | Must be concrete; absence fails closed |
| economic buyer | Separate from beneficiary (Gate C) |
| authority holders | Who can approve / block |
| transition market | Current → desired state gap |
| coordination gap | Why the market has not cleared |
| service-first wedge | Smallest testable intervention |
| distribution | How the wedge reaches the user |
| completion proof | Externally checkable success |
| failure modes | How the opportunity dies |

## Pipeline bind

```text
SIGNAL → BLOCKED TRANSITION → OUTCOME MODEL → OPPORTUNITY THESIS
  → INTERVENTION → VALIDATION → SCORECARD → ROUTING
```

## Outputs

- `NeonGenieOpportunityPacket`
- optional commercial / agentic packets when co-triggered
- `NeonGenieRunReceipt`

Schema: `schemas/opportunity-packet.schema.json`

## CLEAR rules

- High monetization cannot override missing user, proof, or authority.
- Weak signals stay `SPECULATIVE` until evidence density rises.
