# Audit-First Delivery Profile

Generate diagnostic packages that make the implementation decision legible without manipulating the client.

## Triggers

client audit, cost of inaction, diagnostic package, implementation offer, roadmap for stakeholder.

## Runes

- `RUNE.NG.AUDIT.CURRENT_STATE`
- `RUNE.NG.AUDIT.COST_OF_INACTION`
- `RUNE.NG.AUDIT.TARGET_STATE`
- `RUNE.NG.AUDIT.ROADMAP`
- `RUNE.NG.AUDIT.OFFER_MAP`

## Packet contents

| Section | Rule |
|---------|------|
| current-state cartography | Prefer `OBSERVED` / cited workspace facts |
| observed gaps | Explicit, falsifiable |
| cost of inaction | Quantified **or** qualified; never fabricate $ (Gate J) |
| target architecture | Bound to product intent if product profile co-loaded |
| intervention sequence | Ordered, reversible where possible |
| validation gates | External proof for each major step |
| optional implementation offer | Clearly optional; no pressure tactics |
| evidence manifest | Full source list |

## Offer map discipline

- Separate diagnosis from sales.
- Do not invent urgency or losses.
- Implementation remains operator/downstream choice.

## Outputs

- `AuditDeliveryPacket`
- `NeonGenieRunReceipt`

Schema: `schemas/audit-delivery-packet.schema.json`
