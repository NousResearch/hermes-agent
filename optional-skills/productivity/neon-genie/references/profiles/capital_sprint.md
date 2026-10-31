# Capital Sprint Profile

Design and run **time-bounded capital raises** for nonprofits, membership orgs, and community institutions — with completion proof, governance gates, and learning-ledger feedback.

Operator-supplied org cases are **private**. Do not name real beneficiaries, donors, or institutions in shared corpus, profiles, or examples unless the operator explicitly publishes them.

## Triggers

fundraise, capital sprint, annual fund, donation drive, membership drive,
peer-to-peer fundraising, raise money by deadline, 501c3 campaign,
Agentic Giving campaign, donor sprint, nonprofit capital raise.

## Signal classes

- economic friction (weak gift funnel relative to membership/program revenue);
- institutional friction (board/donor trust, disclosure, receipt parity);
- technology frontier (agent-readable impact objects, consent-bound completion);
- repeated workarounds (stalled annual fund pages, zero-dollar P2P hosts).

## Runes

- `RUNE.NG.CAPITAL.INTAKE`
- `RUNE.NG.CAPITAL.SPRINT_DESIGN`
- `RUNE.NG.CAPITAL.IMPACT_OBJECT`
- `RUNE.NG.CAPITAL.WARM_NETWORK`
- `RUNE.NG.CAPITAL.MESSAGE_FRAMES`
- `RUNE.NG.CAPITAL.COMPLETION_PROOF`
- `RUNE.NG.CAPITAL.LEARN`

## Required fields

| Field | Rule |
|-------|------|
| org identity | Legal name, EIN, tax status; absence fails closed |
| deadline | Hard date; absence fails closed |
| floor + stretch | Dollar targets the org will publicly stand behind |
| revenue mix | Membership / program / gifts / grants (at least directional) |
| donation rails | Existing pages/processors (Every.org, Give Lively, Stripe, etc.) |
| warm network classes | Member / alumni / corporate / creator / cold |
| impact object | Machine-readable campaign card (goal, unit cost, beneficiaries, deadline) |
| completion proof | Funds received + receipt issued + donor/CRM record updated |
| authority | Advisory only unless human explicitly authorizes spend/publish |

## Pipeline bind

```text
ORG SIGNAL → DEADLINE + FLOOR → REVENUE MIX + RAILS
  → IMPACT OBJECT → SPRINT DESIGN (7/14/30)
  → WARM-NETWORK MAP → MESSAGE FRAMES → P2P HOST KIT
  → COMPLETION PROOF CHECKLIST → SCOREBOARD → LEARNING LEDGER
```

## Role map (Gate C — required separation)

Separate **beneficiary**, **donor/user**, **economic buyer** (often same as donor),
**authorizer** (board / ED), **payer**, and **risk bearer** (org legal entity).
Conflation of beneficiary and donor without disclosure fails CLEAR when AI mediates asks.

## Number discipline

| Situation | Label |
|-----------|--------|
| Cited public raise total / 990 line | `OBSERVED` |
| Model from cited membership × conversion | `INFERRED` |
| Unanchored “we can raise $X” | `SPECULATIVE` |
| No basis after research | `NOT_COMPUTABLE` |

## Governance gates

- No autonomous gift completion without donor consent bounds (ties to #02 spend controls).
- Agent-mediated asks require disclosure (nonprofit AI governance four-pillar bar).
- Do not invent tax deductibility, EIN, or match commitments.
- Stalled annual-fund pages are not completion proof; live funds + receipt are.
- **Privacy:** never commit real org names, donor lists, or campaign URLs into shared repo files unless the operator explicitly opts in to publication.

## Corpus wiring

| Signal file | Use |
|-------------|-----|
| `references/external-signals/2026-07-31-novel-fundraising-analysis.md` | Retention vs acquisition, mid-level gap |
| `references/external-signals/2026-07-31-agentic-giving-substrate.md` | Impact object, discovery, completion, record layers |
| `references/external-signals/2026-07-31-creator-economy-donation-mechanics.md` | Live/event tip paths, fee stacks |
| `references/external-signals/2026-07-31-nonprofit-ai-governance.md` | Disclosure, board comfort |
| `references/external-signals/2026-07-31-nonprofit-agentic-surfaces.md` | Sector adoption gap |
| `references/external-signals/2026-07-31-isenberg-02-agent-spend-controls.md` | Consent-bound agent spend |

## Outputs

- `CapitalSprintPacket`
- optional `ImpactObject` (embedded or standalone)
- `NeonGenieRunReceipt`
- post-deadline learning ledger entry (`do learn`) when proof obtained or failed

Schema: `schemas/capital-sprint-packet.schema.json`

## CLEAR rules

- Deadline + floor missing → fail closed (not a capital sprint).
- High aspirational goal cannot override missing rails, warm network, or completion proof design.
- Membership dues ≠ donations; message frames must separate them.
- Operator case results stay `PROPOSED` in learning ledger until human promotes observations.
- Shared artifacts use generic placeholders (`ORG`, `DEADLINE`, `FLOOR`); real identity stays in operator workspace only.

## Related profiles

Often co-triggered: `evidence_intelligence`, `commercial`, `memetic`, `opportunity_mining`, `agentic_services`.
