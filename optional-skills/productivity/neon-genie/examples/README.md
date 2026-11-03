# Neon Genie examples

Illustrative request briefs and sample packets for Hermes / packaging CLI.  
They are **not** execution authority.

## Install

```bash
./install.sh
```

## Briefs

| File | Use |
|------|-----|
| `product-audit.brief.yaml` | Product architecture + commercial + wayfinder handoff |
| `zero-option.brief.yaml` | Empty skills/access → honest `NOT_COMPUTABLE` |
| `zero-option-with-skills.brief.yaml` | Declared skills → micro-loop stub |
| `fragmentation.brief.yaml` | Multi-portal friction / defrag scan |
| `commercial.brief.yaml` | Pricing scaffold + buyer DataRequest |
| `audit.brief.yaml` | Offline audit diagnostic package |
| `agentic.brief.yaml` | Agentic graph + x402 misfit |
| `memetic.brief.yaml` | Name pressure without promotion override |
| `evidence.brief.yaml` | Evidence intelligence find/request scaffold |
| `opportunity.brief.yaml` | Opportunity mining with completion proof |
| `privacy-external-research.brief.yaml` | Explicit `privacy.mode` + purpose-bound consents (no global disable) |

## Sample packets

| File | Validate |
|------|----------|
| `packets/sample-opportunity.packet.json` | `do validate --type opportunity` |
| `packets/sample-receipt.packet.json` | `do validate --type receipt --strict-authority` |
| `packets/sample-privacy-context.json` | `do validate --type privacy` |
| `packets/sample-external-action.json` | Schema: external action required fields |
| `packets/sample-purpose-bound-consent.json` | Schema: purpose_bound consent only |

Privacy boundary: `PRIVACY.md`, `python scripts/neon_genie.py do privacy --json`.

## Packaging recipes

```bash
python scripts/neon_genie.py do recipe --list
python scripts/neon_genie.py do recipe --name product-audit
python scripts/neon_genie.py do recipe --name zero-option
python scripts/neon_genie.py do recipe --name zero-option-executable
python scripts/neon_genie.py do recipe --name fragmentation
python scripts/neon_genie.py do recipe --name commercial
python scripts/neon_genie.py do recipe --name audit
python scripts/neon_genie.py do recipe --name agentic
python scripts/neon_genie.py do recipe --name memetic
python scripts/neon_genie.py do recipe --name evidence
python scripts/neon_genie.py do recipe --name opportunity
python scripts/neon_genie.py do doctor
```

Request envelope: `templates/request.yaml`. Gallery: `examples/gallery/`.
