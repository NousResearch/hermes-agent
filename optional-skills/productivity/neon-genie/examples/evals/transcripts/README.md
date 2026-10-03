# Golden prose transcripts

Exemplars of how Neon Genie should run **OPEN → ALIGN → ASCEND → CLEAR → SEAL** in conversation.

These are **not** live model outputs. They are skill-contract goldens: labels, research/find discipline, `DataRequest`s, fail-closed gates, and advisory authority.

## Index

| Transcript | Scenario | Profiles | Key lesson |
|------------|----------|----------|------------|
| `01-zero-option-empty.md` | No skills/access, zero capital | core, zero_option | Honest `NOT_COMPUTABLE` (Gate G) |
| `02-product-audit.md` | Product coherence + handoff | core, product_architecture, commercial, wayfinder_handoff | Intent freeze + private access request |
| `03-fragmentation.md` | Multi-portal friction | core, fragmentation | Defrag only if burden < value; access NC |
| `04-commercial-missing-buyer.md` | Pricing ask without buyer map | core, commercial | Gate C / role separation + DataRequest |
| `05-offline-audit.md` | Client audit offline | core, audit_delivery | No OBSERVED from model prior (Gate N) |
| `06-agentic-x402-misfit.md` | Agentic + x402 on consulting | core, agentic_services | Gate F ornamental x402 REJECT |
| `07-memetic-cannot-promote.md` | Viral name, weak evidence | core, memetic | Gate D memetic cannot promote |
| `08-evidence-intelligence.md` | Public find + private CRM | core, evidence_intelligence | Find/request + claim ledger |
| `09-opportunity-mining.md` | First-cash blocked transition | core, opportunity_mining | Thesis + completion_proof + DataRequest |

## Rubric

See `rubric.md`. Structural checks:

```bash
python scripts/neon_genie.py do transcripts
# or
python scripts/check_transcripts.py
```

## How to use

1. Operators: read a transcript before a similar Hermes run.
2. Skill authors: keep transcripts aligned when changing doctrine.
3. CI: `do transcripts` must PASS.
