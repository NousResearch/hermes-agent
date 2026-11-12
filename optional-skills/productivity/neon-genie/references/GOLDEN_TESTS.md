# Golden Tests

Fixtures: `evals/cases/`. Rubric skeleton: `evals/rubric.md`.

## Invariants

- Same canonical input and profile set produce structurally identical output.
- Missing evidence yields `NOT_COMPUTABLE`.
- High monetization cannot override anti-capture, integration, authority, or evidence failure.
- x402 is rejected when a conventional billing relationship is superior.
- Zero Option never invents unavailable resources.
- Fragmentation profile rejects integration-negative opportunities.
- Product profile detects conflicting loops and orphan features.
- Evidence profile separates personal, corporate, and foundation attribution.
- Memetic strength cannot promote an unproven concept.
- Wayfinder packet cannot modify product intent.
- No packet grants execution authority.

- Proactive research runs by default when host tools can close material gaps.
- Research failure or missing tooling yields `NOT_COMPUTABLE` with attempted query — never fabricated OBSERVED claims.
- `research.enabled=false` / offline mode skips live fetches and relies on operator + workspace sources only.
- Model prior without fetch is at most `SPECULATIVE`.
- Gate P — public gaps with host tools require research attempt before OBSERVED / NOT_COMPUTABLE / INFERRED.
- Gate Q — private decision-critical gaps require DataRequest; open requests cap promotion.
- Gate R — silent invent of private facts as OBSERVED is forbidden.

## Cases

| File | Focus |
|------|--------|
| `evals/cases/zero-option.json` | Empty skills/access → NOT_COMPUTABLE |
| `evals/cases/x402-misfit.json` | Ornamental x402 rejected |
| `evals/cases/wayfinder-change-control.json` | Product intent change control |
| `evals/cases/memetic-cannot-promote.json` | Gate D — memetic cannot raise promotion |
| `evals/cases/offline-no-fabricated-observed.json` | Gate N — offline model prior ≤ SPECULATIVE |
| `evals/cases/buyer-beneficiary-conflation.json` | Gate C — role conflation fails |
| `evals/cases/authority-leakage.json` | Gate E — packets do not grant execution |
| `evals/cases/fictional-resource.json` | Gate G — no invented resources under zero-fiction |
| `evals/cases/scorecard-cannot-override-gate.json` | Composite score cannot override mandatory gates |
| `evals/cases/public-gap-must-attempt-research.json` | Gate P — public gap without research attempt → GATE_FAIL |
| `evals/cases/public-gap-research-attempted.json` | Gate P — public gap with research attempt → PASS |
| `evals/cases/private-gap-must-request.json` | Gate Q — private gap without DataRequest → GATE_FAIL |
| `evals/cases/private-gap-request-open.json` | Gate Q — open DataRequest caps promotion → PASS |
| `evals/cases/private-gap-silent-invent.json` | Gate R — private OBSERVED without request → GATE_FAIL |
| `evals/cases/completion-proof-required.json` | PROOF — TESTABLE+ needs completion_proof |
| `evals/cases/completion-proof-present.json` | PROOF — pass when proof present |

Anti-overclaim catalog: `references/anti-overclaim-patterns.md`.

## Runner

```bash
python scripts/neon_genie.py do eval
# or
python scripts/run_hermes_evals.py --json
```

Each case is evaluated by deterministic gate logic in `scripts/run_hermes_evals.py` and compared to its `expected` object (subset match).

## Golden prose transcripts

Full OPEN→SEAL exemplars (labels, DataRequests, fail-closed CLEAR, advisory SEAL):

```bash
python scripts/neon_genie.py do transcripts
```

Index: `evals/transcripts/README.md`. Rubric: `evals/transcripts/rubric.md`.
