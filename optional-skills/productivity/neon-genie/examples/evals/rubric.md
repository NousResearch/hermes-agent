# Neon Genie Eval Rubric (skeleton)

Wave 1 captures invariants for later automated runners. Fixtures live in `evals/cases/`.

## Global invariants

1. Missing executable evidence → `NOT_COMPUTABLE` (never fabricate).
2. High monetization or memetic strength cannot override authority, evidence, integration, or anti-capture gate failures.
3. x402 rejected when ornamental or when conventional billing is superior.
4. Zero Option never invents unavailable skills, capital, or access.
5. Wayfinder execution packet cannot modify product intent; intent changes require Neon Genie review.
6. No packet grants execution, spending, or publishing authority.
7. Same canonical input + profile set → structurally stable labeled outputs (Wave 2+ determinism checks).
8. Proactive research is default; offline / `research.enabled=false` skips live fetches.
9. Model prior without fetch is at most `SPECULATIVE`.
10. Gate P — public material gaps with host tools require a research attempt before OBSERVED / NOT_COMPUTABLE / INFERRED.
11. Gate Q — private decision-critical gaps require a DataRequest; open requests cap promotion.
12. Gate R — silent invent of private facts as OBSERVED is forbidden.

## Fixture index

| Case | Expectation |
|------|-------------|
| `zero-option.json` | No skills/access → NOT_COMPUTABLE |
| `x402-misfit.json` | Ornamental x402 rejected |
| `wayfinder-change-control.json` | Intent change blocked without Neon review |
| `memetic-cannot-promote.json` | High memetic score cannot raise promotion past failed evidence |
| `offline-no-fabricated-observed.json` | Offline model prior cannot be OBSERVED |
| `buyer-beneficiary-conflation.json` | Collapsed commercial roles fail Gate C |
| `authority-leakage.json` | Packets never grant spend/publish/execute |
| `fictional-resource.json` | Gate G — invented assets under no-fiction fail |
| `scorecard-cannot-override-gate.json` | High score cannot clear failed mandatory gate |
| `public-gap-must-attempt-research.json` | Gate P — public gap without research attempt → GATE_FAIL |
| `public-gap-research-attempted.json` | Gate P — public gap with research attempt → PASS |
| `private-gap-must-request.json` | Gate Q — private decision-critical gap without DataRequest → GATE_FAIL |
| `private-gap-request-open.json` | Gate Q — DataRequest open + blocks_promotion → PASS, promotion_capped |
| `private-gap-silent-invent.json` | Gate R — private OBSERVED without DataRequest → GATE_FAIL |
