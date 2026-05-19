# Vendored Ouroboros Interview/Seed subset

Upstream: https://github.com/Q00/ouroboros
Commit: `b6169f82a12407722522cc5cf04e1a85a4ac5de7`

Purpose: BO-062 replaces the Hermes-local fake `/ouro-intake` Seed brain with a vendored upstream Interview/Seed/Core subset plus a thin Hermes wrapper.

Included intentionally:
- `core/seed.py` copied from upstream: immutable `Seed`, `SeedMetadata`, `BrownfieldContext`, `OntologySchema`, `OntologyField`, `EvaluationPrinciple`, `ExitCondition`.
- `bigbang/seed_generator.py` copied/adapted from upstream SeedGenerator parsing/build semantics; Hermes gateway uses structured confirmed values or Hermes-agent structured extraction instead of issuing provider calls in this no-restart/no-secret slice.
- `auto/ledger.py`, `auto/gap_detector.py`, `auto/grading.py`, `auto/seed_reviewer.py`, `auto/seed_repairer.py` copied from upstream with import paths rewritten to the vendored namespace.
- `bigbang/interview.py` minimal vendored subset preserving upstream `InterviewStatus`, `InterviewRound`, and `InterviewState` shapes needed by gateway-mode intake persistence.
- Interview/Seed Closer prompt assets for behavioral reference.

Excluded intentionally:
- upstream execution/orchestrator/runners; Hermes/Kanban remains execution authority.
- provider-specific runtime adapters; gateway mode uses Hermes wrapper and must not dispatch workers.
