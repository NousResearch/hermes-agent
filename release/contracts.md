# Release Pipeline Contracts

## Invariants

- Stages are numbered and ordered.
- Each stage has a `CONTEXT.md`, a deterministic `run.sh`, and stable artifact paths.
- Scripts are idempotent: rerunning a stage overwrites only that stage's artifacts.
- No network, no interactive prompts, no speculative infrastructure.
- Human checks are explicit gating notes in `CONTEXT.md`; scripts do not block on them.

## Shared Artifacts

Artifacts live under the stage directory and are named in `CONTEXT.md`.

Downstream reuse:
- `01_audit/audit.json` is the single source of truth for change surface.
- `02_plan/plan.json` is the single source of truth for release scope.
- `03_validate/validation.json` is the pass/fail contract for release readiness.
- `04_communicate/changelog.md` and `notes.md` are generated from audit + plan.
- `05_ship/ship_manifest.json` references approved upstream artifacts.

## Failure Mode

On invalid input or missing dependencies, scripts write a deterministic
`<stage>/error.json` and exit non-zero. CI should treat any stage error as
pipeline failure.
