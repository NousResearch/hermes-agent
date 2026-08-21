# Hermes ICM Release Pipeline

Numbered headless stages with explicit contracts, deterministic rebuild scripts,
and focused tests. Every stage can run unattended and produces stable artifacts
for CI review.

## Stages

1. `01_audit` — inspect repo state, change surface, and risk flags.
2. `02_plan` — derive release scope, bump candidate, and validation plan.
3. `03_validate` — run deterministic checks against the planned scope.
4. `04_communicate` — generate changelog draft and release notes.
5. `05_ship` — assemble ship manifest from approved artifacts.

## Contracts

See `contracts.md`.

## Usage

```bash
# run all stages in order
for s in release/01_audit release/02_plan release/03_validate release/04_communicate release/05_ship; do
  bash "$s/run.sh"
done
```

## Tests

```bash
python -m pytest tests/release/test_release_pipeline.py -q
```
