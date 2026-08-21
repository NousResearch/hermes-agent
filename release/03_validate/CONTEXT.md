# Stage 03: Validate

## Inputs

- `02_plan/plan.json`
- Repo root at `/home/gfardad/Projects/hermes-agent`

## Process

1. Read `02_plan/plan.json`; fail if missing or `ok != true`.
2. Run deterministic validations:
   - stage contract presence for all numbered stages
   - deterministic script executable bits
   - focused test discovery for `tests/release/`
3. Write `validation.json`.

## Outputs

- `validation.json`

## Human Check

None required for CI. Gate is artifact pass/fail plus focused test results.
