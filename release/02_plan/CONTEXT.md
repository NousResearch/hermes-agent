# Stage 02: Plan

## Inputs

- `01_audit/audit.json`
- Repo state at `/home/gfardad/Projects/hermes-agent`
- Optional env: `ICM_PLAN_VERSION` — override candidate version string.
  Default: derived from current git tags or `0.0.0-audit`.

## Process

1. Read `01_audit/audit.json`; fail if missing or `ok != true`.
2. Derive:
   - `version_candidate`
   - `scope_summary`
   - `changed_files`
   - `risk_flags`
   - `validation_gates`
3. Write `plan.json` deterministically.

## Outputs

- `plan.json`

## Human Check

Confirm `version_candidate` and `validation_gates` before running `03_validate`.
