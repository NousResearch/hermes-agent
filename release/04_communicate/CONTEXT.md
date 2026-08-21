# Stage 04: Communicate

## Inputs

- `01_audit/audit.json`
- `02_plan/plan.json`

## Process

1. Read upstream audit and plan artifacts.
2. Generate deterministic `changelog.md` and `notes.md` from their contents.
3. Do not call network endpoints.

## Outputs

- `changelog.md`
- `notes.md`

## Human Check

Review generated changelog/notes for accuracy before ship.
