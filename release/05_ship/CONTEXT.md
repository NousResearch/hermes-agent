# Stage 05: Ship

## Inputs

- `02_plan/plan.json`
- `03_validate/validation.json`
- `04_communicate/changelog.md`
- `04_communicate/notes.md`

## Process

1. Read upstream artifacts.
2. Verify validation pass.
3. Write `ship_manifest.json` referencing approved upstream artifact paths.
4. Do not push tags, create releases, or mutate git.

## Outputs

- `ship_manifest.json`

## Human Check

Only execute publish/push actions after explicit approval outside this scaffold.
