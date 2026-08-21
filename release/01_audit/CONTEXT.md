# Stage 01: Audit

## Inputs

- Repo root at `/home/gfardad/Projects/hermes-agent`
- Git working tree with current HEAD and dirty state
- Optional env: `ICM_AUDIT_REF` — override audit base ref/commit.
  Default: `HEAD`.

## Process

1. Resolve base ref from `ICM_AUDIT_REF` or `HEAD`.
2. Collect:
   - `head_commit`
   - `head_branch`
   - `dirty`
   - `changed_files`
   - `changed_langs`
   - `risk_flags`
3. Write `audit.json` deterministically.
4. Do not modify source files.

## Outputs

- `audit.json`

## Human Check

Review `audit.json` for unexpected large change surface or high-risk paths
before continuing to `02_plan`.
