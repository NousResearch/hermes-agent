# Workspace Config Governance

## Purpose

This document is the central operating guide for managing `.env`,
`.env.example`, and `.gitignore` drift across the local Viber Project
workspace.

The first phase is a control plane: scan, report, and track issues. It does not
rewrite project secrets or silently edit project ignore files.

## Plain Language Model

- `.env` = real local secret/config values. Do not commit and do not print.
- `.env.example` = public template that shows required key names with safe
  placeholder values.
- schema/registry = the tracking table that explains what each key is for.
- `.gitignore` = the safety net that keeps secrets and generated files out of
  git.
- comply table = numeric progress table that prevents AI from claiming work is
  complete without evidence.

## Commands

Audit the whole workspace:

```bash
venv/bin/python scripts/workspace_config_audit.py \
  --root "/Users/rattanasak/Documents/Viber Project" \
  --max-depth 5 \
  --max-findings 40
```

Get a short summary only:

```bash
venv/bin/python scripts/workspace_config_audit.py \
  --root "/Users/rattanasak/Documents/Viber Project" \
  --max-depth 5 \
  --summary-only
```

Generate a Markdown report:

```bash
venv/bin/python scripts/workspace_config_audit.py \
  --root "/Users/rattanasak/Documents/Viber Project" \
  --max-depth 5 \
  --format markdown \
  --max-findings 40
```

Check phase compliance:

```bash
venv/bin/python scripts/workspace_config_comply.py \
  --issue-file .hermes/issues/phase-009-workspace-config-governance.md
```

Dry-run safe remediation:

```bash
venv/bin/python scripts/workspace_config_remediate.py \
  --root "/Users/rattanasak/Documents/Viber Project" \
  --max-depth 5
```

Apply reversible `.gitignore` remediation and write central reports:

```bash
venv/bin/python scripts/workspace_config_remediate.py \
  --root "/Users/rattanasak/Documents/Viber Project" \
  --max-depth 5 \
  --apply \
  --report-dir docs/workspace-config-reports
```

Generate central env schema and next remediation queue:

```bash
venv/bin/python scripts/workspace_config_schema.py \
  --root "/Users/rattanasak/Documents/Viber Project" \
  --max-depth 5 \
  --output-dir docs/workspace-config-reports
```

Apply managed `.env.example` blocks from the schema:

```bash
venv/bin/python scripts/workspace_config_apply_examples.py \
  --schema docs/workspace-config-reports/env-schema.json \
  --apply \
  --report-dir docs/workspace-config-reports
```

## Finding Codes

| code | meaning | owner action |
|---|---|---|
| `gitignore-env-missing` | Project has env files, but `.gitignore` does not clearly ignore secret env files. | Add managed ignore block later. |
| `tracked-env-file` | Git already tracks a secret-bearing `.env` file. | Rotate secret if needed, remove from git history separately, and stop tracking. |
| `env-key-unused` | Key exists in secret env files but is not seen in code or examples. | Review whether it is stale, dynamic, or missing from references. |
| `code-key-missing-from-env-example` | Code references an env key that is missing from templates. | Add to `.env.example` or future schema if real. |

## Phase Rule

Before reporting a phase as complete:

1. Run the tests listed in the issue file.
2. Run a real workspace smoke check.
3. Run the compliance checker.
4. Report localhost/VPS status explicitly. For this control plane there is no
   server surface, so the correct result is `not_applicable` unless a future UI
   or service is added.

## Next Remediation Order

After the control plane is verified, remediate projects in this order:

1. Projects with tracked env files.
2. Projects missing `.gitignore` env coverage.
3. Projects with many code keys missing from `.env.example`.
4. Projects with many unused env keys.

Every remediation must be a separate phase with its own issue file and fresh
verification.

## Completed Remediation - Phase 010

The first remediation pass applied only the safe `.gitignore` managed block.
It wrote changes to:

- `/Users/rattanasak/Documents/Viber Project/Office Project/500K Project/.gitignore`
- `/Users/rattanasak/Documents/Viber Project/Office Project/Support Center/.gitignore`

The post-apply dry-run reported `0` remaining `.gitignore` writes. Central
non-secret reports were generated under:

```text
docs/workspace-config-reports/
```

Do not use the generated env registry as approval to auto-edit `.env.example`
files. Treat it as review input for the next phase.

## Completed Schema - Phase 011

The central schema pass generated:

- `docs/workspace-config-reports/env-schema.json`
- `docs/workspace-config-reports/env-schema.md`
- `docs/workspace-config-reports/next-remediation-queue.md`

The schema is intentionally conservative. It classifies keys into:

- `present_in_example` = already represented in `.env.example`
- `missing_from_example` = code references the key, but template does not list it
- `unused_env` = key exists in secret env files but was not found in code/templates
- `review` = ambiguous, example-only, or low-confidence item

Use `next-remediation-queue.md` as the input for the next phase. Do not
bulk-apply every row automatically; project-level template edits should use the
schema status and confidence fields.

## Completed `.env.example` Remediation - Phase 012

Managed `.env.example` blocks were applied across the workspace. The process
was intentionally reversible:

- Human-authored `.env.example` lines were preserved.
- Generated keys were isolated between `hermes-managed` markers.
- Values were left blank.
- Secret values were never copied from `.env`.

Final schema status after remediation:

| status | count |
|---|---:|
| `present_in_example` | 4537 |
| `missing_from_example` | 0 |
| `unused_env` | 333 |
| `review` | 344 |

The remaining audit findings are `unused_env` info items only. Those should be
handled in a later cleanup phase because removing real env keys requires
project-by-project review.
