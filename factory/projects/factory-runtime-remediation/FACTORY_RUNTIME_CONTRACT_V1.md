# Factory Runtime Contract v1 — Repo-First Enforcement

## Purpose

This contract turns Jean's correction into runtime behavior: a Factory project is not complete because a worker says it is complete. It is complete only when Factory DB state, project-local repo artifacts, documentation index, commit checkpoints, gates, and evidence agree.

## Contract scope implemented in this increment

### RFC-1: Required docs must be indexed

The required Factory document pack is already defined in `FACTORY_REQUIRED_DOCS`. Before this increment, reconciliation checked whether each file existed. That was not enough: builders enter a project through the repository, and `DOCUMENTATION_INDEX.md` is their canonical map.

New behavior:

- If required docs are physically present but missing from `DOCUMENTATION_INDEX.md`, reconciliation emits `docs_not_indexed`.
- The generated recovery task is `R2b — Reconciliation: update DOCUMENTATION_INDEX.md`.
- Critical readiness for high/critical projects blocks with `documentation index missing required docs: ...`.

### RFC-2: Project-local artifacts need a git checkpoint

A repo-first Factory must leave normal software-company history. Untracked or modified docs under the project artifact directory are not valid delivery evidence because they are not preserved in commits.

New behavior:

- Reconciliation runs `git status --porcelain -- <artifact_dir>` for the project-local Factory docs path.
- If git reports modified/untracked artifact files, reconciliation emits `uncommitted_project_artifacts`.
- The generated recovery task is `R2c — Reconciliation: commit project-local Factory artifacts`.
- Critical readiness for high/critical projects blocks with `uncommitted project-local factory artifacts: ...`.

### RFC-3: Explicit exception only

Commit checkpoint enforcement can be waived only by project metadata such as:

- `repo_commit_waived`
- `commit_checkpoint_waived`
- `uncommitted_artifacts_waived`

These are intended for Jean-authorized exceptional cases, not as a normal path.

## Runtime touch points

| File | Change |
|---|---|
| `hermes_cli/factory_pg.py` | Added doc-index validator, git porcelain artifact check, new anomaly codes, critical readiness enforcement. |
| `tests/hermes_cli/test_factory_canonical_runtime.py` | Added RED/GREEN tests for docs-not-indexed, uncommitted artifacts, and critical readiness blocking. |
| `factory/projects/factory-runtime-remediation/*` | Added this contract, project global vision, and updated sprint/task/delivery docs. |

## New reconciliation anomaly codes

| Code | Meaning | Recovery owner |
|---|---|---|
| `docs_not_indexed` | Required docs exist but are missing from `DOCUMENTATION_INDEX.md`. | `factory-reporter` |
| `uncommitted_project_artifacts` | Project-local docs/artifacts have no clean git commit checkpoint. | `factory-reporter` |

## Why this matters for recursive autonomy

The Factory can now repair itself with the same standard it will apply to client projects:

1. Open/select repo/worktree.
2. Create/maintain documentation pack.
3. Index docs for builder context.
4. Implement with tests.
5. Detect repo drift through reconciliation.
6. Require commit checkpoint before delivery readiness.

That is the minimum loop for autonomous recursive improvement: the system changes itself, but the same gates catch its own drift.

## Validation evidence

Initial focused verification in this increment:

```text
/home/jean/Projects/hermes-agent-original/venv/bin/python -m pytest tests/hermes_cli/test_factory_canonical_runtime.py -q
20 passed in 1.66s

/home/jean/Projects/hermes-agent-original/venv/bin/python -m pytest tests/hermes_cli/test_factory_canonical_runtime.py tests/tools/test_factory_tools.py -q
22 passed, 1 warning in 7.80s
```
