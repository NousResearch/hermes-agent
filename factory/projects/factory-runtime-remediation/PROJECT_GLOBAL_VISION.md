# Project Global Vision — Factory Runtime Remediation

## Canonical objective

Convert Zeus Software Factory from prompt-driven task execution into a repo-first, spec-driven, event-driven autonomous software organization where every project has professional documentation, commit checkpoints, deterministic gates, and builder context before implementation starts.

## Source-of-truth model

| Layer | Canonical responsibility |
|---|---|
| Project repository | Professional project record: PRD/spec, ADRs, sprint plan, task graph, QA/security/delivery evidence, builder context, and commit history. |
| Factory DB (`agent_core_postgres` / `zeus_agent.factory`) | Operational truth: projects, lanes, tasks, runs, gates, events, alerts, human questions, and reconciliation anomalies. |
| Git commits | Immutable evidence that docs/code changed through normal software-company checkpoints. |
| Notion/dashboard | Human PM/readability projection only; never replaces repo or Factory DB. |
| Skills/wiki/memory | Methodology/procedural memory; not live project status. |

## Current increment

| Field | Value |
|---|---|
| Project ID | `factory-runtime-remediation` |
| Lane | `factory-runtime-remediation-runtime-contract-v1` |
| Branch/worktree | `factory/factory-runtime-contract-v1` / `/home/jean/workspace/hermes-factory-runtime-contract-v1` |
| Active focus | Repo-first Runtime Contract v1 |
| Runtime status before increment | Other autonomous non-Factory project paused; no active Factory worker run detected. |
| Documentation entry point | `factory/projects/factory-runtime-remediation/DOCUMENTATION_INDEX.md` |

## Non-negotiable invariants

1. No Factory project should be considered delivery-ready only because its DB tasks say done.
2. Required project-local documents must exist and must be listed in `DOCUMENTATION_INDEX.md`.
3. Project-local Factory artifacts must be committed or explicitly waived by Jean with reason.
4. Builder engines must receive repository documentation paths, not only a prose prompt.
5. Reconciliation anomalies are first-class work, not UI noise.
6. Zeus can bootstrap recursively, but self-approval must be replaced by deterministic tests/gates and independent review where possible.

## Adopted external patterns

| Source | Adopted pattern |
|---|---|
| n8n | Typed operations, state-machine discipline, heartbeat/watchdog thinking, structural validation before execution. |
| BMAD-METHOD | PRD → architecture → sprint/story → dev-story → adversarial review → correct-course discipline. |
| spec-kit | Spec-driven repo workflow, constitution, feature docs, checklists, git checkpoints, agent-context as builder entry point. |

## Increment context

This increment implements the first code-level enforcement of Jean's correction: repo docs and commits are not optional. The reconciler and critical readiness gate now detect two forms of repo drift that were previously invisible:

- Required docs exist on disk but are missing from `DOCUMENTATION_INDEX.md`.
- Project-local Factory artifacts are modified/untracked in git, meaning there is no commit checkpoint for the claimed source of truth.

## Next expected increments

1. Generate a builder context bundle (`BUILDER_CONTEXT.md` / `.factory/current-context.json`) per task.
2. Add structured `run_result.json` as the worker output contract.
3. Promote Factory operations to typed events/reducers instead of free-form status transitions.
4. Add workflow YAML for Hybrid Factory phases with gates, fan-out/fan-in, and resume semantics.
5. Surface repo-first anomalies in the dashboard so Jean sees DB/repo/commit drift clearly.
