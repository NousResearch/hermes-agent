# Legacy OpenClaw / Workspace Factory Audit

Date: 2026-05-26
Primary factory repo: `/home/jean/Projects/hermes-agent-original`
Legacy/workspace repo reviewed: `/home/jean/Projects/hermes-workspace-su`

## Executive decision

Do not revive the old OpenClaw factory as the operational source of truth. Reuse only the pieces that improve the SitioUno Software Factory:

- semantic worker roster and role descriptions;
- proof-bearing checkpoints;
- mission/orchestrator loop concepts;
- workflow UI ideas only when they are rebuilt inside the official Hermes dashboard; do not revive the separate Workspace SU surface.

The new source of truth is:

- Hermes Kanban for operational task flow;
- `software_factory` progress DB for audit, metrics, gates, artifacts, and benchmarks;
- isolated git branches/worktrees per method lane;
- Zeus as factory orchestrator.

## Reviewed components

### Hermes Agent original

| Component | Path | Decision | Notes |
|---|---|---|---|
| Kanban DB / multi-board | `hermes_cli/kanban_db.py` | Reuse | Mature SQLite board, multi-board isolation, profile-aware worker routing. Factory should layer on top instead of replacing it. |
| Kanban CLI | `hermes_cli/kanban.py` | Reuse | Human/operator surface for boards and tasks. |
| Kanban tools | `tools/kanban_tools.py` | Reuse | Agent tool surface for dispatcher-spawned workers and orchestrators. |
| Kanban plugin | `plugins/kanban/` | Reuse | Existing dispatcher/worker foundation. |
| OpenClaw migration references | release notes / `hermes claw migrate` | Archive only | Historical migration support; not a factory runtime. |

### Hermes Workspace SU / OpenClaw-era workspace

| Component | Path | Decision | Notes |
|---|---|---|---|
| Semantic roster | `swarm.yaml` | Reuse/migrate | Good role taxonomy. Map to factory agents: orchestrator, builders, reviewer, QA, ops, strategist. |
| Workspace agent contract | `AGENTS.md` | Reuse/migrate | Useful operating rules: Builder implements, Reviewer gates, QA verifies, Orchestrator routes. |
| Orchestrator loop | `src/routes/api/swarm-orchestrator-loop.ts` | Reuse concept only | Checkpoint parsing and stale-worker detection are useful, but state writes must move to factory DB/Kanban. |
| Orchestration client/API | `src/server/orchestration-client.ts`, `src/routes/api/orchestration.ts` | Do not port as separate surface | If needed later, reimplement against official Hermes dashboard APIs instead of Workspace SU. |
| Workflow templates | `src/screens/gateway/lib/workflow-templates.ts` | Migrate/rename | Replace `clawsuite:workflow-templates` naming and add Zeus/BMAD factory templates. |
| Swarm UI screens | `src/screens/swarm2/*`, `src/screens/workflows/*` | Do not run separately | Keep ideas only as future official-dashboard functionality. |
| Memory swarm mission logs | `memory/swarm/missions/*` | Archive | Useful examples, not source of truth for new factory. |

## Required cleanup before production use

1. Rename active user-facing `clawsuite`/`openclaw` labels to `sitiouno-factory` or `factory`.
2. Keep legacy docs under `docs/legacy-openclaw/` if needed for traceability.
3. Ensure every new workflow has gates, evidence requirements, DB event logging, and independent review.
4. Do not let legacy swarm state become a competing source of truth.
5. Introduce factory board/lane names rather than numbered-only swarm lanes.

## Migration mapping

| Legacy worker | New factory role |
|---|---|
| `orchestrator` | `factory-orchestrator` |
| `builder` | `claude-builder`, `codex-builder` |
| `reviewer` | `quality-reviewer`, `security-reviewer` |
| `qa` | `qa-verifier` |
| `researcher` | `product-analyst` / research support |
| `ops-watch` | deterministic jobs / `devops-release` |
| `strategist` | `solution-architect` advisory lane |
| `km-agent` | knowledge/artifact curator (v2) |

## Factory v1 direction

Implement the foundation in Hermes Agent first:

- `hermes factory` CLI;
- local deterministic DB fallback for tests/offline operation;
- Postgres `software_factory` schema for production metrics;
- agent tools for project/lane/task/gate/event recording;
- Kanban integration in the next phase.

Only after this foundation is stable should any UI be added inside the official Hermes dashboard.
