# Hermes Code Mode

Hermes Code Mode now includes the base console plus the P0 Engineering Control
Plane foundation, adapted to the current main-compatible architecture.

## Included Capabilities

- CLI Code Mode commands:
  - `/code`
  - `/web`
  - `/workspace`
  - `/session`
  - `/approvals`
  - `/skills-code`
- Code Mode state and events in `state.db`.
- P0 services:
  - `ExecutionPolicyEngine`
  - `ArtifactLedger`
  - `AgentOrchestrator`
  - `WorktreeService` (safe capability/checkpoint foundation)
  - `SkillDiscoveryService` (`SKILL.md` + `scripts/` + `resources/`)
  - `RepoKnowledgeService` (`AGENTS.md` + docs guidance detection)
- Normalized Code Mode event envelope persisted to `code_events`.

## Execution Policy Risk Classes

- `safe_readonly`
- `safe_local_write`
- `network`
- `git_write`
- `secret_sensitive`
- `remote_mutating`
- `destructive`
- `production_sensitive`

Destructive commands are blocked, higher-risk classes are approval candidates,
and command text is secret-redacted before reporting.

## Artifact Categories

- `task_intake`
- `prd_lite`
- `acceptance_criteria`
- `architecture_note`
- `adr`
- `implementation_plan`
- `command_log`
- `diff_summary`
- `test_report`
- `review_report`
- `deploy_plan`
- `deploy_report`
- `memory_update`

Artifacts can link to workspace/session/orchestrated run/command metadata.

## Orchestrator States

- `intake`
- `discovery`
- `product_framing`
- `architecture`
- `planning`
- `approval`
- `implementation`
- `validation`
- `review`
- `ready_for_pr`
- `completed`
- `cancelled`
- `failed`

P0 provides state validation and persistence only. No autonomous execution loop
is enabled in this port.

## API Endpoints

Existing Code Mode endpoints:

- `GET /api/code/status` (public read-only)
- `GET /api/code/workspaces`
- `GET /api/code/sessions`
- `GET /api/code/events`

P0 endpoints:

- `GET /api/code/artifacts`
- `POST /api/code/artifacts`
- `GET /api/code/sessions/{code_session_id}/artifacts`
- `GET /api/code/orchestrator/runs`
- `POST /api/code/orchestrator/runs`
- `GET /api/code/orchestrator/runs/{run_id}`
- `POST /api/code/orchestrator/runs/{run_id}/transition`
- `POST /api/code/policy/assess-command`
- `GET /api/code/workspaces/{workspace_id}/git/capabilities`
- `GET /api/code/workspaces/{workspace_id}/git/worktrees`
- `GET /api/code/skills/discovered`
- `GET /api/code/workspaces/{workspace_id}/repo-knowledge`
- `POST /api/code/workspaces/{workspace_id}/repo-knowledge/bootstrap`

## State / Schema

Code Mode state lives in `state.db`, schema version `13`.

Code Mode + P0 tables:

- `code_workspaces`
- `code_sessions`
- `code_events`
- `code_artifacts`
- `code_orchestrated_runs`
- `code_run_transitions`
- `code_checkpoints`

No separate SQLite database is used for P0.

## Skill and Repo Guidance Discovery

- `/skills-code` now reports discovered skills and workspace-local `SKILL.md`
  capability.
- Repo knowledge supports:
  - repo-root `AGENTS.md`
  - docs guidance folders:
    - `docs/architecture/`
    - `docs/engineering/`
    - `docs/operations/`

Bootstrap creates `AGENTS.md` only when missing and never overwrites.

## Startup

Start backend:

```bash
python -m hermes_cli.main dashboard --port 9119
```

Then use `/code` or `/web` in the CLI.

## Deferred (Not in P0)

- P1 GitHub integration (GitHub App auth, webhooks, ChatOps, sync).
- SSH/VPS automation.
- Desktop app behavior.
- `hermesWeb/` UI integration.

`hermesWeb/` is absent in this checkout, and no large frontend work was added
under deprecated `web/`.
