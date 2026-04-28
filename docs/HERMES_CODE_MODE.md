# Hermes Code Mode

Hermes Code Mode now includes the base console, P0 Engineering Control Plane,
P1 GitHub integration foundation, and P2 approval governance, adapted to the
current architecture.

## Included Capabilities

- CLI Code Mode commands:
  - `/code`
  - `/web`
  - `/workspace`
  - `/session`
  - `/approvals`
  - `/skills-code`
  - `/github`
- Code Mode state and events in `state.db`.
- P0 services:
  - `ExecutionPolicyEngine`
  - `ArtifactLedger`
  - `AgentOrchestrator`
  - `WorktreeService` (safe capability/checkpoint foundation)
  - `SkillDiscoveryService` (`SKILL.md` + `scripts/` + `resources/`)
  - `RepoKnowledgeService` (`AGENTS.md` + docs guidance detection)
- Normalized Code Mode event envelope persisted to `code_events`.
- GitHub integration services:
  - `github_integration`
  - `github_webhooks`
  - `github_sync`
  - `github_chatops`
- P2 approval governance service:
  - `approval_governance`
  - persistent approval lifecycle for Code Mode actions

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

P1 GitHub endpoints:

- `GET /api/code/github/status`
- `GET /api/code/github/installations`
- `GET /api/code/github/repositories`
- `POST /api/code/github/repositories/sync`
- `GET /api/code/github/repositories/{owner}/{repo}`
- `GET /api/code/github/repositories/{owner}/{repo}/issues`
- `GET /api/code/github/repositories/{owner}/{repo}/pulls`
- `POST /api/code/github/webhooks`
- `POST /api/code/github/chatops/{command_id}/run`
- `POST /api/code/github/comments`
- `POST /api/code/github/pull-requests/prepare`

P2 approval endpoints:

- `GET /api/code/approvals`
- `POST /api/code/approvals`
- `GET /api/code/approvals/{approval_id}`
- `POST /api/code/approvals/{approval_id}/approve`
- `POST /api/code/approvals/{approval_id}/reject`
- `POST /api/code/approvals/{approval_id}/cancel`
- `POST /api/code/approvals/expire`
- `GET /api/code/approvals/summary`

## State / Schema

Code Mode state lives in `state.db`, schema version `15`.

Code Mode + P0 tables:

- `code_workspaces`
- `code_sessions`
- `code_events`
- `code_approval_requests`
- `code_artifacts`
- `code_orchestrated_runs`
- `code_run_transitions`
- `code_checkpoints`
- `github_app_installations`
- `github_repositories`
- `github_branches`
- `github_issues`
- `github_pull_requests`
- `github_webhook_deliveries`
- `github_chatops_commands`
- `github_status_reports`

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

## Deferred

- SSH/VPS automation.
- Desktop app behavior.
- `hermesWeb/` UI integration.
- Autonomous coding loop from GitHub events.
- Auto-merge / force-push / destructive GitHub write flows.
- HermesWeb approval panel (dashboard UI) for lifecycle actions.

`hermesWeb/` is absent in this checkout, and no large frontend work was added
under deprecated `web/`.
