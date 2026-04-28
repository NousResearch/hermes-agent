# Hermes Code Mode

Hermes Code Mode is the base coding-console foundation for Hermes Agent. This
port keeps the feature aligned with the current `origin/main` architecture and
intentionally does not include the later P0 Engineering Control Plane or P1
GitHub integration.

## What Is Included

- CLI status console through `/code`.
- Code Mode slash commands:
  - `/code`
  - `/web`
  - `/workspace`
  - `/session`
  - `/approvals`
  - `/skills-code`
- Minimal backend API endpoints under `/api/code/*`.
- SQLite state support for Code Mode workspace, session, and event metadata.
- Graceful degraded behavior when the backend, state database, or Git metadata
  is unavailable.

## CLI Commands

`/code` shows the active provider, model, profile, workspace path, Git branch,
backend reachability, dashboard URL, schema version, and Code Mode state counts.

`/web` shows the local dashboard URL and launch command. In this checkout,
`hermesWeb/` is not present, so richer Code Cockpit UI work is deferred.

`/workspace` lists stored Code Mode workspaces from state when available, and
falls back to the current working directory and Git branch.

`/session` lists stored Code Mode sessions from state when available.

`/approvals` reports that approval orchestration is deferred to the P0 port.

`/skills-code` reports installed slash skill visibility and notes that
repository `SKILL.md` discovery is deferred to the P0 port.

## Backend API

The current base port exposes:

- `GET /api/code/status`
- `GET /api/code/workspaces`
- `GET /api/code/sessions`
- `GET /api/code/events`

`/api/code/status` is public and read-only so the CLI can detect whether the
dashboard backend is online. Other `/api/code/*` endpoints use the existing
dashboard session-token middleware.

## State

Code Mode state lives in `state.db` with schema version `12`.

Tables:

- `code_workspaces`
- `code_sessions`
- `code_events`

Fresh databases create these tables directly. Existing databases migrate through
the normal declarative reconciliation path and the v12 migration step.

## Startup

Start the dashboard backend with:

```bash
python -m hermes_cli.main dashboard --port 9119
```

Then run `/code` or `/web` in the CLI to see backend reachability and launch
hints.

## Deferred Work

This port intentionally does not include:

- P0 ArtifactLedger.
- P0 AgentOrchestrator.
- P0 ExecutionPolicyEngine.
- P0 Worktree/checkpoint service.
- P0 `SKILL.md` discovery bridge.
- P0 RepoKnowledgeService.
- P1 GitHub App integration, webhooks, or ChatOps.
- SSH/VPS behavior.
- Desktop app behavior.
- A large deprecated `web/` migration.

`hermesWeb/` is absent in this checkout, so Code Cockpit UI work is deferred to
a later frontend-specific port.
