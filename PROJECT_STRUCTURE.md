# PROJECT_STRUCTURE.md

Stable system map for this local Hermes Agent checkout.

## 1. Project Purpose

Hermes Agent is a personal AI agent platform. The same core can run through CLI/TUI, messaging gateway, desktop app, cron automations, plugins, skills, and multiple terminal backends. It supports persistent memory, skill learning, tool use, subagents, and external model providers.

For this checkout, the product north star is persistent, cost-aware AI
orchestration: Hermes receives a high-level goal and turns it into a verified
outcome by coordinating replaceable worker AIs, long-running processes,
isolated Git workflows, bounded retries, evidence-based verification, and
selective escalation to expensive expert models.

## 2. Load-Bearing Entry Points

- `run_agent.py`: core `AIAgent` conversation loop and runtime orchestration.
- `model_tools.py`: model tool discovery and function-call handling.
- `toolsets.py`: toolset definitions and core tool list.
- `cli.py`: interactive CLI orchestration.
- `hermes_cli/`: command-line subcommands, setup wizard, plugin loading, dashboard assets.
- `agent/`: provider adapters, transports, prompt/system behavior, tool execution helpers, compression, verification, and turn lifecycle.
- `gateway/`: messaging gateway, sessions, platform adapters, relay, and delivery logic.
- `tools/`: built-in tool implementations and terminal environments.
- `plugins/`: bundled plugin surfaces and providers.
- `skills/` and `optional-skills/`: built-in and optional procedural skills.
- `cron/`: scheduler and automation jobs.
- `ui-tui/` and `tui_gateway/`: React Ink TUI and Python JSON-RPC backend.
- `web/`, `website/`, `apps/desktop/`: dashboard/docs/desktop frontends.
- `tests/`: pytest suite.

## 3. Build Structure: Persistent AI Orchestration

Current architecture direction: Kanban-first V2.1. Do not add a second
orchestrator database, polling workflow engine, or core model tool unless the
existing Kanban/skill/plugin/CLI surfaces cannot carry the work. Hermes Kanban
is the workflow engine; skills, config, provider routing, terminal tools, and
Git/worktree helpers are the orchestration layer around it.

Build layers to preserve when adding capability:

1. Goal intake/specification: turn a high-level user goal into concrete scope,
   acceptance criteria, constraints, and task graph.
2. Persistent task/workflow state: keep job status, owner, attempt count,
   blockers, artifacts, branch/worktree, budget, and next action outside any
   single model context window.
3. Model/worker routing: choose replaceable AI lanes by cost, capability,
   latency, and risk. Cheap workers do routine execution; expensive models
   handle expertise, audit, and escalation.
4. Long-running process execution: supervise commands that may run for hours,
   capture stdout/stderr/exit codes, and resume or diagnose after process death
   or restart.
5. Failure classification and bounded retry: identify root cause, split
   independent failures into tasks, retry only within policy, then block or
   escalate with evidence.
6. Isolated Git execution: assign work to explicit branches/worktrees, avoid
   cross-worker collisions, and record who changed what.
7. Verification evidence: require tests/checks/artifacts before a worker can
   claim success.
8. Commit/merge policy: commit only verified work, merge through the intended
   integration path, and keep metadata traceable to task/model/worker.
9. Expert audit/escalation: use Codex/Claude or another expensive expert only
   when risk, failure class, architecture review, or regression audit justifies
   the cost.
10. Reporting/recovery: summarize cost, model calls, attempts, artifacts,
    commits, unresolved blockers, and the next action so Hermes can continue
    after context loss or user absence.

## 4. Configuration And State

- User config: `~/.hermes/config.yaml`
- Secrets: `~/.hermes/.env` or local `.env` when explicitly used. Do not commit.
- Logs: `~/.hermes/logs/`
- Local repo artifacts: `.ai-runs/` if created by guarded work, ignored by git.
- Source examples: `.env.example`, `cli-config.yaml.example`

## 5. Dependency / Runtime Shape

- Python package: `pyproject.toml`, `uv.lock`, `setup.py`
- Python requirement: `>=3.11,<3.14`
- Preferred Python installer: `uv`
- Node workspaces: root `package.json`, `apps/*`, `ui-tui`, `ui-tui/packages/*`, `web`
- Node requirement: `>=20`
- Main scripts: `hermes`, `hermes-agent`, `hermes-acp`

## 6. Development Setup Paths

Preferred upstream managed layout:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Native Windows user install:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
hermes setup
hermes doctor
```

Manual source checkout fallback:

```bash
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
source ~/.hermes/venvs/hermes-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

## 7. Testing Strategy

- Always prefer `scripts/run_tests.sh` for Python tests.
- Target first, broaden only when risk justifies it.
- Use temp `HERMES_HOME` patterns already present in tests when touching config, memory, sessions, or gateway behavior.
- Avoid live network/API/provider calls in tests unless they are explicitly marked integration.
- Do not write change-detector tests for model catalogs, config version literals, or provider counts.

## 8. High-Risk Areas

- Prompt/system-message construction and compression.
- Tool schema, core tool list, and model-call payload construction.
- Session persistence, memory, and cross-session search.
- Gateway auth, allowed users, webhooks, relay, and message delivery.
- Secrets loading, config migration, and profile paths.
- Terminal backends and subprocess/process-tree management, especially on Windows.
- Package metadata, lockfile, lazy dependencies, and setup/update flows.

## 9. Safe Change Boundaries

Relatively safe:

- Local docs and handoff files.
- Narrow tests for a verified bug.
- Small CLI/help text fixes.
- Local setup notes.

Needs caution:

- Provider/model resolution, config defaults, prompt content, gateway platform behavior, plugin discovery, packaging metadata, dependency pins.

High risk:

- New core tools, broad refactors in `run_agent.py` or `cli.py`, live gateway/provider testing, installer/update logic, auth/session migrations.
