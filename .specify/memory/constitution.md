# Hermes Agent Constitution

## Core Principles

### I. Conversation-First Multi-Platform Agent

Hermes Agent is a multi-platform AI agent runtime (CLI, Discord, Telegram, Slack, Matrix, and 15+ more). Every feature must preserve the core loop: user message → AIAgent.run_conversation() → tool calls → response. The agent loop is synchronous, interrupt-checking, and budget-aware. Features that break the loop or add unbounded iteration are out of scope.

### II. Tool Discovery via Registry

All tools register at import time via `tools/registry.py`. No hardcoded tool lists — `model_tools.py` imports the registry, triggers discovery, and exposes tool schemas to providers. Adding a tool means creating a file under `tools/`, calling `registry.register()`, and nothing else. Removing a tool means removing the file. The registry is the single source of truth.

### III. Plugin Extensibility Over Core Modification

When a feature can be a plugin (`plugins/`), it must be a plugin. Plugins are: memory providers, context engines, model providers, kanban dispatchers, platform adapters, observability hooks. Core (`run_agent.py`, `cli.py`, `model_tools.py`, `toolsets.py`) changes require spec-kit entry and PR review. The plugin API is the extension point; the core is not.

### IV. Testing: E2E Default, Unit When Justified

Per chitin spec 020 §1.2, e2e is the default test layer. New features get e2e tests first. Unit tests are justified by a named subsection ("Why not e2e?") in the spec. Test discovery: `scripts/run_tests.sh` probes `.venv` → `venv` → `$HOME/.hermes/hermes-agent/venv`. All tests must pass on the shared venv path since worktrees share it with the main checkout.

### V. Platform Parity

Every messaging platform (Telegram, Discord, Slack, Matrix, etc.) must have feature parity for slash commands, tool execution, and session management. `GATEWAY_KNOWN_COMMANDS` and `resolve_command()` enforce this centrally. A command available in CLI must have a gateway dispatch path unless explicitly marked `cli_only`. Adding a slash command requires: `CommandDef` in `hermes_cli/commands.py`, handler in `cli.py`, handler in `gateway/run.py` — all three, not two of three.

## Spec-First Development

Constitution §1.1 from the chitin governance layer applies: no implementation branch without a merged spec PR. Tickets promoted to `ready` must reference a spec-kit entry (`Spec: NNN-<slug>`). Emergency fixes exempt. The `.specify/specs/` directory is the canonical registry, tracked by `INDEX.md`.

## Architecture Constraints

- **File dependency chain**: `tools/registry.py` → `tools/*.py` → `model_tools.py` → `run_agent.py`/`cli.py`. Never introduce circular imports.
- **Config**: `~/.hermes/config.yaml` for settings, `~/.hermes/.env` for API keys only. Never embed secrets in config YAML.
- **Logs**: `~/.hermes/logs/` — `agent.log` (INFO+), `errors.log` (WARNING+), `gateway.log` when gateway runs. Profile-aware via `get_hermes_home()`.
- **Worktree discipline**: Primary checkout on `main` is read-only for AI workers. All edits happen in worktrees (`~/workspace/hermes-agent-<slug>` branch, push from any checkout). Workers never push directly to `main`.

## Governance

- Spec-kit constitution supersedes all other practices for feature work.
- Amendments require documentation, operator approval, and a migration plan.
- The `specify` CLI (v0.8.12+) is the canonical tool for spec lifecycle operations (`specify init`, `specify check`, spec creation).
- This constitution derives authority from the operator (Jared Pleva). Amendments are ratified via PR.

**Version**: 1.0.0 | **Ratified**: 2026-05-20 | **Last Amended**: 2026-05-20