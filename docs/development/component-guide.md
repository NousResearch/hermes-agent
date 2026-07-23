# Hermes Component Guide

This is the detailed routing map for contributors. Universal engineering rules
live in the repository root `AGENTS.md`; subsystem-specific rules live in
nested `AGENTS.md` files.

## Core conversation path

- `run_agent.py`: `AIAgent` construction and agent loop.
- `model_tools.py`: tool discovery, schema assembly, and function dispatch.
- `toolsets.py`: named tool bundles and `_HERMES_CORE_TOOLS`.
- `hermes_state.py`: SQLite-backed session state and search.
- `agent/`: providers, prompt construction, memory, caching, compression,
  checkpoints, and supporting services.

The message loop must preserve strict role alternation and a byte-stable system
prompt for the life of a conversation. Agent-level tools such as todo and
memory may be intercepted before generic function dispatch.

## CLI and commands

- `cli.py`: classic interactive CLI orchestrator.
- `hermes_cli/main.py`: entry point and profile override.
- `hermes_cli/commands.py`: slash-command registry.
- `hermes_cli/config.py`: defaults, setup metadata, and merged configuration.
- `hermes_cli/curses_ui.py`: canonical interactive-menu implementation.

Read `hermes_cli/AGENTS.md` before changing these surfaces.

## Tools

Tools register through `tools/registry.py`. Built-in tools are discovered from
`tools/*.py`, but discovery does not expose them to an agent: their names must
also belong to a toolset in `toolsets.py`.

All handlers return JSON strings. Optional tools use requirement checks and are
absent when prerequisites are unavailable. State paths use
`get_hermes_home()`, and schema text uses `display_hermes_home()` for
profile-correct paths.

Prefer extending an existing tool, a CLI command plus skill, a gated tool,
plugin, or MCP server before adding a permanent core schema.

## User interfaces

- `ui-tui/`: Ink/React terminal UI.
- `tui_gateway/`: Python JSON-RPC backend for TUI clients.
- `web/`: dashboard frontend.
- `apps/desktop/`: Electron desktop client.
- `acp_adapter/`: editor integration through ACP.

The dashboard embeds the TUI through a PTY. Desktop is a separate client using
the shared gateway protocol. Read the nested guides under `ui-tui/` and
`apps/desktop/`.

## Gateway

- `gateway/run.py`: lifecycle, platform startup, command interception, and
  background services.
- `gateway/session.py`: conversation routing and active sessions.
- `gateway/platforms/`: platform adapters.
- `gateway/config.py`: gateway-side configuration.

Adapters with unique credentials should acquire and release scoped token locks.
Control commands that must work while an agent is blocked need to bypass both
the base-adapter active-session queue and the runner's active-agent guard.

## Extensions

- `plugins/`: bundled plugin implementations and provider families.
- `skills/`: default skills.
- `optional-skills/`: opt-in skills.
- MCP catalog: reusable external structured tools.

Read the nested plugin and skill guides before editing these trees.

## Durable and background work

- `tools/delegate_tool.py`: process-local child-agent delegation.
- `cron/jobs.py` and `cron/scheduler.py`: durable scheduled work.
- `plugins/kanban/`: durable multi-agent board and dispatcher.
- `agent/curator.py`: lifecycle maintenance for agent-created skills.

Background delegation is detached from a turn but does not survive process
restart. Use cron or another durable mechanism when restart survival matters.
Leaf delegates cannot call `delegate_task`, `clarify`, `memory`,
`send_message`, or `cronjob`, but retain `execute_code` for programmatic tool
calling. Orchestrator delegates may spawn children within configured depth and
concurrency limits.
Kanban workers are board-isolated, and the dispatcher normally runs inside the
gateway.

## Configuration and state

`HERMES_HOME` is profile-specific. Use `get_hermes_home()` for runtime state and
`display_hermes_home()` in user-visible messages. Only profile discovery itself
is anchored to the default home.

User behavior is configured in `config.yaml`; `.env` contains secrets only.
Configuration has multiple loaders, so changes must be verified in each runtime
surface that consumes them.

## Testing

Read `tests/AGENTS.md`. Use `scripts/run_tests.sh`, temporary homes, behavioral
assertions, and real integration paths where discovery or configuration is
involved.
