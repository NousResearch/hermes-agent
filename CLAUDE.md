# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` (root, ~70KB) is the canonical deep guide — Contribution Rubric,
Footprint Ladder, per-subsystem details, full pitfalls list. This file is the
orientation layer; reach into AGENTS.md for depth.

## What Hermes Is

A personal AI agent running the **same core** across a CLI, a messaging gateway
(~20 platforms), a TUI, an Electron desktop app, and an ACP server. It learns
across sessions (memory + skills), spawns subagents, runs cron jobs, and drives a
real terminal + browser. Python core; TypeScript for TUI (Ink), dashboard
(`web/`), and desktop (`apps/desktop/`).

## Two constraints that govern almost every change

1. **Prompt caching is sacred.** A long conversation reuses a cached prefix every
   turn. Don't mutate past context, swap toolsets, or rebuild the system prompt
   mid-conversation (sole exception: context compression). Keep strict message-role
   alternation. Slash commands that change system-prompt state must be cache-aware:
   defer to next session by default, opt-in `--now` for immediate effect.
2. **Narrow waist; capability at the edges.** Every model tool ships on *every* API
   call, so a new *core tool* is the last resort. Walk the Footprint Ladder
   (AGENTS.md): extend existing code → CLI command + skill → service-gated tool
   (`check_fn`) → plugin → MCP catalog → core tool.

Also: **no new `HERMES_*` env vars for non-secrets** (`.env` = secrets only;
behavior goes in `config.yaml`); **no change-detector tests** (assert invariants,
don't snapshot model lists / config versions / counts).

## Commands

```bash
# Tests — ALWAYS use the wrapper; never call pytest directly. It enforces CI
# parity: creds unset, TZ=UTC, LANG=C.UTF-8, per-test subprocess isolation.
scripts/run_tests.sh                                   # full suite
scripts/run_tests.sh tests/agent/test_foo.py::test_x   # one test
scripts/run_tests.sh --no-isolate tests/foo/           # faster, for debugging
scripts/run_tests.sh -- -v --tb=long                   # pass-through pytest flags

ruff check .   # blocking lint gate (only PLW1514 enforced); `ty check` is advisory

./hermes [--tui|gateway|doctor]   # run from the checkout (== installed `hermes`)

# Dev: source .venv/bin/activate; uv pip install -e ".[all,dev]"
```

TypeScript surfaces are npm workspaces (`apps/*`, `ui-tui`, `web`). TUI dev from
`ui-tui/`: `npm run dev|build|typecheck|test`. Run desktop vitest from repo root.

## Architecture

**Agent loop** — `run_agent.py` → `AIAgent.run_conversation()` (synchronous):
builds the system prompt, calls an OpenAI-compatible API, loops on tool calls until
a text response. Any OpenAI-compatible provider works (resolved at init: Nous
Portal OAuth, OpenRouter, or custom endpoint).

**Tools self-register** via `registry.register()` in `tools/*.py`
(`tools/registry.py`), auto-discovered by `model_tools.py`. But a tool is only
*exposed* if its name is in a toolset in `toolsets.py` (`_HERMES_CORE_TOOLS` is the
base bundle). Handlers return a JSON **string**.

**Slash commands** are defined once in `COMMAND_REGISTRY` (`hermes_cli/commands.py`);
CLI/gateway dispatch, help, Telegram/Slack menus, and autocomplete all derive from
it. Skill commands inject as a **user message** (not system prompt) to keep caching.

**Surfaces (one core):** `cli.py` (`HermesCLI`, Rich + prompt_toolkit); `ui-tui/`
(Ink) + `tui_gateway/` (JSON-RPC backend); the dashboard (`web/`) **embeds the real
TUI over a PTY** — don't reimplement chat in React; `apps/desktop/` (separate
Electron surface → `tui_gateway`); `gateway/` (`run.py` + `session.py` +
`platforms/<name>.py`, see `gateway/ADDING_A_PLATFORM.md`); `acp_adapter/`.

**Edges:** `plugins/` (memory/model providers, etc., runtime-discovered), `skills/`
+ `optional-skills/`, `optional-mcps/`. **Terminal backends:** `tools/environments/`
(`BaseEnvironment` ABC → local/docker/ssh/singularity/modal/daytona). **State:**
`hermes_state.py` (`SessionDB`, SQLite + FTS5). Subagents: `tools/delegate_tool.py`.

## Pitfalls

- **Profiles:** never hardcode `~/.hermes` / `Path.home() / ".hermes"` for state —
  use `get_hermes_home()` (code) / `display_hermes_home()` (user-facing) from
  `hermes_constants`. Each profile sets its own `HERMES_HOME` at startup. Tests must
  not write to `~/.hermes/` (`conftest.py` redirects it).
- **No `offset`/`limit` on instructional tools** (skills/prompts) — models read page
  1 and skip the rest.
- **No cross-toolset tool names in schema descriptions** (may be disabled → halluc.
  calls); add such refs dynamically in `get_tool_definitions()` (`model_tools.py`).
- **Gateway has two message guards** — a command that must reach the runner while the
  agent is blocked (e.g. approvals) must bypass both and dispatch inline.
- **Deps are exact-pinned** in `pyproject.toml` (supply-chain); provider deps live in
  extras + lazy-install (`tools/lazy_deps.py`). Bump pin + run `uv lock`.
- **Verify the premise before "fixing"** — many limitations are intentional design.
  Reproduce on `main`, cite the exact line, check intent with `git log -p -S`.
