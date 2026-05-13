# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Primary Reference

**`AGENTS.md`** is the authoritative development guide for this repo (~990 lines, kept current). Read it for architecture, conventions, pitfalls, and contribution patterns. This file is a quick-orientation overlay — when in doubt, defer to `AGENTS.md`.

## What this repo is

`hermes-agent` — Nous Research's self-improving AI agent. Python 3.11+, exact-pinned deps, packaged via `pyproject.toml` (extras: `[all]`, `[dev]`, `[termux]`). Distributed as the `hermes` CLI. The runtime stores its state in `$HERMES_HOME` (default `~/.hermes`); this repo is typically cloned at `~/.hermes/hermes-agent/`.

## Common commands

```bash
# Setup (one-shot, idempotent)
./setup-hermes.sh                 # creates .venv, installs .[all], symlinks ~/.local/bin/hermes

# Activate venv (prefer .venv, fall back to venv)
source .venv/bin/activate

# Tests — ALWAYS use the wrapper, never raw pytest
scripts/run_tests.sh                                  # full suite, CI-parity
scripts/run_tests.sh tests/gateway/                   # one directory
scripts/run_tests.sh tests/agent/test_foo.py::test_x  # one test
scripts/run_tests.sh -v --tb=long                     # pass-through flags

# TUI (Ink/Node) — only when touching ui-tui/
cd ui-tui && npm install && npm run dev
npm run type-check  # tsc --noEmit
npm run lint        # eslint
npm test            # vitest

# Run the agent locally
./hermes              # auto-detects venv, no `source` needed
hermes gateway start  # messaging gateway (Telegram/Discord/Slack/...)
hermes doctor         # diagnose install
```

The test wrapper enforces hermetic CI parity (unsets `*_API_KEY`/`*_TOKEN`, `TZ=UTC`, `LANG=C.UTF-8`, `-n 4` xdist workers, temp `HERMES_HOME`). Bypassing it has caused multiple "works locally, fails in CI" incidents — don't.

## High-level architecture

Five load-bearing entry points you'll touch most often:

- **`run_agent.py`** — `AIAgent` class, the core synchronous conversation loop. ~60-param `__init__`. The loop lives in `run_conversation()`: iterate model → handle tool calls → repeat until `max_iterations` or no tool calls. Reasoning content lives in `assistant_msg["reasoning"]`.
- **`model_tools.py`** — tool orchestration: `discover_builtin_tools()`, `handle_function_call()`, pre/post-tool plugin hook dispatch. Importing this module triggers `discover_plugins()` as a side effect.
- **`toolsets.py`** — `TOOLSETS` dict + `_HERMES_CORE_TOOLS` (the default bundle every platform inherits). Auto-discovery imports each `tools/*.py`, but a tool is only exposed if its name appears in a toolset here.
- **`cli.py`** — `HermesCLI` interactive orchestrator (~11k LOC). `process_command()` dispatches slash commands by canonical name resolved via `resolve_command()` from `hermes_cli/commands.py` (central `COMMAND_REGISTRY`).
- **`hermes_state.py`** — `SessionDB`, SQLite with FTS5 for session search.

Subsystems and where they live:

- **`tools/`** — built-in tools. Each file calls `registry.register(...)` at import time. Adding a tool: create the file *and* add the name to a toolset in `toolsets.py`. Both steps required.
- **`gateway/`** — messaging runner (`run.py`) + per-platform adapters in `gateway/platforms/` (telegram, discord, slack, whatsapp, signal, matrix, email, webhook, ...). Two message guards (`base.py` queue + `run.py` interceptor) — control commands (`/stop`, `/new`, `/approve`, ...) must bypass both.
- **`plugins/`** — two parallel plugin surfaces:
  - General plugins (`plugins/<name>/` with `register(ctx)`): lifecycle + tool-call hooks, custom tools, CLI subcommands. Discovered by `PluginManager`.
  - Memory providers (`plugins/memory/<name>/`): separate orchestrator (`agent/memory_manager.py`), implements `MemoryProvider` ABC.
  - Model providers (`plugins/model-providers/<name>/`): lazy, last-writer-wins discovery via `providers/__init__.py._discover_providers()`. User plugins override bundled ones.
  - **Hard rule:** plugins MUST NOT modify core files (`run_agent.py`, `cli.py`, `gateway/run.py`, `hermes_cli/main.py`). Extend the generic plugin surface instead.
- **`skills/`** (default-active) vs **`optional-skills/`** (shipped but installed explicitly via `hermes skills install official/<category>/<skill>`). Heavy-dep or niche skills go in `optional-skills/`.
- **`ui-tui/`** (Ink/React, TypeScript) + **`tui_gateway/`** (Python JSON-RPC stdio backend) — modern TUI activated by `hermes --tui`. TypeScript owns the screen; Python owns sessions/tools/model calls. Dashboard embeds the real TUI via PTY (`hermes_cli/pty_bridge.py`) — do NOT re-implement chat in React.
- **`cron/`** — jobs.py + scheduler.py. Agents schedule via `cronjob` tool; users via `hermes cron <verb>`. 3-minute hard interrupt on cron sessions; file lock at `~/.hermes/cron/.tick.lock`.
- **`agent/`** — provider adapters, memory manager, context engine, curator (skill lifecycle), compression, auxiliary client (per-task LLM overrides).
- **`environments/` + `tinker-atropos/`** — RL training (Atropos). Optional submodule.

## Critical policies (do not violate)

- **Prompt caching must not break.** Never alter past context, change toolsets, or rebuild system prompts mid-conversation. The only exception is context compression. Slash commands that mutate system-prompt state must default to deferred invalidation with an opt-in `--now` flag.
- **Profile-safe paths.** Always `get_hermes_home()` for code paths and `display_hermes_home()` for user-facing strings — both from `hermes_constants`. Never hardcode `~/.hermes` or `Path.home() / ".hermes"`. This breaks multi-profile installs (PR #3575 fixed five such bugs).
- **Tests must not write to `~/.hermes/`.** The `_isolate_hermes_home` autouse fixture in `tests/conftest.py` redirects `HERMES_HOME`. Profile tests also need to mock `Path.home()` — see `tests/hermes_cli/test_profiles.py`.
- **No change-detector tests.** Don't assert specific model names, config version literals, or enumeration counts. Assert *relationships and invariants* instead (e.g. "every model in catalog has a context-length entry"). See `AGENTS.md` § Testing for the full pattern.
- **Tool handlers MUST return a JSON string.** Use `display_hermes_home()` in schema descriptions that reference paths (schemas are built at import time, after profile override is applied).
- **Don't hardcode cross-tool references in schema descriptions.** A schema mentioning another tool by name causes hallucinated calls when that tool is disabled. Add cross-refs dynamically in `get_tool_definitions()`.
- **Skill vs tool decision:** almost always skill. Only make it a tool when it needs custom auth flows, binary/streaming data, or precise non-LLM logic. See `CONTRIBUTING.md` § "Should it be a Skill or a Tool?".

## Adding things — quick map

- **Slash command:** add `CommandDef` to `COMMAND_REGISTRY` in `hermes_cli/commands.py`, then a handler in `cli.py::process_command()` and (if gateway-available) `gateway/run.py`. Aliases need only the registry tuple.
- **Tool (core):** `tools/<name>.py` with `registry.register(...)` + add the name to a toolset in `toolsets.py`. For user-local tools, prefer a plugin in `~/.hermes/plugins/<name>/`.
- **Config key:** `DEFAULT_CONFIG` in `hermes_cli/config.py`. Bump `_config_version` only if migrating/transforming existing user config (new keys merge automatically).
- **Secret (.env):** `OPTIONAL_ENV_VARS` in `hermes_cli/config.py`. Non-secret settings belong in `config.yaml`, not `.env`.
- **Skin/theme:** add to `_BUILTIN_SKINS` in `hermes_cli/skin_engine.py`, or drop YAML at `~/.hermes/skins/<name>.yaml`.

## Known pitfalls

- `simple_term_menu` is legacy fallback only — new interactive menus must use `hermes_cli/curses_ui.py` (see `tools_config.py` as canonical pattern).
- Don't use ANSI `\033[K` in spinner/display code — leaks as literal `?[K` under `prompt_toolkit`'s `patch_stdout`. Pad with spaces instead.
- `_last_resolved_tool_names` in `model_tools.py` is a process-global, temporarily mutated by `delegate_tool.py` during child runs.
- Before squash-merging, rebase the PR branch onto current `main` — stale branches silently revert unrelated fixes. Verify with `git diff HEAD~1..HEAD` post-merge.
