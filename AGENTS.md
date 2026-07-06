# AGENTS.md — Hermes Agent development protocol

Read this before editing Hermes Agent. Keep this file short: it is auto-injected into agent context and should stay under the 20k-char project-context cap. Full pre-compaction text is preserved in the timestamped backup beside this file.

## Product invariants
- Hermes Agent is a multi-surface personal agent: CLI/TUI, gateway platforms, desktop app, dashboard, cron, delegation, skills, memory, plugins, MCP.
- Per-conversation prompt caching is sacred. Do not mutate past context, toolsets, skills, memory, or system prompt mid-conversation except via the existing compression path.
- The core is a narrow waist. Prefer capability at the edges: existing code → CLI command + skill → service-gated tool → plugin → MCP catalog → new core tool only as last resort.
- Preserve strict message role alternation. Never inject synthetic same-role messages into the loop.
- Use `get_hermes_home()` for state/config paths and `display_hermes_home()` for user-facing paths. Do not hardcode `~/.hermes` except in docs/examples.

## Before changing behavior
1. Verify the premise against current code and runtime. Reproduce the symptom on current main when fixing a bug.
2. Check intent with `git log -p -S '<symbol or phrase>'` before reversing an apparent omission or restriction.
3. Prefer E2E/temp-`HERMES_HOME` validation for resolution chains, config propagation, security boundaries, remote backends, and file/network I/O.
4. Do not add speculative hooks, dead code, or extension points without a real consumer.

## Repository map
- `run_agent.py` — `AIAgent`, core conversation loop, prompt/tool-call iteration.
- `model_tools.py` — tool discovery/dispatch, tool definition assembly.
- `toolsets.py` — toolset definitions and core tool bundles.
- `cli.py`, `hermes_cli/` — classic CLI, slash commands, config/setup/plugin CLI.
- `agent/` — prompt builder, memory manager, compression, model routing, credential pooling, skills, curator.
- `tools/` — built-in tools; auto-discovered through `tools/registry.py` but only exposed when wired into a toolset.
- `gateway/` and `plugins/platforms/` — messaging adapters and gateway runtime.
- `cron/` — durable scheduled jobs.
- `plugins/` — bundled plugin surfaces: memory providers, model providers, kanban, context/image providers, etc.
- `ui-tui/`, `tui_gateway/` — Ink TUI and JSON-RPC backend.
- `apps/desktop/`, `apps/shared/`, `web/` — desktop and web surfaces.
- `tests/` — pytest suite; use the project runner.

## Adding tools / capabilities
- For local or niche capabilities, create a plugin under the user's plugin dir or an external plugin repo, not core.
- Built-in/core tool path:
  1. Create `tools/<name>.py`, register with `tools.registry.registry.register(...)`, and return JSON strings from handlers.
  2. Add the tool name to an appropriate toolset in `toolsets.py`; registration alone does not expose it.
  3. Add a `check_fn` / `requires_env` so unavailable integrations do not appear.
- Tool schema descriptions must not hard-reference tools from other toolsets; add dynamic cross-tool guidance in `get_tool_definitions()` if truly needed.

## Config, env, secrets
- Non-secret behavior settings go in `config.yaml` / `DEFAULT_CONFIG` (`hermes_cli/config.py`), not `.env`.
- `.env` is for secrets only. Add new secret env vars to `OPTIONAL_ENV_VARS` with metadata.
- Adding a key to an existing config section normally does not need a `_config_version` bump; migrations/renames do.
- Config loaders differ: `load_cli_config()` (classic CLI), `load_config()` (subcommands/setup), and gateway raw YAML. Verify the surface you changed.

## Profiles and persistent state
- `_apply_profile_override()` sets `HERMES_HOME` before imports. Code must be profile-safe by using `get_hermes_home()`.
- Profile operations are home-anchored: profiles live under `Path.home()/.hermes/profiles`, intentionally outside the active profile home.
- Tests that mock `Path.home()` must also set `HERMES_HOME`.
- Gateway adapters with unique credentials should use scoped locks (`gateway.status.acquire_scoped_lock`).

## Testing
- Always use the canonical runner, not direct pytest:
  ```bash
  scripts/run_tests.sh
  scripts/run_tests.sh tests/path/test_file.py::test_name -v --tb=short
  ```
- The runner provides CI parity: temp `HERMES_HOME`, credentials scrubbed, UTC/C.UTF-8, xdist/subprocess isolation.
- Do not write change-detector tests for model catalogs, config version literals, or list counts. Assert behavior/invariants instead.
- Windows-specific tests need explicit platform guards. If mocking OS, patch `sys.platform` and `platform.system()/release()/mac_ver()` consistently.

## Dependency policy
- PyPI deps require upper bounds (`>=floor,<next_major`; pre-1.0 use a narrow minor ceiling).
- Git deps and GitHub Actions must be pinned by commit SHA when applicable.
- Run `uv lock` after dependency changes.

## UI / surface-specific rules
- TUI is Ink + `tui_gateway` over stdio JSON-RPC. Dashboard `/chat` embeds the real TUI through a PTY bridge; do not reimplement the main chat transcript/composer in React dashboard.
- Desktop is a separate Electron/React chat surface. Slash command curation lives in `apps/desktop/src/lib/desktop-slash-commands.ts`; do not hide skill/quick-command extensions when pruning built-in noise.
- New interactive CLI menus should use `hermes_cli/curses_ui.py`, not new `simple_term_menu` usage.
- Spinner/display code must not use ANSI erase-to-EOL `\033[K`; use space padding.

## Gateway / background invariants
- Gateway has two active-session guards (base adapter pending queue and runner command interception). Commands that must work while an agent is blocked, such as `/stop`, `/approve`, `/deny`, must bypass both.
- Background process notifications are governed by `display.background_process_notifications` / `HERMES_BACKGROUND_NOTIFICATIONS`.
- Cron sessions are framed separately and should preserve role alternation; cron normally skips memory.
- `delegate_task` background children are process-local, not durable. Use cron or tracked background terminal processes for durable work.

## PR / review taste
- Wanted: real bug fixes with reproduction, edge expansion with setup/config UX, god-file refactors into focused modules, behavior-contract tests, cache-safe changes.
- Rejected: speculative infra, new core tools when file/terminal/skill/plugin works, new `.env` behavior settings, telemetry without opt-in, plugin-specific core patches, in-tree third-party product plugins/providers that should be standalone.
- Preserve contributor authorship when salvaging external work.

## Operational pitfalls
- Squash-merging stale branches can silently revert unrelated main fixes; update branch first and inspect `git diff HEAD~1..HEAD` after merge.
- Missing `__init__.py` files may be intentional to avoid package shadowing; do not restore blindly.
- `_last_resolved_tool_names` in `model_tools.py` is process-global and saved/restored around delegation.
- Tests must never write to real `~/.hermes/`; rely on the test fixtures and `HERMES_HOME`.

## Reference
- User docs: https://hermes-agent.nousresearch.com/docs
- CLI truth: `hermes --help`, `hermes <command> --help`, `hermes_cli/main.py`, `hermes_cli/commands.py`.
- Full pre-compaction instructions: see the `.token-cost-backup-<timestamp>` copy next to this file.
