# Hermes Agent - Root Development Guide

This file is intentionally short. The detailed rules live next to the code they
govern. When touching a subdirectory, read that directory's `AGENTS.md` before
editing files there.

Hermes and Codex do not load project rules in exactly the same way. Codex reads
the applicable `AGENTS.md` hierarchy for the files it edits. Hermes loads the
git-root-to-current-working-directory `AGENTS.md` hierarchy at session start and
discovers deeper subdirectory hints progressively through tool calls. Keep
critical global rules here; keep local implementation details in scoped files.

## Development Environment

```bash
# Prefer .venv; fall back to venv if that's what your checkout has.
source .venv/bin/activate   # or: source venv/bin/activate
```

`scripts/run_tests.sh` probes `.venv` first, then `venv`, then
`$HOME/.hermes/hermes-agent/venv` for worktrees that share a venv with the main
checkout.

## Project Map

File counts shift constantly. Trust the filesystem over this map.

| Path | Purpose | Local guide |
| --- | --- | --- |
| `run_agent.py` | `AIAgent` class and root conversation entry points | this file |
| `model_tools.py` | tool schema filtering and dispatch bridge | this file |
| `toolsets.py` | toolset definitions and `_HERMES_CORE_TOOLS` | this file |
| `cli.py` | classic interactive CLI orchestrator | this file + `hermes_cli/AGENTS.md` |
| `hermes_state.py` | SQLite session store and FTS5 search | this file |
| `agent/` | provider adapters, prompt assembly, memory, compression, LSP | `agent/AGENTS.md` |
| `hermes_cli/` | subcommands, setup, config, plugins loader, dashboard server | `hermes_cli/AGENTS.md` |
| `tools/` | built-in tool implementations and registry | `tools/AGENTS.md` |
| `gateway/` | messaging gateway and platform adapters | `gateway/AGENTS.md` |
| `plugins/` | repo-shipped plugins and plugin backends | `plugins/AGENTS.md` |
| `skills/`, `optional-skills/` | bundled and optional skills | `skills/AGENTS.md`, `optional-skills/AGENTS.md` |
| `ui-tui/`, `tui_gateway/` | Ink TUI and Python JSON-RPC backend | `ui-tui/AGENTS.md`, `tui_gateway/AGENTS.md` |
| `cron/` | scheduled jobs store and scheduler | `cron/AGENTS.md` |
| `tests/` | pytest suite | `tests/AGENTS.md` |

## Dependency Chain

```text
tools/registry.py  (no deps - imported by all tool files)
       ^
tools/*.py  (each calls registry.register() at import time)
       ^
model_tools.py  (imports tools/registry + triggers tool discovery)
       ^
run_agent.py, cli.py, batch_runner.py, environments/
```

Keep this direction acyclic. If a tool needs a capability from higher in the
stack, move the shared code down or use a plugin/hook boundary.

## Root Entry Points

`AIAgent` lives in `run_agent.py`. The constructor has many parameters; the
subset normally relevant to integrations is:

```python
class AIAgent:
    def __init__(self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        api_mode: str = None,
        model: str = "",
        max_iterations: int = 90,
        enabled_toolsets: list = None,
        disabled_toolsets: list = None,
        quiet_mode: bool = False,
        save_trajectories: bool = False,
        platform: str = None,
        session_id: str = None,
        skip_context_files: bool = False,
        skip_memory: bool = False,
        credential_pool=None,
        # ... plus callbacks, routing, budgets, prefill, service tier, etc.
    ): ...

    def chat(self, message: str) -> str: ...
    def run_conversation(self, user_message: str, system_message: str = None,
                         conversation_history: list = None, task_id: str = None) -> dict: ...
```

The synchronous conversation loop in `run_conversation()` calls the model,
dispatches tool calls through `handle_function_call()`, appends tool results, and
returns once the model emits final content. Reasoning content is stored in
`assistant_msg["reasoning"]`.

`model_tools.py` owns tool definition filtering and the call into
`tools.registry`. `toolsets.py` owns the exposed toolset names. Adding a core
tool requires both a registered `tools/<name>.py` module and a toolset entry.

`cli.py` owns the classic prompt_toolkit CLI shell. More detailed CLI command,
config, setup, profile, skin, and dashboard rules live in `hermes_cli/AGENTS.md`.

`hermes_state.py` owns persistent session storage: SQLite sessions, messages,
metadata, and FTS5 search. Session history is separate from curated memory.

## Global Policies

### Harness maintenance

Prefer durable harness improvements over one-off prompt instructions:
directory-local `AGENTS.md`, skills with linked references, hooks, tests, LSP
diagnostics, cron/curator review, and plugins. Revisit these rules every few
months; stale rules can become model drag as model behavior improves.

### Profile-safe paths

Use `get_hermes_home()` from `hermes_constants` for all state paths. Use
`display_hermes_home()` for user-facing messages. Do not hardcode
`~/.hermes` or `Path.home() / ".hermes"` in code that reads or writes Hermes
state. Profiles depend on `HERMES_HOME` being applied before imports.

Profile operations are HOME-anchored intentionally: `_get_profiles_root()` uses
`Path.home() / ".hermes" / "profiles"` so any active profile can list all
profiles.

Tests that mock `Path.home()` must also set `HERMES_HOME`.

### Prompt caching

Do not change past context, toolsets, memories, or system prompts mid-turn unless
the existing compression path does it intentionally. Slash commands that mutate
system-prompt state should default to deferred invalidation and offer an explicit
`--now` style path only when the codebase already supports it.

### Dependency pinning

All dependencies need upper bounds.

| Source type | Treatment | Example |
| --- | --- | --- |
| PyPI package | `>=floor,<next_major` | `"httpx>=0.28.1,<1"` |
| pre-1.0 PyPI package | `>=floor,<0.(minor+2)` | `"pkg>=0.29,<0.31"` |
| Git URL | commit SHA | `git+https://...@<40-char-sha>` |
| GitHub Actions | commit SHA + version comment | `uses: actions/checkout@<sha>  # v4` |
| CI-only pip | exact pin | `pyyaml==6.0.2` |

When changing `pyproject.toml`, regenerate `uv.lock`.

### Configuration

Non-secret settings belong in `config.yaml`; secrets belong in `.env`.

Add config defaults in `hermes_cli/config.py::DEFAULT_CONFIG`. Bump
`_config_version` only for migrations that transform existing user config.
Adding a new key to an existing section is covered by deep-merge defaults.

There are three loader paths:

| Loader | Used by |
| --- | --- |
| `load_cli_config()` in `cli.py` | classic CLI mode |
| `load_config()` in `hermes_cli/config.py` | subcommands, setup, tools |
| direct YAML load in `gateway/` | gateway runtime |

If CLI and gateway disagree, check which loader you changed.

### Tools vs plugins

Prefer plugins for local or third-party tools. Use built-in/core tools only for
capabilities that should ship in the base system. Core tool handlers must return
JSON strings and must be exposed through a toolset before agents can call them.

Plugins must not hardcode themselves into core files. If a plugin needs a new
capability, expand the generic plugin surface.

### Background process notifications

Gateway background process notifications are controlled by
`display.background_process_notifications` or `HERMES_BACKGROUND_NOTIFICATIONS`:
`all`, `result`, `error`, or `off`.

## Known Pitfalls

- Do not introduce new `simple_term_menu` usage. Use curses-based UI helpers.
- Do not use `\033[K` in spinner/display code; use space padding.
- `_last_resolved_tool_names` in `model_tools.py` is process-global and is
  saved/restored around subagent execution.
- Do not hardcode cross-tool references in schema descriptions. Add dynamic
  description adjustments in `model_tools.py` when a reference depends on the
  active toolset.
- Gateway control/approval commands must bypass both the base adapter message
  queue and the runner-level interrupt guard. See `gateway/AGENTS.md`.
- Before wiring unused code into a live path, E2E test the actual import and
  resolution chain against a temp `HERMES_HOME`.
- Squash-merging stale branches can revert unrelated fixes. Check the final diff.

## Testing

Always prefer:

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/path_or_file.py -q
scripts/run_tests.sh tests/path_or_file.py::test_name -v --tb=long
```

Do not call `pytest` directly unless you have a specific reason. The wrapper
sets CI-like environment, temp HOME/HERMES_HOME behavior, UTC locale, xdist, and
the in-tree subprocess isolation plugin. More detail lives in `tests/AGENTS.md`.
