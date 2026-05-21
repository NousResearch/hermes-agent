# Terminal Defaults Source-of-Truth Operational Contract

## Context

Phase 2 makes terminal runtime defaults and config-to-environment bridging explicit. The goal is to prevent drift between documented config, CLI/TUI config loading, gateway startup, and `tools/terminal_tool.py` runtime behavior.

This document is an operational contract for maintainers. It records the source of truth, entrypoint-specific cwd rules, environment serialization rules, and the installed WSL runtime expectations that must remain stable unless a future task deliberately updates this contract and its tests.

## Source of truth

The canonical terminal config implementation lives in `hermes_cli.terminal_config`:

- `DEFAULT_TERMINAL_CONFIG` owns terminal defaults.
- `default_terminal_config()` returns a deep copy of the defaults.
- `normalize_terminal_config(raw)` owns user-config normalization.
- `resolve_cli_terminal_cwd(...)` owns CLI/TUI cwd resolution.
- `resolve_gateway_terminal_cwd(...)` owns gateway cwd resolution.
- `terminal_env_values(config, *, include_secrets=False)` owns config-to-`TERMINAL_*` environment serialization.

Callers must not duplicate terminal default literals or independently serialize nested terminal config keys.

## Normalization contract

`backend` is the documented and canonical terminal backend key. `env_type` remains a legacy compatibility key because older runtime paths and environment names use `TERMINAL_ENV`.

`normalize_terminal_config(raw)` must preserve these invariants:

1. Missing, `null`, and non-dict terminal config normalize to `DEFAULT_TERMINAL_CONFIG` values.
2. Partial dict config merges over defaults.
3. Normalized config contains both `backend` and `env_type`.
4. `backend` and `env_type` have the same normalized value.
5. If both `backend` and `env_type` are present and non-empty, `backend` wins.
6. Mutable defaults such as env passthrough lists, Docker env maps, volumes, and extra args are deep-copied so normalized configs cannot alias the shared defaults or raw input.

## CLI/TUI cwd contract

CLI/TUI config loading normalizes terminal config, resolves cwd with `resolve_cli_terminal_cwd(...)`, then bridges with `terminal_env_values(..., include_secrets=True)`.

Cwd rules:

- Local backend uses the Hermes invocation cwd for `TERMINAL_CWD`.
- For local backend, an explicit configured `terminal.cwd` is intentionally ignored in favor of the invocation cwd.
- Non-local backends preserve explicit configured cwd values.
- Placeholder or missing non-local cwd values may resolve to no `TERMINAL_CWD` export so sandbox backends can keep their own safe defaults.
- Placeholders are `.`, `auto`, and `cwd`.
- A CLI/TUI process running inside a gateway process must not overwrite the gateway-selected `TERMINAL_CWD` marker path.

The installed WSL config may set `terminal.cwd: /home/shlee`, but a direct CLI invocation from another directory still exports that invocation directory for local terminal commands.

## Gateway cwd contract

Gateway startup normalizes terminal config, resolves cwd with `resolve_gateway_terminal_cwd(...)`, and bridges with `terminal_env_values(...)` without secrets.

Cwd rules:

1. An explicit configured `terminal.cwd` wins.
2. Placeholder or missing cwd first falls back to an existing non-placeholder `TERMINAL_CWD`.
3. If no existing non-placeholder `TERMINAL_CWD` exists, gateway uses `MESSAGING_CWD` as provided.
4. If `MESSAGING_CWD` is absent, gateway falls back to `Path.home()`.
5. Placeholders are `.`, `auto`, and `cwd`. Existing `TERMINAL_CWD` placeholders are ignored, but `MESSAGING_CWD` is not placeholder-filtered by current code.

Gateway and CLI cwd behavior intentionally differ: gateway prioritizes stable service/runtime cwd, while direct CLI prioritizes the user's invocation cwd for local work.

## Terminal tool runtime contract

`tools/terminal_tool.py` reads terminal runtime settings from `TERMINAL_*` environment variables.

- `TERMINAL_ENV` selects the backend.
- `TERMINAL_CWD` selects cwd when exported.
- `TERMINAL_TIMEOUT` and `TERMINAL_LIFETIME_SECONDS` select runtime timing values.
- When `TERMINAL_CWD` is absent for the local backend, the terminal tool uses the live process cwd.
- Sandbox and remote backends preserve backend-specific cwd safety behavior, including ignoring unsafe host/relative paths where appropriate.

The terminal tool may use `default_terminal_config()` for fallback values, but it must not become a competing source of defaults.

## Environment serialization and secret handling

`terminal_env_values()` owns serialization from normalized config to process environment variables.

Rules:

- Scalar values serialize with `str(value)`.
- Lists and dicts serialize as JSON.
- `backend` serializes to `TERMINAL_ENV` and takes precedence over legacy `env_type` if both are present.
- Sensitive mappings, such as `sudo_password` to `SUDO_PASSWORD`, are omitted by default.
- Sensitive mappings require explicit `include_secrets=True`.
- Gateway must call `terminal_env_values()` without `include_secrets=True`.
- CLI/TUI may include sensitive terminal mappings only when deliberately calling `include_secrets=True` for process env bridging where legacy terminal behavior requires it.

Do not log, display, or commit secret values while testing this bridge.

## Installed WSL runtime contract

The installed user config at `/home/shlee/.hermes/config.yaml` remains explicit and must not be edited by Phase 2 source changes:

```yaml
terminal:
  backend: local
  cwd: /home/shlee
  timeout: 180
  lifetime_seconds: 300
```

Expected installed display:

```text
◆ Terminal
  Backend:      local
  Working dir:  /home/shlee
  Timeout:      180s
```

Expected runtime env bridge when loaded from `/home/shlee` with `HERMES_HOME=/home/shlee/.hermes` and outside gateway mode:

- normalized terminal backend: `local`
- normalized terminal env_type: `local`
- normalized terminal cwd: `/home/shlee`
- normalized terminal timeout: `180`
- normalized terminal lifetime_seconds: `300`
- `TERMINAL_ENV=local`
- `TERMINAL_CWD=/home/shlee`
- `TERMINAL_TIMEOUT=180`
- `TERMINAL_LIFETIME_SECONDS=300`

## Verification checklist

Before declaring this contract or related terminal-default source-of-truth work complete, run fresh verification:

```bash
python -m pytest tests/hermes_cli/test_terminal_config_normalization.py tests/hermes_cli/test_ignore_user_config_flags.py tests/hermes_cli/test_placeholder_usage.py tests/gateway/test_config_cwd_bridge.py tests/tools/test_terminal_env_config.py tests/tools/test_terminal_config_env_sync.py tests/cli/test_cwd_env_respect.py tests/tools/test_terminal_task_cwd.py -q -o 'addopts='
/home/shlee/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main config show | awk '/◆ Terminal/{flag=1; print; next} flag && /^◆ /{flag=0} flag{print}'
git diff --check
git status --short
```

Also run a runtime env bridge probe from `/home/shlee` that sets `HERMES_HOME=/home/shlee/.hermes`, clears `_HERMES_GATEWAY`, imports `cli`, sets `cli._hermes_home = Path('/home/shlee/.hermes')`, calls `load_cli_config()`, and prints only non-secret terminal config/env values.

## Out of scope

- Editing `/home/shlee/.hermes/config.yaml` during Phase 2 source work.
- Changing production code as part of this documentation task.
- Changing sandbox backend behavior without a dedicated implementation plan and tests.
- Printing secrets or dumping full config/env maps in verification output.
