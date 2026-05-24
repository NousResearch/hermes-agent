# CLI, Config, Dashboard, and Plugin Loader Guide

This directory owns CLI subcommands, setup/config management, profile handling,
the plugin manager, skin/theme data, dashboard routes, and many user-facing
management commands. `cli.py` at repo root is the classic interactive shell and
shares these rules.

## Slash Commands

All slash commands are defined in `hermes_cli/commands.py` as
`COMMAND_REGISTRY` entries. Downstream surfaces derive from this registry:

- classic CLI dispatch in `HermesCLI.process_command()`
- gateway known-command and help generation
- Telegram bot menu
- Slack subcommand routing
- autocomplete and CLI help

Adding an alias only requires updating the existing `CommandDef.aliases`.

Adding a new command normally requires:

1. Add `CommandDef(...)` in `hermes_cli/commands.py`.
2. Add the classic CLI handler in `cli.py`.
3. Add a gateway handler in `gateway/run.py` if the command is available there.
4. Persist settings with the existing config helpers, not ad hoc YAML writes.

## Config

`hermes_cli/config.py::DEFAULT_CONFIG` is the canonical default tree for most
subcommands. The gateway has direct YAML-loading paths; if a new setting behaves
differently in gateway and CLI, check both.

Rules:

- Non-secret settings go in `config.yaml`.
- Secrets, API keys, and tokens go in `.env` via `OPTIONAL_ENV_VARS`.
- Do not bump `_config_version` for a simple new default key.
- Bridge legacy env vars from config in code when backward compatibility is
  required.

Working directory rules:

- CLI mode uses `os.getcwd()`.
- Messaging/gateway mode uses `terminal.cwd` and bridges it to `TERMINAL_CWD`.
- `MESSAGING_CWD` and `.env` `TERMINAL_CWD` are deprecated as canonical config.

## Profiles

Profiles are isolated Hermes homes. `_apply_profile_override()` must run before
imports that read profile-scoped paths. Use `get_hermes_home()` for state and
`display_hermes_home()` for messages.

Profile management itself is HOME-anchored so any active profile can see all
profiles.

## Plugin Manager

`hermes_cli/plugins.py` discovers general plugins from:

- `~/.hermes/plugins/`
- `./.hermes/plugins/`
- pip entry points

General plugins can register lifecycle hooks, tools, slash commands, CLI
commands, and plugin-local skills. Discovery happens as a side effect of
importing `model_tools.py`; paths that need plugin state without importing
`model_tools.py` must call `discover_plugins()` explicitly.

Do not hardcode plugin-specific logic in core. Expand the generic plugin
surface if a plugin needs a capability.

## Shell Hooks

`agent/shell_hooks.py` bridges `config.yaml` `hooks:` entries into the same
plugin hook manager. `hermes_cli/hooks.py` provides:

- `hermes hooks list`
- `hermes hooks test`
- `hermes hooks revoke`
- `hermes hooks doctor`

Unseen shell hooks require consent unless `--accept-hooks`,
`HERMES_ACCEPT_HOOKS=1`, or `hooks_auto_accept: true` is set. This matters for
gateway, cron, and headless runs.

Hook scripts read JSON from stdin. Blocking a `pre_tool_call` should return the
supported block shape consumed by `get_pre_tool_call_block_message()`.

## Skin and Theme System

`hermes_cli/skin_engine.py` is data-driven. Add built-in skins to
`_BUILTIN_SKINS`; user skins live in `~/.hermes/skins/<name>.yaml`.

Skins customize banner colors, response box border, spinner faces/verbs/wings,
tool prefixes, per-tool icons, branding text, and prompt symbol. Missing values
inherit from `default`.

## Dashboard

The dashboard can provide structured React UI around the TUI, but it must not
rebuild the primary chat transcript or composer. `/chat` embeds the real
`hermes --tui` through the PTY bridge.

Keep dashboard sidebars, inspectors, and status views independent from the PTY
child session. Failures in supporting UI should not break the terminal pane.

## Kanban CLI

`hermes_cli/kanban.py` wires `hermes kanban` verbs such as `init`, `create`,
`list`, `show`, `assign`, `link`, `unlink`, `comment`, `complete`, `block`,
`unblock`, `archive`, `tail`, `watch`, `stats`, `runs`, `log`, `dispatch`,
`daemon`, and `gc`.

Keep CLI board operations aligned with `tools/kanban_tools.py`. Worker agents
should use `kanban_*` tools rather than shelling out to `hermes kanban`.

## Interactive UI

Do not add new `simple_term_menu` usage. Use stdlib curses UI helpers such as
`hermes_cli/curses_ui.py`; see `hermes_cli/tools_config.py` for the pattern.
