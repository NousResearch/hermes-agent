# Hermes CLI and Configuration Guide

These instructions apply under `hermes_cli/`.

## Commands

Slash commands are registered through `hermes_cli/commands.py`; do not add
another dispatch system. Commands declare aliases, gateway availability,
argument policy, and optional configuration gates. Keep aliases and discovery
data in the registry so CLI, gateway, help, and completion remain consistent.

Interactive menus should use `hermes_cli/curses_ui.py`. Do not add new
`simple_term_menu` call sites; existing uses are legacy fallback only.

## Configuration

Behavioral settings belong in `config.yaml`. `.env` is only for credentials:
API keys, tokens, and passwords.

For a new configuration key:

1. add it to `DEFAULT_CONFIG` in `hermes_cli/config.py`;
2. use deep-merge defaults for ordinary additions;
3. bump `_config_version` only for an active migration or structural change;
4. verify every runtime loader that consumes the setting.

Loaders differ:

- `load_cli_config()` serves interactive CLI mode;
- `load_config()` serves setup, tools, and most subcommands;
- gateway runtime also reads YAML through `gateway/run.py` and
  `gateway/config.py`.

If CLI and gateway behavior disagree, trace the loaders before adding another
fallback.

Settings that internally require environment variables should be bridged from
`config.yaml`; user documentation must still point to YAML.

## Profile-safe paths

Use `get_hermes_home()` for state and `display_hermes_home()` for printed paths.
Do not hardcode `~/.hermes`.

Profile enumeration is intentionally anchored at
`Path.home() / ".hermes" / "profiles"` so every active profile can see its
siblings.

## Skins

Skins are data interpreted by `hermes_cli/skin_engine.py`. Add built-in skins
to `_BUILTIN_SKINS`; user skins load from `$HERMES_HOME/skins/*.yaml`.
Missing fields inherit from the default skin.

Do not add special-case rendering branches for a skin. New visual choices
should be represented in `SkinConfig` and consumed generically.

Validate CLI changes through the actual command entry point and configuration
loader, not only by constructing internal functions directly.
