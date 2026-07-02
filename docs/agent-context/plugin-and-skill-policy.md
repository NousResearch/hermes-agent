# Hermes Plugin and Skill Policy

This reference preserves plugin, provider, tool, config, and skill rules formerly embedded in `AGENTS.md`.

## Adding new tools

Before adding any tool, apply the Footprint Ladder from `contribution-rubric.md`.

For custom or local-only capabilities, do **not** edit Hermes core. Prefer the plugin route:

```text
~/.hermes/plugins/<name>/plugin.yaml
~/.hermes/plugins/<name>/__init__.py
```

Register tools with `ctx.register_tool(...)`. Plugin toolsets can be enabled/disabled without editing `tools/` or `toolsets.py`.

Built-in/core tools are only for capabilities that should ship in the base system. They require:

1. `tools/your_tool.py` with `registry.register(...)`
2. adding the tool name to `toolsets.py`

All tool handlers must return JSON strings. Use `get_hermes_home()` for persistent state and `display_hermes_home()` in user-facing schema/output text.

## Dependency pinning

All dependencies need upper bounds or fixed SHAs:

| Source type | Treatment |
|---|---|
| PyPI package | `>=floor,<next_major` |
| Git URL | commit SHA |
| GitHub Action | commit SHA + comment |
| CI-only pip | `==exact` |

When changing dependencies, update lockfiles and keep supply-chain risk bounded.

## Adding configuration

Behavior settings belong in `config.yaml`; secrets belong in `.env`.

For `config.yaml` options:

1. Add to `DEFAULT_CONFIG` in `hermes_cli/config.py`.
2. Bump `_config_version` only for active migrations/transformations, not simple new keys.

Know the three config loader paths:

| Loader | Used by |
|---|---|
| `load_cli_config()` | CLI mode |
| `load_config()` | `hermes tools`, setup, most subcommands |
| direct YAML load | gateway runtime |

If the CLI sees a key but the gateway does not, or the reverse, check loader coverage.

## Plugins

General plugins are discovered from:

- `~/.hermes/plugins/`
- `./.hermes/plugins/`
- pip entry points

Plugins can register lifecycle hooks, tools, and CLI commands.

Hard rule: plugins must not modify core files such as `run_agent.py`, `cli.py`, `gateway/run.py`, or `hermes_cli/main.py`. If a plugin needs more capability, expand the generic plugin surface instead of special-casing the plugin in core.

## Memory-provider plugins

Memory providers implement the `MemoryProvider` ABC and integrate through `agent/memory_manager.py`.

Policy: no new in-tree memory providers. New memory backends should ship as standalone plugin repos installed into `~/.hermes/plugins/` or via pip entry points.

## Model-provider plugins

Model providers live under `plugins/model-providers/<name>/` and call `providers.register_provider(ProviderProfile(...))`.

Discovery order:

1. bundled provider plugins
2. user provider plugins under `$HERMES_HOME`
3. legacy `providers/<name>.py`

User plugins of the same name override bundled ones.

## Skills

Two surfaces:

- `skills/` — built-in, loadable by default.
- `optional-skills/` — heavier/niche, installed explicitly.

New or modernized skills must keep listing footprint small:

- `description` ≤ 60 characters
- one sentence
- ends with a period
- no marketing words
- do not repeat the skill name

Skill details belong in `SKILL.md` and linked reference files, not always-on project context.
