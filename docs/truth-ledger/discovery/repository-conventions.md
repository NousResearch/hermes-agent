# Repository and plugin convention inventory (T1)

Date: 2026-07-17
Task: t_a2374086
Workspace: /Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2

## Scope and method

Read-only codebase discovery for plugin registration, config schema behavior, plugin tests, command wiring, profile path rules, packaging, and docs.

## Executive summary

- Hermes plugin discovery is manifest-driven (`plugin.yaml`/`plugin.yml`) with path-derived keys and explicit plugin kinds (`standalone`, `backend`, `exclusive`, `platform`, `model-provider`) in `hermes_cli/plugins.py:275-313`.
- Non-bundled plugins are opt-in via `plugins.enabled`; `plugins.disabled` has higher precedence (`hermes_cli/plugins.py:1378-1467`, `tests/plugins/test_disk_cleanup_plugin.py:618-633`).
- Slash-command and CLI-command plugin surfaces are both wired and already integrated across CLI, gateway, and TUI (`hermes_cli/plugins.py:502-579`, `cli.py:8961-8977`, `gateway/run.py:10037-10051`, `tui_gateway/server.py:11843-11853,13101-13120`).
- Profile-safe filesystem conventions are centralized in `get_hermes_home()`/`display_hermes_home()` (`hermes_constants.py:55-110`, plus many call sites).
- Packaging includes bundled plugin manifests in both wheel and sdist channels (`pyproject.toml:338-354`, `MANIFEST.in:5-10`) and is enforced by tests (`tests/test_packaging_metadata.py:120-156`, `tests/test_project_metadata.py:235-254`).

## 1) Plugin registration/discovery conventions

1. Plugin sources and precedence
- Discovery sources are bundled, user, project (env-gated), and pip entry points (`hermes_cli/plugins.py:5-17,1319-1369`).
- Collision precedence is last-writer-wins by source order (`hermes_cli/plugins.py:16-17,1371-1383`).

2. Required plugin layout
- Directory plugins require both `plugin.yaml` and `__init__.py` with `register(ctx)` (`hermes_cli/plugins.py:19-21`).
- Scanner supports:
  - flat: `<root>/<plugin>/plugin.yaml`
  - one-level category: `<root>/<category>/<plugin>/plugin.yaml`
  - max depth capped at two path segments (`hermes_cli/plugins.py:1486-1559`, `tests/hermes_cli/test_plugin_scanner_recursion.py:73-127`).

3. Plugin kinds and routing
- Valid kinds are defined in `_VALID_PLUGIN_KINDS` and `PluginManifest.kind` docs (`hermes_cli/plugins.py:275,291-306`).
- `exclusive` and `model-provider` are recorded but not loaded by general PluginManager (`hermes_cli/plugins.py:1395-1423`).
- Bundled `backend` autoloads; bundled `platform` registers deferred lazy loaders (`hermes_cli/plugins.py:1425-1446`).

4. Opt-in behavior
- Non-bundled plugins are gated by `plugins.enabled`; `plugins.disabled` overrides (`hermes_cli/plugins.py:1378-1393,1452-1466`).
- Tests confirm default-disabled bundled standalone plugin and enable/disable precedence (`tests/plugins/test_disk_cleanup_plugin.py:593-633`).
- Migration logic grandfathered existing user plugins into `plugins.enabled` when opt-in became default (`hermes_cli/config.py:5734-5797`).

## 2) Config schema and plugin-related config behavior

1. Plugin discovery config semantics
- `_get_enabled_plugins()` returns `None` when key missing/malformed; comments define this as opt-in default behavior (`hermes_cli/plugins.py:241-254`).
- `_get_disabled_plugins()` preserves explicit deny-list behavior (`hermes_cli/plugins.py:225-237`).

2. Platform env schema auto-injection from manifests
- `requires_env`/`optional_env` from `plugins/platforms/*/plugin.yaml` are injected into `OPTIONAL_ENV_VARS` for `hermes config` UX (`hermes_cli/config.py:8316-8409`).
- Developer docs describe rich env metadata in platform plugin manifests (`website/docs/developer-guide/adding-platform-adapters.md:41-62,330-364`).

3. Plugin-specific trust gates
- Tool override by non-bundled plugins requires `plugins.entries.<plugin_id>.allow_tool_override: true` (`hermes_cli/plugins.py:409-423,448-470`).

## 3) Command and tool wiring conventions

1. Slash commands
- Registration API: `ctx.register_command(...)` (`hermes_cli/plugins.py:527-579`).
- Autocomplete/menus include plugin commands (`hermes_cli/commands.py:480-509,2042-2055`).
- Dispatch paths:
  - CLI: `cli.py:8961-8977`
  - Gateway: `gateway/run.py:10037-10051`
  - TUI gateway: `tui_gateway/server.py:11843-11853,13101-13120`

2. CLI subcommands
- Registration API: `ctx.register_cli_command(...)` (`hermes_cli/plugins.py:502-523`).
- Main argparse wiring loads plugin CLI commands dynamically, no hardcoded plugin commands in main (`hermes_cli/main.py:13133-13175`).

3. Tools
- Registration API: `ctx.register_tool(...)` delegates to `tools.registry` and tracks plugin tools (`hermes_cli/plugins.py:389-444`).
- Plugin toolsets are surfaced via plugin tool discovery (`hermes_cli/plugins.py:2422-2463`).

## 4) Profile path and persistence conventions

- Canonical profile-scoped root is `get_hermes_home()`; default fallback is platform-native home (`hermes_constants.py:46-110`).
- Profile-aware display path should use `display_hermes_home()` (usage examples in CLI/gateway outputs, e.g. `cli.py:8783`, `gateway/session.py:601`).
- Model-provider user plugin path is explicitly `$HERMES_HOME/plugins/model-providers/<name>/` (`providers/__init__.py:5-7,91-99,163-171`).

## 5) Packaging and distribution conventions

- Wheel package-data includes plugin manifests, READMEs, and dashboard assets (`pyproject.toml:338-354`).
- Sdist includes plugin manifests recursively (`MANIFEST.in:5-10`).
- Tests enforce both channels:
  - `tests/test_packaging_metadata.py:120-156`
  - `tests/test_project_metadata.py:235-254`

## 6) Documentation convention inventory

- General plugin surface: `website/docs/developer-guide/plugins/index.md` (routing map + opt-in/debug notes, e.g. lines `13-36`, `330-356`).
- Model-provider plugin lifecycle: `website/docs/developer-guide/model-provider-plugin.md:15-35`.
- Platform-adapter plugin path and env schema surface: `website/docs/developer-guide/adding-platform-adapters.md:31-62`.

## 7) Recommended paths for Truth Ledger work

1. If implemented as a bundled standalone plugin in this repo
- Code: `plugins/truth-ledger/`
  - required: `plugin.yaml`, `__init__.py`
  - optional support files under same directory
- Focused tests: `tests/plugins/test_truth_ledger_plugin.py`
- Discovery docs/artifacts: `docs/truth-ledger/discovery/`

2. If implemented as a user-installed plugin (no core tree change)
- Runtime install path: `$HERMES_HOME/plugins/truth-ledger/`
- Enablement: add key to `plugins.enabled` or run `hermes plugins enable truth-ledger` (aligned with opt-in behavior from `hermes_cli/plugins.py:1452-1467` and docs `website/docs/developer-guide/plugins/index.md:353-355`).

3. If implemented as model-provider style extension
- Path shape: `$HERMES_HOME/plugins/model-providers/<name>/` with module-level `register_provider(...)` (`providers/__init__.py:8-15,140-149`).

## 8) Uncertainties / flags for orchestrator

1. Approved-plan file path in card not found at stated location
- Tried: `/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2/.hermes/plans/2026-07-17_143520-truth-ledger-option-2.md`
- Result: not present during this run.
- Impact: this inventory is grounded in repository source and current docs/tests; no direct quotes from the referenced plan file.

2. `plugins` root key is migration- and loader-handled
- The behavior is clearly defined in loader and migration code (`hermes_cli/plugins.py:241-254`, `hermes_cli/config.py:5734-5797`).
- If strict schema docs are needed for a separate validation tool, add an explicit schema note in downstream tasking.

## 9) Fresh-worker reproducibility checklist

- Re-scan plugin rules: `hermes_cli/plugins.py` around lines `1319-1467` and `1486-1559`.
- Recheck opt-in tests: `tests/plugins/test_disk_cleanup_plugin.py:593-633`.
- Recheck scanner recursion tests: `tests/hermes_cli/test_plugin_scanner_recursion.py:73-153`.
- Recheck package-data + manifest guards: `pyproject.toml:338-354`, `MANIFEST.in:5-10`, `tests/test_packaging_metadata.py:120-156`.
- Recheck command dispatch points: `cli.py:8961-8977`, `gateway/run.py:10037-10051`, `tui_gateway/server.py:11843-11853,13101-13120`.
