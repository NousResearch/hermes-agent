# plugins/ — plugin kinds, compat contract, in-tree policy

Applies on top of the root `AGENTS.md`. Authoring guide + canonical compat contract:
`website/docs/developer-guide/plugins/index.md`. Per-kind guides: `memory-provider-plugin.md`,
`model-provider-plugin.md`, `context-engine-plugin.md`, `image-gen-provider-plugin.md`, ...

## Plugins never touch core (Teknium, May 2026)

Plugins live in their own directory and work within the ABCs / hooks / `ctx` surface we provide.
A plugin MUST NOT modify `run_agent.py`, `cli.py`, `gateway/run.py`, `hermes_cli/main.py`, etc.
If it needs a capability the framework lacks, widen the **generic** plugin surface (new hook, new
ctx method) and have the plugin use it — never hardcode plugin-specific logic into core (PR #5295
removed 95 lines of hardcoded honcho argparse from `main.py`). Plugin setup goes through
`hermes memory setup` → `provider.post_setup(hermes_home, config)`, never a parallel top-level
command. A hook with no concrete consumer is speculative infrastructure and is rejected (root).

## What may live in this tree (policy)

- **No new in-tree memory providers (May 2026).** `plugins/memory/` is closed (honcho, mem0,
  supermemory, byterover, hindsight, holographic, openviking, retaindb stay; bug fixes welcome). New
  backends ship as standalone repos implementing the same `MemoryProvider` ABC, discovered through
  the same path, integrated via `hermes memory setup` / `post_setup()`.
- **No new third-party-product plugins (June 2026).** Observability/metrics backends, vendor SaaS
  connectors, analytics dashboards, paid-service tie-ins ship as standalone plugin repos
  (`~/.hermes/plugins/` or pip entry point) promoted in Discord `#plugins-skills-and-skins`. Reason:
  every absorbed product is our maintenance burden against a fast-moving core for a backend we don't
  own. `observability/`, `kanban/`, `disk-cleanup/` are precedent, not an invitation. Closing such a
  PR is a coupling decision, not a quality judgment.
- Reference/docs-companion plugins (`example-dashboard`, `strike-freedom-cockpit`,
  `plugin-llm-example`, `plugin-llm-async-example`) live in
  [`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins), not here.

## Plugin catalog (`plugin-catalog/`, Sep 2026)

The ONLY discovery system for out-of-tree plugins. One YAML per entry, 40-hex SHA pin mandatory,
human-merged via PR (`plugin-catalog/README.md` = admission policy; `plugin-catalog-ci.yml` clones
each changed entry at its pin and runs `hermes plugins validate`). `removed.yaml` is the kill list —
every install path (CLI, dashboard, TUI) refuses matches (repo URLs compared by canonical
`host/owner/repo`, so `git@`/`ssh://`/`www.` spellings match); only the CLI has a loud
`--allow-removed`, which is recorded on the install record and is the only thing that exempts an
installed plugin from the same check at `update`, `enable` and load (`gate_manifest`). Catalog
provenance lives on the installer-owned `.install-metadata.json` record (`catalog` block, sha =
checked-out commit), NEVER in the tree: the in-tree `.hermes-catalog.json` is a convenience copy the
Desktop reads for "Install here"; Python never trusts it (a repo can ship a forged one).
Code: `hermes_cli/plugin_catalog.py` (loader, live refresh from
`/docs/api/plugin-catalog.json` published by the docs build, in-tree fallback),
`hermes_cli/plugins_cmd_catalog.py` (resolution, `.hermes-catalog.json` provenance sidecar,
search/info/validate, re-pin on `update`, dashboard/TUI payloads). Never add a second name index:
bare names resolve through the catalog or error.

## Plugin kinds and their discovery systems

| Kind | Where | Discovery | Notes |
|---|---|---|---|
| General | `plugins/<name>/`, `~/.hermes/plugins/`, `./.hermes/plugins/`, pip entry points | `PluginManager` (`hermes_cli/plugins.py`), later-wins | `register(ctx)` registers hooks (`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`), tools (`ctx.register_tool`), CLI subcommands (`ctx.register_cli_command` — argparse tree wired into `hermes` at startup, no `main.py` change) |
| Memory provider | `plugins/memory/<name>/` | `plugins/memory/__init__.py`: bundled → `$HERMES_HOME/plugins/` → `./.hermes/plugins/` (opt-in `HERMES_ENABLE_PROJECT_PLUGINS`) → `hermes_agent.memory_providers` entry points; **bundled-first** | Activated by name via `memory.provider`, so a dropped-in dir must not shadow a shipped one (reverse of general later-wins). Enumerates without importing. Implements `MemoryProvider` ABC (`agent/memory_provider.py`), orchestrated by `agent/memory_manager.py`: `sync_turn`, `prefetch`, `shutdown`, optional `post_setup`. `cli.py` with `register_cli(subparser)` is wired by `discover_plugin_cli_commands()` — only for the ACTIVE provider, so `hermes --help` stays clean |
| Model provider | `plugins/model-providers/<name>/` | `providers/__init__.py._discover_providers()`, **lazy**, on first `get_provider_profile()`/`list_providers()`; bundled → `$HERMES_HOME/plugins/model-providers/` → legacy `providers/<name>.py` | `__init__.py` calls `providers.register_provider(ProviderProfile(...))` at load; **last-writer-wins** so a user plugin overrides a bundled profile. `PluginManager` records `kind: model-provider` manifests but does NOT import them (would double-instantiate); manifests without `kind:` are auto-coerced by source heuristic (`register_provider` + `ProviderProfile`) |
| Context engine / image-gen / others | `plugins/context_engine/`, `plugins/image_gen/`, ... | ABC + orchestrator + per-plugin directory | Plug into `agent/context_engine.py`, `agent/image_gen_provider.py` |
| Platform adapters | `plugins/platforms/<name>/adapter.py` | gateway | Token-lock and scoped-secret rules in `gateway/AGENTS.md` (`irc`, `feishu` are canonical) |

**Discovery timing pitfall:** `discover_plugins()` runs only as a side effect of importing
`model_tools.py`. Code that reads plugin state without importing `model_tools.py` first must call
`discover_plugins()` explicitly (idempotent). Hooks are invoked from `model_tools.py` (pre/post
tool) and `run_agent.py` (lifecycle). Auxiliary LLM calls (titling, compression, MoA, vision, ...)
fire `pre_auxiliary_call`/`post_auxiliary_call` from `agent/auxiliary_hooks.py` (payload = the
`*_api_request` shape + `aux_task`); they never fire the turn-scoped `pre/post_api_request` (#79733). When a plugin changes a default, add a migration guard keyed
on an "existing config" signal (`_explicitly_configured`) so existing users keep the old default.

**Lifecycle hooks fire under the owning profile's scope, and the caller binds it.**
`on_session_start`/`on_session_end`/`sync_turn`/`shutdown` are invoked from the turn (bound) AND
from eviction, shutdown, `tui_gateway` teardown and cron completion (bound by the caller via
`_run_release_in_profile_scope`, `_session_profile_runtime_scope`, `_profile_cron_scope`). One
process serves several profiles, so a provider never caches `hermes_home` from `initialize()` as
"the" home — key state by the home it is handed per call (`hermes_home_key()`) — and never reads
`os.environ` for credentials (`agent.secret_scope.get_secret`; a `check_fn` too). Background
work starts via `agent.memory_provider.spawn_context_thread`, never a bare `threading.Thread`,
or the worker runs with no scope and fails closed (or writes into the launch profile's tenant).
Platform plugins never mutate `os.environ`: YAML goes to `PlatformConfig.extra` through
`_shared.apply_yaml_bridge`, gates through `platform_gate_env` (`gateway/AGENTS.md`).

## Native plugin compatibility contract (summary — canonical text in the docs page)

Compatibility is a **behavior contract**, not a monolithic `PLUGIN_API_VERSION`, a manifest-wide
native `api:` match, or version literals on unrelated payloads. Documented surfaces stay additive:

- Hook payload data is added as **keyword fields**; callbacks are signature-inspected so old narrow
  signatures receive only the fields they declare and `**kwargs` callbacks get the full payload.
- Never remove or rename `PluginContext` methods; new parameters are optional with defaults and
  keyword-only where possible.
- Unknown native manifest fields are ignored.
- New provider methods get default implementations; optional callback kwargs are
  signature-inspected, not forwarded unconditionally.
- A local schema version exists only for a capability with a wire or persisted contract, and old
  state/config/session replay is preserved or migrated.
- Deprecations: once-per-process warning, documented replacement + migration note, ≥ 2 subsequent
  minor releases before removal.
- Compat tests load **frozen plugins through the real discovery path** and assert outcomes — never
  exact registry/catalog counts, source-reading tests, or "a global version literal changed".

## Sep 2026 decomposition compat window (ends 2026-09-14)

PR #102117 moved internals into `<stem>_<topic>` siblings. Old import paths resolve through
`PLUGIN-COMPAT` `__getattr__` blocks (listed in `COMPAT_MANIFEST.md` / `compat_manifest.json`)
until `hermes_cli.plugin_compat.COMPAT_REMOVAL_DATE`, when the commit that added them is reverted.
`hermes_cli/plugin_compat.py` is the single source: `scan_plugin` (AST scan), `compat_report`
(hits across enabled external plugins, cached to `HERMES_HOME/.plugin-compat-report.json`,
refreshed by discovery), `removal_in_effect`, `warn_once`. Surfaces: CLI banner notice,
`hermes plugins compat` (shows affected user plugins), `hermes doctor`, post-update notices, the
TUI/Desktop `plugins.compat_report` RPC. After the date `PluginManager` skips a hitting plugin
unless `plugins.allow_deprecated_imports: true`. **In-tree code and tests never use compat paths**
(`scripts/check_compat_pointers.py` in CI; `-W error::hermes_cli.plugin_compat.HermesPluginCompatWarning`).
External-plugin compat is handled ONCE here — never add per-PR re-export shims.

## Tests

`tests/plugins/`. Load through real discovery with a temp `HERMES_HOME`; assert behaviour (tool
registered, hook fired with expected kwargs), not counts. Opt-in telemetry rule applies to plugins
too: no attribution tag ships by default (`tests/plugins/memory/test_hindsight_provider.py`).
