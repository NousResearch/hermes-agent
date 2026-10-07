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

- **No new in-tree memory providers (May 2026).** `plugins/memory/` is closed (byterover,
  holographic, openviking, retaindb stay; bug fixes welcome). hindsight (Sep 2026), honcho,
  supermemory and mem0 (Oct 2026) moved to the plugin catalog — `plugin-catalog/<name>.yaml`, auto-installed by
  `hermes_cli/memory_provider_migration.py` for homes still configured for them. Host-side code a
  catalog provider still relies on (the `honcho_host_block` config storage kind, profile clone /
  rename / update-sync hooks) resolves the provider's modules through
  `plugins.memory.import_provider_module`, never a `plugins.memory.<name>` import. New
  backends ship as standalone repos implementing the same `MemoryProvider` ABC, discovered through
  the same path, integrated via `hermes memory setup` / `post_setup()`.
- **Other features that leave core** (Home Assistant platform + toolset, Oct 2026 →
  `plugin-catalog/homeassistant.yaml`) keep their names/config/env and get a row in
  `hermes_cli/left_core_migration.py::LEFT_CORE` (catalog plugin + read-only "this home uses it"
  predicate); `hermes update` and agent/gateway start install the plugin for homes that used it.
  Installing and converting core-era state are separate steps: the toolset-scope conversion runs
  once per home and row (`_left_core_scoped` in config.yaml) even when the plugin is already there.
  The automatic install also happens once per home and row (`_left_core_installed`, written only
  once the plugin dir exists): a later `hermes plugins remove` sticks, a failed install retries.
  Core special cases become `PlatformEntry` seams the plugin sets (`trusted_inbound`,
  `display_tier`, `shared_env_prefixes`), never a name check.
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
human-merged via PR (`plugin-catalog/README.md` = admission policy, mirrored word for word in
`website/docs/developer-guide/plugins/catalog-submission.md` with the submission guide; `plugin-catalog-ci.yml` clones
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
| General | `plugins/<name>/`, `~/.hermes/plugins/`, `./.hermes/plugins/`, pip entry points | `PluginManager` (`hermes_cli/plugins.py`), later-wins | `register(ctx)` registers hooks (`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`), tools (`ctx.register_tool`), CLI subcommands (`ctx.register_cli_command` — argparse tree wired into `hermes` at startup, no `main.py` change), language packs (`provides_locales` in the manifest auto-registers `locales/<lang>[.tui|.desktop].yaml` via `ctx.register_locale_dir`; manifest-only packs need no `__init__.py`; `hermes_cli/plugin_validate_locales.py` validates them) |
| Memory provider | `plugins/memory/<name>/` | `plugins/memory/__init__.py`: bundled → `$HERMES_HOME/plugins/` → `./.hermes/plugins/` (opt-in `HERMES_ENABLE_PROJECT_PLUGINS`) → `hermes_agent.memory_providers` entry points; **bundled-first** | Activated by name via `memory.provider`, so a dropped-in dir must not shadow a shipped one (reverse of general later-wins). Enumerates without importing. Implements `MemoryProvider` ABC (`agent/memory_provider.py`), orchestrated by `agent/memory_manager.py`: `sync_turn`, `prefetch`, `shutdown`, optional `post_setup`. `cli.py` with `register_cli(subparser)` is wired by `discover_plugin_cli_commands()` — only for the ACTIVE provider, so `hermes --help` stays clean |
| Model provider | `plugins/model-providers/<name>/` | `providers/__init__.py._discover_providers()`, **lazy**, on first `get_provider_profile()`/`list_providers()`; bundled → `$HERMES_HOME/plugins/model-providers/` → legacy `providers/<name>.py` | `__init__.py` calls `providers.register_provider(ProviderProfile(...))` at load; **last-writer-wins** so a user plugin overrides a bundled profile. `PluginManager` records `kind: model-provider` manifests but does NOT import them (would double-instantiate); manifests without `kind:` are auto-coerced by source heuristic (`register_provider` + `ProviderProfile`) |
| Context engine / image-gen / others | `plugins/context_engine/`, `plugins/image_gen/`, ... | ABC + orchestrator + per-plugin directory | Plug into `agent/context_engine.py`, `agent/image_gen_provider.py` |
| Platform adapters | `plugins/platforms/<name>/adapter.py` | gateway | Token-lock and scoped-secret rules in `gateway/AGENTS.md` (`irc`, `feishu` are canonical) |

**Discovery timing pitfall:** `discover_plugins()` runs only as a side effect of importing
`model_tools.py`. Code that reads plugin state without importing `model_tools.py` first must call
`discover_plugins()` explicitly (idempotent). Hooks are invoked from `model_tools.py` (pre/post
tool) and `run_agent.py` (lifecycle). A non-forced `discover_plugins()` short-circuits on `_discovered`: every
mid-run load path (install/enable/update on any surface, `reload-plugins` verb) runs
`discover_plugins(force=True)`, and `PluginManager.on_plugin_loaded` fires from inside that sweep for the
newly loaded plugins with an activation summary (`hermes_cli/plugins_activation.py`: handlers live now;
tools/prompt next session; `deferred.mcp_servers` until `mcp.reload`). Never emit that event from an RPC. Auxiliary LLM calls (titling, compression, MoA, vision, ...)
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

## Plugin host boundary (`plugins.isolation: host`)

`in_process` (default) imports user plugins into Hermes. `host` runs every non-bundled plugin in a
per-profile host process (`hermes_cli/plugin_host.py` parent side, `plugin_host_child.py` child,
`plugin_host_wire.py` framed JSON protocol). The child hands plugins a `RemotePluginContext`; Hermes
registers proxies (tools, hooks, commands, provider objects as subclasses of the ABC it checks).
Invariants:

- **Every user-code import path routes to the host or refuses** — general plugins
  (`plugins_loader`), memory / context engine / cron providers (`load_instance`), model-provider
  profiles (`plugin_host_profiles.py`: credential-free extraction cached by file fingerprint, because
  discovery runs while `hermes_cli.config`/`auth` import; overrides run by name in the host), dashboard
  APIs (ASGI bridge). `plugins/plugin_loader.load_plugin_module` refuses any other synthetic-namespace
  import. A new user-code loader MUST do the same.
- **What cannot cross is one table:** `plugin_isolation.HOST_UNSUPPORTED_CTX_METHODS` (live gateway /
  adapter objects). The runtime refusal and the static audit (`plugin_isolation_audit.py`, shown by
  `hermes plugins validate/show`) both read it; `evals/plugin_isolation/audit_catalog.py` re-measures
  the catalog.
- **A host never spawns a host** (`HERMES_PLUGIN_HOST_PROCESS=1` makes `isolation_mode()` in-process
  there) and must not start while a module import holds its lock (model-provider discovery is the
  known case).
- **The isolated party cannot opt out when an operator pins it:** `isolation_mode()` / `host_launcher()`
  read through the managed overlay (`load_config_readonly`), so `/etc/hermes/config.yaml` beats the
  profile's own config. Never switch them to a raw user-config read.
- Plugins stay tenant-unaware: no new API, same `ctx`. Per-profile env comes from
  `served_profile_child_env`, so a host sees only its own profile's secrets.

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

## Internal import paths are not API

PR #102117 moved internals into `<stem>_<topic>` siblings. The temporary compat layer that kept the
old import paths alive for external plugins was removed after its 2026-09-14 window; an old path now
raises `ImportError`, surfaced as the plugin's load error in `hermes plugins list`. Plugins build on
`ctx` and the documented ABCs. Never add re-export shims for an internal move. `hermes_cli/plugin_compat.py`
survives only as three inert stubs that already-running pre-removal updaters import.

## Tests

`tests/plugins/`. Load through real discovery with a temp `HERMES_HOME`; assert behaviour (tool
registered, hook fired with expected kwargs), not counts. Opt-in telemetry rule applies to plugins
too: no attribution tag ships by default.
