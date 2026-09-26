# Plugins

The root guidance applies. The canonical authoring and compatibility contract is `website/docs/developer-guide/plugins/index.md`. Provider-specific guides live beside it.

## Ownership policy

A plugin stays inside its directory and uses ABCs, hooks, and `PluginContext`. Plugin-specific behavior never enters `run_agent.py`, `cli.py`, `gateway/run.py`, or another core facade. When a real plugin needs a missing capability, extend the generic plugin contract and use it from the plugin.

New third-party products and memory backends ship as standalone plugin repositories through `~/.hermes/plugins/` or package entry points. Existing in-tree integrations may receive fixes. Reference/demo plugins live in the maintained example-plugin repository rather than this tree.

Plugin setup uses the existing owning command, such as `hermes memory setup` and `provider.post_setup`, not a parallel top-level flow.

## Catalog and provenance

`plugin-catalog/` is the discovery index for out-of-tree plugins. Each entry pins a full commit SHA and passes catalog validation. `removed.yaml` is the deny list. Only the CLI's explicit override may install a removed entry, and that decision persists on the installer-owned record.

Trust `.install-metadata.json` for catalog provenance. An in-repository `.hermes-catalog.json` is display convenience and cannot establish its own origin. Bare names resolve through the catalog or fail. Do not add another name index.

## Discovery and lifecycle

General plugins, memory providers, model providers, and platform adapters use different discovery precedence. Preserve the documented behavior in `website/docs/developer-guide/plugins/index.md`. Do not unify them by accident.

- General plugin discovery is triggered with `model_tools.py` import. A caller that reads plugin state earlier invokes `discover_plugins()` explicitly.
- Install, enable, update, and reload paths force rediscovery. `on_plugin_loaded` fires inside that sweep and reports which parts are live now or deferred. RPC wrappers do not emit it independently.
- Model-provider discovery remains lazy and must not also instantiate through the general manager.
- A default change for existing users has an explicit-config migration guard.
- Auxiliary model calls emit auxiliary hooks, not main-turn API hooks.

Lifecycle hooks run under the owning profile's full scope. Providers key process state by `hermes_home_key()`, read credentials through `agent.secret_scope`, and start background work with `agent.memory_provider.spawn_context_thread`. A provider does not cache its initialization home as the process-wide home. Platform plugin YAML goes to `PlatformConfig.extra` and uses gateway shared scope-aware readers rather than mutating `os.environ`.

## Compatibility

Native compatibility is behavioral and additive:

- Add hook payload fields as keywords and signature-filter them for older callbacks.
- Preserve `PluginContext` method names. New parameters are optional and preferably keyword-only.
- Ignore unknown native manifest fields.
- Give new provider methods default implementations and signature-filter optional callback arguments.
- Add a local schema version only for a wire or persisted contract, with replay or migration for older data.
- Deprecations warn once, name the replacement, document migration, and remain for the documented support interval.

Frozen plugin fixtures load through real discovery and assert outcomes. In-tree code imports defining modules and never uses compatibility pointers. `COMPAT_MANIFEST.md`, `compat_manifest.json`, and `hermes_cli/plugin_compat.py` own any temporary external-import compatibility and its enforcement.

## Tests

Run `tests/plugins/` through `scripts/run_tests.sh` with a temporary `HERMES_HOME`. Test actual discovery, registration, hook payload filtering, profile scope, and migration behavior. Do not assert registry counts or read source text. Telemetry and attribution remain opt-in.