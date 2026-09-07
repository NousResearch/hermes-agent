# plugins governance

Read repository-root `AGENTS.md` and its required routing table before acting.
These are supplemental current-lineage rules, not exceptions to global rules.
Read this file completely; automatic hints may truncate it. Scope follows the
behavior affected, including tests, docs and callers outside this directory.

## Plugins

Hermes has two plugin surfaces. Both live under `plugins/` in the repo so
repo-shipped plugins can be discovered alongside user-installed ones in
`~/.hermes/plugins/` and pip-installed entry points.

### General plugins (`hermes_cli/plugins.py` + `plugins/<name>/`)

`PluginManager` discovers plugins from `~/.hermes/plugins/`, `./.hermes/plugins/`,
and pip entry points. Each plugin exposes a `register(ctx)` function that
can:

- Register Python-callback lifecycle hooks:
  `pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`,
  `on_session_start`, `on_session_end`
- Register new tools via `ctx.register_tool(...)`
- Register CLI subcommands via `ctx.register_cli_command(...)` — the
  plugin's argparse tree is wired into `hermes` at startup so
  `hermes <pluginname> <subcmd>` works with no change to `main.py`

Hooks are invoked from `model_tools.py` (pre/post tool) and `run_agent.py`
(lifecycle). **Discovery timing pitfall:** `discover_plugins()` only runs
as a side effect of importing `model_tools.py`. Code paths that read plugin
state without importing `model_tools.py` first must call `discover_plugins()`
explicitly (it's idempotent).

#### Native plugin compatibility policy

The canonical contract and deprecation policy live in
`website/docs/developer-guide/plugins/index.md#native-plugin-compatibility-contract`.
Compatibility is enforced as a behavior contract, not through a monolithic
`PLUGIN_API_VERSION`, a manifest-wide native `api:` match, or version literals
on unrelated payloads. Keep documented plugin surfaces additive:

- add hook payload data as keyword fields; signature-inspect callbacks so old
  narrow signatures receive only fields they declare, while `**kwargs`
  callbacks receive the complete payload;
- do not remove or rename `PluginContext` methods; make new parameters optional
  with defaults and keyword-only where possible;
- ignore unknown native manifest fields;
- give new provider methods default implementations, and signature-inspect
  optional callback kwargs rather than forwarding them unconditionally;
- use a local schema version only for a capability with a wire or persisted
  contract, and preserve old state/config/session replay or ship a migration.

Deprecations require a once-per-process warning, a documented replacement and
migration note, and at least two subsequent minor releases before removal.
Compatibility tests must load frozen plugins through the real discovery path
and assert outcomes. Do not replace these with exact registry/catalog counts,
source-reading tests, or assertions that a global version literal changed.

### Memory-provider plugins (`plugins/memory/<name>/`)

Separate discovery system for pluggable memory backends. Current built-in
providers include **honcho, mem0, supermemory, byterover, hindsight,
holographic, openviking, retaindb**.

Discovery covers the same four sources as the general `PluginManager` —
bundled, `$HERMES_HOME/plugins/`, `./.hermes/plugins/` (opt-in via
`HERMES_ENABLE_PROJECT_PLUGINS`), and `hermes_agent.memory_providers` entry
points — but with **bundled-first** precedence, the reverse of the general
system's later-wins order: a memory provider is activated by name, so a
dropped-in directory must not be able to shadow a shipped one. Discovery
enumerates without importing; nothing runs until `memory.provider` names it.

Each provider implements the `MemoryProvider` ABC (see `agent/memory_provider.py`)
and is orchestrated by `agent/memory_manager.py`. Lifecycle hooks include
`sync_turn(turn_messages)`, `prefetch(query)`, `shutdown()`, and optional
`post_setup(hermes_home, config)` for setup-wizard integration.

**CLI commands via `plugins/memory/<name>/cli.py`:** if a memory plugin
defines `register_cli(subparser)`, `discover_plugin_cli_commands()` finds
it at argparse setup time and wires it into `hermes <plugin>`. The
framework only exposes CLI commands for the **currently active** memory
provider (read from `memory.provider` in config.yaml), so disabled
providers don't clutter `hermes --help`.

**Rule (Teknium, May 2026):** plugins MUST NOT modify core files
(`run_agent.py`, `cli.py`, `gateway/run.py`, `hermes_cli/main.py`, etc.).
If a plugin needs a capability the framework doesn't expose, expand the
generic plugin surface (new hook, new ctx method) — never hardcode
plugin-specific logic into core. PR #5295 removed 95 lines of hardcoded
honcho argparse from `main.py` for exactly this reason.

**No new in-tree memory providers (policy, May 2026):** the set of
built-in memory providers under `plugins/memory/` is closed. New memory
backends must ship as **standalone plugin repos** that users install
into `~/.hermes/plugins/` (or via pip entry points) — they implement
the same `MemoryProvider` ABC, register through the same discovery
path, and integrate via `hermes memory setup` / `post_setup()` without
landing in this tree. PRs that add a new directory under
`plugins/memory/` will be closed with a pointer to publish the
provider as its own repo. Existing in-tree providers stay; bug fixes
to them are welcome.

**No new third-party-product plugins in-tree (policy, June 2026):** the
same rule applies beyond memory providers. Plugins that integrate
someone else's product or project — observability/metrics backends,
vendor SaaS connectors, analytics dashboards, paid-service tie-ins —
must ship as **standalone plugin repos** that users install into
`~/.hermes/plugins/` (or via pip entry points). They register through
the existing plugin discovery path and use the ABCs/hooks/ctx surface
we expose; nothing special is needed in core. The reason is
maintenance load: every product we absorb into the tree becomes our
burden to keep working against a fast-moving core, for a backend we
don't own. Promote standalone plugins in the Nous Research Discord
(`#plugins-skills-and-skins`). PRs that add such a directory under
`plugins/` are closed with a pointer to publish it as its own repo —
this is a coupling decision, not a quality judgment. (The
`observability/`, `kanban/`, `disk-cleanup/`, etc. directories already
in the tree are existing precedent, not an invitation to add more
third-party-product plugins alongside them.)

### Model-provider plugins (`plugins/model-providers/<name>/`)

Every inference backend (openrouter, anthropic, gmi, deepseek, nvidia, …)
ships as a plugin here. Each plugin's `__init__.py` calls
`providers.register_provider(ProviderProfile(...))` at module load.
`providers/__init__.py._discover_providers()` is a **lazy, separate
discovery system** — scanned on first `get_provider_profile()` or
`list_providers()` call, NOT by the general PluginManager.

Scan order:
1. Bundled: `<repo>/plugins/model-providers/<name>/`
2. User: `$HERMES_HOME/plugins/model-providers/<name>/`
3. Legacy: `<repo>/providers/<name>.py` (back-compat)

User plugins of the same name override bundled ones — `register_provider()`
is last-writer-wins. This lets third parties swap out any built-in
profile without a repo patch.

The general PluginManager records `kind: model-provider` manifests but does
NOT import them (would double-instantiate `ProviderProfile`). Plugins
without an explicit `kind:` get auto-coerced via a source-text heuristic
(`register_provider` + `ProviderProfile` in `__init__.py`).

Full authoring guide: `website/docs/developer-guide/model-provider-plugin.md`.

### Dashboard / context-engine / image-gen plugin directories

`plugins/context_engine/`, `plugins/image_gen/`, etc. follow the same
pattern (ABC + orchestrator + per-plugin directory). Context engines
plug into `agent/context_engine.py`; image-gen providers into
`agent/image_gen_provider.py`. Reference / docs-companion plugins
(`example-dashboard`, `strike-freedom-cockpit`, `plugin-llm-example`,
`plugin-llm-async-example`) live in the
[`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins)
companion repo, not in this tree.

<!-- End of relocated baseline guidance. -->
