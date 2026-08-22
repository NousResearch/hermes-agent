# Architecture Decision Records

## 2026-07-13: Scope plugin manager state by Hermes home/profile (keyed cache)

Status: Accepted

Context:
Hermes supports multiple profiles via different Hermes home directories.
Homes are switched two ways in a running process: the `HERMES_HOME`
environment variable (single-profile CLI/gateway processes), and the
context-local `set_hermes_home_override()` (`hermes_constants.py`), which
the multiplexed gateway worker (`gateway/run.py`'s `_profile_scope`) and
subagent/embedded callers use to serve several profiles from one
long-lived process. The override is a `ContextVar` and deliberately does
**not** mutate `os.environ`, since that would leak one profile's home
into every other concurrent task in the same process.

The plugin manager was a process-global single-slot singleton
(`_plugin_manager`). User-installed plugins are discovered from
`get_hermes_home() / "plugins"`, and context-engine plugins (e.g.
`hermes-lcm`) capture profile-scoped state — such as the LCM database
path — at registration time. A single-slot cache meant:

1. Switching homes via `set_hermes_home_override()` was invisible to a
   naive "did `HERMES_HOME` change" check, so the singleton silently kept
   serving the first profile's manager to every other profile in the
   process.
2. Even when a fresh `PluginManager` *was* created for a new home, plugin
   modules are imported into `sys.modules` as `hermes_plugins.<slug>` by
   `_load_directory_module`, and only that top-level module was ever
   replaced. A same-slug plugin's *relative* imports
   (`from . import state`) are cached separately under
   `hermes_plugins.<slug>.<submodule>`, and Python's import machinery
   resolves those from `sys.modules` first — so a profile switch could
   silently keep serving a previous profile's already-imported submodule
   code/state instead of re-executing the new profile's plugin.

Decision:
- Replace the single-slot singleton with a cache keyed on the *resolved*
  Hermes home path (`_plugin_managers_by_home: Dict[Path, PluginManager]`).
  `get_plugin_manager()` resolves the current home via `get_hermes_home()`
  (which itself already consults `get_hermes_home_override()` before
  `os.environ`), so both the env-var and context-local override paths are
  covered uniformly.
- `_plugin_manager` (the old single-slot name) is kept as a thin "last
  manager returned" pointer purely for backward compatibility with
  existing test code that does
  `monkeypatch.setattr(plugins_mod, "_plugin_manager", some_manager)`.
  When that name is monkeypatched to a manager the keyed cache doesn't
  know about, `get_plugin_manager()` treats it as an explicit injection
  and adopts it into the cache under the *current* resolved home, rather
  than discarding it.
- Both `PluginManager._load_directory_module` (initial/`force=True`
  reload within the same home) and the shared `_clear_plugin_submodules`
  helper (profile switch / test teardown) evict `sys.modules[module_name]`
  **and every name prefixed with `module_name + "."`** before a plugin
  slug is (re-)imported, so relative-import submodules can never survive
  a reload or a home switch.
- Test isolation (`tests/conftest.py`'s `_hermetic_environment` fixture)
  calls a new `_reset_plugin_managers_for_tests()` helper that drops the
  entire keyed cache and purges every plugin submodule from `sys.modules`
  between tests, instead of only resetting the single-slot pointer.

Consequences:
- Per-profile LCM instances (and any other context-engine plugin) use
  their own `{home}/lcm.db` regardless of whether the profile switch went
  through `HERMES_HOME` or `set_hermes_home_override()`.
- Plugin discovery remains cached within a profile for normal
  performance, and re-entering a previously-seen profile reuses its
  cached manager instead of rebuilding from scratch.
- Sequential *and* interleaved profile switching — in tests, the gateway
  multiplexer worker, or embedded callers using the context-local
  override — no longer leaks context-engine state, plugin module state,
  or stale relative-import submodules across profiles.
- Regression coverage exercises the real production path
  (`set_hermes_home_override()`) rather than only the env-var path, and
  includes a dedicated relative-import leak test.

## 2026-08-21: Configured intent does not confer effective capability

Status: Proposed

Context:
Hermes expresses intended capability through configuration, manifests,
catalog entries, authenticated accounts, remembered routes, deployment
plans, process identifiers, and resolved paths. Those values can select a
candidate, but runtime revocation, credential rotation, policy changes,
ambiguous ownership, process replacement, optional component absence, or
substrate failure can invalidate the candidate before the side effect.

Several concrete paths already depend on this distinction: browser control
must observe live Developer Mode revocation (#91695), Nix startup must wait for
the actual bind target (#91720), optional Desktop plugin entries must represent
absence without operational failure (#91828), durable gateway routing needs a
current credential claimant and adapter generation (#89252), and deployment
admission remains incomplete until physical consumers execute the admitted
plan without reconstructing or widening it (#91316).

Decision:
- Adopt [Effective Capability at Consuming Boundaries](effective-capability-contract.md)
  as the cross-cutting contract.
- Preserve the state progression `configured -> resolvable -> authenticated ->
  policy-admitted -> currently effective -> exercised -> settled`; no earlier
  state implies a later one.
- Require the component that owns the side effect to obtain or revalidate a
  current, scope-bound proof for the exact actor, operation, target, namespace,
  generation, policy, and realized substrate.
- Consume that proof atomically with the effect, or revalidate immediately
  before the effect after any asynchronous, retry, redirect, process, or lock
  boundary that can invalidate it.
- Represent absence, denial, ambiguity, unavailability, and staleness as
  distinct outcomes instead of overloading a configured/effective boolean.
- Keep exercise separate from settlement; operation acceptance or exit status
  does not prove terminal postconditions.
- Treat the contract as semantic rather than mandating one universal runtime
  class. Each subsystem retains its existing implementation owner and native
  types.

Consequences:
- Configured booleans, display names, catalog membership, successful adjacent
  API calls, path/PID/file existence, and old-generation capability verdicts
  may describe or select candidates but cannot independently authorize
  mutation.
- Legacy and inferred values may remain available for display, diagnostics, or
  explicitly read-only degradation. They fail closed for mutation when exact
  authority is absent.
- Capability-sensitive tests must reach the real consuming boundary and include
  relevant revocation, replacement, rotation, duplicate-claim, and stale-proof
  cases.
- The decision composes #90866, #90142, #90144, #90145, #90049, #90150,
  #90200/#90230, and #91230 rather than creating another architecture owner.
