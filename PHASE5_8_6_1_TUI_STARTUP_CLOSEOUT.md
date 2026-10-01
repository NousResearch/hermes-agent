# Phase 5.8.6.1 — TUI startup and session rehydration closeout

Base: `3f3ddeeefc` (5.8.5.7 gateway boundary).
Branch/worktree: `refactor/phase5-8-runtime-consumers`.

## Ownership changes

- Shared, non-domain configured-alias and selection-fact projections moved from
  `gateway/model_aliases.py` and `gateway/session_model_facts.py` to
  `application_model_aliases.py` and `application_model_facts.py`. Existing
  gateway consumers now use those final shared application owners; no forwarding
  import or copy remains.
- `tui_gateway/model_startup_route.py` supplies TUI launch precedence/config
  facts to canonical `models.selection*` and `providers.routing`. Configured
  route keys and aggregator-native model IDs retain their startup meanings.
- URL-bearing launch aliases retain their own endpoint and credential. Explicit
  provider overrides can retain an alias endpoint but never borrow its key.
- Legacy bare-custom session identity recovery now uses
  `providers.configured.configured_custom_identity`; owned managed-runtime
  endpoint detection remains an application acquisition fact.
- Named custom identities are routable only when an exact effective provider or
  configured entry actually exists. The registry's generic `custom` profile
  cannot make a deleted `custom:<name>` entry appear valid.
- Rehydration no longer imports CLI model-derived API-mode logic. Existing
  credential acquisition supplies non-secret route inputs, and
  `providers.routing.resolve_invocation_route` decides the live URL, API mode,
  and runtime kind. A stored mode from another model is not an authority.
- The only `hermes_cli.runtime_provider` import in
  `tui_gateway/agent_factory.py` is `resolve_runtime_provider`, retained
  exactly as Phase 6 credential/runtime-acquisition debt.

## Verification

- Consolidated TUI startup, alias, model-route and persisted-session focused set: **14 passed**.
- Live stale/renamed-provider resume regression set: **3 passed**.
- Gateway launch and architecture boundary set: **29 passed**.
- `ruff check`, targeted `py_compile`, and `git diff --check` are required
  final hygiene gates; results are recorded in the 5.8.6.1 commit closeout.

## Known unrelated inherited failure

`tests/tui_gateway/test_profiles_inherit_launch_model.py` reaches an existing
dashboard import of `is_local_endpoint` from `agent.model_metadata`, although
its current owner is `models.metadata.context`. Neither module is changed by
5.8.6.1. Record this against the dashboard/web cut (5.8.6.6), not as an excuse
to expand TUI startup migration into dashboard implementation.

## Next

5.8.6.2 — TUI model-switch application coordinator. The TUI picker,
configuration and metadata consumers remain 5.8.6.3.
