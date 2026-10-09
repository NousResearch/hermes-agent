# Phase 5.8.6.3 — TUI configuration, picker and metadata closeout

Base: `f9eaae780a` (5.8.6.2 TUI switch coordinator).
Branch/worktree: `refactor/phase5-8-runtime-consumers`.

## Ownership cut

- `tui_gateway/entry.py` and the classic CLI TUI loop now schedule the
  single `application_picker_prewarm.prewarm_picker_cache_async` service
  instead of calling the CLI model-switch provider module. The service captures
  the current profile context, launches exactly one daemon worker per process,
  and invokes the existing shared inventory path against the SAME
  provider-model disk cache. No parallel TUI cache was added.
- `tui_gateway/methods_config.py` obtains its `config.get provider` rows
  from `application_provider_listing`: stable presentation order comes from
  the provider descriptor projection; identity and aliases come from the
  canonical provider registry; authentication flags remain acquisition facts.
  `hermes_cli.models.list_available_providers` imports this same owner.
- The Bots/profile editor now uses `application_model_selection_guards` for
  its confirmation handshake, matching the live TUI switch.
- The network-free session-create model/provider conflict check lives in
  `models.selection_conflict` over the canonical curated model tables and
  provider identity. The previous `hermes_cli.models_validate` implementation
  was removed; the old CLI validator imports the exact lower-domain function.
  Custom/aggregator/undecidable names remain permissive.
- TUI fast-mode/session metadata was already using
  `models.metadata.fast_mode`; source audit and existing regression tests
  confirmed no duplicate metadata migration was needed.
- No TUI configuration, picker or profile module imports CLI-owned model
  selection, provider normalization, private model catalogues or model-switch
  prewarming.

## Behavioural coverage

- Picker prewarming, TUI startup ordering, heartbeat and orphan sweep:
  **19 passed**.
- External-process/plugin provider listing, live picker and fast-mode
  session scope: **13 passed**.
- Profile guard handshake: **4 passed**, after targeting the shared warning
  owner and isolating its test fixture from unrelated dashboard server imports.
- New static vendor conflict, configured provider payload and AST architecture
  regression tests supplement the existing session-create guard checks.
- Full TUI configuration, profile, session-create and runtime architecture
  integration suite: **50 passed**.
- Ruff and targeted Python compilation passed on the migrated sources and
  regression tests. Staged-diff whitespace gate runs before commit.

## Strictly scoped remaining work

- `hermes_cli.inventory` remains the existing shared application inventory
  entry point. Internally it still delegates part of live catalogue assembly to
  legacy CLI picker code. Phase 5.8.7 must address provider/plugin discovery
  ownership; Phase 5.9 must close and remove any surviving CLI semantic bridge.
  Moving its whole live catalogue/cache substrate is not duplicated in this
  TUI consumer step. No second catalogue authority was introduced.
- The dashboard import of removed
  `agent.model_metadata.is_local_endpoint` remains Phase 5.8.6.6 work,
  independently confirmed when the profile test attempted to load the whole
  dashboard. The profile fixture now exercises the TUI handshake without
  treating this pre-existing dashboard error as a TUI change.
- Credential acquisition/OAuth/pool state remains Phase 6.

Next: 5.8.6.4 — ACP catalogue hard cut.
