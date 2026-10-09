# Phase 5.8.6.5 — ACP session model-switch hard cut

Base: `9f3ebbc5e7` (5.8.6.4 ACP catalogue).
Branch/worktree: `refactor/phase5-8-runtime-consumers`.

## Ownership

- `acp_adapter/model_switch_resolution.py` owns the ACP session selection
  request. It gathers caller-owned configuration, alias and existing-session
  facts and selects with `models.selection.select_explicit_model`. Provider
  identity and the per-model wire route come exclusively from `providers`
  and `providers.routing.resolve_invocation_route`; the shared application
  validation policy checks admission. ACP no longer imports the CLI model
  switch coordinator or TUI/gateway session-switch applications.
- The single `hermes_cli.runtime_provider.resolve_runtime_provider` call is
  the explicit Phase 6 *credential acquisition* seam. It is not model/route
  authority. ACP reuses the current live endpoint and credential/pool only
  when appropriate; the same-host/different-path alias case requires fresh
  acquisition. A qualified provider may preserve an alias URL but cannot
  borrow the alias's other-provider credential. Missing credentials are
  client-invalid selection; an internal unscoped-secret bug remains internal.
- `acp_adapter/server.py::_switch_model` owns the session transaction:
  resolves once, rebuilds the agent once, and passes the same validated
  runtime directly to `SessionManager._make_agent`. That factory skips
  its independent credential lookup when handed resolved switch material,
  but unchanged new-session/fork paths retain their existing acquisition.
  It carries existing session MCP/enabled/disabled toolsets into the rebuild.
- Canonical routing recalculates `api_mode` for every target model, even on
  a preserved same-provider endpoint. No process-global model environment or
  YAML setting is changed by ACP model switching.
- Admission while a turn or command is active remains forbidden; synchronous
  resolution/rebuild stay on the ACP worker thread, never the event loop.
  Queued prompts drain *after* the ACP RPC response is queued.
- `SessionManager.save_session(strict=True)` is used only for the ACP switch
  commit. An agent-build or propagated persistence failure restores the
  previous in-memory agent/model. Other existing persistence remains
  best-effort. The DB's underlying update/replace operations were not
  refactored into one physical database transaction here; failures after
  partial underlying writes cannot be promised a fully atomic disk rollback.
- Shared application validation now exports
  `validate_model_switch` publicly. TUI/gateway enrichment uses that same
  owner, and gateway tests target the final name without a compatibility
  forwarding shim.

## Verification

- Canonical ACP resolver, runtime handoff, invalid-parameter mapping,
  thread/busy/queue isolation, MCP preservation and AST ownership:
  **22 passed, 1 dashboard-only test deselected**.
- Broader ACP commands, session, model state and catalogue:
  **59 passed, 2 existing Windows symlink tests deselected**.
- TUI/gateway switch and shared model architecture:
  **36 passed**.
- Ruff and targeted Python compilation passed. Staged Git whitespace gate
  runs at commit closeout.
- Additional targeted cases cover same-host/different-path credential
  isolation, qualified alias URLs, credential pool reuse, missing credentials,
  strict persistence propagation and restoration on commit failure.

## Later phases (deliberately excluded)

- 5.8.6.6: dashboard model assignment and the inherited
  `agent.model_metadata.is_local_endpoint` import fault. The mixed ACP +
  dashboard regression is deselected until that cut.
- 5.8.7/5.9: legacy CLI catalogue validation/discovery leaf ownership and
  remaining app callers. The shared application validator still invokes
  `hermes_cli.models_validate.validate_requested_model`; this is not an ACP
  model-switch coordinator or a second canonical selection/route authority.
- Phase 6: provider-specific credential acquisition, refresh, pool/OAuth
  ownership. The ACP cut intentionally does not reimplement those sources.

Next: 5.8.6.6 — dashboard main-model assignment hard cut.
