# Phase 5.8.6.6 — Dashboard model-assignment hard-cut closeout

Base: `46a04d35c3` (5.8.6.5 ACP transaction).
Branch/worktree: `refactor/phase5-8-runtime-consumers`.

## Ownership and behavior

- `application_dashboard_model_selection.py` owns dashboard main-model request
  interpretation and acquisition facts. It uses `models.selection` for explicit
  model identity, `providers` for plugin/configured provider identity and
  `providers.routing` for model-specific endpoint/wire routing. It no longer
  invokes the CLI model-switch coordinator. Shared
  `application_model_switch_enrichment.validate_model_switch` performs
  model/chat admission; the existing CLI-hosted *validation leaf* remains
  separate follow-up debt, not another selection/route authority.
- `hermes_cli.web_server_config` now uses this application coordinator for
  main assignment, including the profile editor. Built-in provider aliases
  come from the canonical registry; user-declared bare `providers:` keys
  retain their stored identity, and legacy `custom:<name>` entries keep their
  durable identity. Unconfigured named-custom/unknown providers are rejected
  before credential acquisition. Missing-provider authentication maps to
  HTTP 400; internal secret-scoping faults are not mislabeled as bad input.
- Only `hermes_cli.runtime_provider.resolve_runtime_provider` supplies
  Phase 6 acquisition facts. A dashboard-submitted bare-custom endpoint is
  authoritative over stale current/env routing; a bare-custom alias endpoint
  receives no credential from another custom host. Explicit provider aliases
  may supply their URL but cannot silently donate an alias credential to
  another provider.
- Global persistence is now the shared
  `application_model_switch_persistence.apply_model_selection` policy.
  Main `model.default`, provider, URL, wire mode and context-pin/credential
  cleanup use the same updates as CLI, gateway and TUI. The existing
  profile-scoped configuration mutation lock, whole-config save, raw credential
  pointer/template preservation, Nous defaults, custom endpoint registration
  and stale auxiliary reporting remain application-owned and unchanged.
- The flat Settings Model field now uses
  `application_dashboard_model_detection.py`: canonical static/native-vendor
  model facts and credential-gated, unambiguous provider inference. Unknown
  names do not cause unauthorized paid-provider switches, and an existing
  custom endpoint is never automatically replaced by a guessed vendor.
  Cache-backed *live* current-provider detection is not reimplemented here;
  that provider inventory ownership belongs to 5.8.7.
- The dashboard's obsolete
  `agent.model_metadata.is_local_endpoint` import is replaced by the
  canonical `models.metadata.context` owner, including its context-length
  probe tests. Dashboard expensive-model confirmation now reads the existing
  shared `application_model_selection_guards` owner. Network preflight
  still occurs off the event loop and before the profile's config write lock.
  Auxiliary/MoA mutations are deliberately unaffected.

## Verification

- Canonical dashboard selection and flat Settings inference, normalized
  registered/custom-provider identity, credential-pointer/raw-template
  preservation, four-surface model persistence parity, dashboard metadata and
  confirmation behavior, ACP/dashboard rejection and profile confirmation
  have targeted behavioral regressions.
- AST boundary assertions prevent the dashboard main assignment from
  reimporting `hermes_cli.model_switch` or rebuilding its own persistence
  grammar.
- Final focused dashboard, normalization, persistence, confirmation,
  metadata, profile and ACP interaction suite: **55 passed**.
- Independent model architecture, gateway/TUI canonical switch and routing
  suite: **36 passed**. Total disjoint targeted checks: **91 passed**.
- Ruff, Python compilation and the unstaged Git whitespace gate passed.
  The staged diff gate runs immediately before commit.
- A separate, unrelated Cron configuration-guard test requiring
  pytest-asyncio was not part of the final gate because the bounded test
  runner deliberately disables pytest plugin autoload; no dashboard tests
  were excluded from the final 55-case run.

## Strictly deferred

- 5.8.6.7 web/desktop audit: other dashboard model *read/presentation* paths
  (recommended defaults, inventory and picker) still contain application
  fact-acquisition helpers under `hermes_cli`. Do not claim those have
  migrated simply because model assignment has one application owner.
- 5.8.7/5.9: underlying legacy CLI live catalogue and validation leaves,
  including `hermes_cli.models_validate.validate_requested_model`.
- Phase 6: credential source/acquisition/pool/OAuth ownership, presently
  accessed through the narrow runtime-acquisition/credential-check seams.

Next: 5.8.6.7 — web/desktop residual consumer audit.
