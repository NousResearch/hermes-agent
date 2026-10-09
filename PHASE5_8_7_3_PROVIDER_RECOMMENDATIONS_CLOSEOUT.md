# Phase 5.8.7.3 — Provider metadata and Nous recommendation closeout

Base: `888435ca3f` (Phase 5.8.7.2 catalogue hard cut).
Branch: `refactor/phase5-8-runtime-consumers`.

## Ownership changes

- **Nous provider plugin** now owns the public Portal HTTP request and provider
  response parsing, including gzip negotiation. Its `resolve_aux_model` hook
  obtains supplied application facts without importing `hermes_cli.models`.
  `providers.nous_recommendations.recommended_aux_model` continues to own
  pure free/paid and vision/compaction recommendation extraction.
- **Nous shared model catalogue** moved to
  `models.catalog_nous_recommendations`: one process/disk cache scoped by
  Hermes profile and effective Portal endpoint, original 10-minute TTL,
  original `cache/nous_recommended_cache.json` disk format and atomic merge,
  cross-process reuse, explicit refresh and stale-on-network-failure fallback.
  A failed refresh does **not** renew the last successful result's freshness.
  No lower-domain module reads account secrets or imports CLI auth.
- **Application facts** moved to `application_nous_recommendations`:
  resolve the authenticated Portal base URL and collect the existing
  profile-scoped free-tier fact. The existing single 180-second free-tier
  account cache remains in the application-side `hermes_cli.models` path
  until the Phase 6 credential cut; no parallel tier cache was added.
- **Consumer cutover:** application default selection, CLI curated-list
  augmentation, validation and picker warming now consult the application
  source and single canonical catalogue. Final default selection remains
  `models.selection`. Deleted the obsolete Nous catalogue cache/fetch and
  Portal auth-URL helper from `hermes_cli.models`; no internal CLI forwarding
  catalogue façade or duplicated in-memory/disk cache was retained.
- **Actual:** hosted/loopback `/v1` endpoint normalization and loopback
  identification are canonical in `providers.route_identity`. The provider
  plugin imports normalization directly from that lower domain; its only
  remaining `hermes_cli.auth` call obtains credentials. The existing auth
  public surface re-exports the same canonical functions; it does not own
  an alternate implementation.
- **Copilot/OpenRouter:** source inspection confirmed both already consume
  `models.metadata.github` and `models.metadata.reasoning`, respectively.
  Kept their provider-specific wire formatting unchanged and added focused
  tests of capability lookup, effort handling, unknown metadata, mandatory
  reasoning and the CLI-semantic import boundary.

## Verification

- Canonical ownership, cache lifecycle, gzip transport, Nous tier-specific
  recommendations, Copilot/OpenRouter and plugin import boundary: **6 passed**.
- Nous model/validation regressions (including numeric Portal model names):
  **20 passed; 131 unrelated tests deselected**.
- Initially failing legacy test imports and numeric Portal model-name handling
  repaired; their targeted rerun: **2 passed**.
- Free-account sign-in/default integration rerun: **3 passed, 3 deselected**.
- Focused Actual hosted/loopback/profile fetch rerun: **3 passed, 20 deselected**.
- Broad defaults/disk-cache/Actual regression output: **31 passed,
  2 platform skips**. The job wrapper reported a timeout during shutdown,
  so the stdout result is recorded without asserting a successful process exit.
- Targeted Python compilation and Ruff checks: **passed**.
- Production source audit: no remaining `fetch_nous_recommended_models`
  or plugin imports of `hermes_cli.models` in the scoped trees.

This step does not change credential acquisition, token refresh or account
entitlement ownership (Phase 6). It does not claim the full repository
baseline or a successful Arcana structural scan.

Next: **5.8.7.4** — platform-plugin presentation and warning cutover.
