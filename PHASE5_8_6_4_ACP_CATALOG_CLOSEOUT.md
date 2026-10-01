# Phase 5.8.6.4 — ACP catalogue hard-cut closeout

Base: `69ec283fc4` (5.8.6.3 TUI picker). Branch:
`refactor/phase5-8-runtime-consumers`.

## Canonical ownership

- `models/catalog_configured.py` now owns the pure parsing and
  discovery/allowlist flags for both modern `providers:` and legacy
  `custom_providers:` declarations. CLI model-switch code imports the same
  final functions; no duplicate declaration grammar remains there.
- `models/catalog_endpoint.py` is a public read-only endpoint catalogue
  query. It uses the canonical Ollama classifier, supports native
  `/api/tags` plus generic `/models` and `/v1/models` fallback, honours
  Anthropic-style authentication and caller-specific headers, rejects HTTP
  redirects and non-HTTP schemes, and never persists credentials or results.
  In-process observations are bounded to 96 keys, encrypted-secret material
  is not stored in key strings (SHA-256 fingerprints), native positives
  expire after 5 minutes, generic positives after 15 minutes and failures
  after 15 seconds. This is a scoped observation cache, not a second
  provider registry; the existing shared inventory disk cache remains.
- ACP `model_catalog.py` no longer imports private
  `hermes_cli.model_switch`, `model_switch_providers`, `models_local`
  or CLI provider-label implementations. Effective provider labels, identity
  and `custom:<name>` references derive from `providers` and `models`.
- The application still reads config/credential facts in its own scope.
  `agent.secret_scope.get_secret` replaces ambient process environment
  reads for key-env named endpoints (fail-closed under multiplexing).
  Application-resolved custom TLS verify/bundle settings are passed to the
  public query; the domain never imports application trust/config modules.
- ACP preserves disabled-entry filtering, declared-model fallback on failed
  discovery, raw configured list allowlists despite compatibility projection
  converting lists into metadata mappings, keyless native Ollama discovery,
  authoritative empty native lists, explicit `custom:<name>` choice IDs
  and canonical/custom slug collisions. A failed native probe cannot admit an
  otherwise undeclared, uncredentialed generic proxy.
- ACP's existing `_ModelCatalog` read-only row projection retains its
  current selection, semantic deduplication and per-provider cap. Built-in
  rows continue through the existing shared `hermes_cli.inventory`
  application projection; no competing inventory was created.

## Verification

- Focused post-TLS query/ACP architecture, admission and named-provider
  tests: **31 passed**; Ruff and targeted Python compilation passed.
- CLI picker, configured-provider routing, secret scope and model boundary
  regressions: **119 passed**, one unrelated defective test excluded.
- Expanded post-TLS ACP/session suite: **71 passed, 2 deselected**; the two
  deselections are known Windows symlink-normalization failures.
- New tests cover no-redirect behavior, Anthropic auth, base URL fallback,
  native empty vs explicit pins, hashed/cache invalidation and negative TTL,
  key-scope isolation, custom TLS policy, and AST ownership checks.

## Deferred, intentionally

- The legacy shared inventory's live provider catalogue assembly still
  reaches CLI code internally. Final provider/plugin discovery and unified
  disk-cache ownership are 5.8.7/5.9 work; do not claim those dependencies
  have been removed simply because ACP consumes the shared inventory.
- `acp_adapter/server.py::_switch_model` remains application transaction
  cut 5.8.6.5, not this read-only catalogue step.
- Dashboard's stale `agent.model_metadata.is_local_endpoint` import is
  already recorded for 5.8.6.6. Credential acquisition remains Phase 6.
- Existing unrelated test defects: Windows symlink canonical-CWD tests in
  `tests/acp_adapter/test_session.py` (two `require_symlinks` cases);
  `tests/hermes_cli/test_model_switch_custom_providers.py` has an undefined
  `providers_mod` in `test_list_splits_comma_chain_custom_provider_model`.
  These were not altered as part of the ACP catalogue cut.

Next: 5.8.6.5 — ACP live session model-switch transaction.
