# Phase 6.2 — Canonical authentication contracts

These are final-owner data contracts, not a second credential manager or an
adapter to the old pool. Storage/pool/OAuth moves remain steps 6.3–6.5, and
runtime consumers switch in 6.6. No production selection behaviour changes
in 6.2.

## Request and application boundary

- `auth.context.CredentialScope` captures an explicit absolute profile home
  using the existing `hermes_home_key` identity. Optional execution ID binds
  feedback/leases within that home. No ambient profile fallback or new scope
  binder is introduced.
- `AuthSettings` carries resolved external-login policy and pool strategy for
  the requested provider. Existing configuration code still owns defaults,
  parsing, validation and writes.
- `AuthContext` carries that scope, settings and an application-supplied
  `read_secret(scope, name)` callable. The existing scope-aware secret owner
  must service it; a scoped miss must not borrow the launch profile's value.
  Supply a fresh context when settings change, rather than installing a
  global settings snapshot.
- `auth.credentials.CredentialRequest` contains canonical provider ID,
  explicit endpoint scope, application context and optional model ID.
  Routing supplies the provider and canonical route identity. Authentication
  neither selects providers/models nor resolves their aliases.

`EndpointScope.endpoint` is the already-canonical base URL for HTTP, or an
explicit provider SDK/process identity for non-HTTP routes. No route inference
or duplicate URL normalization is introduced. Custom routes also supply their
durable configured provider identity, including when two entries share a URL.
Requests for `custom` or `custom:...` reject missing custom identity. Durable
custom slugs using another provider protocol must also be supplied in
`custom_provider_id`; that mapping belongs to the routing/config boundary.

No provider list or registry is snapshotted at import/construction time.
Provider IDs from `ProviderProfile.name` are accepted unchanged, including
plugins registered later. Canonicalization and profile-specific discovery
remain with `providers/` and the caller's already-bound scope.

## Selection, leases and failure

`CredentialIdentity` is provider + endpoint/custom identity + profile/execution
scope + existing credential ID. Success/failure feedback must use this handle,
never infer the used row from a mutable pool cursor. Model ID stays on the
captured request for model-specific cooldown reporting.

`CredentialSelection` returns that identity, provenance, redacted runtime
`CredentialMaterial`, and an optional `CredentialLease`. The existing pool
allocates and releases leases; the contract creates no pool or lease registry.
A lease for another credential or scope is rejected. Material preserves
opaque provider-specific client options; it is not an OAuth grant or a
persisted row and must not be logged or serialized.

`CredentialResult` is the union of a selection and `CredentialFailure`.
Failures expose distinct `CredentialStatus` outcomes:

| Status | Meaning |
|---|---|
| selected | Usable runtime credential selected; required refresh writes succeeded |
| not_configured | No applicable configured credentials in the requested scope |
| exhausted | Applicable credentials are temporarily unavailable under existing cooldown policy |
| invalid | Applicable credentials are terminally invalid and cannot currently serve the request |
| refresh_failed | Refresh failed; do not reinterpret this as an empty pool |
| persistence_failed | Required credential write failed; do not claim durable rotation |

Failure identity is optional when no specific row is known. Existing error
codes, re-login signals and optional recovery epoch time remain available.
Diagnostic text and credential material are excluded from repr. There is no
new retry policy or change to persisted status strings.

## Slice verification

Run the canonical `scripts/run_tests.sh` against `tests/auth/`, packaging,
existing provider-boundary, OAuth stampede/profile-fork/deferred-refresh,
store-encoding, and provider-registry import-order/mid-discovery regressions.
6.2 validates the contracts and keeps the old runtime green; enforcement inside
pool selection and refresh persistence is verified during their later cuts.

Verification on Windows, 2026-10-01, via the canonical hermetic runner:
- Contracts, boundary, packaging and pool/OAuth/provider-registry regression
  group: **89 passed across 10 files** (28 new contract cases).
- Credential lifecycle and multiplex cloud-client group: **9 passed across
  2 files**.
- No runtime consumers, configuration keys or persisted credential formats
  changed in this slice.
