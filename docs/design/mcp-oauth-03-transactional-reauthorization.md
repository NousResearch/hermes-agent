# MCP OAuth Chunk 3 — Transactional Reauthorization

Status: design proposal (not yet implemented; design-review updates applied 2026-09-03)
Depends on: unified lifecycle service (Chunk 2)
Fix target: GitHub issue #76590
Architecture: [`../architecture/mcp-oauth-credential-store-architecture.md`](../architecture/mcp-oauth-credential-store-architecture.md) §4.1, §5.3, §6.2, §6.3, §8.2, §14, §15, §18
Design-review updates: [`../requirements/mcp-oauth-design-review-approaches.md`](../requirements/mcp-oauth-design-review-approaches.md) (F-2 behavior, F-4 commit order; forward notes F-0/F-3, F-5)

## Purpose

Eliminate failed-reauthorization credential loss by running fresh OAuth flows against isolated in-memory staged state. The active credential remains unchanged until a complete replacement is validated and committed.

This chunk removes rollback rather than attempting to make rollback infer which files belong to which concurrent flow.

## Staged adapter

Add `StagedOAuthStorageAdapter`, implementing the asynchronous storage contract expected by the MCP SDK.

```python
class StagedOAuthStorageAdapter:
    def __init__(self, *, seed_client=None, seed_metadata=None): ...
    async def get_tokens(self) -> None: ...
    async def set_tokens(self, tokens) -> None: ...
    async def get_client_info(self): ...
    async def set_client_info(self, client) -> None: ...
    def load_oauth_metadata(self): ...
    def save_oauth_metadata(self, metadata) -> None: ...
    def build_bundle(self, identity) -> LegacyOAuthState: ...   # architecture §5.3; transitional return type
```

`get_tokens()` always returns `None` at flow start. Client and metadata may be seeded from active validated state to reuse a working dynamic registration and correct token endpoint. New SDK writes remain in the adapter.

No staged secret is written to disk or Keychain.

`build_bundle` (architecture §5.3) returns the transitional `LegacyOAuthState` shape in this chunk; Chunk 4 gives its token record `original_expires_in`, and Chunk 5 makes it return `OAuthCredentialBundle`. The wall-clock plausibility guard on load is added in Chunk 4 (architecture §4.2, §4.3).

## Administrative lock

Introduce a cross-process per-profile/per-server lock under:

```text
HERMES_HOME/runtime/mcp-oauth-locks/<identity-digest>.admin.lock
```

`<identity-digest>` is the SHA-256 of the canonical identity — `hermes_home_key()`-canonicalized `profile_home` plus normalized server URL and name (architecture §4.1, established in Chunk 1). This is the first use of the digest; Chunk 5 reuses it for credential filenames.

The lifecycle service acquires it before loading seed state and holds it through authorization commit or abort, including the one retry per stage that F-2 adds — an immediate retry for a failed pre-token sub-step, a ~2 s wait before the probe re-check (architecture §6.2, §6.3), a bounded step and never a backoff loop. A competing explicit authorization, migration, or deletion returns `reauthorization_in_progress` after a short bounded wait.

Runtime reads remain available. Runtime refresh concurrency is fully revision-safe in Chunk 4; during this transitional chunk, the lifecycle service rechecks active token state immediately before commit and logs a safe warning if it changed.

## Flow

```text
1.  Acquire administrative lock.
2.  Load active state for seed client/metadata.
3.  Construct staged adapter with no token.
4.  Construct a non-cached OAuth provider using the staged adapter.
5.  Run discovery, registration, browser callback, and token exchange.
    - Transient failure (HTTP 5xx / connection error / timeout / 429) at any sub-step:
      one immediate retry of that sub-step, then abort with authorization_endpoint_unavailable.
    - Definitive failure (HTTP 400 / invalid_grant / invalid_client / unsupported
      registration): abort with a permanent code.
6.  Require a staged access token.
7.  Probe authenticated MCP behavior. Classify the outcome (architecture §6.3):
    - authenticated: proceed to commit.
    - rejected (HTTP 401/403 / invalid_token): retry once after ~2 s; still rejected,
      abort loudly with reauthorization_required.
    - indeterminate (HTTP 5xx / timeout / 429): retry once after ~2 s; still
      indeterminate, proceed to commit with probe=deferred.
8.  Validate staged identity/client/metadata/token coherence.
9.  Commit staged state through the compatibility backend, in the order
    metadata -> client -> token, each an atomic os.replace.
10. Evict the cached runtime provider.
11. Release the lock.
```

The compatibility backend in this chunk still writes separate legacy records. The fixed
metadata -> client -> token order means a concurrent reader that observes the new token also
observes new-or-compatible client and metadata. This chunk guarantees *no destructive failure*
only — a reader mid-commit can still see an incoherent triple, and the one benign residual
interleaving (old token + new client, while the dynamic registration is unchanged) is closed only
in Chunk 5, which replaces all of this with one atomic bundle (architecture §18, Phase 2).

## Failure behavior

The flow *succeeds* once a staged access token is obtained, the probe returns `authenticated` or `indeterminate` (step 7), and coherence validation passes (step 8). The probe outcome is part of commit validation, not failure handling: `authenticated` and `indeterminate` both commit — the latter flagged `probe=deferred` — while `rejected` (after one ~2 s retry) aborts.

The **failures** are: a pre-token step failure (step 5), cancellation, callback timeout, a `rejected` probe, and a coherence-validation failure. Every one of them:

- Discards the staged adapter.
- Leaves active files untouched.
- Releases callback/listener resources.
- Releases the administrative lock.
- Returns a typed lifecycle error (`authorization_endpoint_unavailable`, `authorization_cancelled`, `authorization_timeout`, `reauthorization_required`, or `invalid_staged_bundle`).

The active credential is therefore never modified except by a positive commit — whether that commit is `probe=authenticated` or `probe=deferred`.

The following APIs are removed from authorization paths:

- Pre-flow durable `manager.remove`.
- OAuth state snapshot for reauthorization.
- `restore(..., only_if_absent=True)`.
- Unconditional snapshot restore.

Compatibility snapshot helpers may remain only for unrelated migration code until Chunk 7.

## Provider construction seam

Refactor provider construction so the lifecycle service can supply an explicit storage adapter without inserting the staged provider into the runtime manager cache.

```python
def build_oauth_provider(
    identity,
    server_url,
    oauth_config,
    storage: OAuthStorageAdapter,
    interaction: OAuthInteraction,
) -> OAuthClientProvider: ...
```

The runtime manager calls the same builder with `ActiveOAuthStorageAdapter`.

## Commit validation

Commit requires:

- Non-empty access token.
- Supported token type.
- Valid expiry if supplied.
- Expected issuer/resource identity.
- Coherent client authentication method.
- Valid callback state and redirect URI.
- Authentication probe outcome is `authenticated` or `indeterminate` — not `rejected` (which, after one ~2 s retry, aborts). An `indeterminate` outcome commits with `probe=deferred` recorded on the `mcp_oauth.reauth_committed` event.

A public MCP server that never challenges the request does not count as OAuth-authorized without a staged token.

## Tests

Reuse the Chunk 0 failure matrix and invert the invariant:

- Failure after metadata discovery preserves active state.
- Failure after client registration preserves active state.
- Callback cancellation/timeout preserves active state.
- Token-exchange HTTP 400 `invalid_grant`: abort, active state preserved.
- Token-exchange HTTP 500: one retry, then `authorization_endpoint_unavailable`, active state preserved.
- Probe `rejected` (HTTP 401): one ~2 s retry, then abort, active state preserved.
- Probe `indeterminate` (HTTP 503): one ~2 s retry, then commit — active state replaced with the new token, `AuthorizationResult.probe == "deferred"`.
- A `probe=deferred` credential whose token is in fact invalid: first runtime use routes into 401 recovery without fresh-token-rejection logging (Chunk 2 behavior).
- Successful flow replaces state and evicts provider cache.
- A concurrent reader during the metadata → client → token commit never observes a destructive state; the benign old-token+new-client interleaving is documented, not asserted absent.
- Two CLI/dashboard/RPC reauthorization attempts serialize.
- Explicit deletion cannot enter while reauthorization holds the lock.
- Process termination before commit leaves active credentials usable.

## Demonstration

```text
OLD active bundle
  ├── staged client=PARTIAL
  ├── staged metadata=PARTIAL
  └── injected failure

Result: OLD active bundle unchanged
```

Then:

- Complete the same flow and show the active credentials change to `NEW` only after an `authenticated` probe.
- Run the flow with an `indeterminate` probe (MCP server 503) and show the active credentials change to `NEW` with `probe=deferred`; the first runtime request then succeeds or routes into 401 recovery without a browser prompt.

## Non-goals

- Do not add the final versioned bundle format or the transitional revision envelope (Chunk 4).
- Do not add `original_expires_in` or the wall-clock plausibility guard (Chunk 4).
- Do not add identity-digest *credential* filenames (Chunk 5); the digest is used here only for the lock filename.
- Do not add Keychain.
- Do not solve stale refresh writes except for a pre-commit transitional check; Chunk 4 owns CAS.
- Do not retain rollback as a secondary mechanism.

## Completion criteria

- The Chunk 0 destructive scenarios all preserve active credentials.
- All three interactive surfaces use staged authorization.
- No pre-flow durable delete exists.
- Aborted flows create no durable partial client or metadata record.
- A transient OAuth-endpoint failure yields `authorization_endpoint_unavailable` and preserves the active credential; only a positive commit (`probe=authenticated` or `probe=deferred`) modifies it.
- The compatibility backend commits in metadata → client → token order.
- Issue #76590's failed-reauthorization path is demonstrably fixed.
