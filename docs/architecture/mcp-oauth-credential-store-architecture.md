# MCP OAuth Credential Store Architecture

Status: Proposed (design-review feedback incorporated 2026-09-03)
Audience: Hermes maintainers and contributors
Related requirements: [`../requirements/mcp-oauth-credential-store-requirements.md`](../requirements/mcp-oauth-credential-store-requirements.md)
Related review: [`../requirements/mcp-oauth-design-review-findings.md`](../requirements/mcp-oauth-design-review-findings.md), [`../requirements/mcp-oauth-design-review-approaches.md`](../requirements/mcp-oauth-design-review-approaches.md)

## 1. Summary

Hermes will replace surface-specific MCP OAuth persistence and rollback logic with one shared credential lifecycle library. The library will own credential identity, bundle validation, backend selection, refresh concurrency, transactional reauthorization, migration, deletion, and non-secret diagnostics.

All Hermes surfaces will use the same service:

```text
CLI ───────────────┐
Dashboard ─────────┤
Desktop/TUI RPC ───┼── MCP OAuth Lifecycle Service ── Credential Store
Gateway/runtime ───┤                                  ├── File backend
Cron/background ───┘                                  └── Apple Keychain backend
```

The central architectural rule is:

> The active credential bundle remains readable and unchanged while a fresh OAuth flow runs. Reauthorization writes to isolated staged state and replaces the active bundle only after a valid token has been obtained.

This removes rollback from the normal reauthorization design. A failed or abandoned flow has nothing to roll back because it never modifies active credentials.

## 2. Architectural decisions

### 2.1 One shared lifecycle service

The shared lifecycle service is the only production entry point for MCP OAuth credential operations. CLI, dashboard, TUI/Desktop RPC, gateway startup, refresh, reconnect, removal, and migration call this service rather than manipulating token artifacts.

The service separates durable credential operations from in-memory MCP provider management:

- Credential store: durable bundles and concurrency.
- Lifecycle service: authorization, refresh, migration, and policy.
- Provider manager: cached in-memory MCP SDK OAuth providers.
- UI adapters: browser progress and callback transport only.

### 2.2 One coherent bundle per credential

Token, dynamic client registration, authorization-server metadata, and issuer binding are committed as one versioned bundle. Independent live token, client, and metadata files are legacy inputs, not the target data model.

### 2.3 In-memory staged reauthorization

Fresh OAuth flows use an in-memory `StagedOAuthStorageAdapter` compatible with the MCP SDK storage interface. It may be seeded with reusable client registration and metadata from the active bundle, but it deliberately exposes no active token to the SDK.

Consequences:

- The SDK is forced through fresh authorization.
- Partial discovery or dynamic registration remains process-local.
- Failed flows cannot damage the active bundle.
- A process crash before commit leaves no persistent partial state.
- Temporary OAuth secrets do not need a staging file.

### 2.4 Optimistic concurrency for refresh

Normal token refresh loads a bundle and its revision, requests new tokens, and commits with compare-and-swap. A stale refresh cannot overwrite a newer reauthorization or refresh.

### 2.5 Exclusive coordination for administrative operations

Explicit reauthorization, deletion, and migration acquire the same per-profile, per-server cross-process administrative lock. Only one of these operations may run at a time for a credential identity.

The lock is held for the full browser reauthorization flow because competing explicit flows would otherwise present ambiguous account and scope choices. Runtime reads continue. Runtime refresh may continue and is protected by revision checks; a successful explicit reauthorization intentionally supersedes a refresh based on the previous credential.

### 2.6 Backend selection is profile-scoped and fail-closed

The configured backend is resolved per Hermes profile. An explicitly selected secure backend never silently falls back to plaintext storage.

The compatibility behavior is:

- Missing setting: use the legacy-compatible file backend until migration is explicitly selected.
- `file`: use the versioned file backend.
- `apple-keychain`: use Apple Keychain or return a typed availability/locked error.
- `auto` on macOS: select Apple Keychain and fail closed if it is unavailable.
- `auto` on platforms without an implemented secure backend: select the file backend and report that selection in diagnostics.

Adding Windows and Linux secure backends can change `auto` for new installations through a versioned configuration migration, not a silent runtime behavior change.

## 3. Component model

The implementation should introduce a focused package:

```text
tools/mcp_oauth_store/
├── __init__.py
├── models.py            # identity, bundle, stored bundle, revisions
├── errors.py            # typed backend/lifecycle errors
├── base.py              # backend protocol
├── factory.py           # profile-scoped backend selection
├── locks.py             # cross-process administrative and CAS locks
├── file_backend.py      # versioned atomic file bundle storage
├── apple_keychain.py    # macOS generic-password backend
├── sdk_adapter.py       # active and staged MCP SDK storage adapters
├── interaction.py       # OAuthInteraction protocol and CLI/dashboard/RPC adapters
├── lifecycle.py         # authorize, refresh, delete, status; outcome classifier
├── migration.py         # legacy mcp-tokens import
└── diagnostics.py       # non-secret status projection
```

Existing `tools/mcp_oauth.py` remains responsible for OAuth protocol helpers during migration, but persistence responsibilities move into this package. `tools/mcp_oauth_manager.py` receives providers from the lifecycle integration and no longer owns durable deletion.

### 3.1 Dependency direction

```text
Hermes surfaces
      │
      ▼
OAuthLifecycleService
      ├──────────────► MCPOAuthManager (memory only)
      │
      ├──────────────► MCP SDK provider
      │                    │
      │                    ▼
      │              OAuthStorageAdapter
      │
      ▼
OAuthCredentialStore protocol
      ├── FileOAuthCredentialStore
      └── AppleKeychainOAuthCredentialStore
```

Backends do not import UI, gateway, CLI, or MCP transport modules. Surface code does not import concrete backends.

## 4. Domain model

### 4.1 OAuth identity

```python
@dataclass(frozen=True)
class OAuthIdentity:
    profile_home: Path
    server_name: str
    server_url: str
```

The domain object carries the profile-scoped Hermes home as a `Path`, the type `get_hermes_home()` returns and every other Hermes caller works with. `profile_home` is not a display label.

`profile_id` is not a field. It is a *derived backend key*: the SHA-256 digest of the canonical identity — `canonicalize(profile_home)` plus the normalized `server_url` and `server_name`. Backends use this digest for filenames, Keychain account strings, and lock filenames, so arbitrary server names cannot become filesystem paths or Keychain account identifiers.

Profile-home canonicalization is a single shared function with the semantics of `hermes_constants.hermes_home_key()` — `os.path.normcase(str(Path(home).expanduser().resolve(strict=False)))`. The OAuth digest and the existing plugin/registry scope key must not drift; `MCPOAuthManager._key()` and `hermes_cli.profiles.profile_matches_home()` are migrated onto the same helper (both currently omit `normcase`). Canonicalization *follows the path*: a profile directory that is deleted and recreated (backup restore, profile reset) resolves to the same digest and keeps its credentials — an inode- or UUID-file-based identity would orphan them, so it is deliberately not used.

Known limitations of path-string identity, both out of scope for the current design:

- A case-insensitive filesystem accessed through spellings that differ only in case (`~/.hermes` vs `~/.Hermes`) produces different digests off Windows (`normcase` lowercases only on Windows). Pathological; matches existing `_key()` behavior.
- Cross-namespace access (a container and the host sharing one profile volume at different mount paths) is not reconcilable by canonicalization. If that topology is ever supported, the fix is an explicit operator-set `mcp.oauth.credential_identity` string in profile config, used verbatim (hashed) when present — not a cleverer auto-derivation.

The serialized bundle also retains the normalized server URL and discovered issuer. These values are validated on load to prevent a token from being used with a different MCP or authorization server.

Normalization rules:

- Profile home: `expanduser()` then `resolve(strict=False)` then `normcase` (the `hermes_home_key()` helper). Resolving requires the directory to exist; it does by the time any OAuth operation runs.
- Server name: preserved for display, normalized only for comparison where existing Hermes server-name rules require it.
- Server URL: lowercase scheme and hostname, default port removed, fragment removed, and trailing slash normalized without changing a meaningful path.
- Issuer: normalized according to OAuth issuer comparison requirements; no substring or suffix matching.

A parametrized contract test asserts that `{tilde, trailing slash, embedded ``..``, symlinked parent, ``/var`` vs ``/private/var``}` spellings of one profile home all yield an identical digest across two independent calls.

### 4.2 Credential bundle

```python
@dataclass(frozen=True)
class OAuthCredentialBundle:
    schema_version: int
    identity: OAuthIdentity
    issuer: str | None
    protected_resource: dict | None
    authorization_server: dict | None
    client: OAuthClientRecord | None
    tokens: OAuthTokenRecord
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class StoredBundle:
    bundle: OAuthCredentialBundle
    revision: str
```

`OAuthTokenRecord` contains access token, optional refresh token, token type, scopes, `accepted_at_utc`, absolute `expires_at`, and `original_expires_in`.

```text
expires_at = accepted_at_utc + expires_in
```

The conversion uses the wall-clock UTC time at which Hermes accepts the token response. Both the absolute `expires_at` and `original_expires_in` — the relative lifetime exactly as the provider returned it, in seconds — are persisted. `original_expires_in` is authoritative for the load-time plausibility guard (§4.3) and is never recomputed on load; a refresh response replaces it with the new grant's value. Persisting `accepted_at_utc` alongside preserves the original lifetime needed to calculate the proportional refresh window after restart.

Persisted expiration is always an absolute UTC timestamp. Monotonic time is used for waits, retry delays, and timeout measurement within a running process, but it is not persisted because it has no meaning after restart.

The revision is a cryptographically random 128-bit value encoded as lowercase hexadecimal. It is generated for every successful mutation and is not derived from token material.

### 4.3 Expiration states and safety window

The lifecycle service classifies a token with a known expiration into three states:

- `valid`: current time is before the refresh-due threshold.
- `refresh_due`: current time is at or after the refresh-due threshold but before absolute expiration.
- `expired`: current time is at or after absolute expiration.

The refresh-due threshold includes a safety window so Hermes does not begin a request with a token likely to expire in flight:

```text
token_lifetime = expires_at - accepted_at_utc
refresh_window = min(60 seconds, token_lifetime × 10%)
refresh_due_at = expires_at - refresh_window
```

The safety window is clamped to zero for invalid or non-positive lifetimes. Implementations may refresh earlier when a provider explicitly requires it, but they shall not treat a token as valid beyond `expires_at`.

A token for which the provider supplies neither `expires_in` nor another trustworthy expiration signal has `unknown` expiration. Hermes may use it until the provider rejects it. A 401 or equivalent authentication rejection then triggers at most one coordinated reload/refresh attempt when a refresh token exists; otherwise the result is `reauthorization_required`.

**Wall-clock plausibility guard.** State is recalculated from `expires_at` on every load (§7.1). Because `expires_at` is wall-clock UTC, a backward clock step between acceptance and use (NTP correction, manual change, VM snapshot restore) would make a token look valid longer than it is. Before classifying from `expires_at`, the lifecycle service checks the persisted anchors:

```text
if original_expires_in is None:            -> unknown
if now < accepted_at_utc:                  -> unknown        # clock behind acceptance
elapsed = now - accepted_at_utc
if elapsed < 0 or elapsed > original_expires_in * CLOCK_SLACK:  -> unknown
otherwise classify normally from expires_at
```

`CLOCK_SLACK` is `2.0` — wide enough to absorb legitimate NTP drift and refresh-token re-anchoring, narrow enough to trip on a gross step. The guard only *demotes* to `unknown` (an already-safe state: use until rejection, then bounded recovery per §7.1); it never shortens `expires_at`, so it cannot cause a forced-refresh storm. Residual risk: a small backward step (seconds to minutes, within the slack band) still passes and can make a token look valid slightly longer than it is; the §7.1 rejection-recovery path is the backstop.

### 4.4 Refresh-token merge rule

When a refresh response omits `refresh_token`, the lifecycle service copies the refresh token from the loaded bundle. An explicitly returned refresh token, including a rotated token, replaces the previous value.

The merge occurs before bundle validation and compare-and-swap.

## 5. Public interfaces

### 5.1 Store protocol

```python
class OAuthCredentialStore(Protocol):
    backend_name: str

    def load(self, identity: OAuthIdentity) -> StoredBundle | None: ...

    def create(
        self,
        identity: OAuthIdentity,
        bundle: OAuthCredentialBundle,
    ) -> StoredBundle: ...

    def compare_and_swap(
        self,
        identity: OAuthIdentity,
        expected_revision: str,
        bundle: OAuthCredentialBundle,
    ) -> StoredBundle: ...

    def replace_authorized(
        self,
        identity: OAuthIdentity,
        bundle: OAuthCredentialBundle,
    ) -> StoredBundle: ...

    def delete(self, identity: OAuthIdentity) -> bool: ...

    def administrative_lock(
        self,
        identity: OAuthIdentity,
        *,
        timeout: float,
    ) -> ContextManager[None]: ...
```

`replace_authorized` may only be called while holding the identity's administrative lock. It exists for successful explicit authorization, which intentionally replaces the old active credential even if a runtime refresh changed its revision during the browser flow.

`compare_and_swap` is used for refresh and other non-administrative mutations.

### 5.2 Lifecycle service

```python
class OAuthLifecycleService:
    def load_for_runtime(
        self,
        identity: OAuthIdentity,
        expected_issuer: str | None = None,
    ) -> StoredBundle | None: ...

    async def authorize(
        self,
        identity: OAuthIdentity,
        oauth_config: Mapping[str, object],
        interaction: OAuthInteraction,
    ) -> AuthorizationResult: ...

    async def persist_refresh(
        self,
        identity: OAuthIdentity,
        loaded: StoredBundle,
        response: OAuthToken,
    ) -> StoredBundle: ...

    def delete(
        self,
        identity: OAuthIdentity,
        *,
        revoke_remote: bool = False,
    ) -> DeletionResult: ...

    def status(self, identity: OAuthIdentity) -> OAuthCredentialStatus: ...
```

`OAuthInteraction` abstracts how an authorization URL is delivered and how its callback is received. CLI loopback, dashboard callback, and remote Desktop callback relay implement this interface. They do not own storage or rollback.

`StoredBundle` is the final loaded-credential handle. Chunks 2–4 use a transitional `StoredState` in these signatures (wrapping legacy state, then gaining a revision); Chunk 5 replaces it with `StoredBundle`.

### 5.3 MCP SDK storage adapters

Two adapters implement the asynchronous storage interface expected by the MCP SDK:

`ActiveOAuthStorageAdapter`

- Reads the active bundle through the selected store.
- Is used for ordinary MCP runtime initialization and refresh.
- Routes refreshed token persistence through `persist_refresh`.
- Never directly deletes credentials.

`StagedOAuthStorageAdapter`

- Exists only for one authorization transaction.
- Holds token, client registration, and metadata in memory.
- Starts without tokens.
- May start with validated client and metadata copied from the active bundle.
- Tracks which values were produced or changed by the new flow.
- Exposes `build_bundle()` only after a token has been obtained.

## 6. Reauthorization flow

### 6.1 Sequence

```text
Surface          Lifecycle           Store/lock          Staged adapter       IdP/MCP
   │                 │                    │                     │                 │
   │ authorize()     │                    │                     │                 │
   ├────────────────►│ acquire admin lock │                     │                 │
   │                 ├───────────────────►│                     │                 │
   │                 │ load active bundle│                     │                 │
   │                 ├───────────────────►│                     │                 │
   │                 │ create staged adapter, no tokens         │                 │
   │                 ├─────────────────────────────────────────►│                 │
   │                 │ run MCP OAuth provider                   │ discover/auth   │
   │                 ├─────────────────────────────────────────►├────────────────►│
   │                 │                    │                     │ token/client/meta│
   │                 │ validate staged coherent bundle          │◄────────────────┤
   │                 │ replace_authorized()                     │                 │
   │                 ├───────────────────►│                     │                 │
   │                 │ evict cached provider                    │                 │
   │                 │ release lock       │                     │                 │
   │ success         │                    │                     │                 │
   │◄────────────────┤                    │                     │                 │
```

### 6.2 Failure behavior

The authorization flow crosses five network boundaries, only the last of which holds a staged token: (1) protected-resource / authorization-server discovery, (2) dynamic client registration, (3) browser authorization, (4) token exchange, (5) the post-exchange authentication probe against the MCP server. Failures are classified on two axes — *position* (pre-token vs post-token) and *kind* (definitive rejection vs indeterminate) — rather than treated as one undifferentiated "probe failed".

**Pre-token failures (steps 1–4).** No staged token exists, so there is nothing to commit; the flow always aborts. The abort *kind* determines the surfaced error:

- Definitive (HTTP 400, `invalid_grant`, `invalid_client`, unsupported registration): abort with a permanent typed error — `reauthorization_required` or a configuration error the surface can act on.
- Indeterminate (HTTP 5xx, connection reset, DNS failure, timeout at a discovery/registration/token endpoint; HTTP 429): one immediate retry of the failed sub-step, then abort with the transient `authorization_endpoint_unavailable` error ("retry shortly", not "authentication failed"). The step-3 authorization code is single-use and short-lived, so a browser redo after the retry is protocol-mandated, not policy. A `429` `Retry-After` interval is surfaced verbatim; the flow does not sit on the administrative lock waiting it out.

**Post-token failures (step 5).** A staged token from a completed code exchange is in hand. See §6.3.

On any abort:

1. The staged adapter is discarded.
2. The administrative lock is released.
3. The active bundle remains unchanged.
4. The cached provider remains valid unless the failure independently proved it unusable.
5. The surface receives a typed, humanizable error carrying the kind (`rejected` vs `endpoint_unavailable`).

There is no snapshot restore operation.

### 6.3 Successful commit validation

Before `replace_authorized`, the lifecycle service verifies the staged bundle:

- A non-empty access token exists (a provider that exposes public tools without ever challenging the request is not authorized unless the staged adapter received a token).
- Token type is supported.
- The staged issuer matches discovered/configured expectations.
- The client record is coherent with configured pre-registration or dynamic registration.
- Redirect URI and client authentication method are present when required.
- Absolute expiry is valid when supplied.

**Authentication probe outcome.** The post-exchange probe against the MCP server is classified, not treated as pass/fail:

- `authenticated` (probe succeeded and a staged token is present): commit.
- `rejected` (HTTP 401/403, or an `invalid_token` body): retry once after ~2 s first — a fresh token being refused is often authorization-server-to-resource-server replication lag or minor clock skew, which a short wait clears. Still rejected: abort loudly with `reauthorization_required` and a "check clock sync / token audience" hint. Another browser round will not fix a genuinely refused fresh token.
- `indeterminate` (HTTP 5xx, timeout, or 429 from the MCP server): retry once after ~2 s, then **commit the staged bundle** and record `probe=deferred` on the `mcp_oauth.reauth_committed` event. The MCP server being unreachable is not evidence against a token from a verified code exchange; the runtime's 401-recovery path (§7.1) discovers a genuinely bad token on first real use.

A single ~2 s retry is the entire budget for both `rejected` and `indeterminate`, so the administrative lock is never held across a backoff loop.

## 7. Refresh flow

### 7.1 Expiration evaluation

Before an authenticated resource request, the lifecycle service evaluates the loaded token state:

- `valid`: send the request with the current access token.
- `refresh_due` or `expired`, with a refresh token: coordinate a refresh before sending the request.
- `expired`, without a refresh token: return `reauthorization_required` without launching a browser in a background context.
- `unknown`: send the request and use bounded rejection recovery if the provider returns an authentication failure.

A response rejecting an apparently valid or unknown-lifetime token causes one coordinated credential reload. If another process has already committed a different valid access token, Hermes retries once with that token. Otherwise, if the bundle is refreshable, Hermes performs one refresh attempt. The original request is not placed in an unbounded authentication retry loop.

The refresh path classifies a failed token-endpoint response through the same `classify_outcome` helper the authorization path uses (§6.2): `invalid_grant` is definitive and maps to `reauthorization_required`; HTTP 5xx / timeout is indeterminate and returns a retryable failure that preserves stored state.

A bundle committed with `probe=deferred` (§6.3) reaches the runtime unverified by design. Its first authentication rejection is expected, not anomalous: the runtime routes it straight into the reload/refresh recovery above and surfaces `reauthorization_required` if that fails, without the elevated logging reserved for a fresh-token rejection during commit.

Wall-clock movement may change expiration classification after a restart or between requests. Hermes recalculates state from `expires_at` on every bundle load and before refresh-sensitive operations, applying the plausibility guard in §4.3. In-process timers use monotonic time and are advisory only; the persisted absolute expiration remains authoritative.

### 7.2 Sequence

```text
Runtime             Lifecycle                 Store                 Token endpoint
   │ load              │                        │                         │
   ├──────────────────►├───────────────────────►│                         │
   │ bundle + revision │◄───────────────────────┤                         │
   │ refresh request   │────────────────────────────────────────────────►│
   │ token response    │◄────────────────────────────────────────────────┤
   │                   │ merge refresh token   │                         │
   │                   │ compare_and_swap(rev) │                         │
   │                   ├───────────────────────►│                         │
```

If compare-and-swap reports `revision_conflict`, the lifecycle service reloads the active bundle:

- If it contains a different valid access token, the runtime retries with it.
- If it is still expired but refreshable, one bounded retry may refresh from the newer revision.
- If identity or issuer changed, the runtime returns `identity_mismatch` or `reauthorization_required` without deleting credentials.

No refresh failure deletes the bundle automatically. Confirmed `invalid_grant` marks the runtime result as requiring reauthorization; explicit deletion remains a separate operation.

## 8. Concurrency model

### 8.1 Lock classes

The architecture uses two lock scopes:

1. **Administrative lock:** held across explicit reauthorization, migration, or deletion for one credential identity.
2. **Mutation lock:** held briefly by a backend while reading a revision and atomically writing/replacing/deleting a bundle.

Both locks are cross-process. In-process locks may reduce contention but do not replace them.

### 8.2 Lock location

Lock files contain no secrets and live under:

```text
HERMES_HOME/runtime/mcp-oauth-locks/<identity-digest>.admin.lock
HERMES_HOME/runtime/mcp-oauth-locks/<identity-digest>.mutation.lock
```

POSIX uses `fcntl.flock`. Windows uses `portalocker`, which Hermes already supports for cross-process MCP locking. Lock acquisition has a bounded timeout and maps contention to `reauthorization_in_progress` or `backend_locked` as appropriate.

The same locks coordinate file and Keychain backends because they are scoped to the Hermes profile, not the backend artifact.

### 8.3 Operation matrix

| Operation | Administrative lock | Mutation lock | Revision check |
|---|---:|---:|---:|
| Load | No | Backend-dependent brief read | No |
| Revision probe | No | Backend-dependent brief read | No |
| Token refresh commit | No | Yes | Yes |
| Initial authorization | Yes | Commit only | Create-if-absent |
| Explicit reauthorization | Yes | Commit only | No; explicit replacement |
| Migration | Yes | Destination write/delete | Conflict policy |
| Explicit deletion | Yes | Yes | No |
| In-memory eviction | No | No | No |

**Revision-probe cadence.** The provider manager (§12.1) checks whether the stored revision changed only at provider-rebuild decision points — before an authorization flow, during 401 recovery, and on explicit refresh or status — never on the per-request read path. Each cached provider entry additionally holds the last-observed revision under a short in-memory TTL (default 10 s) with exponential backoff and last-known-good fallback on probe failure, so an auth-sensitive request never forces a backend round trip (a `security` subprocess spawn on the Keychain backend). If profiling ever shows the subprocess probe itself is a hot-path cost, the escalation is Security.framework bindings for load/probe, which does not change the store protocol.

### 8.4 Crash guarantees

- Crash before staged commit: active bundle unchanged.
- Crash during file commit: atomic replace yields old or new complete bundle.
- Crash during Keychain replacement: Keychain operation yields old or new item; read-back verification detects uncertain outcomes.
- Crash after commit before provider eviction: disk-change/revision detection rebuilds the provider on the next request.

The revision travels *inside* the same serialized envelope as the bundle it identifies (§9.2), committed by the one atomic write. There is no separate revision manifest, so no crash window can leave the revision and the state it describes mismatched. The transitional Chunk 4 backend, which carries a revision alongside not-yet-unified legacy records, achieves the same property by embedding the revision in the token record's own JSON and writing it with that record's atomic replace — see `../design/mcp-oauth-04-expiration-refresh-concurrency.md`.

## 9. File backend

### 9.1 Layout

The final file backend stores one bundle per identity:

```text
HERMES_HOME/mcp-credentials/
├── v1/
│   └── <identity-digest>.json
└── index.v1.json
```

The bundle file is the source of truth. `index.v1.json` contains only non-secret diagnostic mapping — identity digest, profile display identifier, server display name, schema version, and a backend-visible status timestamp. The digest, server name, and schema version are rebuildable from bundle headers; the profile identifier and timestamp are index-only conveniences. The index is not required for loading a known identity.

Directory mode is `0700`; bundle files are `0600` on POSIX.

### 9.2 Serialization

The JSON envelope contains:

```json
{
  "schema_version": 1,
  "revision": "opaque-random-revision",
  "bundle": {
    "identity": { "server_name": "...", "server_url": "..." },
    "issuer": "https://issuer.example",
    "protected_resource": {},
    "authorization_server": {},
    "client": {},
    "tokens": {},
    "created_at": "2026-09-01T12:00:00Z",
    "updated_at": "2026-09-01T12:00:00Z"
  }
}
```

The serialized `identity` block holds only `server_name` and `server_url` — not `profile_home`. Profile scope is the file's location (`HERMES_HOME/mcp-credentials/`) and its digest-derived name; the digest is recomputed from the requesting identity on load and a mismatch finds no file (`credential_not_found`). The in-memory `OAuthCredentialBundle.identity` is a full `OAuthIdentity`, reconstructed on load by pairing the serialized `server_name`/`server_url` with the requesting `profile_home`.

`tokens` serializes `OAuthTokenRecord` in full, including `accepted_at_utc`, `expires_at`, and `original_expires_in` (§4.2).

Unknown schema versions fail with `migration_required`; corrupt JSON fails with a typed corruption error and is not deleted automatically. A `server_url` that disagrees with the runtime expectation fails with `identity_mismatch`; a stored `issuer` that disagrees fails with `issuer_mismatch`.

### 9.3 Atomic write

Under the mutation lock, the backend:

1. Reads and validates the current revision when compare-and-swap is requested.
2. Creates a random same-directory temporary file with `O_EXCL` and mode `0600`.
3. Writes the full envelope — the new revision and the bundle it identifies, together — flushes it, and calls `fsync`.
4. Atomically replaces the destination with `os.replace`.
5. `fsync`s the parent directory on POSIX where supported.
6. Releases the mutation lock.

Readers observe a complete old or new file. Because the revision and its bundle share the one envelope and the one `os.replace`, no reader and no crash can pair a revision with state it does not describe.

## 10. Apple Keychain backend

### 10.1 Item identity

Each credential bundle is stored as a macOS generic-password item:

- Service: `com.nousresearch.hermes.mcp-oauth.v1`
- Account: `<identity-digest>`
- Label: `Hermes MCP OAuth — <server display name>`
- Password data: UTF-8 JSON serialization of the complete versioned envelope

Service and account naming are stable across Hermes updates and do not depend on Electron bundle signing.

### 10.2 Access mechanism

The Python backend invokes the macOS `security` tool with argument arrays and bounded subprocess timeouts. It does not execute through a shell. This keeps the backend available to CLI, TUI, gateway, cron, and Desktop-launched Python processes without coupling it to Electron `safeStorage`.

If later reliability testing favors direct Security.framework bindings, the concrete implementation may change without changing the store protocol or item naming.

### 10.3 Operations

- Load uses `security find-generic-password` for the exact service/account.
- Create/update uses the Keychain generic-password replacement operation while holding the mutation lock.
- Delete targets the exact service/account.
- Every write is followed by read-back validation of identity, revision, and schema.

Keychain denial, lock, timeout, missing command, malformed payload, and duplicate-item ambiguity map to typed backend errors. The backend never falls back to a file after selection.

### 10.4 User prompts and headless operation

Keychain access policy should permit credentials created by Hermes command-line processes to be read by subsequent Hermes Python processes owned by the same logged-in user. Architecture tests must verify behavior for:

- Terminal CLI.
- LaunchAgent/gateway process.
- Desktop-launched local gateway.
- Locked login Keychain.
- Headless session without an unlocked user Keychain.

If access would trigger an interactive Keychain prompt in a background context, the operation returns `backend_locked`/`interaction_required`; it does not hang indefinitely or create a plaintext copy.

## 11. Backend factory and configuration

Configuration is read from the active profile:

```yaml
mcp:
  oauth:
    credential_store: apple-keychain
    backend_timeout_seconds: 10
    reauthorization_lock_timeout_seconds: 1
```

Only `credential_store` is required initially. Timeouts may use internal defaults until a demonstrated need warrants public configuration; they must not be introduced as new environment variables.

`OAuthCredentialStoreFactory`:

1. Resolves the active profile with `get_hermes_home()`.
2. Reads the profile-scoped setting.
3. Applies compatibility/default selection.
4. Constructs one backend instance per profile/backend combination.
5. Performs a non-destructive availability check.
6. Returns typed errors without fallback when the selected backend is unavailable.

Backend instances may be cached in process, but no cached instance may capture the wrong profile after a profile override.

## 12. Lifecycle integration

### 12.1 MCP provider manager

`MCPOAuthManager` becomes an in-memory cache only:

- `get_or_build_provider` uses an `ActiveOAuthStorageAdapter` supplied by the lifecycle service.
- `evict` removes a cached provider and never touches durable storage.
- Durable `remove` moves to `OAuthLifecycleService.delete`.
- Disk mtime watching becomes backend-neutral revision watching.

The manager's cache key remains profile plus server identity, canonicalized through the shared `hermes_home_key()` helper (§4.1) so it and the credential digest cannot disagree about which profile a request belongs to.

Each cached entry remembers the last-observed storage revision. **Invariant: the revision is probed only at rebuild decision points** — before an authorization flow, during 401 recovery, and on explicit refresh or status — never on the runtime read path that serves a resource request. The remembered revision is held under a short TTL (default 10 s) with backoff and last-known-good fallback (§8.3), so between decision points a process may serve requests from a slightly stale bundle; compare-and-swap still rejects any stale *write*, and a stale *read* costs at most one late refresh, recovered by 401 handling.

### 12.2 CLI

`hermes mcp login` and `hermes mcp reauth` call `OAuthLifecycleService.authorize`. They no longer call `manager.remove`, snapshot files, or implement rollback.

`hermes mcp remove` calls `OAuthLifecycleService.delete`, then removes MCP configuration according to existing command semantics.

### 12.3 Dashboard and Desktop/TUI RPC

Dashboard and RPC OAuth session modules retain:

- Flow/session ID.
- Authorization URL publication.
- Callback delivery and state validation.
- Progress polling and humanized errors.

They delegate credential operations to `OAuthLifecycleService.authorize`. Duplicate storage transaction and rollback code is removed from both paths.

Remote Desktop callback relay sends only callback parameters and session identity to the gateway. It never receives the active or staged credential bundle.

### 12.4 Runtime reconnect

Transport loss, keepalive failure, exhausted retries, and parked-server recovery call only `MCPOAuthManager.evict`. On revival, the manager rebuilds from the active bundle.

Confirmed authentication failures return a typed `reauthorization_required` state. Background contexts do not start a browser and do not delete the bundle.

## 13. Migration architecture

### 13.1 Legacy inputs

Migration recognizes, per server:

```text
HERMES_HOME/mcp-tokens/<server>.json
HERMES_HOME/mcp-tokens/<server>.client.json
HERMES_HOME/mcp-tokens/<server>.meta.json
HERMES_HOME/mcp-tokens/<server>.cimd-off
```

The CIMD refusal marker is operational metadata. It may be represented in the bundle or a separate non-secret policy record, but migration must preserve its behavior.

### 13.2 Trigger

- Missing configuration uses the legacy-compatible file adapter and does not force migration.
- Selecting `file` migrates legacy records into the versioned file bundle.
- Selecting `apple-keychain` or `auto` on macOS migrates legacy records into Keychain.
- An explicit diagnostic/migration command may run migration before normal startup.

### 13.3 Algorithm

Under the administrative lock:

1. Load all legacy artifacts for the configured server.
2. Validate token, client, metadata, server URL, and issuer relationships.
3. Construct a versioned bundle, reconstructing the token time model (§4.2) from the legacy anchors — the file mtime stands in for `accepted_at_utc`, and `original_expires_in` is derived from `expires_at - mtime` or the raw legacy `expires_in`, or left `None`. The `CLOCK_SLACK` cushion (§4.3) absorbs the imprecision.
4. Check the destination.
5. If destination is empty, write the bundle.
6. Read it back and verify all non-secret fields plus secret-value equality in constant-process memory.
7. Delete the legacy secret-bearing artifacts.
8. Record non-secret migration completion.

Migration is idempotent. If interrupted before verified destination write, legacy files remain. If interrupted after destination verification but before cleanup, the next run detects equivalent credentials, verifies again, and completes cleanup.

### 13.4 Conflicts

If valid destination and legacy bundles differ, automatic migration stops with `migration_conflict`. Hermes continues using the configured destination backend and reports the conflict through diagnostics. A user must explicitly choose:

- Keep destination and securely delete legacy.
- Replace destination with legacy.
- Reauthorize and replace both.

Timestamps alone never select a winner.

## 14. Error model

All store and lifecycle errors derive from `MCPOAuthCredentialError` and include a stable code, safe message, backend name, and credential display identity. They do not include serialized responses or secrets.

```text
credential_not_found
credential_ambiguous
backend_unavailable
backend_locked
backend_timeout
interaction_required
credential_corrupt
unsupported_schema
revision_conflict
identity_mismatch
issuer_mismatch
reauthorization_in_progress
reauthorization_required
authorization_cancelled
authorization_timeout
authorization_endpoint_unavailable
invalid_staged_bundle
migration_required
migration_conflict
migration_failed
deletion_failed
```

- `credential_ambiguous` — a backend found more than one stored item matching the identity (e.g. two Keychain generic-password items with the same service and account). The payload is not corrupt, so `credential_corrupt` would misdescribe it; the store must refuse the operation rather than act on an arbitrary match, and diagnostics name the exact remediation.
- `authorization_endpoint_unavailable` — a pre-token step (discovery, registration, token exchange) failed on a transient transport condition (HTTP 5xx / 429 / timeout) that persisted through one retry (§6.2). Distinct from `authorization_timeout`, which is the user not completing the browser step. Surfaces present it as "retry shortly", carrying any `Retry-After` interval.
- `backend_locked` vs `interaction_required` — `backend_locked` is a locked login Keychain the user can unlock once; `interaction_required` is a specific item whose access-control policy would raise an interactive prompt in the current (often background) context. Both fail closed without hanging or writing plaintext; the remediation differs (unlock the Keychain vs re-grant the item or run interactively).

Surfaces translate these errors into presentation appropriate to interactive or background operation. They do not infer deletion, retry, or fallback policy from raw exception text.

## 15. Diagnostics and logging

Structured lifecycle events include:

- `mcp_oauth.bundle_loaded`
- `mcp_oauth.refresh_committed`
- `mcp_oauth.refresh_conflict`
- `mcp_oauth.reauth_started`
- `mcp_oauth.reauth_committed`
- `mcp_oauth.reauth_aborted`
- `mcp_oauth.provider_evicted`
- `mcp_oauth.credential_deleted`
- `mcp_oauth.migration_completed`
- `mcp_oauth.backend_error`

Permitted fields include profile display name, server name, backend, revision prefix, expiry status, error code, and duration. Token values, authorization codes, client secrets, full revisions, and raw provider response bodies are prohibited.

Event-specific fields:

- `mcp_oauth.reauth_committed` carries `probe` — `authenticated` when the post-exchange probe confirmed the token, `deferred` when it was committed past an indeterminate probe outcome (§6.3).
- `mcp_oauth.reauth_aborted` carries `reason` — `rejected` (definitive auth rejection), `endpoint_unavailable` (transient pre-token failure), `cancelled`, or `timeout`.

**Revision-prefix bound.** Any logged or diagnostic revision prefix is at most 8 hexadecimal characters (32 bits). The revision itself is 128 bits and is used only for compare-and-swap; a bounded prefix is enough to correlate two log lines by eye while leaving the value unguessable. A test asserts no emitted field — log or `OAuthCredentialStatus.revision_prefix` — exceeds the bound.

A diagnostic command should expose output equivalent to:

```text
Server: todoist
Profile: default
Backend: apple-keychain
Credential: present
Expiration: refresh_due
Expires: 2026-09-01T18:30:00Z
Refreshable: yes
Issuer binding: valid
Last probe: authenticated
Migration: complete
Reauthorization: idle
```

`Last probe` is `deferred` for a credential committed past an indeterminate probe (§6.3) that no live request has since confirmed. The full projection type is `OAuthCredentialStatus` — see `../design/mcp-oauth-07-migration-diagnostics-cleanup.md`.

## 16. Security boundaries

- Only backend modules handle serialized secret bundles.
- Model-visible file tools remain blocked from credential storage and lock directories.
- Desktop renderer and plugins never receive token bundles.
- Keychain subprocess arguments never contain secret payloads when the tool permits stdin; secret input is passed through a pipe. If a required `security` operation cannot avoid secrets in process arguments, the implementation must use Security.framework instead.
- Temporary buffers are scoped to the operation and not retained in global diagnostics.
- Bundle parsing validates maximum payload size before JSON decoding.
- Identity digests prevent path traversal and unsafe Keychain identifiers.
- Debug bundles redact both legacy and new storage paths.

## 17. Test architecture

### 17.1 Backend contract suite

One parameterized contract suite runs against every backend implementation:

- Create/load round trip.
- Create conflict.
- Compare-and-swap success.
- Compare-and-swap stale revision rejection.
- Authorized replacement under administrative lock.
- Atomic reader behavior during replacement.
- Delete and missing delete.
- Identity and profile isolation.
- Identity-digest stability: `{tilde, trailing slash, embedded ``..``, symlinked parent, ``/var`` vs ``/private/var``}` spellings of one profile home resolve to one digest.
- Corrupt and unsupported bundle behavior.
- Backend locked/unavailable behavior.
- No emitted revision prefix exceeds 8 hex characters.

The Apple Keychain contract suite uses a unique service suffix in platform CI and removes only items created by that test run. It additionally asserts duplicate-item ambiguity: with two generic-password items matching one service/account, `load`, `compare_and_swap`, `replace_authorized`, and `delete` each return `credential_ambiguous` and never operate on an arbitrary match.

### 17.2 Lifecycle tests

Lifecycle tests use a real store implementation and a fake OAuth protocol peer, not source inspection:

- Failed flow after client registration leaves active bundle unchanged.
- Failed flow after metadata discovery leaves active bundle unchanged.
- Cancellation and timeout leave active bundle unchanged.
- Successful flow commits a complete bundle and evicts provider cache.
- Public MCP initialization without a token is not reported as authenticated.
- Refresh without a returned refresh token preserves the previous refresh token.
- Relative `expires_in` is converted to the expected absolute UTC expiration.
- `original_expires_in` is persisted and is not recomputed on load.
- Tokens become `refresh_due` at the specified safety-window threshold.
- Short-lived tokens use ten percent of their lifetime rather than a fixed 60-second window.
- Expired tokens refresh before a resource request when refreshable.
- Expired non-refreshable tokens return `reauthorization_required` without browser interaction.
- Unknown-lifetime tokens remain usable until provider rejection.
- An authentication rejection performs at most one coordinated reload/refresh retry.
- Wall-clock changes do not affect monotonic operation timeouts or create unbounded retries.
- A load where `now` precedes `accepted_at_utc`, or where elapsed exceeds `original_expires_in × CLOCK_SLACK`, classifies as `unknown`; normal NTP-scale drift within the slack band does not.
- Pre-token token-exchange HTTP 500 retries once, then aborts with `authorization_endpoint_unavailable`, active bundle untouched.
- Pre-token token-exchange HTTP 400 `invalid_grant` aborts with a permanent error, no retry storm.
- A 429 at the token endpoint surfaces its `Retry-After` and does not hold the administrative lock waiting.
- Probe HTTP 401 retries once after the short delay, then aborts loudly without committing.
- Probe HTTP 503 retries once, then commits the staged bundle with `probe=deferred`.
- A `probe=deferred` bundle whose token is in fact invalid surfaces `reauthorization_required` on first runtime use via the 401 path, without fresh-token-rejection logging.
- Stale refresh loses to newer reauthorization.
- Refresh during failed reauthorization survives.
- Two explicit reauthorization processes serialize.
- Explicit deletion cannot race with reauthorization.
- Parked reconnect evicts memory only.
- The revision is probed only at rebuild decision points, never on the per-request read path; a stale in-TTL bundle still serves reads and CAS still rejects a stale write.

### 17.3 Surface-routing tests

CLI, dashboard, and TUI/Desktop RPC tests inject the lifecycle service and assert behavioral results. They do not mock or assert direct file removal because surfaces no longer own storage.

### 17.4 Migration tests

- Each valid legacy artifact combination.
- Missing optional client or metadata.
- Invalid/corrupt token.
- Destination write failure.
- Read-back verification failure.
- Crash-resume checkpoints.
- Equivalent destination plus legacy.
- Conflicting destination plus legacy.
- Profile isolation.

## 18. Rollout plan

### Phase 1: abstraction without format change

- Add models, errors, protocol, factory, and lifecycle service.
- Implement a legacy-compatible file backend over the existing files.
- Route all read, refresh, deletion, and status operations through the service.
- Change `MCPOAuthManager` to memory-only eviction semantics.

Exit criterion: no surface directly manipulates `mcp-tokens/`.

### Phase 2: transactional authorization

- Add staged SDK adapter.
- Route CLI, dashboard, and TUI/Desktop RPC authorization through one service.
- Remove snapshot/remove/restore reauthorization logic.
- Add administrative locks and cross-surface concurrency tests.
- Commit order in the compatibility backend is fixed as metadata → client → token, each an atomic `os.replace`, so a concurrent reader that observes the new token also observes new-or-compatible client and metadata.

Exit criterion: failed authorization cannot modify active credentials. This phase guarantees *no destructive failure* only. Because the compatibility backend still writes the token, client, and metadata as separate files, a concurrent reader can still observe an incoherent triple during a commit; the one remaining interleaving (reader between the client write and the token write sees the old token with the new client, benign while the dynamic registration is unchanged) is closed only in Phase 3, when the versioned single bundle lands.

### Phase 3: versioned file bundle and CAS

- Implement versioned single-bundle file backend.
- Add legacy-to-bundle migration.
- Add revision watching and refresh compare-and-swap.

Exit criterion: readers observe coherent bundles and stale refreshes cannot overwrite newer credentials.

### Phase 4: Apple Keychain

- Implement and platform-test Keychain backend.
- Add configuration and diagnostics.
- Add verified migration into Keychain.

Exit criterion: CLI, TUI, gateway, cron, and Desktop-local gateway share Keychain-backed MCP credentials.

### Phase 5: cleanup

- Remove obsolete `HermesTokenStorage` file persistence methods.
- Remove manager durable-delete APIs.
- Remove legacy rollback tests and replace them with lifecycle invariants.
- Update user and security documentation.

## 19. Compatibility and rollback

During rollout, the legacy-compatible backend remains available so a release rollback can still read existing credentials. Once a credential is migrated to a versioned bundle or Keychain, migration completion metadata records the source and destination backend.

Application rollback must not silently export Keychain credentials back to plaintext. A user wishing to downgrade must run an explicit credential export/migration command that explains the security consequence, or reauthorize on the older version.

The architecture introduces no requirement for Electron. Local CLI and gateway operation remain fully supported.

## 20. Requirements traceability

| Requirement area | Architecture sections |
|---|---|
| Shared library and ownership | 3, 5, 12 |
| Identity and profile isolation | 4.1, 11, 13 |
| Coherent credential bundle | 4.2, 9.2 |
| Transactional reauthorization | 2.3, 5.3, 6 |
| Refresh concurrency | 2.4, 4.3, 7, 8 |
| Configurable backends | 2.6, 9, 10, 11 |
| Migration | 13, 18 |
| Explicit deletion | 5.2, 8.3, 12 |
| Observability | 14, 15 |
| Security | 9, 10, 16 |
| Reliability and testing | 8.4, 17 |
| Clock robustness | 4.2, 4.3, 7.1 |
| Probe/commit policy | 6.2, 6.3, 7.1 |
| Runtime hot-path cost | 8.3, 12.1 |

## 21. Consequences

### Positive

- Failed OAuth flows cannot erase active credentials.
- Every Hermes surface follows one lifecycle and error model.
- File and Keychain storage become implementation choices rather than UI-specific behavior.
- Refresh and reauthorization have explicit concurrency semantics.
- Credential state becomes coherent and easier to validate, migrate, and diagnose.
- A transient outage at an authorization or MCP endpoint no longer forces a user to repeat a browser flow whose token is fine.
- The runtime read path performs no per-request backend probe, so the Keychain backend adds no subprocess spawn to an authenticated request.
- A gross wall-clock step is detected and demoted to a safe state rather than silently extending a token's apparent life.

### Costs

- The MCP SDK storage integration must be adapted rather than used as a direct file writer.
- Cross-process locking and Keychain integration require platform tests.
- Migration must support both legacy files and new bundles across multiple releases.
- A full browser reauthorization holds an administrative lock for that server, so a second explicit attempt is rejected or waits.
- The probe-outcome classifier and the wall-clock guard are new logic on the authorization and load paths, each with its own test matrix.
- A bundle committed with `probe=deferred` reaches the runtime unverified; correctness then depends on the 401-recovery path, not on commit-time proof.

These costs are accepted because they replace duplicated destructive control paths with one testable credential lifecycle boundary.
