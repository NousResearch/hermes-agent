# MCP OAuth Credential Store Architecture Requirements

Status: Draft
Audience: Hermes maintainers and contributors
Scope: MCP OAuth credential persistence, refresh, reauthorization, migration, and removal

## 1. Purpose

Hermes currently persists MCP OAuth tokens, dynamic client registration, and authorization-server metadata as separate files under `HERMES_HOME/mcp-tokens/`. The CLI, dashboard, Desktop/TUI RPC gateway, MCP runtime, and reconnect machinery reach this state through related but non-uniform control paths.

This architecture has two user-visible consequences:

1. A failed reauthorization can delete a previously working token and then fail to restore it when the incomplete flow leaves partial client or metadata files behind.
2. MCP OAuth credentials do not participate in the configurable operating-system credential protection used for other Hermes secrets.

This document specifies a shared OAuth credential-store library that all Hermes surfaces must use. The library must support configurable persistence backends, coherent credential bundles, transactional reauthorization, safe concurrent refresh, and migration from the existing file layout.

## 2. Goals

The architecture shall:

- Provide one authoritative library for all MCP OAuth credential lifecycle operations.
- Allow users or administrators to select where MCP OAuth credentials are stored.
- Treat a server's token, OAuth client registration, authorization-server metadata, and issuer binding as one coherent credential bundle.
- Preserve a working credential bundle until a replacement has been successfully authorized and committed.
- Support safe use by the CLI, dashboard, Desktop, TUI, gateway, cron jobs, and MCP runtime.
- Prevent stale or concurrent writers from overwriting newer credentials.
- Provide secure migration from the legacy `mcp-tokens/` file layout.
- Preserve profile isolation through `HERMES_HOME` and the active Hermes profile.
- Make failure modes explicit, observable, and recoverable without silently discarding credentials.

## 3. Non-goals

This specification does not:

- Define provider-specific authorization scopes or consent parameters.
- Replace the MCP SDK's OAuth protocol implementation.
- Define storage for non-MCP model-provider credentials.
- Require Electron `safeStorage` as a backend. Desktop-only storage is insufficient because the gateway, CLI, and TUI must access the same credentials.
- Permit arbitrary plugins to read raw OAuth credentials. Backend extensibility must preserve the existing credential security boundary.

## 4. Terminology

**Credential identity**
A stable key identifying one MCP OAuth credential set. It includes the canonical Hermes profile identity and MCP server name. It may also include a normalized server or issuer identity when needed to prevent credential reuse across different authorities.

**Credential bundle**
The coherent persisted unit containing the OAuth token set, OAuth client registration, authorization-server metadata, issuer binding, schema version, and storage revision.

**Active bundle**
The credential bundle currently available to MCP runtime clients.

**Reauthorization transaction**
An isolated workspace in which a fresh OAuth flow may write tokens, client registration, and metadata without mutating the active bundle before successful completion.

**Storage backend**
An implementation responsible for durable storage, retrieval, locking, atomic replacement, and deletion of credential bundles.

**Revision**
An opaque value that changes whenever the active bundle changes and supports compare-and-swap updates.

## 5. Current-state problem statement

The current file-backed implementation can follow this sequence:

1. Snapshot the existing token, client, and metadata files.
2. Delete the live files to force a fresh OAuth flow.
3. Write a new client registration or metadata file during the incomplete flow.
4. Fail before obtaining a new token.
5. Skip rollback because some OAuth state now exists.
6. Start subsequent non-interactive sessions without a cached token and require manual authorization.

Unconditionally restoring the snapshot is also unsafe: another CLI or runtime may have successfully refreshed or reauthorized the same credential while the first flow was in progress. An unconditional restore could overwrite that newer credential.

The required invariant is therefore:

> A reauthorization attempt must not mutate or delete the active credential bundle until the replacement flow has completed successfully and the replacement can be committed atomically.

## 6. Architectural requirements

### 6.1 Shared library and ownership

REQ-001: Hermes shall provide one shared MCP OAuth credential-store library as the only supported persistence interface for MCP OAuth state.

REQ-002: The CLI, web dashboard, Desktop/TUI RPC gateway, MCP runtime, token-refresh path, reconnect path, startup discovery, cron execution, server removal, and profile management shall use this library.

REQ-003: User-interface and transport layers shall not directly read, write, rename, or delete backend credential artifacts.

REQ-004: The library shall distinguish these operations explicitly:

- Load the active bundle.
- Refresh or update the active bundle conditionally.
- Begin fresh reauthorization.
- Commit successful reauthorization.
- Abort reauthorization.
- Evict in-memory provider state without deleting durable credentials.
- Explicitly revoke or delete durable credentials.

REQ-005: A transient transport, keepalive, discovery, or MCP tool failure shall never invoke durable credential deletion.

### 6.2 Credential identity and profile isolation

REQ-010: Every stored bundle shall be keyed by a credential identity containing the canonical active profile and MCP server name.

REQ-011: Profile resolution shall use `get_hermes_home()` or an explicit profile-scoped Hermes home. Code shall not hardcode `~/.hermes`.

REQ-012: Credentials from one profile shall not be visible to another profile unless an explicit future sharing feature defines that behavior.

REQ-013: The stored bundle shall bind tokens to the expected OAuth issuer or authorization server when that identity is available.

REQ-014: A change in configured MCP URL, OAuth issuer, or pre-registered client identity shall be detected before a stored bundle is used.

REQ-015: An identity mismatch shall produce a typed result requiring reauthorization; it shall not silently reuse or destroy the mismatched bundle.

### 6.3 Credential bundle

REQ-020: The durable unit shall be a versioned credential bundle rather than independently managed token, client, and metadata records.

REQ-021: A credential bundle shall support, at minimum:

- Schema version.
- Credential identity.
- Access token.
- Refresh token when issued.
- Token type.
- Granted scopes.
- Absolute expiration time when known.
- OAuth client ID and client secret when applicable.
- Registered redirect URIs and client authentication method.
- OAuth protected-resource and authorization-server metadata needed for refresh.
- Bound issuer or authorization-server identity.
- Creation and last-update timestamps.
- Opaque storage revision.

REQ-022: Backends may serialize the bundle differently, but all backends shall preserve equivalent semantics.

REQ-023: A refresh response that omits `refresh_token` shall preserve the existing refresh token as required by OAuth refresh semantics.

REQ-024: Secret values shall never be included in normal logs, exception messages, telemetry, debug bundles, or UI payloads.

### 6.4 Transactional reauthorization

REQ-030: Fresh reauthorization shall run inside a reauthorization transaction created by the credential-store library.

REQ-031: The transaction shall expose storage compatible with the MCP OAuth provider while isolating all writes from the active bundle.

REQ-032: Beginning reauthorization shall not delete, truncate, replace, or invalidate the active bundle.

REQ-033: An incomplete transaction may contain partial client registration or metadata without affecting the active bundle.

REQ-034: Successful reauthorization shall require a valid replacement token before commit.

REQ-035: Commit shall atomically replace the active credential bundle with the staged coherent bundle.

REQ-036: Aborting, timing out, cancelling, or crashing during reauthorization shall leave the active bundle unchanged.

REQ-037: Temporary transaction data shall be owner-only and shall be removed after commit, abort, or bounded stale-transaction cleanup.

REQ-038: Explicit reauthorization attempts for the same credential identity shall be serialized across threads and processes, or a second attempt shall return a typed `reauthorization_in_progress` result.

REQ-039: A successful explicit reauthorization may supersede an access-token refresh based on the previous credential, but a failed reauthorization shall never supersede it.

REQ-040: A successful commit shall evict or invalidate the corresponding in-memory OAuth provider so future requests load the committed bundle.

### 6.5 Refresh and optimistic concurrency

REQ-050: Loading a bundle shall return its opaque revision with the credential data.

REQ-051: Token refresh shall update the bundle with compare-and-swap semantics against the revision that was loaded for the refresh.

REQ-052: If the active revision changed before refresh commit, the store shall reject the stale update rather than overwrite newer credentials.

REQ-053: On a revision conflict, the runtime shall reload the active bundle and determine whether the request can be retried with the newer access token.

REQ-054: Bundle updates and replacement shall be atomic from the perspective of readers. Readers shall observe either the complete old bundle or the complete new bundle.

REQ-055: In-process locks may optimize coordination but shall not be the sole protection when multiple Hermes processes can access the same backend.

### 6.6 Configurable storage backends

REQ-060: The credential-store library shall select a backend through profile-scoped Hermes configuration.

REQ-061: Configuration shall support at least these logical values:

```yaml
mcp:
  oauth:
    credential_store: auto
```

The initial supported values shall include:

- `auto`
- `file`
- `apple-keychain` on macOS

Future backends may include Windows Credential Manager/DPAPI, Linux Secret Service, and an encrypted file store.

REQ-062: `auto` shall have documented, deterministic platform selection behavior.

REQ-063: Hermes shall not silently downgrade from a configured secure operating-system backend to plaintext file storage.

REQ-064: If the configured backend is unavailable or locked, Hermes shall return a typed backend error with recovery guidance.

REQ-065: The file backend shall use owner-only directories and files, atomic replacement, durable flush where supported, and cross-process locking.

REQ-066: The Apple Keychain backend shall be accessible to all authorized Hermes Python processes that need the credentials, not solely the Electron application.

REQ-067: Keychain item identity shall be stable across ordinary Hermes updates and shall not depend on an ad-hoc application signature that changes between launches.

REQ-068: Backend configuration and non-secret metadata may live in `config.yaml`; raw token and client-secret values shall not.

REQ-069: Backend implementations shall pass one backend-independent lifecycle contract test suite.

### 6.7 Proposed library contract

The exact API may evolve, but it shall provide behavior equivalent to:

```python
class OAuthCredentialStore(Protocol):
    def load(self, identity: OAuthIdentity) -> StoredBundle | None: ...

    def compare_and_swap(
        self,
        identity: OAuthIdentity,
        expected_revision: str,
        bundle: OAuthCredentialBundle,
    ) -> StoredBundle: ...

    def begin_reauthorization(
        self,
        identity: OAuthIdentity,
    ) -> ReauthorizationTransaction: ...

    def delete(self, identity: OAuthIdentity) -> None: ...


class ReauthorizationTransaction(Protocol):
    @property
    def storage(self) -> OAuthStorageAdapter: ...

    def commit(self) -> StoredBundle: ...
    def abort(self) -> None: ...
```

REQ-070: Callers shall not need backend-specific branches for normal OAuth lifecycle operations.

REQ-071: Backend exceptions shall be normalized into typed library errors such as:

- `credential_not_found`
- `backend_unavailable`
- `backend_locked`
- `revision_conflict`
- `identity_mismatch`
- `reauthorization_in_progress`
- `invalid_staged_bundle`
- `migration_required`
- `migration_failed`

### 6.8 Migration

REQ-080: Hermes shall detect legacy credential artifacts under the active profile's `mcp-tokens/` directory.

REQ-081: Migration shall combine the legacy token, client, metadata, and applicable issuer-binding state into one validated bundle.

REQ-082: Migration to a new backend shall write and read-verify the destination bundle before modifying legacy files.

REQ-083: After verification, legacy secret-bearing files shall be removed or archived according to an explicit migration policy.

REQ-084: A failed migration shall preserve the legacy credentials and report an actionable error.

REQ-085: Migration shall be idempotent and safe to resume after interruption.

REQ-086: If both legacy and destination credentials exist, Hermes shall use revisions and timestamps only as supporting evidence; it shall not silently overwrite one valid credential set with another. The conflict shall be resolved by a deterministic documented rule or explicit user action.

REQ-087: Migration shall preserve profile isolation and shall not scan or import credentials from unrelated profiles.

REQ-088: Hermes shall provide a diagnostic command that reports backend type, migration state, credential identity, and expiry status without revealing secrets.

### 6.9 Explicit deletion and revocation

REQ-090: Durable deletion shall occur only after an explicit user removal/logout action or a narrowly defined, verified security condition.

REQ-091: Transport failures, timeouts, HTTP 5xx responses, unavailable metadata endpoints, and MCP initialization errors shall not delete credentials.

REQ-092: OAuth `invalid_grant`, revoked refresh tokens, or confirmed client invalidation may mark a bundle unusable, but automatic deletion shall follow an explicitly reviewed policy and preserve diagnostic metadata when safe.

REQ-093: Deletion shall remove the complete bundle and associated staged transactions for that credential identity.

REQ-094: When supported, remote token revocation and local deletion shall be reported as separate outcomes.

### 6.10 Observability and diagnostics

REQ-100: Lifecycle logs shall identify the credential identity, backend type, operation, and non-secret result.

REQ-101: Logs shall distinguish in-memory provider eviction, durable deletion, transaction abort, transaction commit, refresh, migration, and revision conflict.

REQ-102: No log shall describe a transaction as authenticated until the staged bundle contains a validated token and commit succeeds.

REQ-103: Repeated background failures shall be parked without interactive browser authorization and without credential deletion.

REQ-104: Diagnostics shall report whether the active bundle exists and whether it is expired or refreshable without printing token values, client secrets, or complete Keychain payloads.

## 7. Security requirements

SEC-001: All backends shall protect confidentiality and integrity against other local users to the extent supported by the operating system.

SEC-002: File-backend directories shall be mode `0700` and secret-bearing files mode `0600` on POSIX systems.

SEC-003: Temporary files shall be created with restrictive permissions before secret content is written.

SEC-004: The library shall reject path traversal in profile and server identifiers.

SEC-005: Keychain item service and account identifiers shall not contain raw secrets.

SEC-006: Credential-store operations shall remain outside model-accessible file and terminal tooling unless the user explicitly performs an approved administrative action.

SEC-007: Debug export and support bundles shall redact credential-store payloads and temporary transaction files.

SEC-008: Backend selection shall fail closed when a specifically configured secure backend cannot protect or retrieve the credential.

SEC-009: Stored bundles shall validate schema, identity, issuer binding, and expected field types before use.

SEC-010: Corrupt bundles shall be quarantined or reported; they shall not be silently treated as authorization revocation and deleted.

## 8. Reliability requirements

REL-001: A process crash at any point before reauthorization commit shall leave the active bundle usable.

REL-002: A process crash during commit shall leave either the old or new complete bundle, never a partial bundle.

REL-003: Stale transactions shall be garbage-collected after a documented retention period without affecting active credentials.

REL-004: Backend operations shall use bounded waits and cancellation-safe cleanup.

REL-005: A locked or temporarily unavailable Keychain shall not cause Hermes to erase cached credentials or create a plaintext replacement.

REL-006: Runtime refresh and cross-process credential changes shall be detected without requiring a complete gateway restart.

REL-007: The library shall preserve refresh tokens when a provider legally omits a replacement refresh token.

## 9. Integration requirements by surface

### CLI

- `hermes mcp login` and `hermes mcp reauth` shall use a reauthorization transaction.
- Failed or cancelled login shall preserve the active bundle.
- `hermes mcp remove` shall use explicit durable deletion.

### Dashboard and Desktop/TUI

- Both shall call the same shared reauthorization service used by the CLI.
- UI session records may track browser progress but shall not own credential rollback logic.
- Remote callback relay shall deliver authorization results to the transaction without receiving raw stored credentials.

### MCP runtime and gateway

- Startup shall load the active bundle through the selected backend.
- Non-interactive contexts shall never initiate browser authorization automatically.
- Reconnect parking shall evict only in-memory provider state.
- Refresh shall use compare-and-swap persistence.

### Profiles and cron

- Each profile shall resolve its configured backend and credential namespace independently.
- Cron and background jobs shall use refreshable active credentials but shall not trigger interactive reauthorization.

## 10. Acceptance criteria

The architecture is complete when all the following are demonstrated:

1. CLI, dashboard, Desktop/TUI, runtime refresh, reconnect, and removal use the shared library.
2. No production call path directly manipulates MCP OAuth token files or Keychain entries outside backend implementations and migration code.
3. A failed reauthorization after dynamic client registration leaves the active bundle byte-for-byte or semantically unchanged.
4. A failed reauthorization after metadata discovery leaves the active bundle unchanged.
5. A successful reauthorization atomically replaces the complete bundle.
6. A concurrent successful token refresh is not overwritten by a failed reauthorization.
7. A stale refresh cannot overwrite a newer successful reauthorization.
8. Two concurrent explicit reauthorization attempts are serialized or one receives `reauthorization_in_progress`.
9. A transient MCP connection failure never deletes durable credentials.
10. File and Apple Keychain backends pass the same lifecycle contract tests.
11. Legacy file credentials migrate without token exposure or loss.
12. An unavailable configured Keychain produces an actionable error and no plaintext fallback.
13. Profile A cannot load, refresh, migrate, or delete profile B's bundle.
14. Logs and diagnostics contain no token or client-secret values.
15. Existing MCP OAuth integrations continue to authenticate, refresh, and reconnect through the new abstraction.

## 11. Required test strategy

Tests shall execute behavior rather than inspect source text.

### Backend contract tests

Every backend shall run tests for:

- Save and load round trip.
- Atomic replacement.
- Compare-and-swap success and conflict.
- Transaction commit and abort.
- Crash/interruption simulation.
- Explicit deletion.
- Identity isolation.
- Corrupt data handling.
- Permission or locked-backend failures.

### Lifecycle integration tests

Integration tests shall use real shared-library imports and temporary profile state to exercise:

- Initial authorization.
- Refresh with and without a returned refresh token.
- Failed reauthorization after partial staged writes.
- Successful reauthorization promotion.
- Concurrent refresh versus reauthorization.
- Concurrent processes using the file backend.
- Provider eviction and reconstruction from the committed bundle.
- Dashboard, TUI/Desktop RPC, and CLI routing through the same lifecycle service.

### Platform tests

- macOS tests shall exercise real Keychain command/API behavior in an isolated test namespace when CI permits.
- File-backend tests shall validate POSIX permissions and atomicity.
- Unsupported platforms shall return typed availability errors rather than silently selecting an unsafe backend.

## 12. Migration and delivery plan

Implementation should proceed in independently reviewable phases:

1. Define the backend-independent bundle, identity, errors, and store protocol.
2. Wrap the current file behavior in `FileOAuthCredentialStore` and route all readers through it without changing the on-disk format.
3. Introduce the shared lifecycle service and remove surface-specific reauthorization logic.
4. Add staged transactions and atomic bundle promotion to the file backend.
5. Add revisioned refresh and cross-process concurrency tests.
6. Add the Apple Keychain backend and backend selection configuration.
7. Add verified, idempotent legacy migration.
8. Remove obsolete snapshot/remove/restore APIs and direct `mcp-tokens/` manipulation after all call sites migrate.

Each phase shall preserve compatibility with existing credentials or include its required migration in the same release.

## 13. Open design decisions

The implementation proposal must resolve these questions before approval:

1. Whether the file backend stores one bundle file or a manifest plus versioned immutable bundle files.
2. The cross-platform locking primitive and Windows behavior.
3. The stable Apple Keychain service/account naming convention and access-control policy.
4. Whether `auto` defaults to OS secure storage for new installations or preserves the file backend until explicit migration.
5. How backend selection interacts with headless macOS sessions where the login Keychain may be locked.
6. The deterministic conflict policy when both legacy and destination backends contain valid but different bundles.
7. Whether remote revocation is attempted during deletion and how partial revocation failures are represented.
8. How long abandoned staged transactions are retained for diagnostics before secure cleanup.

## 14. Compatibility constraints

- Existing configuration without `mcp.oauth.credential_store` shall continue to start with a documented compatibility default.
- The migration shall not require users to reauthorize valid credentials solely because the storage abstraction changed.
- Existing MCP server names and profile paths shall map deterministically to credential identities.
- The architecture shall remain usable without Desktop or Electron installed.
- The architecture shall support local and remote gateway configurations without transferring stored credential bundles to the Desktop renderer.
