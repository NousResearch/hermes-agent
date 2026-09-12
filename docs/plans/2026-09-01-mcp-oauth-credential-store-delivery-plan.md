# MCP OAuth Credential Store Delivery Plan

Status: Proposed (reconciled with design-review updates 2026-09-03)
Date: 2026-09-01
Related architecture: [`../architecture/mcp-oauth-credential-store-architecture.md`](../architecture/mcp-oauth-credential-store-architecture.md)
Related requirements: [`../requirements/mcp-oauth-credential-store-requirements.md`](../requirements/mcp-oauth-credential-store-requirements.md)
Related design review: [`../requirements/mcp-oauth-design-review-approaches.md`](../requirements/mcp-oauth-design-review-approaches.md) (findings F-0..F-8)

## 1. Delivery strategy

Deliver the architecture as eight independently demonstrable pull requests. Each chunk must improve or preserve production behavior without depending on an unmerged later chunk to remain functional.

| Chunk | Deliverable | Demonstration |
|---|---|---|
| 0 | Rebase and reproduce | Automated reproduction of failed reauthorization deleting tokens on current `main` |
| 1 | Shared library facade | All current file operations route through one store interface, with no format change |
| 2 | Unified lifecycle service | CLI, dashboard, and Desktop/TUI invoke the same authorization service |
| 3 | Transactional reauthorization | Failed OAuth flows cannot modify active credentials |
| 4 | Safe refresh concurrency | Restart- and clock-robust expiration, proactive refresh, CAS revisions, and bounded 401 recovery |
| 5 | Coherent file bundles | One atomic versioned bundle replaces separate live files |
| 6 | Apple Keychain backend | CLI, TUI, gateway, cron, and Desktop share Keychain-backed credentials |
| 7 | Migration and cleanup | Verified legacy migration, diagnostics, and removal of obsolete paths |

## 2. Chunk 0: Rebase and establish the behavioral baseline

Start from current `NousResearch/main`, not the existing month-old proposed-fix branch.

### Deliverables

- A deterministic reproduction using a fake OAuth/MCP provider.
- Tests proving the current defects:

  - Failed reauthorization writes `client.json`, obtains no token, and loses the old token.
  - Dashboard and TUI paths exhibit the same failure.
  - CLI reauthorization deletes credentials without rollback.
  - Transient reconnect must not delete credentials.

This pull request may be tests-only or accompany Chunk 1 if maintainers prefer not to merge expected-failure tests.

### Demonstration

```text
Before fix:
old token present → failed reauthorization → old token missing
```

### Merge gate

- The reproduction executes against current `main` through real lifecycle imports.
- The test does not inspect source text.
- Failure evidence identifies the exact destructive path for each surface.

## 3. Chunk 1: Introduce the shared credential-store facade

Wrap the existing file layout without changing behavior or storage format.

### Deliverables

- `OAuthIdentity` (carries `profile_home: Path`; the identity digest is derived, not a field).
- Legacy-compatible credential model.
- Typed errors.
- `OAuthCredentialStore` protocol.
- `LegacyFileOAuthCredentialStore`.
- Profile-scoped backend factory.
- Centralize profile-home canonicalization on the shared `hermes_home_key()` helper (also migrate `MCPOAuthManager._key()` and `profiles.profile_matches_home()` onto it).
- Backend contract tests.
- Route all token, client, and metadata reads and writes through the facade.

Keep these legacy files operational:

```text
mcp-tokens/server.json
mcp-tokens/server.client.json
mcp-tokens/server.meta.json
```

### Demonstration

- Existing authenticated MCP integrations still work.
- CLI and gateway load the same credential through the library.
- Profile A cannot see Profile B's token.
- No user migration occurs.

### Merge gate

- No behavioral regression.
- No direct persistence calls outside the backend and temporary compatibility shims.
- The backend contract suite passes against the legacy-compatible implementation.

## 4. Chunk 2: Introduce the unified lifecycle service

Centralize authorization policy before changing transaction behavior.

### Deliverables

- `OAuthLifecycleService`.
- `OAuthInteraction` abstraction.
- Loopback CLI interaction.
- Dashboard interaction adapter.
- Desktop/TUI callback-relay adapter.
- Shared status and deletion operations.
- Failure classifier and stable typed lifecycle codes, including `authorization_endpoint_unavailable` for a transient endpoint failure; `AuthorizationResult` gains its final shape (a `probe` outcome field).
- Convert `MCPOAuthManager` to memory-only eviction semantics.

At this stage, the service may internally preserve the existing persistence flow. The accomplishment is eliminating three independently evolving implementations.

### Demonstration

```text
CLI login ──────┐
Dashboard auth ─┼── same authorize() implementation
TUI auth ───────┘
```

### Merge gate

- Each surface produces the same typed outcome for success, cancellation, browser timeout, a transient authorization-endpoint failure, and a missing token.
- Reconnect invokes memory eviction only.
- Explicit removal invokes durable deletion.
- UI session records own browser progress, not credential rollback.

## 5. Chunk 3: Add transactional staged reauthorization

This chunk provides the user-visible fix for the current failed-reauthorization bug.

### Deliverables

- `StagedOAuthStorageAdapter`.
- Fresh authorization begins without exposing the active token to the SDK.
- Client registration and metadata remain in memory during the flow.
- Successful flow validates and commits staged state.
- Failed, cancelled, or timed-out flow discards staged state; a `rejected` probe aborts after one retry.
- An indeterminate MCP probe commits the staged bundle rather than discarding it, recorded as probe-deferred.
- A transient pre-token endpoint failure is retried once, then returns `authorization_endpoint_unavailable` without touching the active credential.
- The compatibility backend commits legacy records in a fixed order (metadata, then client, then token).
- Per-credential administrative lock shared by CLI, dashboard, and TUI.

### Demonstration

```text
Active token: OLD
    │
    ├── failed reauthorization with partial client registration
    │
    └── active token remains OLD

Active token: OLD
    │
    ├── successful reauthorization obtains NEW
    │
    └── active token becomes NEW

Active token: OLD
    │
    ├── new token obtained; MCP probe times out
    │
    └── active token becomes NEW (probe-deferred)
```

### Merge gate

- Active credentials change only on a positive commit (probe-confirmed or probe-deferred); every genuine failure leaves them byte-for-byte unchanged.
- Two explicit reauthorization attempts serialize or return `reauthorization_in_progress`.
- Public MCP initialization without a token is not reported as authenticated.
- Existing rollback code is removed, not retained as a fallback.

This is the target chunk for closing issue #76590.

## 6. Chunk 4: Add expiration policy and refresh concurrency

### Deliverables

- Persist `accepted_at_utc`, `expires_at`, and `original_expires_in` (the last never recomputed on load).
- Implement expiration states:

  - `valid`
  - `refresh_due`
  - `expired`
  - `unknown`

- Implement the refresh window:

```text
refresh_window = min(60 seconds, token_lifetime × 10%)
```

- Apply a wall-clock plausibility guard before classification: an implausible elapsed time demotes to `unknown` (demote-only, never shortens `expires_at`).
- Add bundle revisions, carried inside the token record (unified into one bundle in Chunk 5).
- Commit refresh using compare-and-swap.
- Preserve omitted refresh tokens; a refresh with a new `expires_in` re-anchors `original_expires_in`.
- Bound 401 recovery to one reload/refresh retry.

### Demonstration

- Advance a fake clock to `refresh_due_at`; the next request refreshes before being sent.
- Run two concurrent refreshers; only one revision wins, and the loser reloads it.
- Complete reauthorization while an old refresh is in flight; the stale refresh cannot overwrite the new credential.
- An unknown-lifetime token works until a simulated 401, then receives one coordinated recovery attempt.

### Merge gate

- No unbounded authentication loops.
- No refresh failure deletes credentials.
- Background execution returns `reauthorization_required` without opening a browser.
- Short-lived tokens use ten percent of their lifetime rather than a fixed 60-second window.
- Refresh responses that omit `refresh_token` retain the current refresh token.
- A gross backward wall-clock step demotes classification to `unknown` rather than extending apparent validity.

## 7. Chunk 5: Introduce the coherent versioned file backend

Change the storage format only after lifecycle behavior is stable.

### Deliverables

```text
mcp-credentials/v1/<identity-digest>.json
```

- One bundle containing token, client, metadata, issuer, expiry, and revision (no stored `profile_id`; identity is the digest-named file's location).
- Atomic same-directory temporary write and `os.replace`; the revision lives inside the envelope, never a separate manifest.
- Mutation locks.
- Directory mode `0700` and file mode `0600` on POSIX.
- Parent-directory `fsync` where supported.
- Backend-neutral revision watching replaces token-file mtime watching; the revision is consulted only at rebuild decision points, held under a short in-memory TTL, never probed per request.

Migration may initially be lazy: read legacy state, construct a bundle in memory, and write the new format only after verification.

### Demonstration

- A reader loop running during refresh observes either the complete old bundle or complete new bundle, never mixed state.
- Kill a writer before replacement; the old bundle remains usable.
- Kill it after replacement; the new bundle remains usable.

### Merge gate

- Legacy credentials remain readable.
- No reauthorization is required solely because of the format change.
- Cross-process file tests run with real processes.
- Corrupt or unknown-version bundles are reported and not deleted automatically.

## 8. Chunk 6: Add Apple Keychain

### Deliverables

- `AppleKeychainOAuthCredentialStore`.
- Stable service and account naming; the account is the identity digest, validated against the requesting identity on load.
- Exact-item load, replace, and delete.
- Read-back verification.
- Typed handling for locked, unavailable, denied, and timed-out Keychain operations, and for duplicate items matching one identity (`credential_ambiguous`, fail closed — never act on an arbitrary match).
- Profile-scoped configuration:

```yaml
mcp:
  oauth:
    credential_store: apple-keychain
```

### Demonstration matrix

- Authenticate from CLI, then use the MCP from TUI.
- Authenticate from Desktop, then use it from the gateway.
- Refresh from cron, then use the refreshed credential in an existing interactive session.
- Lock Keychain and show an actionable error with no plaintext fallback.
- Inspect `mcp-tokens/` and confirm no token file was written.

### Merge gate

- Real macOS integration tests use an isolated Keychain namespace.
- Secrets do not appear in subprocess arguments, logs, or debug bundles.
- The backend works without Electron installed.
- A locked or unavailable configured Keychain never creates a plaintext replacement.
- A revision-safe request never spawns `security` on the per-request path.
- Two items matching one identity fail closed rather than one being chosen.

## 9. Chunk 7: Migration, diagnostics, and cleanup

### Deliverables

- Idempotent migration from legacy files to a versioned file bundle or Keychain, carrying the full token time model (including `original_expires_in`).
- Destination write and read-back verification before legacy deletion.
- Conflict handling when destination and legacy credentials differ.
- A safe diagnostic command, for example:

```bash
hermes mcp credentials status todoist
```

- An explicit backend migration command, for example:

```bash
hermes mcp credentials migrate --to apple-keychain
```

- A `hermes mcp credentials repair <server> --keep <item-id>` command to resolve duplicate Keychain items (never auto-selects).

- Remove obsolete:

  - `HermesTokenStorage.snapshot()`.
  - `HermesTokenStorage.restore()`.
  - Manager disk-deleting `remove()` APIs.
  - Direct `mcp-tokens/` manipulation.
  - Surface-specific rollback logic.

### Demonstration

```text
Legacy files present
→ migrate
→ Keychain bundle verified
→ legacy secret files removed
→ CLI/TUI/gateway continue without reauthorization
```

### Merge gate

- Interrupted migration resumes safely.
- Conflicting credentials, and duplicate Keychain items, are never resolved automatically.
- Downgrade does not silently export Keychain credentials to plaintext.
- Repository search confirms only migration code reads the legacy paths.
- Diagnostics reveal no token, authorization code, client-secret value, or more than an 8-hex revision prefix.

## 10. Proposed pull-request sequence

1. `test(mcp): reproduce cross-surface OAuth credential loss`
2. `refactor(mcp): introduce shared OAuth credential store`
3. `refactor(mcp): unify OAuth lifecycle across CLI dashboard and TUI`
4. `fix(mcp): stage reauthorization before credential replacement`
5. `fix(mcp): make token expiry and refresh revision-safe`
6. `feat(mcp): add atomic versioned OAuth bundle backend`
7. `feat(mcp): add Apple Keychain OAuth credential backend`
8. `feat(mcp): migrate legacy OAuth credentials and remove old storage`

Each pull request must demonstrate a behavioral invariant. Abstraction, Keychain integration, migration, and rollback fixes should not be combined into one large change because that would be difficult to review, bisect, and safely revert.

## 11. Cross-chunk rules

- Use `get_hermes_home()` for profile-scoped paths, canonicalized through `hermes_home_key()` where an identity or digest is derived.
- Use `scripts/run_tests.sh`; do not invoke pytest directly.
- Preserve existing credentials or include verified migration in the same chunk.
- Do not add model-visible tools for credential management.
- Do not add user-facing non-secret configuration through environment variables.
- Tests must execute behavior and must not inspect source text.
- Durable credential deletion remains explicit and separate from in-memory provider eviction.
- A chunk is not complete until its demonstration works through real imports against an isolated temporary `HERMES_HOME`.
