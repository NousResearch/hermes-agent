# MCP OAuth Chunk 6 — Apple Keychain Backend

Status: design proposal (not yet implemented; design-review updates applied 2026-09-03)
Depends on: versioned bundle store protocol (Chunk 5)
Platform: macOS
Architecture: [`../architecture/mcp-oauth-credential-store-architecture.md`](../architecture/mcp-oauth-credential-store-architecture.md) §2.6, §4.1, §5.1, §8.3, §8.4, §10, §11, §14, §17.1
Design-review updates: [`../requirements/mcp-oauth-design-review-approaches.md`](../requirements/mcp-oauth-design-review-approaches.md) (F-6 duplicate-item ambiguity, F-1 probe cost, F-0 identity)

## Purpose

Provide a secure MCP OAuth storage backend available to every local Hermes Python surface: CLI, TUI, gateway, cron, Desktop-launched gateway, and background runtime. It must not depend on Electron `safeStorage`.

## Keychain item model

Store each complete versioned credential envelope as a generic-password item:

```text
Service: com.nousresearch.hermes.mcp-oauth.v1
Account: <identity-digest>
Label:   Hermes MCP OAuth — <server display name>
Secret:  UTF-8 JSON versioned bundle envelope
```

`<identity-digest>` is the SHA-256 of the canonical identity — `hermes_home_key()`-canonicalized `profile_home` plus normalized server URL and name (architecture §4.1) — the same digest Chunk 5 uses for the bundle filename and Chunk 3 for lock filenames.

The stable service/account identity does not depend on Hermes application signing or install path. Profile isolation is the account name: on load the backend recomputes the digest from the requesting identity and matches it against the account, the same way the file backend matches it against the filename. The envelope itself carries no profile identifier (Chunk 5 removed `profile_id`); its `server_name` / `server_url` are validated against the runtime expectation for server binding.

## Backend interface

`AppleKeychainOAuthCredentialStore` implements the same contract suite as the file backend:

```python
load
create
compare_and_swap
replace_authorized
delete
administrative_lock
```

Cross-process administrative and mutation locks remain profile-scoped files under `HERMES_HOME/runtime/mcp-oauth-locks/`. They contain no secrets and coordinate Keychain callers consistently with the file backend.

## Access implementation

Initial implementation may invoke `/usr/bin/security` with argument arrays, no shell, bounded timeouts, and a minimal environment.

Secret-handling rule:

- Secret JSON must not appear in process arguments, logs, or exception text.
- If `security` cannot perform a required non-interactive write with secret input through stdin, implement that operation with Security.framework bindings instead of passing `-w <secret>`.

Probe cost: the stored revision is checked only at rebuild decision points (before a flow, during 401 recovery, on explicit refresh or status), and between those the provider entry serves from the Chunk 5 in-memory revision cache (10 s TTL, backoff, last-known-good). A resource request therefore never spawns `security`. If profiling shows the `security find-generic-password` spawn is itself a hot-path cost at those decision points, the escalation is Security.framework bindings for `load` and the revision probe — the same CLI→framework path already used for the secret-write case.

Duplicate-item detection (see Operations) needs an "all matches" query, which the `security` CLI does not offer; `SecItemCopyMatching` with `kSecMatchLimitAll` is the expected mechanism.

The backend adapter hides these choices from lifecycle callers.

## Operations

Every operation first resolves how many items match the exact service/account:

- Zero: `credential_not_found` (for load/delete) or proceed (for create).
- Exactly one: proceed.
- More than one: **`credential_ambiguous`** — refuse; never operate on an arbitrary match. The error message names the exact remediation (`security delete-generic-password -s com.nousresearch.hermes.mcp-oauth.v1 -a <account>` for each stale item). A `hermes mcp credentials repair` command is deferred to Chunk 7.

### Load

- Resolve match count (above).
- Query the single item; bound output size.
- Parse and validate the envelope.
- Verify schema and revision; verify the recomputed identity digest equals the account name (a stored envelope whose identity disagrees with its account is `credential_corrupt`).
- Map a missing item to `credential_not_found`.

### Create and replacement

- Hold mutation lock.
- Resolve match count; `credential_ambiguous` if more than one existing item.
- Verify create/revision precondition.
- Replace the generic-password payload (one atomic Keychain operation; the revision travels inside the envelope, no separate revision item).
- Read back the exact item.
- Verify the account digest, revision, and complete bundle equality.
- Report uncertain write outcomes rather than assuming success.

### Delete

- Hold administrative and mutation locks.
- Resolve match count; `credential_ambiguous` if more than one item (do not delete an arbitrary one).
- Delete the single exact service/account item.
- Report local deletion independently from optional remote revocation.

## Configuration

```yaml
mcp:
  oauth:
    credential_store: apple-keychain
```

`auto` on macOS selects the same backend. An unavailable, locked, denied, or interaction-required Keychain returns a typed error. There is no plaintext fallback after selection.

Missing configuration remains on compatibility file behavior until explicit migration policy changes it.

## Availability probe

The factory performs a non-destructive probe that distinguishes:

- Supported and accessible.
- Login Keychain locked.
- User interaction required.
- `security`/framework unavailable.
- Permission denied.
- Operation timeout.

The probe must not create a persistent test credential during ordinary startup. Platform integration tests use isolated test items.

## Headless and background behavior

Background callers use bounded subprocess/framework operations. If Keychain access would prompt or hang:

- Return `backend_locked` or `interaction_required`.
- Preserve existing Keychain item.
- Do not start browser authorization.
- Do not create file credentials.

User guidance directs the operator to unlock the login Keychain or select/migrate to an explicitly chosen backend.

## Desktop boundary

Desktop renderer and plugins never receive bundle data. The local Python gateway accesses Keychain directly. Desktop RPC carries only authorization progress and callback parameters.

Remote gateways use the backend configured on the remote host; Desktop's local Keychain is not copied to the remote machine.

## Tests

### Backend contract

Run in a unique service namespace or account prefix:

- Create/load/replace/delete.
- CAS conflict across processes.
- Identity isolation: a recomputed digest that does not match the account fails closed; `{tilde, trailing slash, embedded ``..``, symlinked parent, ``/var`` vs ``/private/var``}` spellings of one profile home resolve to one account.
- Duplicate-item ambiguity: with two generic-password items matching one service/account, `load`, `compare_and_swap`, `replace_authorized`, and `delete` each return `credential_ambiguous` and never touch an arbitrary match; the error names the `security delete-generic-password` remediation.
- Read-back verification failure.
- Locked/denied/timeout mapping.
- No file fallback.

Cleanup deletes only exact test items created by the run.

### Cross-surface integration

- Authenticate via CLI, consume via TUI.
- Authenticate via Desktop, consume via gateway.
- Refresh via cron, reload in a live runtime session.
- Restart Hermes and load without reauthorization.
- Locked Keychain produces actionable failure without token loss.

### Secret exposure

- Captured subprocess arguments contain no bundle or token.
- Logs and errors contain no secrets; any revision value in a log line or error is truncated to at most 8 hex characters (bound established in Chunk 4).
- Debug export excludes Keychain payloads.
- Renderer-facing RPC payloads contain no secrets.

## Demonstration

Configure `apple-keychain`, authorize a fake/local MCP through CLI, restart the gateway, and call the MCP from TUI. Verify:

- Keychain exact item exists.
- No `mcp-tokens/<server>.json` or plaintext bundle exists.
- Refresh persists to Keychain.
- A second process observes the new revision.

## Non-goals

- Do not use Electron `safeStorage`.
- Do not add Windows or Linux secure backends here.
- Do not silently migrate without verified destination write.
- Do not make Keychain contents model-visible.
- Do not add a `hermes mcp credentials repair` command for duplicate items here; the `credential_ambiguous` error names the manual `security` remediation, and the repair command is Chunk 7.
- Do not auto-delete a duplicate item.

## Completion criteria

- All local Hermes Python surfaces share Keychain credentials.
- Backend contract tests pass on macOS.
- Secure-backend failures are typed and fail closed, including `credential_ambiguous` for duplicate items.
- Identity is validated by recomputing the digest against the account name; a mismatch fails closed.
- The revision probe never spawns `security` on the per-request path.
- No secret is exposed through argv, logs, UI, or fallback files.
- Ordinary Hermes updates do not change Keychain item identity.
