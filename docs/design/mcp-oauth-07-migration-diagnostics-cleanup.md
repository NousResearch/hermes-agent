# MCP OAuth Chunk 7 — Migration, Diagnostics, and Cleanup

Status: design proposal (not yet implemented; design-review updates applied 2026-09-03)
Depends on: file bundle and Apple Keychain backends (Chunks 5–6)
Finalizes: shared OAuth credential-store rollout
Architecture: [`../architecture/mcp-oauth-credential-store-architecture.md`](../architecture/mcp-oauth-credential-store-architecture.md) §4.1, §4.3, §6.3, §13, §14, §15, §18 (Phase 5), §19
Design-review updates: [`../requirements/mcp-oauth-design-review-approaches.md`](../requirements/mcp-oauth-design-review-approaches.md) (F-7 revision-prefix bound, F-6 repair command, F-2 probe state, F-5 migration)

## Purpose

Safely move valid legacy credentials into the configured backend, expose non-secret operational diagnostics, and remove obsolete persistence and rollback paths after every production caller uses the shared architecture.

## Migration sources

Per configured MCP server and active profile:

```text
HERMES_HOME/mcp-tokens/<server>.json
HERMES_HOME/mcp-tokens/<server>.client.json
HERMES_HOME/mcp-tokens/<server>.meta.json
HERMES_HOME/mcp-tokens/<server>.cimd-off
```

Migration never scans unrelated profiles. Server enumeration comes from profile configuration plus exact known legacy mappings.

## Migration commands

Proposed CLI:

```bash
hermes mcp credentials status [server]
hermes mcp credentials migrate --to file [server]
hermes mcp credentials migrate --to apple-keychain [server]
hermes mcp credentials resolve-conflict <server> --keep destination|legacy|reauthorize
hermes mcp credentials repair <server> --keep <item-id>
```

Names may align with existing CLI conventions, but operations remain CLI/lifecycle functions rather than new model-visible tools.

### Duplicate-item repair

The Keychain backend returns `credential_ambiguous` when more than one generic-password item
matches one service/account (Chunk 6); its error names the manual `security delete-generic-password`
remediation. `hermes mcp credentials repair` is the guided form:

- Enumerate the matching items — for each, show the label, modification date, and 8-hex revision
  prefix. No secret values.
- With `--keep <item-id>`, remove every other matching item under the credential's administrative
  lock; without it, print the list and exit.
- Never auto-select — this mirrors `resolve-conflict --keep` and the "do not auto-select between
  conflicting valid credentials" non-goal.
- File backend: not applicable (one path per identity); the command reports nothing to repair.

## Migration algorithm

Under the credential's administrative lock:

```text
1. Read all legacy artifacts without modifying them.
2. Validate token, client, metadata, MCP URL, and issuer relationships.
3. Reconstruct the token time model from the legacy file, using its mtime as the
   acceptance proxy: expires_at from documented compatibility rules, and
   original_expires_in ~= round((expires_at - mtime).total_seconds()) — or the raw
   legacy expires_in when there is no expires_at, or None when neither is present.
4. Construct the versioned bundle per the Chunk 5 schema: no profile_id in the
   identity block (server_name + server_url only); original_expires_in in tokens.
5. Load destination.
6. Resolve empty/equivalent/conflicting destination state.
7. Write destination (the backend's atomic write; destination digest is
   hermes_home_key()-canonicalized, architecture §4.1).
8. Read back and validate complete equality.
9. Record non-secret migration completion.
10. Securely remove legacy secret-bearing artifacts.
```

The `original_expires_in` reconstruction is best-effort; the `CLOCK_SLACK = 2.0` cushion in the
load-time plausibility guard (architecture §4.3) absorbs its imprecision, and a healthy migrated
token is not demoted to `unknown`.

Migration completion is idempotent. A crash before verified destination write leaves legacy files untouched. A crash after verification but before cleanup is resumed by recognizing an equivalent destination bundle.

## Conflict policy

Destination and legacy states are:

- `destination_empty`: migrate automatically after validation.
- `equivalent`: verify destination and finish legacy cleanup.
- `conflicting`: stop with `migration_conflict`.
- `legacy_invalid`: preserve artifacts, report corruption, require reauthorization or explicit cleanup.
- `destination_invalid`: fail closed; do not overwrite automatically.

Timestamps never choose a winner. Explicit resolution options:

- Keep destination, then remove legacy.
- Replace destination with validated legacy.
- Reauthorize into destination, then remove legacy.

## Cleanup semantics

After destination read-back verification, remove exact legacy files. Because portable secure deletion cannot be guaranteed on modern filesystems, documentation must say "remove" rather than promise physical overwriting.

No broad glob or recursive delete is used. Cleanup validates the exact profile token directory and sanitized server artifact names.

## Diagnostic projection

```python
@dataclass(frozen=True)
class OAuthCredentialStatus:
    server_name: str
    profile_display: str
    backend: str
    present: bool
    expiration_state: str
    expires_at: datetime | None
    refreshable: bool
    issuer_binding: str
    revision_prefix: str | None      # at most 8 hex characters
    last_probe: str | None           # "authenticated" | "deferred"
    migration_state: str
    reauthorization_state: str
    error_code: str | None           # includes authorization_endpoint_unavailable, credential_ambiguous
```

`revision_prefix` is at most an 8-hex-character prefix of the 128-bit revision — enough to
correlate two lines, not enough to brute-force the space. A test asserts that no emitted field —
this projection or any log line — carries more than 8 hex characters of a revision.

`last_probe` is `deferred` when the credential was committed past an indeterminate authentication
probe (F-2, architecture §6.3) and no live request has since confirmed it; `authenticated`
otherwise; `None` before any authorization.

Diagnostics never include access token, refresh token, client secret, authorization code, more
than an 8-hex revision prefix, or raw provider response.

Example:

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

Ambiguous credential:

```text
Server: todoist
Backend: apple-keychain
Credential: ambiguous — 2 Keychain items match; run `hermes mcp credentials repair todoist`
Error: credential_ambiguous
```

## Automatic startup behavior

- Missing backend setting: continue legacy-compatible behavior; report migration availability.
- Configured destination empty with valid legacy state: run verified migration according to configuration policy.
- Conflict: use configured valid destination, report conflict, do not delete either state.
- Locked secure backend: fail closed without using legacy plaintext as an implicit fallback.
- Background contexts never prompt for migration conflict resolution.

## Production cleanup

After call-site audit and migration coverage, remove:

- Durable path logic from `HermesTokenStorage`.
- Snapshot/restore APIs used for reauthorization.
- Manager methods that combine eviction and durable deletion.
- Surface-specific OAuth persistence and rollback.
- Token-file mtime semantic watching.
- Direct legacy artifact reads outside migration/compatibility modules.
- Documentation claiming MCP OAuth tokens always live in `mcp-tokens/`.

Token-file mtime *semantic* watching is removed — the provider cache invalidates on revision change (Chunk 5, probed at decision points under a 10 s TTL). A backend may still use mtime or inode as a cheap probe hint, which is not what is being removed here.

Keep a time-bounded legacy reader only if the supported upgrade window requires it. Mark it with an explicit removal release/migration version rather than leaving permanent dual behavior.

## Downgrade behavior

Hermes never silently exports Keychain credentials to plaintext for an older release. Downgrade options are:

- Explicitly migrate to the file backend with a security warning.
- Reauthorize using the older release.

Migration metadata remains non-secret and allows newer Hermes to diagnose prior backend selection after reinstall or rollback.

## Tests

- Every valid combination of legacy token/client/metadata.
- Legacy refresh response without refresh token.
- Migration reconstructs `original_expires_in` from `expires_at` − mtime, from a raw legacy `expires_in`, and yields `None` when neither is present; a healthy migrated token is not demoted to `unknown` by the wall-clock guard.
- Corrupt legacy JSON; identity mismatch fails closed (recomputed digest ≠ destination filename/account).
- Destination write/read-back failure preserves legacy.
- Crash/resume before and after destination verification.
- Equivalent destination completes cleanup.
- Conflicting destination requires explicit choice.
- Exact deletion targets only selected server/profile.
- Locked Keychain does not fall back to legacy file at runtime.
- `credential_ambiguous` surfaces in `status`; `repair --keep` removes only the non-kept items under the admin lock and never auto-chooses.
- A `probe=deferred` credential shows `last_probe: deferred` in `status`.
- No diagnostic field or log line carries more than an 8-hex revision prefix.
- Diagnostics redact all secrets.
- Repository behavioral tests prove no non-migration production path reads legacy files.

## Demonstration

```text
legacy token files
    │
    ├── migrate to Keychain
    ├── read-back verify
    ├── remove exact legacy files
    └── restart CLI/TUI/gateway without reauthorization
```

Also demonstrate a conflicting destination remains untouched and produces actionable status without exposing credential values.

## Non-goals

- Do not auto-select between conflicting valid credentials, or between duplicate Keychain items (`repair` always requires `--keep`).
- Do not promise forensic secure erasure.
- Do not add model-visible credential tools.
- Do not retain permanent dual-write compatibility.
- Do not emit more than an 8-hex revision prefix anywhere.

## Completion criteria

- Migration is verified, idempotent, profile-safe, and resumable, and carries `original_expires_in`.
- Diagnostics are useful without exposing secrets; the revision prefix is bounded to 8 hex and tested.
- `hermes mcp credentials repair` resolves duplicate Keychain items without auto-selecting.
- A probe-deferred credential is visible in `status`.
- Keychain-to-plaintext downgrade is explicit.
- Direct legacy storage manipulation is confined to migration/temporary compatibility code.
- All obsolete rollback and ambiguous durable-delete APIs are removed.
