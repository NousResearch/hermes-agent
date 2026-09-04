# MCP OAuth Chunk 5 — Versioned Atomic File Bundles

Status: design proposal (not yet implemented; design-review updates applied 2026-09-03)
Depends on: revision-safe refresh (Chunk 4)
Supersedes: independent live token/client/metadata files for the `file` backend
Architecture: [`../architecture/mcp-oauth-credential-store-architecture.md`](../architecture/mcp-oauth-credential-store-architecture.md) §4.1, §8.3, §8.4, §9, §12.1, §18
Design-review updates: [`../requirements/mcp-oauth-design-review-approaches.md`](../requirements/mcp-oauth-design-review-approaches.md) (F-1 revision TTL, F-0 identity, F-5 field; F-4, F-7, F-8)

## Purpose

Replace independently written legacy OAuth artifacts with one versioned, atomically replaced credential bundle. Readers must observe a coherent old or new state, never a token paired with unrelated client or metadata.

## Final file layout

```text
HERMES_HOME/mcp-credentials/
├── v1/
│   └── <identity-digest>.json
└── index.v1.json
```

The bundle file is authoritative. The optional index contains only non-secret diagnostic mapping and is rebuildable.

The identity digest is SHA-256 over a canonical, length-delimited encoding of the canonicalized profile home, server name, and normalized server URL (architecture §4.1). "Canonicalized profile home" is `hermes_constants.hermes_home_key()` — `normcase(str(Path(home).expanduser().resolve(strict=False)))` — the same helper the provider cache key and `profile_matches_home()` use, so this filename and the Chunk 3 lock filename (`<identity-digest>.admin.lock`) derive from an identical digest. The length-delimited encoding prevents path traversal and ambiguous concatenation.

## Envelope schema

```json
{
  "schema_version": 1,
  "revision": "random-128-bit-hex",
  "bundle": {
    "identity": {
      "server_name": "todoist",
      "server_url": "https://ai.todoist.net/mcp"
    },
    "issuer": "https://todoist.com",
    "protected_resource": {},
    "authorization_server": {},
    "client": {},
    "tokens": {
      "access_token": "...",
      "refresh_token": "...",
      "token_type": "Bearer",
      "scopes": [],
      "accepted_at_utc": "...",
      "expires_at": "...",
      "original_expires_in": 3600
    },
    "created_at": "...",
    "updated_at": "...",
    "cimd_rejected": false
  }
}
```

The `identity` block carries no `profile_id`. Profile scope is the file's location under
`HERMES_HOME/mcp-credentials/` and its digest-derived name; the digest is recomputed from the
requesting identity on load and a mismatch simply finds no file (`credential_not_found`). This
matches how `HermesTokenStorage` scopes today (directory + filename, no in-file identity echo).
`server_name` and `server_url` are retained and validated against the runtime's expectation to
prevent a token being used with a different MCP or authorization server (architecture §4.1).

`original_expires_in` is the relative lifetime exactly as the provider returned it (Chunk 4); it is
persisted, never recomputed on load, and replaced by a refresh that supplies a new `expires_in`.

Pydantic or equivalent typed validation enforces field types, URL/issuer identity, maximum sizes, and schema version before use.

## Backend API

Replace compatibility record methods with bundle operations:

```python
class FileOAuthCredentialStore:
    def load(identity) -> StoredBundle | None: ...
    def create(identity, bundle) -> StoredBundle: ...
    def compare_and_swap(identity, expected_revision, bundle) -> StoredBundle: ...
    def replace_authorized(identity, bundle) -> StoredBundle: ...
    def delete(identity) -> bool: ...
    def administrative_lock(identity, *, timeout) -> ContextManager[None]: ...
```

This is the full `OAuthCredentialStore` protocol from architecture §5.1. `replace_authorized` may only be called while holding the identity's administrative lock. `compare_and_swap` takes the mutation lock and verifies revision.

## Atomic write protocol

Under the mutation lock:

1. Load and validate the current envelope when revision checking is required.
2. Serialize the complete new envelope in memory — the new `revision` and the `bundle` it identifies, together.
3. Create a random same-directory temporary file with `O_EXCL`, mode `0600`.
4. Write, flush, and `fsync` the temporary file.
5. Atomically replace the destination with `os.replace`.
6. `fsync` the parent directory on POSIX where supported.
7. Remove any leftover temporary file after failure.

The revision lives in the envelope and is committed by this one `os.replace`. There is never a separate revision manifest — this generalizes Chunk 4's embedded-revision approach to the whole bundle (architecture §8.4), so no reader and no crash can pair a revision with state it does not describe.

Directory mode is `0700`; `secure_parent_dir` must preserve existing repository security invariants.

## Read protocol

Known-identity reads require no directory scan:

1. Resolve identity digest.
2. Read a bounded payload.
3. Decode JSON.
4. Validate schema and identity.
5. Validate issuer binding against runtime expectation.
6. Return immutable bundle plus revision.

Corrupt or unsupported bundles produce typed errors and are not deleted or overwritten automatically.

## Revision watching

Replace token-file mtime watching with revision watching (architecture §8.3, §12.1):

- Each cached provider entry remembers the last-observed revision.
- The revision is probed only at rebuild decision points — before an authorization flow, during 401 recovery, and on explicit refresh or status. It is **never probed on the per-request read path** that serves a resource request.
- Between decision points the entry serves from the remembered revision, held under a short in-memory TTL (default 10 s) with exponential backoff and last-known-good fallback on probe failure. A stale in-TTL read costs at most one late refresh, recovered by the bounded 401 path; compare-and-swap still rejects any stale write.
- A revision change observed at a decision point evicts and rebuilds the provider.
- File mtime or inode may be a cheap probe hint for the file backend but is never the semantic revision.

## Legacy compatibility in this chunk

On load, if no bundle exists but legacy files do, invoke the migration reader to construct a validated in-memory bundle. The backend may write the new bundle only through the verified migration procedure.

During the rollout window, legacy files are not updated after a new bundle becomes authoritative. A non-secret migration marker prevents ambiguous dual writes.

This chunk is where reader coherence is finally guaranteed. Chunk 3's fixed metadata → client → token commit order only narrowed the window in which a concurrent reader could see an incoherent legacy triple; the single atomic bundle closes it (architecture §18, Phase 3).

## Index design

`index.v1.json` is optional and contains:

- Identity digest.
- Profile display identifier.
- Server display name.
- Schema version.
- Backend-visible status timestamp.

It contains no URL query credentials, tokens, client secrets, authorization codes, or metadata response bodies. Diagnostics can enumerate configured MCP servers without relying on the index.

## Crash and concurrency tests

- Reader loop during replacement sees only complete envelopes.
- Crash before `os.replace` preserves old destination.
- Crash after `os.replace` exposes valid new destination.
- Two CAS writers from one revision yield one winner.
- Delete versus CAS is serialized by mutation lock.
- Temporary files have `0600` from creation.
- Directory has `0700`.
- Parent `fsync` is attempted on supported POSIX systems.
- Unsupported schema and corrupt payload remain intact for diagnosis.
- Profile and server identity mismatches fail closed: a request whose canonicalized identity digest does not match an existing file returns `credential_not_found`; a bundle whose `server_url` disagrees with the runtime expectation returns `identity_mismatch`; a bundle whose stored `issuer` disagrees returns `issuer_mismatch`.
- `{tilde, trailing slash, embedded ``..``, symlinked parent, ``/var`` vs ``/private/var``}` spellings of one profile home resolve to one digest and one bundle file.
- No emitted field — log line, `mcp_oauth.refresh_conflict` event, or `OAuthCredentialStatus.revision_prefix` — contains more than 8 hex characters of a revision (bound established in Chunk 4).

## Demonstration

Continuously load and validate a credential in one process while another performs hundreds of revisions. Record that every observed token, client ID, issuer, metadata endpoint, and revision belongs to one known complete generation.

## Non-goals

- Do not add encryption to the file backend; it is owner-only plaintext by explicit selection.
- Do not add Apple Keychain in this chunk.
- Do not silently resolve conflicting legacy and bundle credentials.
- Do not support arbitrary user-provided bundle paths.

## Completion criteria

- File backend uses one authoritative bundle per identity.
- CAS and explicit replacement are atomic; the revision is committed inside the envelope, never a separate manifest.
- Provider invalidation uses revisions, probed only at decision points under a 10 s TTL — never per request.
- Concurrent reads are coherent (the F-4 incoherence window is closed here).
- `original_expires_in` is carried in the bundle.
- Existing valid legacy credentials remain usable through verified migration.
- Independent live OAuth record writes are no longer used by the file backend.
