# MCP OAuth Chunk 1 — Shared Credential Store Facade

Status: design proposal (not yet implemented; design-review updates applied 2026-09-03)
Depends on: Chunk 0 behavioral harness
Delivery plan: [`../plans/2026-09-01-mcp-oauth-credential-store-delivery-plan.md`](../plans/2026-09-01-mcp-oauth-credential-store-delivery-plan.md)
Architecture: [`../architecture/mcp-oauth-credential-store-architecture.md`](../architecture/mcp-oauth-credential-store-architecture.md) §4.1, §14, §18 (Phase 2)
Design-review updates: [`../requirements/mcp-oauth-design-review-approaches.md`](../requirements/mcp-oauth-design-review-approaches.md) (F-0, F-3; forward notes F-4, F-5)

## Purpose

Introduce one backend-neutral persistence API while preserving the existing `mcp-tokens/` layout and runtime behavior. This is an ownership refactor, not yet the rollback fix or storage-format migration.

At completion, callers no longer know how token, client, and metadata records are stored. The compatibility backend remains responsible for the existing files.

## Package slice

Create the initial package:

```text
tools/mcp_oauth_store/
├── __init__.py
├── models.py
├── errors.py
├── base.py
├── factory.py
└── legacy_file_backend.py
```

Later chunks add lifecycle, staging, versioned bundles, Keychain, and migration modules.

## Domain objects

`OAuthIdentity` contains canonical profile home, server name, and normalized MCP URL. `LegacyOAuthState` represents the current optional token/client/metadata records without pretending they are atomically coherent.

```python
@dataclass(frozen=True)
class OAuthIdentity:
    profile_home: Path
    server_name: str
    server_url: str


@dataclass(frozen=True)
class LegacyOAuthState:
    tokens: OAuthToken | None
    client: OAuthClientInformationFull | None
    metadata: OAuthMetadata | None
    cimd_rejected: bool = False
```

This transitional model must not become the final bundle schema.

`profile_home` is canonicalized by a single shared function with the semantics of `hermes_constants.hermes_home_key()` — `os.path.normcase(str(Path(home).expanduser().resolve(strict=False)))` — so the OAuth identity and the existing plugin/registry/config scope key cannot drift. Resolution collapses symlinks and `..` because the profile directory exists by the time any OAuth operation runs. Two spellings of one profile home (`~/.hermes`, an absolute path, a trailing slash, a symlinked parent, macOS `/var` vs `/private/var`) therefore produce one identity and hit one legacy file set. The digest-keyed filename (`<identity-digest>.json`) does not arrive until Chunk 5; in Chunk 1 the canonical `profile_home` alone provides this guarantee. See architecture §4.1 for the full rule and its known limits (case-insensitive-filesystem case drift; cross-namespace mounts).

## Store interface for this chunk

```python
class OAuthCredentialStore(Protocol):
    backend_name: str

    def load_state(self, identity: OAuthIdentity) -> LegacyOAuthState: ...
    def set_tokens(self, identity: OAuthIdentity, tokens: OAuthToken) -> None: ...
    def set_client(self, identity: OAuthIdentity, client: ClientInfo) -> None: ...
    def set_metadata(self, identity: OAuthIdentity, metadata: OAuthMetadata) -> None: ...
    def mark_cimd_rejected(self, identity: OAuthIdentity) -> None: ...
    def delete(self, identity: OAuthIdentity) -> bool: ...
```

The interface temporarily permits independent record writes to maintain compatibility. These methods are removed or made private when Chunk 5 introduces coherent bundles. Chunk 1 guarantees only that each individual record is replaced atomically (`os.replace`); ordering the three writes to bound reader incoherence is Chunk 3's responsibility (architecture §18, Phase 2), and a coherent multi-record read is not guaranteed until Chunk 5.

## Legacy backend

`LegacyFileOAuthCredentialStore` ports the current safe filename, JSON validation, absolute-expiry compatibility, permissions, atomic writes, and CIMD marker behavior from `HermesTokenStorage`.

It preserves the provider's raw `expires_in` in the on-disk record (current `set_tokens` behavior — the SDK payload dump plus an added `expires_at`). Chunk 1 does not add the never-rewritten `original_expires_in` field or the wall-clock plausibility guard; those are Chunk 4 (architecture §4.2, §4.3). Not stripping the raw value now lets Chunk 4 adopt it without a format migration.

Paths remain:

```text
HERMES_HOME/mcp-tokens/<safe-server>.json
HERMES_HOME/mcp-tokens/<safe-server>.client.json
HERMES_HOME/mcp-tokens/<safe-server>.meta.json
HERMES_HOME/mcp-tokens/<safe-server>.cimd-off
```

Profile resolution uses the identity's explicit canonical home or `get_hermes_home()` at the factory boundary. No module-level path may capture the wrong profile.

## Factory

The initial factory always returns the legacy file backend. It still reads the future configuration location so later backend addition does not require surface rewiring.

```python
def get_oauth_credential_store(
    *, hermes_home: Path | None = None,
) -> OAuthCredentialStore: ...
```

Unknown configured backend values return a typed `backend_unavailable` error; they do not silently choose files.

## Call-site migration

Replace direct durable operations in:

- `tools/mcp_oauth.py` storage callbacks.
- `tools/mcp_oauth_manager.py` durable-state checks and disk watching.
- `hermes_cli/mcp_config.py` token presence, login, and remove paths.
- Dashboard MCP OAuth code.
- TUI gateway MCP OAuth session code.
- Startup and diagnostics that inspect `mcp-tokens/`.

During this chunk, `HermesTokenStorage` may remain as an MCP SDK adapter, but it delegates every durable operation to the store. It no longer constructs paths itself.

Profile-home canonicalization is centralized in the same pass. `MCPOAuthManager._key()` and `hermes_cli/profiles.py::profile_matches_home()` both currently do `expanduser().resolve(strict=False)` without `normcase`; they move onto the shared `hermes_home_key()` helper so the manager cache key, the profile-match check, and the OAuth identity agree on which profile a request belongs to.

## Typed errors

Introduce stable safe codes:

- `credential_not_found`
- `backend_unavailable`
- `backend_timeout`
- `credential_corrupt`
- `identity_mismatch`
- `deletion_failed`

Errors contain safe identity and backend context, never token payloads.

## Contract tests

Parameterize tests over the store factory:

- Token/client/metadata round trip.
- Missing optional records.
- CIMD marker round trip.
- Explicit deletion removes every legacy artifact.
- Profile isolation.
- Identity stability across path spellings: `{tilde, trailing slash, embedded ``..``, symlinked parent, ``/var`` vs ``/private/var``}` of one profile home produce one `OAuthIdentity` and resolve `load_state` / `set_*` to the same legacy file set.
- Server-name path traversal resistance.
- POSIX directory and file permissions.
- Corrupt record reporting without automatic deletion.
- Atomic replacement of each compatibility record.

## Demonstration

1. Seed legacy files under a temporary profile.
2. Load the MCP provider through production startup code.
3. Refresh a fake token and observe the same legacy file update.
4. Load from a second process/profile-aware entry point.
5. Show that no storage format changed and no reauthorization occurred.

## Non-goals

- Do not add staged reauthorization (Chunk 3), the probe-outcome classifier, or the `authorization_endpoint_unavailable` error (F-2, Chunk 2).
- Do not fix destructive rollback by adding new rollback rules.
- Do not introduce bundle revisions or CAS (Chunk 4), or the identity digest (Chunk 3 for lock filenames, Chunk 5 for credential filenames).
- Do not add the `original_expires_in` field or the wall-clock plausibility guard (F-5, Chunk 4).
- Do not add Keychain or the `credential_ambiguous` error (Chunk 6).
- Do not migrate user files.

## Completion criteria

- Production callers use the store facade for durable OAuth state.
- `HermesTokenStorage` is an adapter, not a persistence implementation.
- Existing integrations and files remain compatible.
- Backend contract tests pass with isolated profiles.
- The Chunk 0 harness still demonstrates the baseline failure.
