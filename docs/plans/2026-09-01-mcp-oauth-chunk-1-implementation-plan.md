# MCP OAuth Chunk 1 Shared Credential Store Facade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every current MCP OAuth persistence operation through one backend-neutral credential-store facade while preserving the legacy `mcp-tokens/` file layout and all current runtime behavior.

**Architecture:** Introduce a focused `tools.mcp_oauth_store` package containing canonical identity/state models, safe typed errors, a synchronous store protocol, a legacy-compatible file backend, and a profile-scoped factory. Keep `HermesTokenStorage` as the asynchronous MCP SDK adapter, but make it delegate every durable operation to the facade; migrate non-SDK callers to the facade or adapter methods so only the legacy backend constructs or mutates credential paths.

**Tech Stack:** Python 3.11+, `dataclasses`, `typing.Protocol`, MCP SDK 2.0 Pydantic models, Hermes profile/config helpers, `pytest`, and `scripts/run_tests.sh`.

**Spec:** [`docs/design/mcp-oauth-01-shared-store-facade.md`](../design/mcp-oauth-01-shared-store-facade.md)

## Global Constraints

- Implement from commit `d2a5ea3502623e85319a6761ed4f3a7c2e71325b` or revalidate every cited symbol against the newer base before editing.
- Preserve these exact files and meanings under the active profile: `<safe-server>.json`, `<safe-server>.client.json`, `<safe-server>.meta.json`, and `<safe-server>.cimd-off` in `HERMES_HOME/mcp-tokens/`.
- Do not add staged reauthorization, rollback changes, coherent bundles, CAS revisions, migration, Apple Keychain, or any other new backend.
- Do not convert the existing strict Chunk 0 expected failures into passes; Chunk 1 is an ownership refactor, not the credential-loss fix.
- Keep `HermesTokenStorage` compatible with the MCP SDK's asynchronous storage interface and with its existing constructor and helper methods.
- Resolve profile state at call time from an explicit canonical `hermes_home` or `get_hermes_home()`; do not capture a profile path at module import time.
- Credential errors and logs may include backend name, canonical profile identity, server name, and normalized server URL, but never tokens, refresh tokens, authorization codes, or client secrets.
- On POSIX, keep `mcp-tokens/` mode `0700`, secret-bearing JSON files mode `0600`, and temporary files mode `0600` before writing content.
- Corrupt JSON or schema-invalid records must remain on disk. The backend raises `credential_corrupt`; the compatibility SDK adapter logs a safe warning and returns `None`, preserving current runtime behavior.
- Tests must exercise behavior through imports and temporary profiles. Do not add source-text/change-detector tests or contact a live OAuth/MCP provider.
- Run tests only through `scripts/run_tests.sh` with a Python environment containing the repository's development and MCP dependencies.
- Raspberry Pi 4 provisioning is not required for implementation or the merge gate. A Linux/ARM smoke run is optional after the contract and CI suites pass and must not block Chunk 1.
- Preserve the unrelated macOS case-collision modification under `contributors/emails/`; do not stage or alter it.

## File and Responsibility Map

| Path | Responsibility |
|---|---|
| `tools/mcp_oauth_store/__init__.py` | Stable public exports for models, errors, protocol, factory, and legacy backend. |
| `tools/mcp_oauth_store/models.py` | Canonical `OAuthIdentity`, transitional `LegacyOAuthState`, and identity construction/URL normalization. |
| `tools/mcp_oauth_store/errors.py` | Safe typed store errors and stable error codes. |
| `tools/mcp_oauth_store/base.py` | Backend-neutral protocol plus a separately named legacy compatibility protocol. |
| `tools/mcp_oauth_store/legacy_file_backend.py` | The only production implementation that constructs or mutates legacy OAuth artifact paths. |
| `tools/mcp_oauth_store/factory.py` | Profile-scoped backend selection; only the compatibility backend is available in Chunk 1. |
| `tools/mcp_oauth.py` | MCP SDK adapter and OAuth helpers delegating every durable operation to the selected store; retains protocol behavior without retaining path ownership. |
| `tools/mcp_oauth_manager.py` | Provider cache and disk-change watcher using adapter/facade observations rather than constructing paths. |
| `hermes_cli/mcp_config.py` | Non-SDK token-presence check through the shared store. |
| `optional-skills/mcp/mcp-oauth-remote-gateway/scripts/diagnose-oauth-mcp.py` | Remote-gateway diagnostic reads and optional refresh persistence through the shared store. |
| `tests/tools/test_mcp_oauth_store_models.py` | Identity, protocol, and safe-error unit contracts. |
| `tests/tools/test_mcp_oauth_store_contract.py` | Backend-independent legacy-state contract and file-backend security/atomicity tests. |
| `tests/tools/test_mcp_oauth_store_factory.py` | Profile/config selection and typed unavailable-backend behavior. |
| `tests/tools/test_mcp_oauth_store_adapter.py` | `HermesTokenStorage` delegation and compatibility behavior. |
| `tests/tools/test_mcp_oauth_store_integration.py` | Production manager/CLI/profile interoperability and unchanged-format demonstration. |

---

### Task 1: Define the Transitional Domain Contract and Safe Errors

**Files:**
- Create: `tools/mcp_oauth_store/__init__.py`
- Create: `tools/mcp_oauth_store/models.py`
- Create: `tools/mcp_oauth_store/errors.py`
- Create: `tools/mcp_oauth_store/base.py`
- Create: `tests/tools/test_mcp_oauth_store_models.py`

**Interfaces:**
- Produces: `OAuthIdentity(profile_home: Path, server_name: str, server_url: str)`.
- Produces: `build_oauth_identity(server_name: str, *, server_url: str = "", hermes_home: str | Path | None = None) -> OAuthIdentity`.
- Produces: `LegacyOAuthState(tokens, client, metadata, cimd_rejected=False)`.
- Produces: `OAuthCredentialStore` and `LegacyOAuthCompatibilityStore` protocols.
- Produces: `OAuthCredentialStoreError` with stable `code`, `backend_name`, and optional safe identity/profile context.
- Consumes: MCP SDK `OAuthToken`, `OAuthClientInformationFull`, and `OAuthMetadata` types.

- [ ] **Step 1: Add identity and safe-error tests that fail before the package exists**

Create `tests/tools/test_mcp_oauth_store_models.py` with these behavioral cases:

```python
from pathlib import Path

import pytest


def test_identity_canonicalizes_profile_and_server_url(tmp_path):
    from tools.mcp_oauth_store import build_oauth_identity

    identity = build_oauth_identity(
        "reports",
        server_url="HTTPS://MCP.EXAMPLE:443/api/",
        hermes_home=tmp_path / "profile" / ".." / "profile",
    )

    assert identity.profile_home == (tmp_path / "profile").resolve(strict=False)
    assert identity.server_name == "reports"
    assert identity.server_url == "https://mcp.example/api"


def test_identity_uses_active_profile_at_call_time(tmp_path, monkeypatch):
    from tools.mcp_oauth_store import build_oauth_identity

    profile_a = tmp_path / "a"
    profile_b = tmp_path / "b"
    monkeypatch.setenv("HERMES_HOME", str(profile_a))
    first = build_oauth_identity("shared")
    monkeypatch.setenv("HERMES_HOME", str(profile_b))
    second = build_oauth_identity("shared")

    assert first.profile_home == profile_a.resolve(strict=False)
    assert second.profile_home == profile_b.resolve(strict=False)


def test_store_error_repr_never_contains_secret_payload(tmp_path):
    from tools.mcp_oauth_store import (
        CredentialCorruptError,
        build_oauth_identity,
    )

    identity = build_oauth_identity("reports", hermes_home=tmp_path)
    error = CredentialCorruptError(
        identity=identity,
        backend_name="legacy-file",
        artifact="tokens",
    )

    rendered = repr(error)
    assert error.code == "credential_corrupt"
    assert "reports" in rendered
    assert "ACCESS_TOKEN_FOR_TEST_ONLY" not in rendered
```

- [ ] **Step 2: Run the new module and verify the missing-package failure**

Run:

```bash
scripts/run_tests.sh tests/tools/test_mcp_oauth_store_models.py -q
```

Expected: the first test fails with `ModuleNotFoundError: No module named 'tools.mcp_oauth_store'` (or collection fails if the implementation uses module-scope imports).

- [ ] **Step 3: Implement canonical transitional models**

In `tools/mcp_oauth_store/models.py`, implement the exact public shape:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from hermes_constants import get_hermes_home
from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata, OAuthToken


def normalize_server_url(value: str) -> str:
    if not value:
        return ""
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if scheme and not hostname:
        raise ValueError("server URL with a scheme requires a hostname")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("server URL has an invalid port") from exc
    default_port = (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    host_literal = f"[{hostname}]" if ":" in hostname else hostname
    authority = host_literal if port is None or default_port else f"{host_literal}:{port}"
    path = parsed.path.rstrip("/") or ""
    return urlunsplit((scheme, authority, path, parsed.query, ""))


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


def build_oauth_identity(
    server_name: str,
    *,
    server_url: str = "",
    hermes_home: str | Path | None = None,
) -> OAuthIdentity:
    home = Path(hermes_home) if hermes_home is not None else Path(get_hermes_home())
    return OAuthIdentity(
        profile_home=home.expanduser().resolve(strict=False),
        server_name=server_name,
        server_url=normalize_server_url(server_url),
    )
```

Reject server names containing NUL and reject URLs that have a scheme but no hostname or a malformed port by raising `ValueError`; preserve ordinary display names verbatim in the identity. Add cases for default ports, IPv6 literals, and invalid ports so normalization cannot silently create a different authority.

- [ ] **Step 4: Implement typed, secret-free errors**

In `tools/mcp_oauth_store/errors.py`, define `StoreErrorCode` as a `Literal` containing:

```python
StoreErrorCode = Literal[
    "credential_not_found",
    "backend_unavailable",
    "backend_timeout",
    "credential_corrupt",
    "identity_mismatch",
    "deletion_failed",
]
```

Implement `OAuthCredentialStoreError` so `str()` and `repr()` are constructed only from the code, backend name, an optional identity's server name/profile home, an optional profile home when selection happens before a server identity exists, and an optional non-secret artifact label. Add concrete subclasses `CredentialNotFoundError`, `BackendUnavailableError`, `BackendTimeoutError`, `CredentialCorruptError`, `IdentityMismatchError`, and `DeletionFailedError`; do not accept raw exception bodies or serialized records as public message fields. `BackendUnavailableError` uses the requested configured value as its safe `backend_name`, so factory diagnostics can identify `file`, `auto`, `apple-keychain`, or an unknown value without inventing an `OAuthIdentity`.

- [ ] **Step 5: Define the base and compatibility protocols**

In `tools/mcp_oauth_store/base.py`, define:

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class OAuthCredentialStore(Protocol):
    backend_name: str

    def load_state(self, identity: OAuthIdentity) -> LegacyOAuthState: ...
    def set_tokens(self, identity: OAuthIdentity, tokens: OAuthToken) -> None: ...
    def set_client(self, identity: OAuthIdentity, client: OAuthClientInformationFull) -> None: ...
    def set_metadata(self, identity: OAuthIdentity, metadata: OAuthMetadata) -> None: ...
    def mark_cimd_rejected(self, identity: OAuthIdentity) -> None: ...
    def delete(self, identity: OAuthIdentity) -> bool: ...


@runtime_checkable
class LegacyOAuthCompatibilityStore(OAuthCredentialStore, Protocol):
    def load_tokens(self, identity: OAuthIdentity) -> OAuthToken | None: ...
    def load_client(self, identity: OAuthIdentity) -> OAuthClientInformationFull | None: ...
    def load_metadata(self, identity: OAuthIdentity) -> OAuthMetadata | None: ...
    def is_cimd_rejected(self, identity: OAuthIdentity) -> bool: ...
    def snapshot(self, identity: OAuthIdentity) -> dict[str, bytes]: ...
    def restore(
        self,
        identity: OAuthIdentity,
        snapshot: dict[str, bytes],
        *,
        only_if_absent: bool = False,
    ) -> None: ...
    def poison_client_registration(self, identity: OAuthIdentity) -> bool: ...
    def clear_tokens_and_metadata(self, identity: OAuthIdentity) -> bool: ...
    def has_tokens(self, identity: OAuthIdentity) -> bool: ...
    def tokens_change_token(self, identity: OAuthIdentity) -> int | None: ...
```

The second protocol is deliberately named as temporary compatibility surface. Its record-specific reads preserve the current rule that corruption in one legacy artifact does not make unrelated valid artifacts unreadable. It also contains behavior that disappears when coherent bundles and transactional reauthorization replace legacy partial-file operations.

- [ ] **Step 6: Export only the stable package surface and run tests**

Export the models, error classes, protocols, and builders from `tools/mcp_oauth_store/__init__.py`. Run:

```bash
scripts/run_tests.sh tests/tools/test_mcp_oauth_store_models.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add tools/mcp_oauth_store/__init__.py tools/mcp_oauth_store/models.py tools/mcp_oauth_store/errors.py tools/mcp_oauth_store/base.py tests/tools/test_mcp_oauth_store_models.py
git commit -m "refactor(mcp): define OAuth credential store contract"
```

---

### Task 2: Port Legacy File Persistence Behind the Backend Contract

**Files:**
- Create: `tools/mcp_oauth_store/legacy_file_backend.py`
- Create: `tests/tools/test_mcp_oauth_store_contract.py`
- Modify: `tools/mcp_oauth_store/__init__.py`

**Interfaces:**
- Consumes: Task 1 `OAuthIdentity`, `LegacyOAuthState`, errors, and compatibility protocol.
- Produces: `LegacyFileOAuthCredentialStore` with `backend_name == "legacy-file"`.
- Produces: private `_LegacyArtifactPaths` and public test-neutral behavior; production callers never receive artifact paths.

- [ ] **Step 1: Write the backend-independent round-trip and isolation tests**

Create `tests/tools/test_mcp_oauth_store_contract.py` with a fixture returning `LegacyFileOAuthCredentialStore()` and identities built with explicit temporary homes. Cover:

```python
def test_legacy_backend_round_trips_complete_optional_state(store, identity):
    tokens = OAuthToken(access_token="ACCESS_TOKEN_FOR_TEST_ONLY", token_type="Bearer")
    client = OAuthClientInformationFull(
        client_id="CLIENT_ID_FOR_TEST_ONLY",
        client_secret="CLIENT_SECRET_FOR_TEST_ONLY",
        redirect_uris=["http://127.0.0.1:8765/callback"],
    )
    metadata = OAuthMetadata.model_validate({
        "issuer": "https://auth.invalid",
        "authorization_endpoint": "https://auth.invalid/authorize",
        "token_endpoint": "https://auth.invalid/token",
        "response_types_supported": ["code"],
    })

    store.set_tokens(identity, tokens)
    store.set_client(identity, client)
    store.set_metadata(identity, metadata)
    store.mark_cimd_rejected(identity)

    loaded = store.load_state(identity)
    assert loaded.tokens == tokens
    assert loaded.client.client_id == client.client_id
    assert loaded.metadata.token_endpoint == metadata.token_endpoint
    assert loaded.cimd_rejected is True
```

Add separate tests for missing optional records, profile A/B isolation, traversal-resistant server names, deletion of all four artifacts, and `delete()` returning `False` when nothing existed.

- [ ] **Step 2: Add security, corruption, and atomicity contract tests**

Add tests that assert:

- `mcp-tokens/` is `0700` and each created JSON/marker file is `0600` on POSIX.
- A corrupt token, client, or metadata record raises `CredentialCorruptError` from the corresponding record-specific compatibility read and from aggregate `load_state()`, names only the artifact label, and leaves the bytes unchanged. A corrupt client must not prevent `load_tokens()` from returning a valid token, and vice versa.
- Replacing each JSON record leaves no `.tmp.*` file and a reader sees the complete old or new JSON, never truncated JSON.
- `snapshot()`/`restore()` round-trip exactly the token/client/metadata bytes but not the CIMD marker, matching current behavior.
- `restore(..., only_if_absent=True)` preserves a concurrently written artifact.
- `poison_client_registration()` removes client and metadata, preserves tokens, and keeps one `.bak` client copy.
- `tokens_change_token()` returns `None` when absent and changes after atomic token replacement.

Guard POSIX mode assertions with `pytest.mark.skipif(os.name != "posix", ...)`; the repository has OS-specific markers but no generic `posix_only` marker. Do not emulate another operating system in-process.

- [ ] **Step 3: Run the contract tests and verify the missing-backend failure**

```bash
scripts/run_tests.sh tests/tools/test_mcp_oauth_store_contract.py -q
```

Expected: the module fails at its first import/use because `LegacyFileOAuthCredentialStore` is not defined.

- [ ] **Step 4: Implement safe paths and restrictive atomic writes**

In `legacy_file_backend.py`, move the behavior—not the public helper names—from `_safe_filename`, `_get_token_dir`, `_read_json`, and `_write_json` into focused private helpers. The path builder must always derive from `identity.profile_home`:

```python
@dataclass(frozen=True)
class _LegacyArtifactPaths:
    tokens: Path
    client: Path
    metadata: Path
    cimd_rejected: Path


def _artifact_paths(identity: OAuthIdentity) -> _LegacyArtifactPaths:
    directory = identity.profile_home / "mcp-tokens"
    stem = _safe_filename(identity.server_name)
    return _LegacyArtifactPaths(
        tokens=directory / f"{stem}.json",
        client=directory / f"{stem}.client.json",
        metadata=directory / f"{stem}.meta.json",
        cimd_rejected=directory / f"{stem}.cimd-off",
    )
```

Create directories with `mode=0o700`; after creation, enforce `chmod(0o700)` on POSIX. Write JSON to a same-directory `O_CREAT | O_EXCL` temporary file with mode `0o600`, call `flush()` and `os.fsync()`, then `os.replace()` and enforce final mode `0o600`. Always remove the temporary file after an exception.

- [ ] **Step 5: Implement state serialization and corruption errors**

Port the existing absolute-expiry compatibility exactly:

- `set_tokens()` stores `expires_at = time.time() + int(expires_in)` when valid.
- `load_state()` removes `expires_at` before MCP model validation and rewrites `expires_in` to remaining non-negative seconds.
- Legacy records lacking `expires_at` use token-file mtime plus `expires_in` as the existing best-effort reference.
- Client records containing a secret and missing/`none` auth method are normalized and rewritten with `client_secret_post`.
- A missing record produces `None`; invalid JSON or invalid MCP schema raises `CredentialCorruptError` without deleting or rewriting the invalid record.

Do not log serialized input or the underlying validation exception from the backend.

- [ ] **Step 6: Implement compatibility operations**

Implement all methods from `LegacyOAuthCompatibilityStore`. Build `load_state()` from the same private record readers used by `load_tokens()`, `load_client()`, and `load_metadata()`. `delete()` attempts every artifact, reports `DeletionFailedError` if an `OSError` prevents deletion, and returns whether at least one artifact existed. `clear_tokens_and_metadata()` removes only those two artifacts, preserving client registration and the CIMD marker for the configured-client-change workflow. Preserve current snapshot/restore and poisoned-client semantics byte-for-byte; create backup and restored files with mode `0600`.

- [ ] **Step 7: Run the backend contract and existing storage tests**

```bash
scripts/run_tests.sh \
  tests/tools/test_mcp_oauth_store_contract.py \
  tests/tools/test_mcp_oauth.py \
  tests/tools/test_mcp_oauth_metadata.py \
  tests/tools/test_mcp_oauth_cold_load_expiry.py \
  -q
```

Expected: the new contract passes; existing tests still run against the pre-delegation adapter and remain green.

- [ ] **Step 8: Commit Task 2**

```bash
git add tools/mcp_oauth_store/legacy_file_backend.py tools/mcp_oauth_store/__init__.py tests/tools/test_mcp_oauth_store_contract.py
git commit -m "refactor(mcp): add legacy OAuth file backend"
```

---

### Task 3: Add the Profile-Scoped Store Factory

**Files:**
- Create: `tools/mcp_oauth_store/factory.py`
- Create: `tests/tools/test_mcp_oauth_store_factory.py`
- Modify: `tools/mcp_oauth_store/__init__.py`

**Interfaces:**
- Consumes: `LegacyFileOAuthCredentialStore`, `BackendUnavailableError`, and `build_oauth_identity` profile semantics.
- Produces: `get_oauth_credential_store(*, hermes_home: str | Path | None = None, configured_backend: str | None = None) -> OAuthCredentialStore`.
- Produces: `resolve_configured_backend(config: Mapping[str, object]) -> str | None`.

- [ ] **Step 1: Write factory selection and profile-resolution tests**

Create tests for these exact cases:

```python
def test_factory_defaults_to_legacy_backend_for_missing_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_store import get_oauth_credential_store

    store = get_oauth_credential_store()
    assert store.backend_name == "legacy-file"


@pytest.mark.parametrize("value", ["file", "auto", "apple-keychain", "bogus"])
def test_unimplemented_or_unknown_backend_fails_closed(tmp_path, value):
    from tools.mcp_oauth_store import BackendUnavailableError, get_oauth_credential_store

    with pytest.raises(BackendUnavailableError) as caught:
        get_oauth_credential_store(hermes_home=tmp_path, configured_backend=value)

    assert caught.value.code == "backend_unavailable"
    assert value in str(caught.value)
```

Also test active-profile config extraction from `mcp.oauth.credential_store`, malformed non-mapping `mcp`/`oauth` values falling back to missing-setting compatibility, explicit `hermes_home` canonicalization, and two factory calls after an active profile switch producing independent backends.

- [ ] **Step 2: Run the new factory tests and verify the missing-symbol failure**

```bash
scripts/run_tests.sh tests/tools/test_mcp_oauth_store_factory.py -q
```

Expected: the first factory test fails because `get_oauth_credential_store` is not exported.

- [ ] **Step 3: Implement fail-closed configuration resolution**

In `factory.py`, implement a pure nested-config reader and a late behavioral config import:

```python
def resolve_configured_backend(config: Mapping[str, object]) -> str | None:
    mcp = config.get("mcp")
    if not isinstance(mcp, Mapping):
        return None
    oauth = mcp.get("oauth")
    if not isinstance(oauth, Mapping):
        return None
    value = oauth.get("credential_store")
    if value is None:
        return None
    return str(value).strip().lower() or None
```

When `configured_backend` is omitted and the requested home equals the active canonical `get_hermes_home()`, call `hermes_cli.config.load_config_readonly()` at function invocation and resolve the setting. An explicit non-active home uses the compatibility default unless its caller supplies `configured_backend`; do not read another profile's config through active-profile globals.

Only missing/empty configuration selects `LegacyFileOAuthCredentialStore` in Chunk 1. Every named value returns `BackendUnavailableError`, including future `file`, `auto`, and `apple-keychain`; this prevents silently promising a backend that this chunk does not implement. Later backend chunks extend this branch without rewiring callers.

- [ ] **Step 4: Export and verify the factory**

```bash
scripts/run_tests.sh \
  tests/tools/test_mcp_oauth_store_factory.py \
  tests/tools/test_mcp_oauth_store_models.py \
  tests/tools/test_mcp_oauth_store_contract.py \
  -q
```

Expected: all store-layer tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add tools/mcp_oauth_store/factory.py tools/mcp_oauth_store/__init__.py tests/tools/test_mcp_oauth_store_factory.py
git commit -m "refactor(mcp): add profile-scoped OAuth store factory"
```

---

### Task 4: Convert `HermesTokenStorage` into an MCP SDK Adapter

**Files:**
- Create: `tests/tools/test_mcp_oauth_store_adapter.py`
- Modify: `tools/mcp_oauth.py`
- Modify: `tools/mcp_oauth_manager.py`
- Modify: `tests/tools/test_mcp_oauth.py`
- Modify: `tests/tools/test_mcp_oauth_metadata.py`
- Modify: `tests/tools/test_mcp_cimd.py`
- Modify: `tests/tools/test_mcp_dashboard_oauth.py`
- Modify: `tests/tools/test_mcp_oauth_manager.py`
- Modify: `tests/fakes/mcp_oauth_peer.py`

**Interfaces:**
- Consumes: `get_oauth_credential_store()`, `build_oauth_identity()`, and `LegacyOAuthCompatibilityStore`.
- Produces: backward-compatible `HermesTokenStorage(server_name, *, hermes_home=None, server_url="", store=None)`.
- Produces: `HermesTokenStorage.tokens_change_token() -> int | None` for the manager's backend-neutral watcher.
- Preserves: all existing async MCP SDK methods and synchronous compatibility methods.

- [ ] **Step 1: Write a recording-store adapter test before changing production code**

Create a minimal `RecordingStore` implementing `LegacyOAuthCompatibilityStore`, then assert that `HermesTokenStorage` delegates each operation with the same canonical identity:

```python
@pytest.mark.asyncio
async def test_adapter_delegates_sdk_reads_and_writes(tmp_path):
    store = RecordingStore()
    storage = HermesTokenStorage(
        "reports",
        hermes_home=tmp_path,
        server_url="https://mcp.invalid/api/",
        store=store,
    )
    token = OAuthToken(access_token="ACCESS_TOKEN_FOR_TEST_ONLY", token_type="Bearer")

    await storage.set_tokens(token)
    assert store.calls[-1][0] == "set_tokens"
    identity = store.calls[-1][1]
    assert identity.profile_home == tmp_path.resolve(strict=False)
    assert identity.server_name == "reports"
    assert identity.server_url == "https://mcp.invalid/api"
```

Add tests for client, metadata, CIMD, deletion, snapshot/restore, poisoned registration, cached-token presence, changed-client cleanup, and change-token delegation. Add corrupt-record tests proving the adapter returns `None` only for the corrupt artifact, continues to return unrelated valid artifacts, and logs only `credential_corrupt`, backend, server, and artifact labels.

- [ ] **Step 2: Run the adapter module and verify constructor incompatibility**

```bash
scripts/run_tests.sh tests/tools/test_mcp_oauth_store_adapter.py -q
```

Expected: tests fail because `HermesTokenStorage` does not accept `store` or `server_url` and still performs direct file I/O.

- [ ] **Step 3: Replace adapter path ownership with store/identity ownership**

Change the constructor to:

```python
def __init__(
    self,
    server_name: str,
    *,
    hermes_home: str | Path | None = None,
    server_url: str = "",
    store: LegacyOAuthCompatibilityStore | None = None,
):
    self._identity = build_oauth_identity(
        server_name,
        server_url=server_url,
        hermes_home=hermes_home,
    )
    selected = store or get_oauth_credential_store(
        hermes_home=self._identity.profile_home,
    )
    if not isinstance(selected, LegacyOAuthCompatibilityStore):
        raise BackendUnavailableError(
            identity=self._identity,
            backend_name=selected.backend_name,
        )
    self._store = selected
```

Keep `_server_name` and `_hermes_home` compatibility attributes only where existing runtime diagnostics require them. Remove `_tokens_path()`, `_client_info_path()`, `_meta_path()`, and `_cimd_rejected_path()` after migrating their callers; do not expose backend artifact paths through the store protocol merely to preserve test helpers. Tests that need exact legacy bytes use `snapshot()` or a narrowly imported backend-private path fixture in the backend contract module.

- [ ] **Step 4: Delegate every adapter operation**

Map methods exactly:

| Adapter method | Store operation |
|---|---|
| `get_tokens()` | `load_tokens(identity)` |
| `set_tokens(tokens)` | `set_tokens(identity, tokens)` |
| `get_client_info()` | `load_client(identity)` |
| `set_client_info(client)` | `set_client(identity, client)` |
| `save_oauth_metadata(metadata)` | `set_metadata(identity, metadata)` |
| `load_oauth_metadata()` | `load_metadata(identity)` |
| `mark_cimd_rejected()` | `mark_cimd_rejected(identity)` |
| `cimd_rejected()` | `is_cimd_rejected(identity)` |
| `remove()` | `delete(identity)` |
| `snapshot()` / `restore()` | compatibility protocol methods |
| `poison_client_registration()` | compatibility protocol method |
| `has_cached_tokens()` | `has_tokens(identity)` |
| `tokens_change_token()` | `tokens_change_token(identity)` |

Catch only `CredentialCorruptError` in the four read methods that historically treated corrupt state as absent. Log the safe typed error and return `None`/`False`; do not catch backend unavailable, timeout, identity mismatch, or deletion errors. The record-specific store reads are required here so one corrupt optional artifact cannot hide valid tokens or another valid artifact.

Add synchronous adapter helpers for existing synchronous OAuth setup code: `cached_client_info()`, `set_client_info_sync(client)`, and `clear_tokens_and_metadata()`. These delegate to the compatibility protocol and share the same safe corruption handling as the async SDK methods; they expose domain objects/results, never paths.

- [ ] **Step 5: Migrate the remaining durable operations in `tools/mcp_oauth.py`**

Replace every out-of-class `_read_json(storage._client_info_path())`, `_write_json(...)`, and token/metadata `unlink()` with the adapter's domain-level helpers. In particular:

- `_cached_redirect_port()`, `_cached_redirect_uri()`, and `_has_cached_client_info()` inspect `storage.cached_client_info()`.
- `_invalidate_tokens_on_client_change()` compares the returned client model and calls `storage.clear_tokens_and_metadata()` when the configured identity changed.
- `_maybe_preregister_client()` calls `storage.set_client_info_sync(client_info)`.

Add focused tests for cached redirect selection, client-presence detection, unchanged-client preservation, changed-client token/metadata removal, and synchronous pre-registration. Preserve their current return values and warning behavior, while removing all direct credential artifact access from these helpers.

- [ ] **Step 6: Pass profile homes and server URLs at SDK construction points**

Update `build_oauth_auth()` and `MCPOAuthManager._build_provider()` to construct the adapter with `server_url=...`. Add canonical `profile_home` and `storage` fields to `_ProviderEntry`; populate `profile_home` from the already-computed manager cache key, then construct `HermesTokenStorage(server_name, hermes_home=entry.profile_home, server_url=entry.server_url)` and retain it on the entry. This prevents a profile switch between cache-key calculation and provider construction from selecting another profile's store. Dashboard and TUI workers already pass through those production provider-building paths; do not create a separate UI backend branch.

- [ ] **Step 7: Replace path-based test setup and run compatibility suites**

Update path-based setup/assertions in `tests/tools/test_mcp_oauth.py`, `tests/tools/test_mcp_oauth_metadata.py`, `tests/tools/test_mcp_cimd.py`, `tests/tools/test_mcp_dashboard_oauth.py`, `tests/tools/test_mcp_oauth_manager.py`, and `tests/fakes/mcp_oauth_peer.py`. Use adapter/store operations for behavioral setup and `snapshot()` for exact legacy-byte oracle capture. Only the backend contract test may import a private artifact-path helper to inject corrupt bytes or assert file modes/names.

```bash
scripts/run_tests.sh \
  tests/tools/test_mcp_oauth_store_adapter.py \
  tests/tools/test_mcp_oauth.py \
  tests/tools/test_mcp_oauth_metadata.py \
  tests/tools/test_mcp_cimd.py \
  tests/tools/test_mcp_dashboard_oauth.py \
  tests/tools/test_mcp_oauth_manager.py \
  tests/tools/test_mcp_oauth_cold_load_expiry.py \
  tests/tools/test_mcp_oauth_bidirectional.py \
  -q
```

Expected: all tests pass with the unchanged legacy layout.

- [ ] **Step 8: Commit Task 4**

```bash
git add tools/mcp_oauth.py tools/mcp_oauth_manager.py tests/tools/test_mcp_oauth_store_adapter.py tests/tools/test_mcp_oauth.py tests/tools/test_mcp_oauth_metadata.py tests/tools/test_mcp_cimd.py tests/tools/test_mcp_dashboard_oauth.py tests/tools/test_mcp_oauth_manager.py tests/fakes/mcp_oauth_peer.py
git commit -m "refactor(mcp): delegate SDK OAuth storage to facade"
```

---

### Task 5: Remove Direct Persistence Knowledge from Non-SDK Callers

**Files:**
- Modify: `tools/mcp_oauth_manager.py:760-855`
- Modify: `hermes_cli/mcp_config.py:401-420`
- Modify: `optional-skills/mcp/mcp-oauth-remote-gateway/scripts/diagnose-oauth-mcp.py`
- Modify: `tests/tools/test_mcp_oauth_manager.py`
- Modify: `tests/hermes_cli/test_mcp_config.py`
- Modify: `tests/skills/test_mcp_oauth_remote_gateway_skill.py`

**Interfaces:**
- Consumes: `HermesTokenStorage.tokens_change_token()` for SDK/provider cache invalidation.
- Consumes: factory plus `build_oauth_identity()` for CLI state presence.
- Consumes: the same facade in the optional remote-gateway diagnostic for reads and explicit `--write` refresh persistence.
- Preserves: `MCPOAuthManager.remove()` durable deletion semantics in Chunk 1 and `evict()` memory-only semantics.

- [ ] **Step 1: Add manager tests that prohibit path construction behaviorally**

Replace direct test setup through `_get_token_dir()` with a `HermesTokenStorage` or facade write. Add a recording adapter test:

```python
@pytest.mark.asyncio
async def test_disk_watch_uses_storage_change_token(monkeypatch):
    manager = MCPOAuthManager()
    adapter = MagicMock()
    adapter.tokens_change_token.side_effect = [101, 101, 202]
    provider = MagicMock()
    provider._initialized = True
    manager._entries[manager._key("srv")] = _ProviderEntry(
        profile_home=tmp_path.resolve(strict=False),
        server_url="https://mcp.invalid",
        oauth_config=None,
        provider=provider,
        storage=adapter,
    )

    assert await manager.invalidate_if_disk_changed("srv") is True
    assert await manager.invalidate_if_disk_changed("srv") is False
    assert await manager.invalidate_if_disk_changed("srv") is True
    assert provider._initialized is False
```

Task 4 adds `profile_home` and `storage` to `_ProviderEntry`. This test locks in that the watcher uses the retained adapter, keeping the watcher backend-neutral and avoiding reconstruction under a different active profile.

- [ ] **Step 2: Add CLI token-presence tests using a recording facade**

Patch `hermes_cli.mcp_config.get_oauth_credential_store` and assert `_oauth_tokens_present()` calls `load_state(build_oauth_identity(name))`. Verify missing tokens return `False`, present tokens return `True`, and `BackendUnavailableError` follows the existing defensive log/permissive return path without exposing error payloads.

- [ ] **Step 3: Run the focused tests and verify they fail on direct-path behavior**

```bash
scripts/run_tests.sh \
  tests/tools/test_mcp_oauth_manager.py \
  tests/hermes_cli/test_mcp_config.py \
  -q
```

Expected: the new tests fail because the manager imports `_get_token_dir`/`_safe_filename` and CLI constructs `HermesTokenStorage` for a non-SDK read.

- [ ] **Step 4: Convert the manager watcher to the adapter observation**

Store the adapter in `_ProviderEntry` and replace:

```python
tokens_path = _get_token_dir(hermes_home) / f"{_safe_filename(server_name)}.json"
mtime_ns = tokens_path.stat().st_mtime_ns
```

with:

```python
change_token = entry.storage.tokens_change_token() if entry.storage is not None else None
if change_token is None:
    return False
```

Retain `last_mtime_ns` for compatibility in Chunk 1 or rename it to `last_store_change_token` in the dataclass and every test/reference within this same task. Compare the opaque integer only; do not interpret it as a filesystem timestamp outside the backend.

- [ ] **Step 5: Convert CLI token presence to the facade**

Implement `_oauth_tokens_present()` as:

```python
def _oauth_tokens_present(name: str) -> bool:
    try:
        identity = build_oauth_identity(name)
        state = get_oauth_credential_store(
            hermes_home=identity.profile_home,
        ).load_state(identity)
        return state.tokens is not None
    except Exception as exc:  # existing defensive boundary
        logger.debug("Could not check OAuth credential state for '%s': %s", name, exc)
        return True
```

Import store symbols at module scope only if doing so creates no CLI/MCP circular import; otherwise use late imports inside the helper and patch the defining modules in tests.

- [ ] **Step 6: Audit remaining production path manipulation**

Convert `diagnose-oauth-mcp.py` before the audit: build the identity from the requested server and optional MCP URL, obtain the profile-scoped store, and read its state rather than opening token/client files. On `--write`, validate a new `OAuthToken` and call `set_tokens()`; report the backend name instead of a credential path. Keep the script read-only without `--write`, retain its mocked network decision tree, and update `tests/skills/test_mcp_oauth_remote_gateway_skill.py` to seed/assert through the facade. This script may run on a Pi later, but its unit tests require no Pi or network.

Run:

```bash
rg -n "_get_token_dir|_safe_filename|mcp-tokens|\.client\.json|\.meta\.json|\.cimd-off" \
  tools/mcp_oauth.py tools/mcp_oauth_manager.py hermes_cli/mcp_config.py \
  hermes_cli/web_server.py tui_gateway/mcp_oauth_sessions.py \
  optional-skills/mcp/mcp-oauth-remote-gateway/scripts/diagnose-oauth-mcp.py
rg -n "mcp-tokens|\.client\.json|\.meta\.json|\.cimd-off" \
  tools/mcp_oauth_store/legacy_file_backend.py
```

Expected production matches:

- The first command emits no credential-path ownership matches in callers (ordinary security-policy comments outside this scope remain unchanged).
- The second command finds the concrete compatibility layout in `legacy_file_backend.py`.
- No path construction, artifact-path compatibility shim, or direct credential-file mutation in `tools/mcp_oauth.py`, manager, CLI, dashboard, or TUI modules.
- Dashboard/TUI snapshot and restore calls remain through `HermesTokenStorage`; their rollback semantics are intentionally unchanged until Chunk 3.

- [ ] **Step 7: Run manager, CLI, dashboard, and TUI regressions**

```bash
scripts/run_tests.sh \
  tests/tools/test_mcp_oauth_manager.py \
  tests/hermes_cli/test_mcp_config.py \
  tests/hermes_cli/test_mcp_dashboard_oauth.py \
  tests/tui_gateway/test_mcp_oauth_client_callback.py \
  tests/skills/test_mcp_oauth_remote_gateway_skill.py \
  -q
```

Expected: all existing non-baseline tests pass.

- [ ] **Step 8: Commit Task 5**

```bash
git add tools/mcp_oauth_manager.py hermes_cli/mcp_config.py optional-skills/mcp/mcp-oauth-remote-gateway/scripts/diagnose-oauth-mcp.py tests/tools/test_mcp_oauth_manager.py tests/hermes_cli/test_mcp_config.py tests/skills/test_mcp_oauth_remote_gateway_skill.py
git commit -m "refactor(mcp): route OAuth callers through credential store"
```

---

### Task 6: Prove Cross-Surface Compatibility and Preserve the Chunk 0 Baseline

**Files:**
- Create: `tests/tools/test_mcp_oauth_store_integration.py`
- Modify: `docs/design/mcp-oauth-01-shared-store-facade.md`

**Interfaces:**
- Consumes: complete shared store package, `HermesTokenStorage`, manager provider construction, CLI presence helper, and Chunk 0 deterministic peer/oracle.
- Produces: merge-gate evidence that all surfaces observe one facade without a format change or user migration.

- [ ] **Step 1: Write the production-path compatibility demonstration**

Create `tests/tools/test_mcp_oauth_store_integration.py` with a temporary-profile scenario that:

1. Seeds the exact legacy token/client/metadata files through `LegacyFileOAuthCredentialStore`.
2. Builds the production manager provider for the same server and initializes it without opening a browser.
3. Asserts the provider loaded the seeded access token and metadata through `HermesTokenStorage`.
4. Calls the adapter's token write with a fake refreshed token.
5. Asserts the same `<safe-server>.json` path changed and no new bundle, Keychain, or migration artifact appeared.
6. Calls the CLI `_oauth_tokens_present()` and observes the same state.
7. Switches to a second temporary profile with the same server name and proves it cannot load profile A's token.

Patch only browser/callback/network boundaries. Do not mock the store factory, backend, adapter, manager, or CLI presence helper in this integration test.

- [ ] **Step 2: Run the new integration module and demonstrate the first missing wiring**

```bash
scripts/run_tests.sh tests/tools/test_mcp_oauth_store_integration.py -q
```

Expected before completing any overlooked wiring: a specific assertion fails at the first production surface that bypasses or mis-scopes the new facade. If the test passes immediately because Tasks 1-5 completed every path, record that as a valid integration-first green result and perform the sensitivity check in Step 3.

- [ ] **Step 3: Perform and restore a sensitivity mutation**

Temporarily make the second profile use profile A's identity or make the CLI helper return `True` unconditionally. Run the integration module and verify the profile-isolation or shared-state assertion fails. Restore the real implementation before continuing; no deliberate failure may remain in the diff.

- [ ] **Step 4: Run the store and OAuth compatibility suites**

```bash
scripts/run_tests.sh \
  tests/tools/test_mcp_oauth_store_models.py \
  tests/tools/test_mcp_oauth_store_contract.py \
  tests/tools/test_mcp_oauth_store_factory.py \
  tests/tools/test_mcp_oauth_store_adapter.py \
  tests/tools/test_mcp_oauth_store_integration.py \
  tests/tools/test_mcp_oauth.py \
  tests/tools/test_mcp_oauth_metadata.py \
  tests/tools/test_mcp_oauth_manager.py \
  tests/tools/test_mcp_oauth_cold_load_expiry.py \
  tests/tools/test_mcp_oauth_bidirectional.py \
  tests/tools/test_mcp_cimd.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Verify the cross-surface neighboring suites**

```bash
scripts/run_tests.sh \
  tests/hermes_cli/test_mcp_config.py \
  tests/hermes_cli/test_mcp_dashboard_oauth.py \
  tests/tools/test_mcp_dashboard_oauth.py \
  tests/tui_gateway/test_mcp_oauth_client_callback.py \
  tests/tools/test_mcp_initial_connect_shutdown.py \
  -q
```

Expected: all existing non-baseline tests pass.

- [ ] **Step 6: Verify Chunk 0 remains an unchanged behavioral baseline**

```bash
scripts/run_tests.sh \
  tests/tools/test_mcp_oauth_reauth_regression.py \
  tests/hermes_cli/test_mcp_reauth_lifecycle.py \
  tests/tui_gateway/test_mcp_oauth_reauth.py \
  -q -rxX
```

Expected at the Chunk 1 boundary:

- Regression harness: `23 passed, 7 xfailed`.
- CLI lifecycle: `3 xfailed`.
- TUI lifecycle: `1 passed, 2 xfailed`.
- Total: `24 passed, 12 strict xfailed`, no XPASS and no failure caused by the facade refactor.

- [ ] **Step 7: Run final ownership, formatting, and leakage audits**

```bash
git diff --check
rg -n "_get_token_dir|_safe_filename|mcp-tokens|\.client\.json|\.meta\.json|\.cimd-off" \
  tools/mcp_oauth_store tools/mcp_oauth.py tools/mcp_oauth_manager.py \
  hermes_cli/mcp_config.py hermes_cli/web_server.py tui_gateway/mcp_oauth_sessions.py \
  optional-skills/mcp/mcp-oauth-remote-gateway/scripts/diagnose-oauth-mcp.py
rg -n "ACCESS_TOKEN_FOR_TEST_ONLY|REFRESH_TOKEN_FOR_TEST_ONLY|CLIENT_SECRET_FOR_TEST_ONLY" \
  tools/mcp_oauth_store tools/mcp_oauth.py tools/mcp_oauth_manager.py hermes_cli/mcp_config.py
```

Expected:

- `git diff --check` emits nothing.
- Concrete path construction/mutation exists only in `legacy_file_backend.py`.
- Production files contain no test sentinel values.
- `git status --short` lists only planned Chunk 1 paths plus the pre-existing unrelated contributor marker.

- [ ] **Step 8: Update the design status and hardware note**

In `docs/design/mcp-oauth-01-shared-store-facade.md`, change status to `Implemented by Chunk 1` and add a verification paragraph recording:

```text
The merge gate uses temporary-profile contract and integration tests on supported CI hosts. Raspberry Pi 4 provisioning is not required; Linux/ARM remote-gateway smoke validation is optional and does not replace the backend contract suite.
```

Do not mark Chunk 2-7 behavior implemented.

- [ ] **Step 9: Commit Task 6**

```bash
git add tests/tools/test_mcp_oauth_store_integration.py docs/design/mcp-oauth-01-shared-store-facade.md
git commit -m "test(mcp): verify shared OAuth store facade"
```

## Final Verification

- [ ] Run the complete focused Chunk 1 and neighboring command set from Tasks 4-6 on the final tree.
- [ ] Run `scripts/run_tests.sh -j 8 -q` if the repository's optional test dependencies and host platform support the full suite; classify unrelated environment/platform failures separately and rerun every affected file in isolation.
- [ ] Confirm the implementation diff contains only the planned package, call-site, test, and Chunk 1 design-status paths.
- [ ] Confirm no Keychain code, migration, coherent bundle schema, CAS, or staged reauthorization entered the diff.
- [ ] Confirm every strict Chunk 0 xfail still fails for `KnownCredentialLoss`, with no XPASS.
- [ ] Capture sanitized demonstration evidence for the pull request: same legacy files, same token visible to manager and CLI, second profile isolated, no browser reauthorization.

## Optional Raspberry Pi 4 Validation

This is a post-merge-confidence activity, not an implementation prerequisite or merge gate. Once a Pi is provisioned, run only after installing the same pinned Python/test dependencies:

```bash
scripts/run_tests.sh \
  tests/tools/test_mcp_oauth_store_contract.py \
  tests/tools/test_mcp_oauth_store_integration.py \
  tests/tools/test_mcp_initial_connect_shutdown.py \
  -q
```

Record Python version, OS/kernel, architecture, filesystem type, permission results, and atomic-replacement results. Do not add Pi-specific production branches; any real platform difference must be expressed through an existing supported-platform boundary and reviewed separately.
