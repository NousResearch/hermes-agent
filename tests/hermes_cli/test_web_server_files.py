"""Tests for the dashboard-managed file browser API."""

from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


def _client_with_app_state():
    prev_auth_required = getattr(web_server.app.state, "auth_required", None)
    prev_bound_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client, prev_auth_required, prev_bound_host


def _restore_app_state(prev_auth_required, prev_bound_host):
    if prev_auth_required is None:
        delattr(web_server.app.state, "auth_required")
    else:
        web_server.app.state.auth_required = prev_auth_required
    if prev_bound_host is None:
        if hasattr(web_server.app.state, "bound_host"):
            delattr(web_server.app.state, "bound_host")
    else:
        web_server.app.state.bound_host = prev_bound_host


def _close_client(client):
    close = getattr(client, "close", None)
    if close is not None:
        close()


@pytest.fixture
def forced_files_client(monkeypatch, tmp_path):
    root = tmp_path / "data"
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))

    client, prev_auth_required, prev_bound_host = _client_with_app_state()
    try:
        yield client, root
    finally:
        _close_client(client)
        _restore_app_state(prev_auth_required, prev_bound_host)


@pytest.fixture
def local_files_client(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))

    client, prev_auth_required, prev_bound_host = _client_with_app_state()
    try:
        yield client, home
    finally:
        _close_client(client)
        _restore_app_state(prev_auth_required, prev_bound_host)














def _seed_file(client, root, name="out/hello.txt"):
    file_path = root / name
    created = client.post(
        "/api/files/upload",
        json={"path": str(file_path), "data_url": "data:text/plain;base64,aGVsbG8="},
    )
    assert created.status_code == 200
    return file_path




def test_download_authenticates_via_query_token(forced_files_client):
    client, root = forced_files_client
    file_path = _seed_file(client, root)

    # Drop the session header so only the ?token= query param authenticates —
    # mirrors a browser/shell-opened download that can't set the session header.
    del client.headers[web_server._SESSION_HEADER_NAME]

    ok = client.get(
        "/api/files/download",
        params={"path": str(file_path), "token": web_server._SESSION_TOKEN},
    )
    assert ok.status_code == 200
    assert ok.content == b"hello"

    assert client.get(
        "/api/files/download", params={"path": str(file_path), "token": "nope"}
    ).status_code == 401
    assert client.get(
        "/api/files/download", params={"path": str(file_path)}
    ).status_code == 401


def test_query_token_does_not_authenticate_other_endpoints(forced_files_client):
    client, root = forced_files_client
    file_path = _seed_file(client, root)

    del client.headers[web_server._SESSION_HEADER_NAME]

    # The query-token escape hatch is scoped to /api/files/download only; it must
    # not unlock the rest of the API surface.
    leaked = client.get(
        "/api/files/read",
        params={"path": str(file_path), "token": web_server._SESSION_TOKEN},
    )
    assert leaked.status_code == 401




# ---------------------------------------------------------------------------
# Streaming multipart upload (/api/files/upload-stream) — NS-501
# ---------------------------------------------------------------------------








def test_stream_upload_cleans_temp_on_cancellation(forced_files_client):
    """A client disconnect mid-stream (asyncio.CancelledError) must not leak a temp file.

    CancelledError is a BaseException, not an Exception, so it bypasses the
    endpoint's ``except`` clauses entirely. The cleanup therefore lives in a
    ``finally`` keyed on a success flag — without it, every aborted large
    upload (the exact NS-501 scenario) would orphan a partial ``.upload`` temp
    file in the target directory. We invoke the endpoint coroutine directly so
    the BaseException propagates instead of being swallowed by the test client.
    """
    import asyncio

    _client, root = forced_files_client
    target = root / "out" / "aborted.bin"
    target.parent.mkdir(parents=True, exist_ok=True)

    class _AbortingUpload:
        """UploadFile stand-in that yields one chunk then aborts like a dropped client."""

        filename = "aborted.bin"

        def __init__(self):
            self._calls = 0

        async def read(self, _size):
            self._calls += 1
            if self._calls == 1:
                return b"partial chunk before the client vanished"
            raise asyncio.CancelledError()

        async def close(self):
            return None

    request = SimpleNamespace()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            web_server.upload_managed_file_stream(
                request=request,
                file=_AbortingUpload(),
                path=str(target),
                overwrite=True,
            )
        )

    # No partial data was promoted into place ...
    assert not target.exists()
    # ... and no .upload temp file was left behind.
    leftovers = [p.name for p in target.parent.iterdir() if ".upload" in p.name]
    assert leftovers == [], f"temp upload files leaked on cancellation: {leftovers}"


def test_sensitive_env_files_hidden_from_listing(forced_files_client):
    """Regression test for #57505: .env files must not appear in directory listings."""
    client, root = forced_files_client

    # Create a regular file and .env variants including shorthand suffixes.
    root.mkdir(parents=True, exist_ok=True)
    regular = root / "config.txt"
    regular.write_text("safe content")
    env_file = root / ".env"
    env_file.write_text("SECRET_KEY=abc123")
    env_local = root / ".env.local"
    env_local.write_text("LOCAL_SECRET=def456")
    env_prod = root / ".env.prod"
    env_prod.write_text("PROD_SECRET=ghi789")

    listing = client.get("/api/files", params={"path": str(root)})
    assert listing.status_code == 200
    names = [e["name"] for e in listing.json()["entries"]]
    assert "config.txt" in names
    assert ".env" not in names
    assert ".env.local" not in names
    assert ".env.prod" not in names












def test_other_credential_store_basenames_blocked(forced_files_client):
    """Regression: the managed-files guard must cover the same credential
    basenames as gateway.platforms.base._ROOT_CREDENTIAL_FILES and
    agent.file_safety.get_read_block_error, not just .env — an operator can
    point the managed root at HERMES_HOME itself (#57505), which contains
    all of these live secret stores."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    for name in (
        "auth.json",
        "auth.lock",
        "credentials",
        "config.yaml",
        ".anthropic_oauth.json",
        "google_token.json",
        "google_oauth_pending.json",
        "google_oauth.json",
        "webhook_subscriptions.json",
        "bws_cache.json",
        "bws_cache.enc.json",
    ):
        p = root / name
        p.write_text("SECRET=abc123")
        assert client.get("/api/files/read", params={"path": str(p)}).status_code == 403, name
        assert client.get("/api/files/download", params={"path": str(p)}).status_code == 403, name

    listing = client.get("/api/files", params={"path": str(root)})
    names = [e["name"] for e in listing.json()["entries"]]
    assert names == []




def test_credential_dir_trees_blocked_on_subdir_descent(forced_files_client):
    """Regression: mcp-tokens/ (live MCP OAuth tokens) and pairing/ are denied
    as whole directory trees by both canonical guards
    (gateway.platforms.base._ROOT_CREDENTIAL_DIRS and
    agent.file_safety). A basename-only check would still expose their
    per-server files (e.g. ``mcp-tokens/github.json``) once the browser
    descends into the subdir. The managed-files guard must block any path with
    a credential-directory component, not just leaf basenames."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    # A per-server MCP token file with a NON-canonical basename that the
    # basename denylist alone would not catch.
    mcp_dir = root / "mcp-tokens"
    mcp_dir.mkdir(parents=True, exist_ok=True)
    mcp_file = mcp_dir / "github.json"
    mcp_file.write_text('{"access_token": "SECRET"}\n')

    pairing_dir = root / "pairing"
    pairing_dir.mkdir(parents=True, exist_ok=True)
    pairing_file = pairing_dir / "device-abc"
    pairing_file.write_text("PAIRING-SECRET\n")

    # The token dirs themselves must not appear in the root listing.
    root_names = [e["name"] for e in client.get(
        "/api/files", params={"path": str(root)}).json()["entries"]]
    assert "mcp-tokens" not in root_names
    assert "pairing" not in root_names

    # Read/download of the per-server files must be denied even though their
    # basenames aren't in _SENSITIVE_MANAGED_FILE_BASENAMES.
    for p in (mcp_file, pairing_file):
        assert client.get("/api/files/read", params={"path": str(p)}).status_code == 403, str(p)
        assert client.get("/api/files/download", params={"path": str(p)}).status_code == 403, str(p)

    # Listing the credential dir itself yields nothing exploitable: every child
    # is filtered because the parent component is a credential dir.
    mcp_listing = client.get("/api/files", params={"path": str(mcp_dir)})
    assert [e["name"] for e in mcp_listing.json()["entries"]] == []


# ---------------------------------------------------------------------------
# Sibling coverage: a credential basename and its on-disk siblings
# ---------------------------------------------------------------------------

# Suffixes that routinely appear beside a credential file on a real install:
# editor/tool backups, interrupted atomic writes, and manual copies made
# before an edit. Each carries the *same plaintext* as the file it shadows.
_SIBLING_SUFFIXES = (".bak", ".tmp", ".orig", ".save", ".1", ".bak.1")


def _assert_not_exposed(client, path, label):
    """A path must be invisible to all three managed-files read routes."""
    listing = client.get("/api/files", params={"path": str(path.parent)})
    assert listing.status_code == 200, label
    names = [e["name"] for e in listing.json()["entries"]]
    assert path.name not in names, f"{label}: listed by /api/files"

    read = client.get("/api/files/read", params={"path": str(path)})
    assert read.status_code == 403, f"{label}: readable via /api/files/read"

    download = client.get("/api/files/download", params={"path": str(path)})
    assert download.status_code == 403, f"{label}: downloadable via /api/files/download"


def _assert_visible(client, path, label):
    """An ordinary file must stay fully usable in the browser."""
    listing = client.get("/api/files", params={"path": str(path.parent)})
    assert listing.status_code == 200, label
    names = [e["name"] for e in listing.json()["entries"]]
    assert path.name in names, f"{label}: over-blocked, missing from /api/files"

    read = client.get("/api/files/read", params={"path": str(path)})
    assert read.status_code == 200, f"{label}: over-blocked on /api/files/read"

    download = client.get("/api/files/download", params={"path": str(path)})
    assert download.status_code == 200, f"{label}: over-blocked on /api/files/download"


def test_credential_basename_siblings_are_never_exposed(forced_files_client):
    """Contract: if a basename is credential material, so are its siblings.

    Asserts the RELATION rather than a frozen list literal — the cases are
    generated from the guard's own declaration
    (``_SENSITIVE_MANAGED_FILE_BASENAMES``), so a future entry added there is
    covered by this test automatically.

    ``.env`` already got this treatment (``.env.bak`` / ``.env.local`` are
    denied by the prefix branch), but every other credential basename was
    matched *exactly*. So ``auth.json`` was denied while ``auth.json.bak`` —
    byte-identical provider keys, written by any editor or interrupted atomic
    save — was listed, readable, and downloadable through the dashboard.
    """
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    for basename in sorted(web_server._SENSITIVE_MANAGED_FILE_BASENAMES):
        for suffix in _SIBLING_SUFFIXES:
            sibling = root / f"{basename}{suffix}"
            sibling.write_text("SECRET_KEY=abc123\n")
            try:
                _assert_not_exposed(client, sibling, sibling.name)
            finally:
                sibling.unlink()


def test_session_transcript_stores_are_never_exposed(forced_files_client):
    """Contract: the application history stores are credential material.

    ``state.db`` is the session transcript store — every message, every tool
    payload, and therefore every secret that ever passed through a tool result.
    The project already classifies it as a secret elsewhere
    (``hermes_cli.backup._SECRET_FILE_NAMES`` and the write-side deny in
    ``agent.file_safety``), and ``response_store.db`` is chmod-0600 on creation
    for exactly this reason (``gateway/platforms/api_server.py``), but the
    managed-files read guard listed and served them.

    The SQLite sidecars are covered too: in WAL mode ``-wal`` holds
    transactions not yet checkpointed into the main DB, and both ``-wal`` and
    ``-shm`` are real files on disk that the rest of the codebase enumerates
    explicitly (``hermes_cli/profiles.py``, ``docker/stage2-hook.sh``).
    """
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    for stem in ("state.db", "response_store.db", "kanban.db"):
        for suffix in ("", "-wal", "-shm", "-journal", ".bak", "-wal.bak"):
            store = root / f"{stem}{suffix}"
            store.write_bytes(b"SQLite format 3\x00secret-transcript")
            try:
                _assert_not_exposed(client, store, store.name)
            finally:
                store.unlink()


def test_ordinary_files_stay_visible(forced_files_client):
    """Negative controls: widening the guard must not break the browser.

    Over-blocking the managed-files browser is a regression, not a fix. These
    names deliberately sit near the denied ones — a user's own ``.db``, a
    longer word that merely starts with a credential basename, and a file whose
    stem extends one without a separator — and must all stay listable,
    readable, and downloadable.
    """
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    for name in (
        "notes.txt",
        "README.md",
        # A user's own SQLite DB that is NOT an application history store.
        "mydata.db",
        "mydata.db-wal",
        # Extends a credential basename with no separator: a distinct file.
        "state.dbf",
        "authentication.md",
        "environment.md",
        "configuration.yaml",
        # Dash suffixes are only sensitive for the SQLite sidecars, so an
        # ordinary dash-suffixed name near a credential basename stays visible.
        "credentials-howto.md",
        "config.yaml-template",
        ".environment",
        # Suffix-only match must not be enough: these end in a sidecar suffix
        # but their stem is not a credential basename.
        "notes.db-wal",
    ):
        ordinary = root / name
        ordinary.write_text("ordinary content\n")
        try:
            _assert_visible(client, ordinary, name)
        finally:
            ordinary.unlink()


