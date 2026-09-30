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
# #85387 — the read-side sensitive guard must also apply to WRITE endpoints.
# upload / upload-stream / mkdir / delete must fail closed on .env,
# config.yaml, credential stores and credential directory trees.
# ---------------------------------------------------------------------------


def test_sensitive_paths_blocked_on_upload(forced_files_client):
    """#85387: /api/files/upload must not create or overwrite sensitive files."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    for name in (".env", "config.yaml", "auth.json", "credentials"):
        target = root / name
        resp = client.post(
            "/api/files/upload",
            json={"path": str(target), "data_url": "data:text/plain;base64,RVZJTD0x"},
        )
        assert resp.status_code == 403, name
        assert not target.exists(), f"sensitive file was created: {name}"

    # A credential-directory component (mcp-tokens/<server>.json) is blocked too.
    nested = root / "mcp-tokens" / "github.json"
    resp = client.post(
        "/api/files/upload",
        json={"path": str(nested), "data_url": "data:text/plain;base64,e30="},
    )
    assert resp.status_code == 403
    assert not nested.exists()

    # Regular files still upload fine.
    ok = root / "notes.txt"
    resp = client.post(
        "/api/files/upload",
        json={"path": str(ok), "data_url": "data:text/plain;base64,aGk="},
    )
    assert resp.status_code == 200
    assert ok.read_text() == "hi"


def test_sensitive_paths_blocked_on_stream_upload(forced_files_client):
    """#85387: the multipart /api/files/upload-stream path is blocked as well."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    target = root / ".env"
    resp = client.post(
        "/api/files/upload-stream",
        files={"file": ("evil.env", b"EVIL=1", "text/plain")},
        data={"path": str(target), "overwrite": "true"},
    )
    assert resp.status_code == 403
    assert not target.exists()
    # No temp upload artifact was left behind.
    assert [p.name for p in root.iterdir() if ".upload" in p.name] == []

    ok = root / "streamed.txt"
    resp = client.post(
        "/api/files/upload-stream",
        files={"file": ("streamed.txt", b"hello", "text/plain")},
        data={"path": str(ok), "overwrite": "true"},
    )
    assert resp.status_code == 200
    assert ok.read_text() == "hello"


def test_sensitive_paths_blocked_on_mkdir(forced_files_client):
    """#85387: mkdir must not create directories that collide with sensitive paths."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    for name in (".env", "mcp-tokens", "pairing"):
        target = root / name
        resp = client.post("/api/files/mkdir", json={"path": str(target)})
        assert resp.status_code == 403, name
        assert not target.exists(), f"sensitive dir was created: {name}"

    ok = root / "workdir"
    assert client.post("/api/files/mkdir", json={"path": str(ok)}).status_code == 200
    assert ok.is_dir()


def test_sensitive_paths_blocked_on_delete(forced_files_client):
    """#85387: delete (a destructive write) must refuse sensitive paths."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)
    env_file = root / ".env"
    env_file.write_text("SECRET_KEY=abc123")

    resp = client.request(
        "DELETE", "/api/files", json={"path": str(env_file), "recursive": False}
    )
    assert resp.status_code == 403
    assert env_file.exists(), "sensitive file was deleted"

    # Deleting an ordinary file still works.
    plain = root / "plain.txt"
    plain.write_text("data")
    assert client.request(
        "DELETE", "/api/files", json={"path": str(plain), "recursive": False}
    ).status_code == 200
    assert not plain.exists()


def test_recursive_delete_blocked_when_directory_contains_sensitive_file(forced_files_client):
    """#85387: recursive delete of a directory containing sensitive files must be rejected."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    parent_dir = root / "project"
    parent_dir.mkdir()
    nested_env = parent_dir / ".env"
    nested_env.write_text("API_KEY=secret")
    plain_file = parent_dir / "app.py"
    plain_file.write_text("print('hello')")

    resp = client.request(
        "DELETE", "/api/files", json={"path": str(parent_dir), "recursive": True}
    )
    assert resp.status_code == 403
    assert parent_dir.exists(), "directory was deleted"
    assert nested_env.exists(), "sensitive child was deleted"
    assert plain_file.exists(), "sibling was deleted"

    # Also check directory containing mcp-tokens subdirectory
    mcp_dir = root / "integration"
    mcp_sub = mcp_dir / "mcp-tokens"
    mcp_sub.mkdir(parents=True)
    mcp_token = mcp_sub / "github.json"
    mcp_token.write_text("{}")

    resp_mcp = client.request(
        "DELETE", "/api/files", json={"path": str(mcp_dir), "recursive": True}
    )
    assert resp_mcp.status_code == 403
    assert mcp_dir.exists()
    assert mcp_token.exists()


def test_symlink_to_sensitive_path_blocked_on_write(forced_files_client):
    """#85387: symlinks pointing to sensitive files must be rejected on upload and delete."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)
    env_file = root / ".env"
    env_file.write_text("SECRET=123")
    symlink_target = root / "symlink_env"

    try:
        symlink_target.symlink_to(env_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported or permitted on this platform")

    # Upload targeting symlink must be blocked
    resp_upload = client.post(
        "/api/files/upload",
        json={"path": str(symlink_target), "data_url": "data:text/plain;base64,TkVX"},
    )
    assert resp_upload.status_code == 403
    assert env_file.read_text() == "SECRET=123"

    # Delete targeting symlink must be blocked
    resp_del = client.request(
        "DELETE", "/api/files", json={"path": str(symlink_target), "recursive": False}
    )
    assert resp_del.status_code == 403
    assert env_file.exists()



