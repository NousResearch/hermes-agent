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


def test_forced_root_file_upload_list_read_delete_roundtrip(forced_files_client):
    client, root = forced_files_client
    file_path = root / "out" / "hello.txt"

    created = client.post(
        "/api/files/upload",
        json={
            "path": str(file_path),
            "data_url": "data:text/plain;base64,aGVsbG8=",
        },
    )
    assert created.status_code == 200
    assert created.json()["entry"]["path"] == str(file_path)
    assert created.json()["locked_root"] == str(root)
    assert created.json()["can_change_path"] is False
    assert file_path.read_text() == "hello"

    listing = client.get("/api/files", params={"path": str(root / "out")})
    assert listing.status_code == 200
    assert listing.json()["path"] == str(root / "out")
    assert listing.json()["parent"] == str(root)
    assert listing.json()["entries"] == [
        {
            "name": "hello.txt",
            "path": str(file_path),
            "is_directory": False,
            "size": 5,
            "mtime": pytest.approx(file_path.stat().st_mtime),
            "mime_type": "text/plain",
        }
    ]

    read = client.get("/api/files/read", params={"path": str(file_path)})
    assert read.status_code == 200
    assert read.json()["data_url"] == "data:text/plain;base64,aGVsbG8="

    deleted = client.request(
        "DELETE",
        "/api/files",
        json={"path": str(file_path)},
    )
    assert deleted.status_code == 200
    assert not file_path.exists()


def test_directory_management_requires_recursive_delete_for_nonempty_dirs(forced_files_client):
    client, root = forced_files_client
    runs_path = root / "runs"
    checkpoints_path = runs_path / "checkpoints"

    created = client.post("/api/files/mkdir", json={"path": str(checkpoints_path)})
    assert created.status_code == 200
    assert checkpoints_path.is_dir()

    listing = client.get("/api/files", params={"path": str(runs_path)})
    assert listing.status_code == 200
    assert listing.json()["entries"][0]["path"] == str(checkpoints_path)
    assert listing.json()["entries"][0]["is_directory"] is True

    non_recursive = client.request(
        "DELETE",
        "/api/files",
        json={"path": str(runs_path), "recursive": False},
    )
    assert non_recursive.status_code == 409

    recursive = client.request(
        "DELETE",
        "/api/files",
        json={"path": str(runs_path), "recursive": True},
    )
    assert recursive.status_code == 200
    assert not runs_path.exists()


def test_forced_root_paths_stay_under_root(forced_files_client, tmp_path):
    client, root = forced_files_client
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not leak")

    traversal = client.get("/api/files", params={"path": "../outside"})
    assert traversal.status_code == 400

    outside_absolute = client.get("/api/files", params={"path": str(outside)})
    assert outside_absolute.status_code == 403

    root_delete = client.request(
        "DELETE",
        "/api/files",
        json={"path": str(root), "recursive": True},
    )
    assert root_delete.status_code == 400

    root.mkdir(exist_ok=True)
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem does not allow directory symlinks")

    escaped = client.get("/api/files", params={"path": str(link)})
    assert escaped.status_code == 403


def test_local_mode_defaults_to_home_and_can_jump_to_absolute_path(local_files_client, tmp_path):
    client, home = local_files_client
    (home / "home.txt").write_text("home")

    default_listing = client.get("/api/files")
    assert default_listing.status_code == 200
    assert default_listing.json()["path"] == str(home)
    assert default_listing.json()["locked_root"] is None
    assert default_listing.json()["can_change_path"] is True
    assert default_listing.json()["entries"][0]["path"] == str(home / "home.txt")

    other = tmp_path / "other"
    other.mkdir()
    (other / "other.txt").write_text("other")

    other_listing = client.get("/api/files", params={"path": str(other)})
    assert other_listing.status_code == 200
    assert other_listing.json()["path"] == str(other)
    assert other_listing.json()["parent"] == str(tmp_path)
    assert other_listing.json()["entries"][0]["path"] == str(other / "other.txt")


def test_gated_local_mode_still_defaults_to_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))

    prev_auth_required = getattr(web_server.app.state, "auth_required", None)
    prev_bound_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = True
    web_server.app.state.bound_host = "0.0.0.0"
    try:
        request = SimpleNamespace(
            app=web_server.app,
            client=SimpleNamespace(host="10.0.0.2"),
            url=SimpleNamespace(hostname="example.com"),
        )
        policy = web_server._managed_files_policy(request, create_root=False)
    finally:
        _restore_app_state(prev_auth_required, prev_bound_host)

    assert policy.default_path == home.resolve()
    assert policy.locked_root is None
    assert policy.can_change_path is True


def test_local_mode_upload_read_mkdir_delete_roundtrip(local_files_client):
    client, home = local_files_client
    folder = home / "workspace"
    file_path = folder / "note.txt"

    created_folder = client.post("/api/files/mkdir", json={"path": str(folder)})
    assert created_folder.status_code == 200
    assert created_folder.json()["locked_root"] is None
    assert created_folder.json()["can_change_path"] is True
    assert folder.is_dir()

    uploaded = client.post(
        "/api/files/upload",
        json={
            "path": str(file_path),
            "data_url": "data:text/plain;base64,bG9jYWw=",
        },
    )
    assert uploaded.status_code == 200
    assert file_path.read_text() == "local"

    read = client.get("/api/files/read", params={"path": str(file_path)})
    assert read.status_code == 200
    assert read.json()["data_url"] == "data:text/plain;base64,bG9jYWw="

    deleted = client.request(
        "DELETE",
        "/api/files",
        json={"path": str(folder), "recursive": True},
    )
    assert deleted.status_code == 200
    assert not folder.exists()


def _seed_file(client, root, name="out/hello.txt"):
    file_path = root / name
    created = client.post(
        "/api/files/upload",
        json={"path": str(file_path), "data_url": "data:text/plain;base64,aGVsbG8="},
    )
    assert created.status_code == 200
    return file_path


def test_download_returns_file_as_attachment(forced_files_client):
    client, root = forced_files_client
    file_path = _seed_file(client, root)

    resp = client.get("/api/files/download", params={"path": str(file_path)})
    assert resp.status_code == 200
    assert resp.content == b"hello"
    disposition = resp.headers["content-disposition"]
    assert "attachment" in disposition
    assert "hello.txt" in disposition


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


def test_hosted_policy_locks_to_opt_data(monkeypatch):
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.setenv("HERMES_HOME", "/opt/data")
    client, prev_auth_required, prev_bound_host = _client_with_app_state()
    try:
        request = SimpleNamespace(
            app=web_server.app,
            client=SimpleNamespace(host="127.0.0.1"),
            url=SimpleNamespace(hostname="127.0.0.1"),
        )
        policy = web_server._managed_files_policy(request, create_root=False)
    finally:
        _restore_app_state(prev_auth_required, prev_bound_host)
        client.close()

    assert str(policy.locked_root) == "/opt/data"
    assert policy.can_change_path is False


# ---------------------------------------------------------------------------
# Streaming multipart upload (/api/files/upload-stream) — NS-501
# ---------------------------------------------------------------------------


def test_stream_upload_roundtrip(forced_files_client):
    """The multipart endpoint writes raw bytes to disk and reports the entry."""
    client, root = forced_files_client
    file_path = root / "out" / "backup.zip"
    payload = b"PK\x03\x04 not really a zip but binary enough \x00\x01\x02"

    created = client.post(
        "/api/files/upload-stream",
        data={"path": str(file_path), "overwrite": "true"},
        files={"file": ("backup.zip", payload, "application/zip")},
    )
    assert created.status_code == 200, created.text
    assert created.json()["entry"]["path"] == str(file_path)
    assert created.json()["locked_root"] == str(root)
    # Bytes land verbatim — no base64 round-trip, no corruption.
    assert file_path.read_bytes() == payload


def test_stream_upload_rejects_oversized_without_clobbering(forced_files_client, monkeypatch):
    """Over-limit uploads return 413 and never overwrite an existing file.

    The size cap is enforced while streaming (not after buffering), and the
    temp-file + atomic-rename design means a rejected upload leaves any
    pre-existing file at the target path untouched.
    """
    client, root = forced_files_client
    file_path = root / "out" / "big.bin"

    # Seed an existing file at the target path.
    seeded = client.post(
        "/api/files/upload-stream",
        data={"path": str(file_path), "overwrite": "true"},
        files={"file": ("big.bin", b"original-contents", "application/octet-stream")},
    )
    assert seeded.status_code == 200
    assert file_path.read_bytes() == b"original-contents"

    # Shrink the cap so a small payload trips it deterministically.
    monkeypatch.setattr(web_server, "_MANAGED_FILE_MAX_BYTES", 8)
    rejected = client.post(
        "/api/files/upload-stream",
        data={"path": str(file_path), "overwrite": "true"},
        files={"file": ("big.bin", b"way too many bytes for the cap", "application/octet-stream")},
    )
    assert rejected.status_code == 413
    # The original file must survive a rejected overwrite.
    assert file_path.read_bytes() == b"original-contents"
    # No stray temp files left behind in the directory.
    leftovers = [p.name for p in file_path.parent.iterdir() if ".upload" in p.name]
    assert leftovers == [], f"temp upload files leaked: {leftovers}"


def test_stream_upload_respects_overwrite_false(forced_files_client):
    client, root = forced_files_client
    file_path = root / "keep.txt"

    first = client.post(
        "/api/files/upload-stream",
        data={"path": str(file_path), "overwrite": "true"},
        files={"file": ("keep.txt", b"first", "text/plain")},
    )
    assert first.status_code == 200

    conflict = client.post(
        "/api/files/upload-stream",
        data={"path": str(file_path), "overwrite": "false"},
        files={"file": ("keep.txt", b"second", "text/plain")},
    )
    assert conflict.status_code == 409
    assert file_path.read_bytes() == b"first"


def test_stream_upload_stays_under_forced_root(forced_files_client):
    """A relative path with traversal can't escape the locked root."""
    client, root = forced_files_client
    escaped = client.post(
        "/api/files/upload-stream",
        data={"path": "../../etc/evil.txt", "overwrite": "true"},
        files={"file": ("evil.txt", b"nope", "text/plain")},
    )
    assert escaped.status_code in (400, 403)


def test_stream_upload_large_file_under_cap_succeeds(forced_files_client, monkeypatch):
    """A multi-chunk payload (larger than the 1 MiB chunk) streams correctly."""
    client, root = forced_files_client
    file_path = root / "multi-chunk.bin"
    # 2.5 MiB exercises the chunked read loop across multiple iterations.
    payload = b"x" * (2 * 1024 * 1024 + 512 * 1024)

    created = client.post(
        "/api/files/upload-stream",
        data={"path": str(file_path), "overwrite": "true"},
        files={"file": ("multi-chunk.bin", payload, "application/octet-stream")},
    )
    assert created.status_code == 200
    assert file_path.stat().st_size == len(payload)
    assert file_path.read_bytes() == payload


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


def test_sensitive_env_files_blocked_read(forced_files_client):
    """Regression test for #57505: .env files must not be readable."""
    client, root = forced_files_client

    root.mkdir(parents=True, exist_ok=True)
    env_file = root / ".env"
    env_file.write_text("SECRET_KEY=abc123")

    resp = client.get("/api/files/read", params={"path": str(env_file)})
    assert resp.status_code == 403


def test_sensitive_env_files_blocked_download(forced_files_client):
    """Regression test for #57505: .env files must not be downloadable."""
    client, root = forced_files_client

    root.mkdir(parents=True, exist_ok=True)
    env_file = root / ".env"
    env_file.write_text("SECRET_KEY=abc123")

    resp = client.get("/api/files/download", params={"path": str(env_file)})
    assert resp.status_code == 403


def test_sensitive_env_suffix_variants_blocked(forced_files_client):
    """Regression: .env.<suffix> shorthand variants (e.g. .env.prod) must also be blocked."""
    client, root = forced_files_client

    root.mkdir(parents=True, exist_ok=True)
    for suffix in ("prod", "dev", "staging.local", "ci"):
        p = root / f".env.{suffix}"
        p.write_text(f"SECRET_{suffix}=abc123")
        assert client.get("/api/files/read", params={"path": str(p)}).status_code == 403
        assert client.get("/api/files/download", params={"path": str(p)}).status_code == 403


def test_sensitive_env_case_insensitive_blocked(forced_files_client):
    """Regression: .ENV / .Env.local casings must be blocked too (case-insensitive FS mounts)."""
    client, root = forced_files_client

    root.mkdir(parents=True, exist_ok=True)
    for name in (".ENV", ".Env.local", ".eNv.PROD"):
        p = root / name
        p.write_text("SECRET=abc123")
        assert client.get("/api/files/read", params={"path": str(p)}).status_code == 403
        assert client.get("/api/files/download", params={"path": str(p)}).status_code == 403


def test_envrc_blocked(forced_files_client):
    """Regression: .envrc (direnv) is a distinct basename from .env.<suffix> and
    was not caught by the old ``== ".env" or startswith(".env.")`` check."""
    client, root = forced_files_client

    root.mkdir(parents=True, exist_ok=True)
    p = root / ".envrc"
    p.write_text("export SECRET_KEY=abc123")

    listing = client.get("/api/files", params={"path": str(root)})
    assert ".envrc" not in [e["name"] for e in listing.json()["entries"]]
    assert client.get("/api/files/read", params={"path": str(p)}).status_code == 403
    assert client.get("/api/files/download", params={"path": str(p)}).status_code == 403


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
        "google_workspace_auth_contexts.json",
        "google_oauth_pending.json",
        "google_oauth.json",
        "webhook_subscriptions.json",
        "bws_cache.json",
    ):
        p = root / name
        p.write_text("SECRET=abc123")
        assert client.get("/api/files/read", params={"path": str(p)}).status_code == 403, name
        assert client.get("/api/files/download", params={"path": str(p)}).status_code == 403, name

    listing = client.get("/api/files", params={"path": str(root)})
    names = [e["name"] for e in listing.json()["entries"]]
    assert names == []


def test_git_credentials_blocked(forced_files_client):
    """Regression: .git-credentials (git's credential-store helper cache) is
    blocked by agent.file_safety; the dashboard guard must cover it too."""
    client, root = forced_files_client

    root.mkdir(parents=True, exist_ok=True)
    p = root / ".git-credentials"
    p.write_text("https://user:token@github.com\n")

    listing = client.get("/api/files", params={"path": str(root)})
    assert ".git-credentials" not in [e["name"] for e in listing.json()["entries"]]
    assert client.get("/api/files/read", params={"path": str(p)}).status_code == 403
    assert client.get("/api/files/download", params={"path": str(p)}).status_code == 403


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


def test_benign_subdir_file_still_browsable(forced_files_client):
    """Positive control: the directory-component guard must NOT over-block a
    benign subdir. A normal file under a normal subdir stays listable/readable."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    sub = root / "notes"
    sub.mkdir(parents=True, exist_ok=True)
    p = sub / "todo.txt"
    p.write_text("buy milk\n")

    listing = client.get("/api/files", params={"path": str(sub)})
    assert "todo.txt" in [e["name"] for e in listing.json()["entries"]]
    assert client.get("/api/files/read", params={"path": str(p)}).status_code == 200
