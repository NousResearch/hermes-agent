"""Tests for the dashboard-managed file browser API."""

from pathlib import Path
import base64
import os
import posixpath
import types
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server, web_server_files
import hermes_cli.web_routers.files as _rt_files


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




@pytest.mark.parametrize("client_fixture", ["local_files_client", "forced_files_client"])
def test_mkdir_creates_a_folder_the_picker_can_list_and_enter(client_fixture, request):
    """The desktop remote folder picker's New folder: mkdir an absolute child of
    the folder it is browsing, then list the parent and navigate into the result."""
    client, root = request.getfixturevalue(client_fixture)
    root.mkdir(exist_ok=True)
    listed = client.get("/api/fs/list", params={"path": str(root)}).json()
    assert "error" not in listed

    created = client.post("/api/files/mkdir", json={"path": str(root / "fresh project")})

    assert created.status_code == 200
    new_dir = created.json()["path"]
    assert (root / "fresh project").is_dir()
    after = client.get("/api/fs/list", params={"path": str(root)}).json()["entries"]
    assert {"name": "fresh project", "path": new_dir, "isDirectory": True} in after
    assert client.get("/api/fs/list", params={"path": new_dir}).json() == {"entries": []}


def test_download_authenticates_via_query_token(forced_files_client):
    client, root = forced_files_client
    file_path = _seed_file(client, root, name="out/demo.mp4")
    active_content = _seed_file(client, root, name="out/page.html")

    # Drop the session header so only the ?token= query param authenticates —
    # mirrors a browser/shell-opened download that can't set the session header.
    del client.headers[web_server._SESSION_HEADER_NAME]

    ok = client.get(
        "/api/files/download",
        params={"path": str(file_path), "token": web_server._SESSION_TOKEN},
    )
    assert ok.status_code == 200
    assert ok.content == b"hello"
    assert ok.headers["content-disposition"].startswith("attachment;")

    playback = client.get(
        "/api/files/download",
        params={"path": str(file_path), "token": web_server._SESSION_TOKEN},
        headers={"Sec-Fetch-Dest": "video", "Range": "bytes=1-3"},
    )
    assert playback.status_code == 206
    assert playback.content == b"ell"
    assert playback.headers["content-disposition"].startswith("inline;")
    assert playback.headers["x-content-type-options"] == "nosniff"

    rejected = client.get(
        "/api/files/download",
        params={"path": str(active_content), "token": web_server._SESSION_TOKEN},
        headers={"Sec-Fetch-Dest": "video"},
    )
    assert rejected.status_code == 415

    assert client.get(
        "/api/files/download", params={"path": str(file_path), "token": "nope"}
    ).status_code == 401
    assert client.get(
        "/api/files/download", params={"path": str(file_path)}
    ).status_code == 401


def test_download_resolves_paths_in_the_originating_profile_session(local_files_client, monkeypatch):
    from pathlib import Path
    from hermes_state import SessionDB

    client, home = local_files_client
    monkeypatch.setattr(Path, "home", lambda: home)
    hermes_home = home / "isolated-hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    session_cwd = home / "project"
    session_cwd.mkdir()
    gateway_cwd = home / "gateway"
    gateway_cwd.mkdir()
    monkeypatch.chdir(gateway_cwd)
    artifact = session_cwd / "report.txt"
    artifact.write_bytes(b"session artifact")
    (gateway_cwd / artifact.name).write_bytes(b"wrong gateway artifact")
    for profile, sid, cwd in [("default", "origin-session", str(session_cwd)),
                              ("other", "other-session", str(gateway_cwd))]:
        db_home = hermes_home if profile == "default" else hermes_home / "profiles" / profile
        db_home.mkdir(parents=True, exist_ok=True)
        (db_home / "config.yaml").write_text("{}", encoding="utf-8")
        db = SessionDB(db_path=db_home / "state.db")
        try:
            db.create_session(sid, source="gui", cwd=cwd)
        finally:
            db.close()
    for route in ("/api/fs/download", "/api/fs/read-data-url"):
        for path in ("./report.txt", "../project/report.txt", str(artifact), artifact.as_uri()):
            response = client.get(route, params={
                "path": path, "profile": "default", "session_id": "origin-session",
            })
            assert response.status_code == 200, response.text
            data = (base64.b64decode(response.json()["dataUrl"].split(",", 1)[1])
                    if route.endswith("read-data-url") else response.content)
            assert data == artifact.read_bytes()
        for profile, session_id in (("other", "origin-session"), ("missing", "origin-session"),
                                    ("default", "missing-session"), ("default", "")):
            response = client.get(route, params={
                "path": str(artifact), "profile": profile, "session_id": session_id,
            })
            assert response.status_code == 404, response.text


def test_stream_requires_header_auth_and_supports_ranges(forced_files_client):
    client, root = forced_files_client
    file_path = _seed_file(client, root, name="out/demo.mp4")

    # Electron's main-process proxy supplies the connection credential as a
    # header. Unlike browser-visible download links, the stream endpoint must
    # not accept credentials in its URL.
    params = {"path": str(file_path)}

    full = client.get("/api/files/stream", params=params)
    assert full.status_code == 200
    assert full.content == b"hello"
    assert full.headers["content-type"] == "video/mp4"
    assert full.headers["content-disposition"].startswith("inline;")
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["x-content-type-options"] == "nosniff"

    partial = client.get(
        "/api/files/stream",
        params=params,
        headers={"Range": "bytes=1-3"},
    )
    assert partial.status_code == 206
    assert partial.content == b"ell"
    assert partial.headers["content-range"] == "bytes 1-3/5"
    assert partial.headers["content-disposition"].startswith("inline;")
    assert partial.headers["x-content-type-options"] == "nosniff"

    head = client.head("/api/files/stream", params=params)
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-length"] == "5"
    assert head.headers["x-content-type-options"] == "nosniff"

    del client.headers[web_server._SESSION_HEADER_NAME]
    assert client.get(
        "/api/files/stream",
        params={"path": str(file_path), "token": web_server._SESSION_TOKEN},
    ).status_code == 401
    assert client.get("/api/files/stream", params=params).status_code == 401


def test_stream_rejects_non_media_active_content(forced_files_client):
    client, root = forced_files_client

    for name in ("out/page.html", "out/image.svg"):
        file_path = _seed_file(client, root, name=name)
        response = client.get("/api/files/stream", params={"path": str(file_path)})
        assert response.status_code == 415


def test_query_token_does_not_authenticate_other_endpoints(forced_files_client):
    client, root = forced_files_client
    file_path = _seed_file(client, root)

    del client.headers[web_server._SESSION_HEADER_NAME]

    # The query-token escape hatch is scoped to downloads only; it must not
    # unlock the rest of the API surface.
    leaked = client.get(
        "/api/files/read",
        params={"path": str(file_path), "token": web_server._SESSION_TOKEN},
    )
    assert leaked.status_code == 401




# ---------------------------------------------------------------------------
# Streaming multipart upload (/api/files/upload-stream) — NS-501
# ---------------------------------------------------------------------------








def test_directory_listing_shows_broken_symlink_placeholder(forced_files_client, tmp_path):
    """Dangling symlinks should appear in listings with broken_link: true."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    valid = root / "valid.txt"
    valid.write_text("hello")

    broken = root / "broken_link"
    outside_broken = root / "outside_broken_link"
    try:
        broken.symlink_to(root / "nonexistent_target")
        outside_broken.symlink_to(tmp_path / "outside_nonexistent_target")
    except OSError:
        pytest.skip("filesystem does not allow symlinks")

    resp = client.get("/api/files", params={"path": str(root)})
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    names = [e["name"] for e in entries]
    assert "valid.txt" in names
    assert "broken_link" in names
    assert "outside_broken_link" in names

    broken_entry = next(e for e in entries if e["name"] == "broken_link")
    assert broken_entry["broken_link"] is True
    assert broken_entry["path"] == str(broken)
    assert broken_entry["size"] is None
    assert broken_entry["mtime"] is None
    assert broken_entry["mime_type"] is None
    assert broken_entry["is_directory"] is False

    outside_entry = next(e for e in entries if e["name"] == "outside_broken_link")
    assert outside_entry["broken_link"] is True
    assert outside_entry["path"] == str(outside_broken)


def test_managed_file_entry_reuses_symlink_stat_snapshot(tmp_path, monkeypatch):
    """A target disappearing after its first stat must not turn the listing into a 500."""
    root = tmp_path.resolve()
    target = root / "target.txt"
    target.write_text("x")
    link = root / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("filesystem does not allow symlinks")

    original_resolve = web_server.Path.resolve
    original_stat = web_server.Path.stat
    resolved_target = original_resolve(link)
    target_stat_calls = 0

    def stable_resolve(path, *args, **kwargs):
        if path == link:
            return resolved_target
        return original_resolve(path, *args, **kwargs)

    def disappearing_target_stat(path, *args, **kwargs):
        nonlocal target_stat_calls
        if path == resolved_target:
            target_stat_calls += 1
            if target_stat_calls == 2:
                raise FileNotFoundError(str(path))
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(web_server.Path, "resolve", stable_resolve)
    monkeypatch.setattr(web_server.Path, "stat", disappearing_target_stat)

    entry = web_server_files._managed_file_entry(
        web_server_files.ManagedFilesPolicy(
            default_path=root,
            locked_root=root,
            can_change_path=False,
        ),
        link,
    )

    assert entry["name"] == "link"
    assert entry["path"] == str(resolved_target)
    assert entry["size"] == 1
    assert target_stat_calls == 1


@pytest.mark.parametrize("client_fixture", ["forced_files_client", "local_files_client"])
def test_directory_listing_skips_vanished_regular_file(request, client_fixture, monkeypatch):
    client, root = request.getfixturevalue(client_fixture)
    root.mkdir(parents=True, exist_ok=True)
    vanished = root / "vanished.txt"
    vanished.write_text("gone", encoding="utf-8")
    (root / "valid.txt").write_text("still here", encoding="utf-8")
    (root / "folder").mkdir()
    original_stat = web_server_files.Path.stat

    def disappearing_stat(path, *args, **kwargs):
        if path == vanished and kwargs.get("follow_symlinks", True):
            vanished.unlink(missing_ok=True)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(web_server_files.Path, "stat", disappearing_stat)
    response = client.get("/api/files", params={"path": str(root)})

    assert response.status_code == 200
    entries = response.json()["entries"]
    assert [entry["name"] for entry in entries] == ["folder", "valid.txt"]
    assert entries[0]["is_directory"] is True
    assert entries[1]["size"] == len("still here")


@pytest.mark.parametrize("error", [PermissionError("denied"), OSError("I/O error")])
def test_directory_listing_does_not_hide_stat_errors(forced_files_client, monkeypatch, error):
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)
    target = root / "unreadable.txt"
    target.write_text("x", encoding="utf-8")
    original_stat = web_server_files.Path.stat

    def unreadable_stat(path, *args, **kwargs):
        if path == target and kwargs.get("follow_symlinks", True):
            raise error
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(web_server_files.Path, "stat", unreadable_stat)
    response = client.get("/api/files", params={"path": str(root)})
    assert response.status_code == 500
    assert "Could not stat path" in response.json()["detail"]


def test_managed_write_result_rejects_vanished_file(tmp_path):
    from fastapi import HTTPException

    policy = web_server_files.ManagedFilesPolicy(tmp_path, tmp_path, False)
    missing = tmp_path / "missing.txt"
    with pytest.raises(HTTPException) as exc_info:
        _rt_files._managed_write_result(policy, missing, str(missing))
    assert exc_info.value.status_code == 500


def test_broken_symlink_placeholder_requires_entry_inside_root(tmp_path):
    from fastapi import HTTPException

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside-link"
    try:
        outside.symlink_to(root / "missing.txt")
    except OSError:
        pytest.skip("filesystem does not allow symlinks")
    policy = web_server_files.ManagedFilesPolicy(root, root, False)
    with pytest.raises(HTTPException) as exc_info:
        web_server_files._managed_file_entry(policy, outside, skip_missing=True)
    assert exc_info.value.status_code == 403


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
            _rt_files.upload_managed_file_stream(
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
        assert client.get("/api/files/stream", params={"path": str(p)}).status_code == 403, name

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
        assert client.get("/api/files/stream", params={"path": str(p)}).status_code == 403, str(p)

    # Listing the credential dir itself yields nothing exploitable: every child
    # is filtered because the parent component is a credential dir.
    mcp_listing = client.get("/api/files", params={"path": str(mcp_dir)})
    assert [e["name"] for e in mcp_listing.json()["entries"]] == []


def test_git_branch_decodes_utf8_under_a_gbk_default_codec(tmp_path, monkeypatch):
    """#83851: the Desktop polls ``/api/fs/default-cwd``; on zh-CN Windows the serve process's default
    subprocess codec is cp936, and git's UTF-8 output (branch names, localized stderr) raised
    UnicodeDecodeError in communicate()'s reader threads on every poll. The branch must round-trip."""
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:
        pytest.skip("git not installed")
    branch = "功能/✅-修复"  # UTF-8 bytes that are illegal multibyte sequences in GBK
    subprocess.run([git, "init", "-q", str(tmp_path)], check=True)
    subprocess.run([git, "-C", str(tmp_path), "symbolic-ref", "HEAD", f"refs/heads/{branch}"], check=True)
    # subprocess resolves an unspecified text-mode codec through _text_encoding() → locale.getencoding()
    # (cp936 on zh-CN Windows); patch that seam since run_tests.sh's PYTHONUTF8=1 short-circuits locale.
    monkeypatch.setattr(subprocess, "_text_encoding", lambda: "gbk")

    assert _rt_files._fs_git_branch(str(tmp_path)) == branch


# --- Credential stores are guarded in BOTH directions -------------------------
# The read side (#57505) refuses to list/read/download credential basenames. Its
# docstring claimed the other direction was covered: "The write endpoints
# (upload/mkdir/delete) are a separate threat class handled by the write-path
# checks." Those checks are path SHAPE only (`..`, locked_root, must-be-absolute,
# regular-file, size caps) — not one of them consults the denylist. So the same
# token that cannot read ~/.hermes/.env could overwrite it (a hostile
# OPENAI_BASE_URL there sends the real key to a third party on the next provider
# call) or delete it outright.


def test_dotenv_cannot_be_overwritten_through_the_file_browser(forced_files_client):
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)
    env_path = root / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-real-key\n", encoding="utf-8")

    resp = client.post(
        "/api/fs/write-text",
        json={"path": str(env_path), "content": "OPENAI_BASE_URL=https://attacker.example\n"},
    )

    assert resp.status_code == 409, resp.text
    assert env_path.read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-real-key\n"


def test_dotenv_cannot_be_deleted_through_the_file_browser(forced_files_client):
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)
    env_path = root / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-real-key\n", encoding="utf-8")

    resp = client.request("DELETE", "/api/files", json={"path": str(env_path)})

    assert resp.status_code == 409, resp.text
    assert env_path.exists()


def test_ordinary_writes_still_work(forced_files_client):
    """The guard must not cost the feature: a non-credential file is still writable,
    creatable and deletable through the same endpoints."""
    client, root = forced_files_client

    # write-text never builds trees by design, so the parent is created first.
    assert client.post("/api/files/mkdir", json={"path": str(root / "sub")}).status_code == 200

    written = client.post(
        "/api/fs/write-text",
        json={"path": str(root / "sub" / "notes.md"), "content": "hello"},
    )
    assert written.status_code == 200, written.text
    assert (root / "sub" / "notes.md").read_text(encoding="utf-8") == "hello"

    deleted = client.request("DELETE", "/api/files", json={"path": str(root / "sub" / "notes.md")})
    assert deleted.status_code == 200, deleted.text
    assert not (root / "sub" / "notes.md").exists()


def test_ssh_backend_write_never_reaches_the_adapter_for_a_credential_path(
    monkeypatch, forced_files_client
):
    """``/api/fs/*`` routes to the profile's SSH workspace adapter when it has one, and
    that adapter resolves and writes in a single call — so a guard placed on the
    RESOLVED target would run after the write already landed. Assert the adapter is
    never asked to write a credential path at all."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    calls: list[tuple] = []

    class _RecordingBackend:
        def write_text(self, path, text, *, max_bytes=None):
            calls.append((path, text))
            return ("/remote/.env", len(text))

    monkeypatch.setattr(_rt_files, "_fs_backend", lambda profile=None: _RecordingBackend())

    resp = client.post(
        "/api/fs/write-text",
        json={"path": "/remote/.env", "content": "OPENAI_BASE_URL=https://attacker.example\n"},
    )

    assert resp.status_code == 409, resp.text
    assert calls == [], f"the remote adapter was asked to write a credential path: {calls}"


def test_write_text_refuses_to_clobber_a_live_database(forced_files_client):
    """The spot editor replaces a file wholesale. On a SQLite database that destroys it —
    and the dashboard process itself is holding state.db open, so `_serve_offline`
    refuses that on the read side while the write side had no equivalent guard.

    Asserted against a real SessionDB (a genuine live connection), not a mock: the guard
    consults the live-connection registry, so a fabricated path would not exercise it.
    """
    from hermes_state import SessionDB

    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "state.db"
    session_db = SessionDB(db_path=db_path)
    try:
        resp = client.post(
            "/api/fs/write-text",
            json={"path": str(db_path), "content": "not a database"},
        )
        assert resp.status_code == 409, resp.text
        # Untouched: still a readable store, not the text we tried to write.
        assert session_db.get_meta("__probe__") is None
    finally:
        session_db.close()


def test_upload_refuses_to_clobber_a_live_database(forced_files_client):
    """Same guard on the upload path, which can also target any writable path."""
    from hermes_state import SessionDB

    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "state.db"
    session_db = SessionDB(db_path=db_path)
    try:
        resp = client.post(
            "/api/files/upload",
            json={
                "path": str(db_path),
                "data_url": "data:application/octet-stream;base64,bm90IGEgZGF0YWJhc2U=",
            },
        )
        assert resp.status_code == 409, resp.text
        assert db_path.read_bytes().startswith(b"SQLite format 3")
    finally:
        session_db.close()


def test_managed_files_guard_is_never_narrower_than_the_canonical_guard():
    """The Files tab's credential guard must not lag behind the canonical one.

    The two hand-listed tests in this file snapshot today's names, so they stay green when
    ``agent.file_safety`` gains a credential store or directory — which is how
    ``vault/`` and ``browser-profile/`` came to be readable here while the guard's own
    comment claims it "mirrors the two canonical guards". This asserts the RELATIONSHIP
    instead of the values: every canonical basename and every canonical credential
    directory must be denied. The Files tab may be STRICTER (it also denies
    config.yaml, .git-credentials and pairing/); it must never be narrower.
    """
    from agent.file_safety import _CREDENTIAL_FILE_NAMES, _READ_DENIED_DIRS

    missing_files = sorted(
        name for name in _CREDENTIAL_FILE_NAMES
        if not _rt_files._is_sensitive_path(Path(name).parent / Path(name).name)
    )
    missing_dirs = sorted(
        name for name, _dir_msg, _file_msg in _READ_DENIED_DIRS
        if not _rt_files._is_sensitive_path(Path("hermes") / name / "anything.json")
    )

    assert not missing_files, (
        f"credential basenames the Files tab would expose: {missing_files}. Add them to "
        f"_SENSITIVE_MANAGED_FILE_BASENAMES."
    )
    assert not missing_dirs, (
        f"credential directories the Files tab would expose: {missing_dirs}. Add them to "
        f"_SENSITIVE_MANAGED_DIR_NAMES — a basename-only check still exposes their contents "
        f"once the browser descends into the subdirectory."
    )


def test_vault_directory_is_not_readable_through_the_files_tab(forced_files_client):
    """``vault.key`` + ``vault.json.enc`` side by side = plaintext, so the whole tree is one
    credential. Asserted over HTTP against a realistic layout, not just the helper."""
    client, root = forced_files_client
    vault = root / "vault"
    vault.mkdir(parents=True)
    (vault / "vault.key").write_text("KEY", encoding="utf-8")
    (vault / "vault.json.enc").write_text("CIPHERTEXT", encoding="utf-8")

    for name in ("vault.key", "vault.json.enc"):
        target = vault / name
        assert client.get("/api/files/read", params={"path": str(target)}).status_code == 403, name
        assert client.get("/api/files/download", params={"path": str(target)}).status_code == 403, name

    entries = client.get("/api/files", params={"path": str(vault)}).json()["entries"]
    assert entries == []
@pytest.mark.require_symlinks
def test_dangling_symlink_does_not_break_directory_listing(forced_files_client):
    """Regression: a dangling symlink in a directory must not 500 the listing.

    Symlinks whose target is currently unavailable (deleted file, unmounted
    external drive, or a Nix/Home Manager profile with /nix/store not mounted)
    should not prevent the rest of the directory from being browsable. The
    broken link is surfaced as an informative entry rather than crashing the
    listing or being misreported as a root escape.
    """
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    real = root / "real.txt"
    real.write_text("still here")

    try:
        (root / "dangling").symlink_to("/nix/store/nonexistent-broken-link-profile")
    except OSError:
        pytest.skip("filesystem does not allow symlinks")

    listing = client.get("/api/files", params={"path": str(root)})
    assert listing.status_code == 200

    entries = {e["name"]: e for e in listing.json()["entries"]}
    assert "real.txt" in entries
    assert entries["real.txt"]["is_directory"] is False

    assert "dangling" in entries
    dangling = entries["dangling"]
    assert dangling["is_directory"] is False
    assert dangling.get("broken_link") is True
    assert dangling["mime_type"] is None
    assert dangling["size"] is None
    assert dangling["mtime"] is None

    # Reading or downloading the broken symlink directly is still an error.
    assert client.get("/api/files/read", params={"path": str(root / "dangling")}).status_code == 404
    assert client.get("/api/files/download", params={"path": str(root / "dangling")}).status_code == 404


# ── Review repairs (#134670): F1 footprint, F2 lifecycle, F3 effective target ──

def test_recursive_delete_refuses_a_protected_descendant(forced_files_client):
    """F1: rmtree acts on the whole tree, so the parent-only sensitive check left
    protected CHILDREN deletable through their ordinary parent."""
    client, root = forced_files_client
    parent = root / "project"
    (parent / "sub").mkdir(parents=True)
    (parent / "notes.txt").write_text("keep", encoding="utf-8")
    env_child = parent / "sub" / ".env"
    env_child.write_text("OPENAI_API_KEY=sk-real\n", encoding="utf-8")

    resp = client.request("DELETE", "/api/files", json={"path": str(parent), "recursive": True})

    assert resp.status_code == 409, resp.text
    assert env_child.read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-real\n"
    assert (parent / "notes.txt").exists(), "an unprotected sibling was removed by the refused delete"


def test_recursive_delete_still_removes_ordinary_trees(forced_files_client):
    client, root = forced_files_client
    parent = root / "project"
    (parent / "sub").mkdir(parents=True)
    (parent / "notes.txt").write_text("gone", encoding="utf-8")

    resp = client.request("DELETE", "/api/files", json={"path": str(parent), "recursive": True})

    assert resp.status_code == 200, resp.text
    assert not parent.exists()


def test_recursive_delete_fails_closed_on_an_unverifiable_footprint(forced_files_client, monkeypatch):
    """F1: an unreadable subtree is a refusal, never 'an empty safe tree'."""
    client, root = forced_files_client
    parent = root / "project"
    parent.mkdir(parents=True)
    (parent / "notes.txt").write_text("keep", encoding="utf-8")

    def _unwalkable(_target):
        raise OSError("EACCES: subtree unreadable")
        yield  # pragma: no cover

    monkeypatch.setattr(_rt_files.os, "walk", _unwalkable)

    resp = client.request("DELETE", "/api/files", json={"path": str(parent), "recursive": True})

    assert resp.status_code == 409, resp.text
    assert parent.exists() and (parent / "notes.txt").exists()


def test_recursive_delete_refuses_a_live_database_descendant(forced_files_client):
    """F1/F2: the registry is keyed per file — a live tracked DB inside the tree must
    block the recursive delete of its parent, with the tree left intact."""
    from hermes_state import SessionDB

    client, root = forced_files_client
    parent = root / "project"
    (parent / "data").mkdir(parents=True)
    session_db = SessionDB(db_path=parent / "data" / "state.db")
    try:
        resp = client.request("DELETE", "/api/files", json={"path": str(parent), "recursive": True})
        assert resp.status_code == 409, resp.text
        assert (parent / "data" / "state.db").exists()
        assert (parent / "data").exists()
    finally:
        session_db.close()


def _lock_probing_os(monkeypatch, record, key):
    """Replace the route module's ``os`` with a proxy that records whether the
    connection-lifecycle registry lock is held by the calling thread."""
    import hermes_cli.sqlite_safe_read as _ssr
    real_os = _rt_files.os
    probe = types.SimpleNamespace(**vars(real_os))

    def _probe_replace(tmp, dst):
        record[key] = _ssr._live_lock._is_owned()
        return real_os.replace(tmp, dst)

    probe.replace = _probe_replace
    monkeypatch.setattr(_rt_files, "os", probe)
    return probe


def test_write_text_replaces_under_the_connection_lifecycle_lock(forced_files_client, monkeypatch):
    """F2 structural: admission and the destructive os.replace are one serialized
    operation — the write runs while the registry lock is owned by this thread."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)
    target = root / "notes.txt"
    target.write_text("old", encoding="utf-8")

    record = {}
    _lock_probing_os(monkeypatch, record, "write-text")

    resp = client.post(
        "/api/fs/write-text", json={"path": str(target), "content": "new"}
    )

    assert resp.status_code == 200, resp.text
    assert record.get("write-text") is True, "os.replace ran outside the lifecycle guard"
    assert target.read_text(encoding="utf-8") == "new"


def test_stream_upload_commits_under_the_connection_lifecycle_lock(forced_files_client, monkeypatch):
    """F2 structural: multipart staging is unguarded, but the final placement runs
    inside the guard."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)
    target = root / "upload.bin"

    record = {}
    _lock_probing_os(monkeypatch, record, "upload-commit")

    resp = client.post(
        "/api/files/upload-stream",
        data={"path": str(target), "overwrite": "true"},
        files={"file": ("upload.bin", b"payload")},
    )

    assert resp.status_code == 200, resp.text
    assert record.get("upload-commit") is True, "the streamed commit ran outside the lifecycle guard"
    assert target.read_bytes() == b"payload"


def test_delete_runs_under_the_connection_lifecycle_lock(forced_files_client, monkeypatch):
    """F2 structural: the unlink itself executes while the registry lock is held."""
    import hermes_cli.sqlite_safe_read as _ssr
    client, root = forced_files_client
    target = root / "notes.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("gone", encoding="utf-8")

    record = {}

    real_shutil = _rt_files.shutil
    real_rmtree = real_shutil.rmtree

    def _probe_rmtree(path, *args, **kwargs):
        record["delete"] = _ssr._live_lock._is_owned()
        return real_rmtree(path, *args, **kwargs)

    probe_shutil = types.SimpleNamespace(**vars(real_shutil))
    probe_shutil.rmtree = _probe_rmtree
    monkeypatch.setattr(_rt_files, "shutil", probe_shutil)

    project = root / "project"
    project.mkdir(parents=True)
    (project / "notes.txt").write_text("gone", encoding="utf-8")

    resp = client.request("DELETE", "/api/files", json={"path": str(project), "recursive": True})

    assert resp.status_code == 200, resp.text
    assert record.get("delete") is True, "rmtree ran outside the lifecycle guard"
    assert not project.exists()


class _WorkspaceBackend:
    """SshWorkspaceFs-shaped fake: ``resolve`` is the same cwd/home expansion
    ``write_text`` performs (review F3)."""

    def __init__(self, cwd):
        self._cwd = cwd
        self.writes = []

    def resolve(self, path):
        raw = str(path or "").strip()
        if raw.startswith("~"):
            raw = "/home/remote" + raw[1:]
        elif not raw.startswith("/"):
            raw = posixpath.join(self._cwd, raw)
        return posixpath.normpath(raw)

    def write_text(self, path, text, *, max_bytes=None):
        effective = self.resolve(path)
        self.writes.append(effective)
        return (effective, len(text))


def test_remote_write_policy_evaluates_the_effective_target(monkeypatch, forced_files_client):
    """F3: a relative basename is expanded against the selected workspace's cwd, so the
    policy must judge the adapter's effective target — refusal/success/refusal for the
    SAME request across protected/ordinary/protected workspaces."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)
    holder = {"backend": _WorkspaceBackend("/home/remote/work")}
    monkeypatch.setattr(_rt_files, "_fs_backend", lambda profile=None: holder["backend"])

    ok = client.post("/api/fs/write-text", json={"path": "notes.txt", "content": "hi"})
    assert ok.status_code == 200, ok.text
    assert holder["backend"].writes == ["/home/remote/work/notes.txt"]

    holder["backend"] = _WorkspaceBackend("/home/remote/.hermes/vault")
    refused = client.post("/api/fs/write-text", json={"path": "notes.txt", "content": "hi"})
    assert refused.status_code == 409, refused.text
    assert holder["backend"].writes == [], "the adapter was invoked for a protected effective target"

    home_alias = client.post("/api/fs/write-text", json={"path": "~/.hermes/.env", "content": "x"})
    assert home_alias.status_code == 409, home_alias.text
    assert holder["backend"].writes == [], "a ~-alias to a credential file reached the adapter"
