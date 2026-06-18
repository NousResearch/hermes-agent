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


def test_directory_listing_shows_broken_symlink_placeholder(forced_files_client):
    """Dangling symlinks should appear in listings with broken_link: true."""
    client, root = forced_files_client
    root.mkdir(parents=True, exist_ok=True)

    valid = root / "valid.txt"
    valid.write_text("hello")

    broken = root / "broken_link"
    nonexistent = root / "nonexistent_target"
    broken.symlink_to(str(nonexistent))

    resp = client.get("/api/files", params={"path": str(root)})
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    names = [e["name"] for e in entries]
    assert "valid.txt" in names
    assert "broken_link" in names

    broken_entry = next(e for e in entries if e["name"] == "broken_link")
    assert broken_entry["broken_link"] is True
    assert broken_entry["size"] is None
    assert broken_entry["mtime"] is None
    assert broken_entry["mime_type"] is None
    assert broken_entry["is_directory"] is False
