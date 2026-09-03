import base64
import os
from pathlib import Path

import pytest

from hermes_cli import web_server

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous_auth_required is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = previous_auth_required


def test_fs_list_sorts_and_hides_noise(client, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "b.txt").write_text("b")
    (root / "a_dir").mkdir()
    (root / "a.txt").write_text("a")
    (root / "node_modules").mkdir()
    (root / ".git").mkdir()

    response = client.get("/api/fs/list", params={"path": str(root)})

    assert response.status_code == 200
    entries = response.json()["entries"]
    assert [entry["name"] for entry in entries] == ["a_dir", "a.txt", "b.txt"]
    assert entries[0] == {"name": "a_dir", "path": str(root / "a_dir"), "isDirectory": True}
    assert all(entry["name"] not in {".git", "node_modules"} for entry in entries)


def test_fs_read_data_url_rejects_over_cap(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "_FS_DATA_URL_MAX_BYTES", 3)
    target = tmp_path / "image.png"
    target.write_bytes(b"1234")

    response = client.get("/api/fs/read-data-url", params={"path": str(target)})

    assert response.status_code == 413


def test_fs_download_streams_file_without_data_url_cap(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "_FS_DATA_URL_MAX_BYTES", 3)
    target = tmp_path / "report with spaces.pdf"
    target.write_bytes(b"123456")

    response = client.get("/api/fs/download", params={"path": str(target)})

    assert response.status_code == 200
    assert response.content == b"123456"
    assert response.headers["content-type"].startswith("application/pdf")
    assert "report%20with%20spaces.pdf" in response.headers["content-disposition"]


def test_fs_download_rejects_sensitive_files(client, tmp_path):
    target = tmp_path / ".env"
    target.write_text("SECRET=1")

    response = client.get("/api/fs/download", params={"path": str(target)})

    assert response.status_code == 403


def test_fs_endpoints_require_auth(tmp_path):
    client = TestClient(web_server.app)
    target = tmp_path / "secret.txt"
    target.write_text("secret")

    list_response = client.get("/api/fs/list", params={"path": str(tmp_path)})
    read_response = client.get("/api/fs/read-text", params={"path": str(target)})
    default_response = client.get("/api/fs/default-cwd")

    assert list_response.status_code == 401
    assert read_response.status_code == 401
    assert default_response.status_code == 401


def test_fs_enforce_scope_allows_path_when_env_unset(client, tmp_path):
    """When HERMES_DASH_FS_ALLOW is unset, behavior is unrestricted (backward compat)."""
    target = tmp_path / "allowed.txt"
    target.write_text("data")

    response = client.get("/api/fs/read-text", params={"path": str(target)})
    assert response.status_code == 200
    assert response.json()["text"] == "data"


def test_fs_enforce_scope_rejects_outside_root(client, tmp_path, monkeypatch):
    """When HERMES_DASH_FS_ALLOW is set, paths outside roots are rejected."""
    monkeypatch.setenv("HERMES_DASH_FS_ALLOW", str(tmp_path))

    outside = Path("/tmp/outside_file")
    outside.write_text("leaked")

    response = client.get("/api/fs/read-text", params={"path": str(outside)})
    assert response.status_code == 403
    assert "outside allowed" in response.json()["detail"].lower()


def test_fs_enforce_scope_allows_inside_root(client, tmp_path, monkeypatch):
    """When HERMES_DASH_FS_ALLOW is set, paths inside roots are allowed."""
    monkeypatch.setenv("HERMES_DASH_FS_ALLOW", str(tmp_path))

    inside = tmp_path / "allowed.txt"
    inside.write_text("data")

    response = client.get("/api/fs/read-text", params={"path": str(inside)})
    assert response.status_code == 200
    assert response.json()["text"] == "data"


def test_fs_enforce_scope_blocks_denied_basenames(client, tmp_path, monkeypatch):
    """Denied basenames (.env, shadow, etc.) are blocked even inside allowed roots."""
    monkeypatch.setenv("HERMES_DASH_FS_ALLOW", str(tmp_path))

    env_file = tmp_path / ".env"
    env_file.write_text("SECRET=1")

    response = client.get("/api/fs/read-text", params={"path": str(env_file)})
    assert response.status_code == 403
    assert "not readable" in response.json()["detail"].lower()


def test_fs_enforce_scope_blocks_symlink_escape(client, tmp_path, monkeypatch):
    """Symlink escapes are blocked because realpath is used."""
    monkeypatch.setenv("HERMES_DASH_FS_ALLOW", str(tmp_path))

    # Create the "outside" target in a sibling directory so the symlink
    # actually escapes the allowed root.
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as _td:
        outside = Path(_td) / "secret"
        outside.write_text("leaked")

        link = tmp_path / "link"
        link.symlink_to(outside)

        response = client.get("/api/fs/read-text", params={"path": str(link)})
        assert response.status_code == 403


def test_fs_enforce_scope_multiple_roots(client, tmp_path, monkeypatch, tmpdir):
    """Multiple roots separated by os.pathsep are supported."""
    monkeypatch.setenv("HERMES_DASH_FS_ALLOW", f"{str(tmp_path)}{os.pathsep}{str(tmpdir)}")

    inside1 = tmp_path / "a.txt"
    inside1.write_text("data1")
    inside2 = Path(tmpdir) / "b.txt"
    inside2.write_text("data2")

    r1 = client.get("/api/fs/read-text", params={"path": str(inside1)})
    r2 = client.get("/api/fs/read-text", params={"path": str(inside2)})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["text"] == "data1"
    assert r2.json()["text"] == "data2"


def test_fs_enforce_scope_empty_env_is_unrestricted(client, tmp_path, monkeypatch):
    """Empty HERMES_DASH_FS_ALLOW means unrestricted (backward compat)."""
    monkeypatch.setenv("HERMES_DASH_FS_ALLOW", "")

    target = tmp_path / "file.txt"
    target.write_text("data")

    response = client.get("/api/fs/read-text", params={"path": str(target)})
    assert response.status_code == 200
