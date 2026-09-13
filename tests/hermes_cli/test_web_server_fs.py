import base64
import sys
from pathlib import Path

import pytest

from hermes_cli import web_server

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


def _list_entries(client, path) -> list:
    response = client.get("/api/fs/list", params={"path": str(path)})
    assert response.status_code == 200, response.text
    return response.json()["entries"]


# Creating symlinks needs elevation (or Developer Mode) on Windows.
requires_symlinks = pytest.mark.skipif(
    sys.platform == "win32", reason="Symlinks require elevated privileges on Windows"
)


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


@requires_symlinks
def test_fs_list_reports_symlinked_directories_as_directories(client, tmp_path):
    """A symlink pointing at a directory must be expandable in the file tree.

    The remote (SSH/URL) Files panel and any client that lists through
    /api/fs/list decides whether a row is expandable from `isDirectory`. A
    symlinked directory reported as a file renders as a dead leaf — the user
    cannot descend into it (e.g. a `~/code -> /mnt/data/code` checkout root).
    """
    root = tmp_path / "project"
    real_dir = root / "real_dir"
    real_dir.mkdir(parents=True)
    (root / "link_to_dir").symlink_to(real_dir)
    (root / "link_to_file").symlink_to(root / "a.txt")
    (root / "a.txt").write_text("a")
    (root / "broken_link").symlink_to(root / "does-not-exist")

    entries = {entry["name"]: entry for entry in _list_entries(client, root)}

    assert entries["real_dir"]["isDirectory"] is True
    assert entries["link_to_dir"]["isDirectory"] is True

    # A symlink to a FILE stays a file — the fix must not mark everything a dir.
    assert entries["link_to_file"]["isDirectory"] is False
    # A dangling symlink is not a directory (and must not raise).
    assert entries["broken_link"]["isDirectory"] is False


@requires_symlinks
def test_fs_list_keeps_symlinked_directories_sorted_with_directories(client, tmp_path):
    root = tmp_path / "project"
    real_dir = root / "zzz_real"
    real_dir.mkdir(parents=True)
    (root / "aaa_link").symlink_to(real_dir)
    (root / "mmm_file.txt").write_text("m")

    names = [entry["name"] for entry in _list_entries(client, root)]

    # Directory-ness drives the grouping, so the symlinked dir sorts above files.
    assert names == ["aaa_link", "zzz_real", "mmm_file.txt"]


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
