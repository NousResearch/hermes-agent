import base64
from pathlib import Path

import pytest

from hermes_cli import web_server
from hermes_cli.web_routers import files as files_router

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


def test_fs_read_data_url_enforces_smaller_requested_cap(client, tmp_path):
    target = tmp_path / "image.png"
    target.write_bytes(b"1234")

    response = client.get(
        "/api/fs/read-data-url", params={"path": str(target), "max_bytes": 3}
    )

    assert response.status_code == 413


def test_fs_read_data_url_rejects_sensitive_files(client, tmp_path):
    target = tmp_path / ".env"
    target.write_text("SECRET=1")

    response = client.get("/api/fs/read-data-url", params={"path": str(target)})

    assert response.status_code == 403


def test_fs_read_data_url_rejects_symlink_outside_source_directory(client, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    markdown = docs / "report.md"
    markdown.write_text("# Report")
    outside = tmp_path / "outside.svg"
    outside.write_text("<svg/>")
    linked_image = docs / "chart.svg"
    linked_image.symlink_to(outside)

    response = client.get(
        "/api/fs/read-data-url",
        params={"path": str(linked_image), "relative_to_file": str(markdown)},
    )

    assert response.status_code == 403


def test_fs_read_data_url_rejects_symlink_swap_before_open(client, tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    markdown = docs / "report.md"
    markdown.write_text("# Report")
    image = docs / "chart.svg"
    image.write_text('<svg id="inside"/>')
    outside = tmp_path / "outside.svg"
    outside.write_text('<svg id="outside"/>')
    real_open = files_router.os.open
    swapped = False

    def swapping_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path) == image and not swapped:
            swapped = True
            image.unlink()
            image.symlink_to(outside)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(files_router.os, "open", swapping_open)

    response = client.get(
        "/api/fs/read-data-url",
        params={"path": str(image), "relative_to_file": str(markdown)},
    )

    assert response.status_code == 403


def test_fs_read_data_url_rejects_source_directory_swap_before_open(client, tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    markdown = docs / "report.md"
    markdown.write_text("# Original")
    image = docs / "chart.svg"
    image.write_text('<svg id="inside"/>')
    replacement = tmp_path / "replacement-docs"
    replacement.mkdir()
    (replacement / "report.md").write_text("# Replacement")
    (replacement / "chart.svg").write_text('<svg id="outside"/>')
    original = tmp_path / "original-docs"
    real_open = files_router.os.open
    swapped = False

    def swapping_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path) == image and not swapped:
            swapped = True
            docs.rename(original)
            replacement.rename(docs)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(files_router.os, "open", swapping_open)

    response = client.get(
        "/api/fs/read-data-url",
        params={"path": str(image), "relative_to_file": str(markdown)},
    )

    assert response.status_code == 403


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
