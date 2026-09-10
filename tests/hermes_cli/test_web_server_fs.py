import base64
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


def test_fs_download_translates_docker_volume_path_to_host(client, tmp_path, monkeypatch):
    host_workspace = tmp_path / "AI-Workspace"
    host_workspace.mkdir()
    target = host_workspace / "generated report.docx"
    target.write_bytes(b"generated-document")
    monkeypatch.setattr(
        "gateway.platforms.base._translate_docker_container_media_path",
        lambda path: target if path == Path("/workspace/projects/generated report.docx") else None,
    )

    response = client.get(
        "/api/fs/download",
        params={"path": "/workspace/projects/generated report.docx"},
    )

    assert response.status_code == 200
    assert response.content == b"generated-document"


def test_fs_download_retries_docker_translation_with_profile_terminal_scope(
    client, tmp_path, monkeypatch,
):
    target = tmp_path / "generated report.docx"
    target.write_bytes(b"isolated-desktop-document")

    from tools.terminal_scope import get_terminal_scope

    def translate(path):
        if path == Path("/workspace/projects/generated report.docx") and get_terminal_scope() is not None:
            return target
        return None

    monkeypatch.setattr(
        "gateway.platforms.base._translate_docker_container_media_path",
        translate,
    )
    monkeypatch.setattr(
        "tools.terminal_scope.build_profile_terminal_scope",
        lambda _home: {"TERMINAL_ENV": "docker", "TERMINAL_DOCKER_VOLUMES": "[]"},
    )

    response = client.get(
        "/api/fs/download",
        params={"path": "/workspace/projects/generated report.docx"},
    )

    assert response.status_code == 200
    assert response.content == b"isolated-desktop-document"


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
