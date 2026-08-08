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


def test_fs_write_text_rejects_memory_md_over_configured_char_limit(client, tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    memories = home / "memories"
    memories.mkdir(parents=True)
    target = memories / "MEMORY.md"
    target.write_text("old memory\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "memory:\n"
        "  memory_char_limit: 5\n"
        "  user_char_limit: 99\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_server, "get_hermes_home", lambda: home)

    response = client.post(
        "/api/fs/write-text",
        json={"path": str(target), "content": "123456"},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "MEMORY.md" in detail
    assert "6/5 chars" in detail
    assert "memory.memory_char_limit" in detail
    assert target.read_text(encoding="utf-8") == "old memory\n"


def test_fs_write_text_rejects_profile_user_md_over_configured_char_limit(client, tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    profile_home = home / "profiles" / "work"
    memories = profile_home / "memories"
    memories.mkdir(parents=True)
    target = memories / "USER.md"
    target.write_text("old user\n", encoding="utf-8")
    (profile_home / "config.yaml").write_text(
        "memory:\n"
        "  memory_char_limit: 99\n"
        "  user_char_limit: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_server, "get_hermes_home", lambda: home)

    response = client.post(
        "/api/fs/write-text",
        json={"path": str(target), "content": "🚀🚀🚀"},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "USER.md" in detail
    assert "3/2 chars" in detail
    assert "memory.user_char_limit" in detail
    assert target.read_text(encoding="utf-8") == "old user\n"
