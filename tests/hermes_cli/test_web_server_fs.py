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


def test_fs_download_rejects_sensitive_files(client, tmp_path):
    target = tmp_path / "auth.json"
    target.write_text("SECRET=1")

    response = client.get("/api/fs/download", params={"path": str(target)})

    assert response.status_code == 403


@pytest.mark.parametrize("endpoint", ["/api/fs/read-text", "/api/fs/read-data-url", "/api/fs/download"])
@pytest.mark.parametrize("relative", ["auth.json", "mcp-tokens/github.json"])
def test_fs_readers_reject_sensitive_paths(client, tmp_path, endpoint, relative):
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("SECRET=1")

    response = client.get(endpoint, params={"path": str(target)})

    assert response.status_code == 403
    assert "SECRET" not in response.text


def test_fs_list_hides_sensitive_entries(client, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text("SECRET=1")
    (root / "auth.json").write_text("{}")
    (root / "mcp-tokens").mkdir()
    (root / "notes.txt").write_text("ok")

    response = client.get("/api/fs/list", params={"path": str(root)})

    assert response.status_code == 200
    assert [entry["name"] for entry in response.json()["entries"]] == [".env", "notes.txt"]


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


def _project_with_env_files(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text("SECRET=1")
    (root / ".env.local").write_text("LOCAL=1")
    (root / ".envrc").write_text("RC=1")
    (root / "auth.json").write_text("{}")
    (root / "notes.txt").write_text("ok")
    return root


def test_fs_list_shows_project_env_files(client, tmp_path):
    """Regression for #121755: remote Desktop tree reads through /api/fs/list,
    so project .env* files must be enumerated (auth.json stays hidden)."""
    root = _project_with_env_files(tmp_path)

    response = client.get("/api/fs/list", params={"path": str(root)})

    assert response.status_code == 200
    names = [entry["name"] for entry in response.json()["entries"]]
    assert ".env" in names
    assert ".env.local" in names
    assert ".envrc" in names
    assert "notes.txt" in names
    assert "auth.json" not in names


@pytest.mark.parametrize("endpoint", ["/api/fs/read-text", "/api/fs/read-data-url", "/api/fs/download"])
@pytest.mark.parametrize("relative", [".env", ".env.local", ".envrc"])
def test_fs_readers_open_project_env_files(client, tmp_path, endpoint, relative):
    """Follow-up from #121755 reporter Alorse: listing alone is not enough -
    opening a project .env must not 403 via _fs_regular_file."""
    target = tmp_path / "project" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("SECRET=1")

    response = client.get(endpoint, params={"path": str(target)})

    assert response.status_code == 200
    if endpoint == "/api/fs/read-text":
        assert "SECRET" in response.text


@pytest.mark.parametrize("relative", ["auth.json", "config.yaml", "mcp-tokens/github.json"])
def test_fs_readers_still_block_non_env_sensitive(client, tmp_path, relative):
    """#57505 guard stays: Hermes credential stores remain unreadable via /api/fs."""
    target = tmp_path / "project" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("SECRET=1")

    response = client.get("/api/fs/read-text", params={"path": str(target)})

    assert response.status_code == 403
    assert "SECRET" not in response.text


def test_fs_project_env_still_blocked_under_hermes_home(client):
    """The gateway's own .env stays blocked even though project .env files open."""
    from hermes_cli.web_routers import files as fs_routes

    home = Path(fs_routes.get_hermes_home())
    assert home.is_dir()
    target = home / ".env"
    target.write_text("SECRET=1")

    try:
        response = client.get("/api/fs/read-text", params={"path": str(target)})
        assert response.status_code == 403
        assert "SECRET" not in response.text

        listed = client.get("/api/fs/list", params={"path": str(home)})
        assert listed.status_code == 200
        assert ".env" not in [entry["name"] for entry in listed.json()["entries"]]
    finally:
        target.unlink(missing_ok=True)
