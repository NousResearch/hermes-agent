"""Locked preview and review routes share the managed-file read boundary."""

import subprocess

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    with TestClient(web_server.app) as client:
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        yield client


@pytest.mark.parametrize("locked", [True, False])
def test_preview_respects_locked_root_and_sensitive_paths(client, monkeypatch, tmp_path, locked):
    root = tmp_path / "root"
    root.mkdir()
    if locked:
        monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    else:
        monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    safe = root / "safe.txt"
    secret = root / "auth.json"
    outside = tmp_path / "outside.txt"
    for path in (safe, secret, outside):
        path.write_text("example", encoding="utf-8")
    for endpoint in ("read-text", "read-data-url", "download"):
        for path in (safe, secret, outside):
            response = client.get(f"/api/fs/{endpoint}", params={"path": str(path)})
            denied = path == secret or (locked and path == outside)
            assert response.status_code == (403 if denied else 200)
    response = client.get("/api/fs/list", params={"path": str(root)})
    names = {entry["name"] for entry in response.json()["entries"]}
    assert "safe.txt" in names
    assert "auth.json" not in names
    assert client.get("/api/fs/list", params={"path": str(tmp_path)}).status_code == (403 if locked else 200)


@pytest.mark.parametrize("locked", [True, False])
@pytest.mark.parametrize("change", ["staged", "unstaged", "deleted", "renamed"])
def test_git_reads_filter_credentials_without_losing_safe_diffs(
    client, monkeypatch, tmp_path, locked, change
):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    secret = root / ".env"
    secret.write_text("PRIVATE_EXAMPLE=old\n", encoding="utf-8")
    archive = root / "archive"
    archive.mkdir()
    (archive / "auth.json").write_text("PRIVATE_EXAMPLE=archived\n", encoding="utf-8")
    (root / "safe.txt").write_text("old\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "initial")
    (archive / "auth.json").unlink()
    archive.rmdir()
    (root / "safe.txt").write_text("SAFE_CHANGE\n", encoding="utf-8")
    if change == "deleted":
        secret.unlink()
    elif change == "renamed":
        git("mv", ".env", "auth.json")
        (root / "auth.json").write_text("PRIVATE_EXAMPLE=new\n", encoding="utf-8")
    else:
        secret.write_text("PRIVATE_EXAMPLE=new\n", encoding="utf-8")
    if change != "unstaged":
        git("add", "-A")
    if locked:
        monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    else:
        monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)

    response = client.get("/api/git/review/commit-context", params={"path": str(root)})
    assert response.status_code == 200
    diff = response.json()["diff"]
    assert "SAFE_CHANGE" in diff
    assert ("PRIVATE_EXAMPLE" in diff) is not locked
    for endpoint in ("/api/git/file-diff", "/api/git/review/diff"):
        response = client.get(endpoint, params={"path": str(root), "file": ".env", "staged": change != "unstaged"})
        assert response.status_code == (403 if locked else 200)
        # A pathspec is not a file name and must not expand into credential files.
        for file in (":(top)*", ".", "archive"):
            response = client.get(endpoint, params={"path": str(root), "file": file, "staged": change != "unstaged"})
            assert "PRIVATE_EXAMPLE" not in response.text

    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(tmp_path / "other"))
    response = client.get("/api/git/review/commit-context", params={"path": str(root)})
    assert response.status_code == 403
