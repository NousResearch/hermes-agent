import base64
import os
from pathlib import Path

import pytest

from hermes_cli import web_server
from hermes_cli.web_routers import files as _fs_routes

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


@pytest.mark.parametrize("endpoint", ["/api/fs/read-text", "/api/fs/read-data-url", "/api/fs/download"])
@pytest.mark.parametrize("relative", [".env", "auth.json", "mcp-tokens/github.json"])
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
    assert [entry["name"] for entry in response.json()["entries"]] == ["notes.txt"]


def test_fs_endpoints_require_auth(tmp_path):
    client = TestClient(web_server.app)
    target = tmp_path / "secret.txt"
    target.write_text("secret")

    list_response = client.get("/api/fs/list", params={"path": str(tmp_path)})
    read_response = client.get("/api/fs/read-text", params={"path": str(target)})
    write_response = client.post(
        "/api/fs/write-text", json={"path": str(target), "content": "nope"}
    )
    default_response = client.get("/api/fs/default-cwd")

    assert list_response.status_code == 401
    assert read_response.status_code == 401
    assert write_response.status_code == 401
    assert default_response.status_code == 401


# ---------------------------------------------------------------------------
# /api/fs/write-text sandbox (#95306): the endpoint also serves remote
# dashboard sessions, so writes must stay inside the workspace roots and away
# from credential / shell-startup targets.
# ---------------------------------------------------------------------------


@pytest.fixture
def sandboxed(monkeypatch, tmp_path):
    """Pin the default write sandbox to a fresh dir and clear root overrides.

    The sandbox is a *subdir* of ``tmp_path`` so the test's own ``tmp_path``
    sits outside it — handy for escape attempts.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(_fs_routes, "_fs_default_cwd", lambda: str(workspace))
    monkeypatch.delenv("HERMES_DASHBOARD_WRITE_ROOTS", raising=False)
    return workspace


def test_fs_write_text_writes_inside_sandbox(client, sandboxed):
    target = sandboxed / "notes.md"

    response = client.post(
        "/api/fs/write-text", json={"path": str(target), "content": "hello"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["byteSize"] == len("hello".encode("utf-8"))
    assert target.read_text(encoding="utf-8") == "hello"


def test_fs_write_text_rejects_escape_outside_sandbox(client, sandboxed, tmp_path):
    outside_dir = tmp_path / "escape"
    outside_dir.mkdir()
    target = outside_dir / "cron-job.sh"

    response = client.post(
        "/api/fs/write-text", json={"path": str(target), "content": "#!/bin/sh\nrm -rf /\n"}
    )

    assert response.status_code == 403
    assert not target.exists()


def test_fs_write_text_rejects_file_url_escape(client, sandboxed, tmp_path):
    outside_dir = tmp_path / "escape-file-url"
    outside_dir.mkdir()
    raw = (outside_dir / "authorized.txt").as_uri()

    response = client.post(
        "/api/fs/write-text", json={"path": raw, "content": "nope"}
    )

    assert response.status_code == 403


def test_fs_write_text_rejects_sensitive_targets_inside_sandbox(client, sandboxed):
    targets = [
        sandboxed / ".env",
        sandboxed / ".env.local",
        sandboxed / "config.yaml",
        sandboxed / "auth.json",
        sandboxed / ".bashrc",
        sandboxed / ".zshrc",
        sandboxed / ".ssh" / "authorized_keys",
        sandboxed / ".gnupg" / "privkey.asc",
        sandboxed / "mcp-tokens" / "server.json",
        sandboxed / "deep" / "pairing" / "code.txt",
    ]

    for target in targets:
        response = client.post(
            "/api/fs/write-text", json={"path": str(target), "content": "x"}
        )

        assert response.status_code == 403, f"{target} should be denied"
        assert not target.exists()


def test_fs_write_text_honors_configured_write_roots(client, sandboxed, monkeypatch, tmp_path):
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    root_a.mkdir()
    root_b.mkdir()
    monkeypatch.setenv(
        "HERMES_DASHBOARD_WRITE_ROOTS", os.pathsep.join([str(root_a), str(root_b)])
    )

    allowed = client.post(
        "/api/fs/write-text", json={"path": str(root_b / "allowed.txt"), "content": "y"}
    )
    denied = client.post(
        "/api/fs/write-text",
        json={"path": str(sandboxed / "elsewhere.txt"), "content": "y"},
    )

    assert allowed.status_code == 200
    assert (root_b / "allowed.txt").read_text(encoding="utf-8") == "y"
    assert denied.status_code == 403
    assert not (sandboxed / "elsewhere.txt").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "../escape/dotdot.txt",
        "..\\escape\\win-backslash.txt",
        "sub/../../escape/deep.txt",
        "C:/Windows/Temp/pwned.txt",
        "\\\\somehost\\share/pwned.txt",
    ],
)
def test_fs_write_text_rejects_escape_matrix(client, sandboxed, tmp_path, relative):
    """Traversal, backslash, Windows-absolute and UNC paths all resolve outside
    the sandbox and must be denied with nothing written to disk."""
    response = client.post(
        "/api/fs/write-text", json={"path": relative, "content": "nope"}
    )
    assert response.status_code == 403, f"{relative} should be denied"
    # No file may land anywhere the traversal pointed at.
    for candidate in (
        sandboxed / "escape",
        tmp_path / "escape",
        sandboxed / ".." / "escape",
    ):
        assert not (candidate / "dotdot.txt").exists()
        assert not (candidate / "win-backslash.txt").exists()
        assert not (candidate / "deep.txt").exists()
        assert not (candidate / "pwned.txt").exists()


def test_fs_write_text_rejects_symlink_escape(client, sandboxed, tmp_path):
    """A symlink inside the sandbox pointing outside it: _fs_path resolves the
    link, so the target lands outside every write root and is rejected."""
    outside = tmp_path / "outside"
    outside.mkdir()
    link = sandboxed / "link.txt"
    try:
        link.symlink_to(outside / "pwned.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    (outside / "pwned.txt").write_text("original", encoding="utf-8")

    response = client.post(
        "/api/fs/write-text", json={"path": str(link), "content": "nope"}
    )

    assert response.status_code == 403
    assert (outside / "pwned.txt").read_text(encoding="utf-8") == "original"


def test_fs_write_text_allows_legitimate_writes(client, sandboxed):
    """False-positive check: the sandbox must not kill the editor's normal
    workflow - new file, deep subdirectory, overwrite of an existing file."""
    new_file = sandboxed / "notes.md"
    deep = sandboxed / "a" / "b" / "c" / "deep.md"
    deep.parent.mkdir(parents=True)
    deep.write_text("old", encoding="utf-8")

    for target, body in ((new_file, "fresh"), (deep, "replaced")):
        response = client.post(
            "/api/fs/write-text", json={"path": str(target), "content": body}
        )
        assert response.status_code == 200, str(target)
        assert target.read_text(encoding="utf-8") == body
