import base64
from contextlib import nullcontext
from pathlib import Path

import pytest

from hermes_cli import ssh_workspace_fs, web_server

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


def test_fs_routes_workspace_reads_to_ssh_backend(client, monkeypatch):
    calls = []

    class FakeSshFs:
        cwd = "/srv/repos"

        def list_dir(self, path, _hidden_names):
            calls.append(("list", path))
            return {
                "entries": [
                    {
                        "name": "project",
                        "path": "/srv/repos/project",
                        "isDirectory": True,
                    }
                ]
            }

        def git_branch(self, path):
            calls.append(("branch", path))
            return "main"

        def git_root(self, path):
            calls.append(("git-root", path))
            return "/srv/repos/project"

        def read_bytes(self, path, *, max_bytes, read_limit=None):
            calls.append(("read", path, max_bytes, read_limit))
            return b"hello", 5, path

        def inspect_file(self, path):
            calls.append(("inspect", path))
            return path, 5

        def stream_file(self, path):
            calls.append(("stream", path))
            return iter((b"hello",))

        def write_text(self, path, content, *, max_bytes):
            calls.append(("write", path, content, max_bytes))
            return path, len(content.encode("utf-8"))

    backend = FakeSshFs()
    monkeypatch.setattr(web_server, "_fs_backend", lambda profile=None: backend, raising=False)

    listing = client.get(
        "/api/fs/list",
        params={"path": "/srv/repos", "profile": "remote-dev"},
    )
    default_cwd = client.get(
        "/api/fs/default-cwd",
        params={"profile": "remote-dev"},
    )
    read_text = client.get(
        "/api/fs/read-text",
        params={"path": "/srv/repos/project/README.md", "profile": "remote-dev"},
    )
    read_data = client.get(
        "/api/fs/read-data-url",
        params={"path": "/srv/repos/project/logo.png", "profile": "remote-dev"},
    )
    write_text = client.post(
        "/api/fs/write-text",
        params={"profile": "remote-dev"},
        json={"path": "/srv/repos/project/README.md", "content": "updated"},
    )
    git_root = client.get(
        "/api/fs/git-root",
        params={"path": "/srv/repos/project/src", "profile": "remote-dev"},
    )
    download = client.get(
        "/api/fs/download",
        params={"path": "/srv/repos/project/report.txt", "profile": "remote-dev"},
    )

    assert listing.status_code == 200
    assert listing.json() == {
        "entries": [
            {
                "name": "project",
                "path": "/srv/repos/project",
                "isDirectory": True,
            }
        ]
    }
    assert default_cwd.status_code == 200
    assert default_cwd.json() == {"cwd": "/srv/repos", "branch": "main"}
    assert read_text.json()["text"] == "hello"
    assert read_data.json()["dataUrl"] == "data:image/png;base64,aGVsbG8="
    assert write_text.json() == {"ok": True, "path": "/srv/repos/project/README.md", "byteSize": 7}
    assert git_root.json() == {"root": "/srv/repos/project"}
    assert download.content == b"hello"
    assert calls == [
        ("list", "/srv/repos"),
        ("branch", "/srv/repos"),
        (
            "read",
            "/srv/repos/project/README.md",
            web_server._FS_TEXT_SOURCE_MAX_BYTES,
            web_server._FS_TEXT_PREVIEW_MAX_BYTES,
        ),
        ("read", "/srv/repos/project/logo.png", web_server._FS_DATA_URL_MAX_BYTES, None),
        ("write", "/srv/repos/project/README.md", "updated", web_server._FS_TEXT_WRITE_MAX_BYTES),
        ("git-root", "/srv/repos/project/src"),
        ("inspect", "/srv/repos/project/report.txt"),
        ("stream", "/srv/repos/project/report.txt"),
    ]


def test_fs_download_rejects_sensitive_remote_path_before_streaming(client, monkeypatch):
    calls = []

    class FakeSshFs:
        def inspect_file(self, path):
            calls.append(("inspect", path))
            return "/srv/repos/project/.env", 9

        def stream_file(self, _path):
            raise AssertionError("sensitive remote content must not be streamed")

    monkeypatch.setattr(web_server, "_fs_backend", lambda profile=None: FakeSshFs(), raising=False)

    response = client.get("/api/fs/download", params={"path": "/srv/repos/project/safe-link", "profile": "remote-dev"})

    assert response.status_code == 403
    assert calls == [("inspect", "/srv/repos/project/safe-link")]


def test_fs_download_streams_remote_file_without_text_source_cap(client, monkeypatch):
    calls = []
    monkeypatch.setattr(web_server, "_FS_TEXT_SOURCE_MAX_BYTES", 3)

    class FakeSshFs:
        def inspect_file(self, path):
            calls.append(("inspect", path))
            return "/srv/repos/project/report.pdf", 6

        def stream_file(self, path):
            calls.append(("stream", path))
            return iter((b"123", b"456"))

    monkeypatch.setattr(web_server, "_fs_backend", lambda profile=None: FakeSshFs(), raising=False)

    response = client.get("/api/fs/download", params={"path": "/srv/repos/project/report.pdf", "profile": "remote-dev"})

    assert response.status_code == 200
    assert response.content == b"123456"
    assert calls == [
        ("inspect", "/srv/repos/project/report.pdf"),
        ("stream", "/srv/repos/project/report.pdf"),
    ]


def test_fs_backend_resolves_ssh_connection_from_profile_env(monkeypatch):
    captured = {}
    expected = object()
    monkeypatch.setattr(web_server, "_config_profile_scope", lambda _profile: nullcontext())
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {"terminal": {"backend": "ssh", "cwd": "/srv/repos"}},
    )
    monkeypatch.setattr(
        web_server,
        "load_env",
        lambda: {
            "TERMINAL_SSH_HOST": "ssh.example",
            "TERMINAL_SSH_USER": "dev",
            "TERMINAL_SSH_PORT": "2222",
        },
    )

    def fake_factory(profile_key, terminal):
        captured.update(profile_key=profile_key, terminal=terminal)
        return expected

    monkeypatch.setattr(ssh_workspace_fs, "get_ssh_workspace_fs", fake_factory)

    assert web_server._fs_backend("Remote-Dev") is expected
    assert captured == {
        "profile_key": "remote-dev",
        "terminal": {
            "backend": "ssh",
            "cwd": "/srv/repos",
            "ssh_host": "ssh.example",
            "ssh_user": "dev",
            "ssh_port": "2222",
        },
    }


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
