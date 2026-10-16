import base64
import urllib.parse
import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from hermes_cli import ssh_workspace_fs, web_server
from hermes_cli.web_routers import files as file_routes

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


@pytest.fixture
def docker_volumes(monkeypatch, tmp_path):
    """Docker backend env whose ``/workspace`` is bind-mounted to a fresh host dir.

    Returns the host dir; writes there are what the agent's ``/workspace/...``
    paths must resolve to."""
    host = tmp_path / "host-workspace"
    host.mkdir()
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{host}:/workspace"]))
    monkeypatch.delenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", raising=False)
    monkeypatch.delenv("TERMINAL_CONTAINER_PERSISTENT", raising=False)
    return host


def _no_scope_builder(_home):
    return {"TERMINAL_ENV": "docker", "TERMINAL_DOCKER_VOLUMES": "[]"}


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

        def write_text(self, path, content, *, max_bytes):
            calls.append(("write", path, content, max_bytes))
            return path, len(content.encode("utf-8"))

        def inspect_file(self, path):
            calls.append(("inspect", path))
            return path, 5

        def stream_file(self, path):
            calls.append(("stream", path))
            yield b"hello"

    backend = FakeSshFs()
    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: backend)

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
            file_routes._FS_TEXT_SOURCE_MAX_BYTES,
            file_routes._FS_TEXT_PREVIEW_MAX_BYTES,
        ),
        ("read", "/srv/repos/project/logo.png", web_server._FS_DATA_URL_MAX_BYTES, None),
        ("write", "/srv/repos/project/README.md", "updated", file_routes._FS_TEXT_WRITE_MAX_BYTES),
        ("git-root", "/srv/repos/project/src"),
        ("inspect", "/srv/repos/project/report.txt"),
        ("stream", "/srv/repos/project/report.txt"),
    ]


def test_fs_backend_resolves_ssh_connection_from_profile_env(monkeypatch):
    captured = {}
    expected = object()
    monkeypatch.setattr(file_routes, "_profile_scope", lambda _profile: nullcontext())
    monkeypatch.setattr(
        file_routes,
        "load_config",
        lambda: {"terminal": {"backend": "ssh", "cwd": "/srv/repos"}},
    )
    monkeypatch.setattr(
        file_routes,
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

    assert file_routes._fs_backend("Remote-Dev") is expected
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


def test_fs_download_rejects_sensitive_resolved_ssh_path_before_streaming(client, monkeypatch):
    class FakeSshFs:
        def inspect_file(self, _path):
            return "/srv/repos/.env", 8

        def stream_file(self, _path):
            pytest.fail("sensitive SSH files must not be streamed")

    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: FakeSshFs())

    response = client.get(
        "/api/fs/download",
        params={"path": "/srv/repos/safe-link", "profile": "remote-dev"},
    )

    assert response.status_code == 403


def test_fs_download_streams_ssh_files_without_preview_cap(client, monkeypatch):
    class FakeSshFs:
        def inspect_file(self, _path):
            return "/srv/repos/report.txt", 6

        def stream_file(self, _path):
            yield b"123"
            yield b"456"

    monkeypatch.setattr(file_routes, "_FS_TEXT_SOURCE_MAX_BYTES", 3)
    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: FakeSshFs())

    response = client.get(
        "/api/fs/download",
        params={"path": "/srv/repos/report.txt", "profile": "remote-dev"},
    )

    assert response.status_code == 200
    assert response.content == b"123456"


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


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("endpoint", ["/api/fs/download", "/api/fs/read-data-url"])
@pytest.mark.parametrize("prefix", ["/", "/cygdrive/", "/mnt/"])
def test_fs_media_shell_paths_deliver_files_and_preserve_access_checks(client, tmp_path, endpoint, prefix):
    """MEDIA may reuse the path a successful Git Bash command used."""
    target = tmp_path / "checklist with spaces #1.txt"
    target.write_bytes(b"Weekend checklist\n[ ] Take a walk\n")
    sensitive = tmp_path / ".env"
    sensitive.write_bytes(b"SECRET=must-not-be-delivered")

    def shell_path(path):
        native = path.as_posix()
        return f"{prefix}{native[0].lower()}{native[2:]}"

    for path in (shell_path(target), str(target), target.as_uri()):
        response = client.get(endpoint, params={"path": path})
        assert response.status_code == 200, response.text
        data = (base64.b64decode(response.json()["dataUrl"].split(",", 1)[1])
                if endpoint.endswith("read-data-url") else response.content)
        assert data == target.read_bytes()

    for path, expected in ((sensitive, 403), (tmp_path / "missing.txt", 404)):
        response = client.get(endpoint, params={"path": shell_path(path)})
        assert response.status_code == expected, response.text
        assert "must-not-be-delivered" not in response.text

    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    SessionDB(db_path=get_hermes_home() / "state.db").close()  # read-only lookups need an initialized store
    invalid_session = client.get(endpoint, params={
        "path": shell_path(target), "profile": "default", "session_id": "missing-session",
    })
    assert invalid_session.status_code == 404, invalid_session.text
    client.headers.pop(web_server._SESSION_HEADER_NAME)
    assert client.get(endpoint, params={"path": shell_path(target)}).status_code == 401


@pytest.mark.platforms("posix")
def test_fs_media_posix_paths_are_not_interpreted_as_windows_drives():
    from hermes_cli.web_server_files import _fs_path

    for path in ("/c/Users/example/checklist.txt", "/cygdrive/c/checklist.txt", "/mnt/c/checklist.txt"):
        assert _fs_path(path) == Path(path).resolve()


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
    default_response = client.get("/api/fs/default-cwd")

    assert list_response.status_code == 401
    assert read_response.status_code == 401
    assert default_response.status_code == 401


def test_fs_read_text_decodes_double_encoded_non_ascii_path(client, tmp_path):
    target = tmp_path / "半导体行业运营分享会纪要.md"
    target.write_text("hello")

    # One percent-encoding on top of HTTP's own query decoding: the server
    # receives the literal "%E5%8D%8A..." string (issue #103425).
    double_encoded = urllib.parse.quote(str(target))

    response = client.get("/api/fs/read-text", params={"path": double_encoded})

    assert response.status_code == 200
    body = response.json()
    assert body["path"] == str(target)
    assert body["text"] == "hello"


def test_fs_read_text_keeps_literal_percent_filename_verbatim(client, tmp_path):
    target = tmp_path / "50%25off.md"
    target.write_text("literal percent name")

    response = client.get("/api/fs/read-text", params={"path": str(target)})

    assert response.status_code == 200
    assert response.json()["text"] == "literal percent name"


def test_fs_write_text_never_decodes_percent_path(client, tmp_path):
    sibling = tmp_path / "report v2.md"
    sibling.write_text("existing")

    response = client.post(
        "/api/fs/write-text",
        json={"path": str(tmp_path / "report%20v2.md"), "content": "new"},
    )

    assert response.status_code == 200
    assert sibling.read_text() == "existing"
    assert (tmp_path / "report%20v2.md").read_text() == "new"
@pytest.mark.parametrize("endpoint", ["read-text", "read-data-url", "download"])
def test_fs_reads_translate_docker_workspace_path_to_host(
    client, docker_volumes, monkeypatch, endpoint,
):
    """#85238: an agent in a docker container emits /workspace/... paths; the
    authenticated fs API runs on the host gateway and must resolve them through
    the same docker volume translator MEDIA delivery uses."""
    target = docker_volumes / "generated report.txt"
    target.write_text("container output", encoding="utf-8")

    response = client.get(f"/api/fs/{endpoint}", params={"path": "/workspace/generated report.txt"})

    assert response.status_code == 200
    if endpoint == "read-text":
        assert response.json()["text"] == "container output"
    elif endpoint == "read-data-url":
        expected = base64.b64encode(b"container output").decode("ascii")
        assert response.json()["dataUrl"] == f"data:text/plain;base64,{expected}"
    else:
        assert response.text == "container output"


def test_fs_reads_translate_after_binding_profile_terminal_scope(client, tmp_path, monkeypatch):
    """`serve --isolated` reaches these routes with no terminal scope bound, so
    the first translation attempt sees the launch env only; the retry under the
    profile's complete terminal policy must find the mount (from #104992)."""
    target = tmp_path / "isolated report.txt"
    target.write_text("isolated output", encoding="utf-8")

    from tools.terminal_scope import get_terminal_scope

    def translate(path, session_key="", *, quiet=False):
        if path == Path("/workspace/isolated report.txt") and get_terminal_scope() is not None:
            return target
        return None

    monkeypatch.setattr(
        "gateway.platforms.base._translate_docker_container_media_path", translate,
    )
    monkeypatch.setattr("tools.terminal_scope.build_profile_terminal_scope", _no_scope_builder)
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    response = client.get("/api/fs/read-text", params={"path": "/workspace/isolated report.txt"})

    assert response.status_code == 200
    assert response.json()["text"] == "isolated output"
    assert get_terminal_scope() is None  # the retry's scope never leaks


def test_fs_read_untranslated_container_path_names_the_agent_sandbox(
    client, docker_volumes,
):
    """A container-shaped path that fails translation under the docker backend
    must 404 with the honest sandbox reason, not a bare missing-file 404."""
    response = client.get("/api/fs/read-text", params={"path": "/workspace/never-written.txt"})

    assert response.status_code == 404
    assert "agent sandbox" in response.json()["detail"]


def test_fs_list_translates_docker_workspace_path_to_host(client, docker_volumes):
    (docker_volumes / "report.txt").write_text("one", encoding="utf-8")
    (docker_volumes / "nested").mkdir()

    response = client.get("/api/fs/list", params={"path": "/workspace"})

    assert response.status_code == 200
    entries = response.json()["entries"]
    assert [(entry["name"], entry["isDirectory"]) for entry in entries] == [
        ("nested", True), ("report.txt", False),
    ]
    # Children carry the host path so the browser's next read/list works.
    assert entries[0]["path"] == str(docker_volumes / "nested")
    assert entries[1]["path"] == str(docker_volumes / "report.txt")


def test_fs_download_maps_root_home_to_persistent_sandbox(client, monkeypatch, tmp_path):
    """#85238 Discord repro: a persistent docker agent writes /root/fileIwant.zip;
    the host copy lives at <sandbox>/docker/default/home. The download click must
    serve that file, not stat the gateway's own /root ("File is not readable")."""
    sandbox = tmp_path / "sandboxes"
    home = sandbox / "docker" / "default" / "home"
    home.mkdir(parents=True)
    (home / "fileIwant.zip").write_bytes(b"PK\x03\x04zip")
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(sandbox))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)

    response = client.get("/api/fs/download", params={"path": "/root/fileIwant.zip"})

    assert response.status_code == 200
    assert response.content == b"PK\x03\x04zip"


@pytest.mark.platforms("windows")
def test_agent_fs_path_translates_before_host_normalization(monkeypatch, tmp_path):
    """On a Windows gateway ``_fs_path("/workspace/x")`` anchors to the current
    drive (``D:\\workspace\\x``) and could serve an unrelated host file; the
    container path must reach the translator verbatim, before normalization."""
    from hermes_cli import web_server_files

    mapped = tmp_path / "x.txt"
    seen = []

    def translate(candidate, session_key="", *, quiet=False):
        seen.append(candidate.as_posix())
        return mapped

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr("gateway.platforms.base._translate_docker_container_media_path", translate)

    def boom(*_a, **_k):
        raise AssertionError("_fs_path must not run when translation succeeds")

    monkeypatch.setattr(web_server_files, "_fs_path", boom)

    assert web_server_files._agent_fs_path("/workspace/x.txt") == mapped
    assert seen == ["/workspace/x.txt"]


@pytest.mark.platforms("windows")
def test_agent_fs_path_unmapped_container_path_404s_on_windows(monkeypatch):
    """A container path that misses translation must not fall back to the drive."""
    from fastapi import HTTPException

    from hermes_cli import web_server_files

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)

    def boom(*_a, **_k):
        raise AssertionError("an unmapped container path must not reach _fs_path")

    monkeypatch.setattr(web_server_files, "_fs_path", boom)
    with pytest.raises(HTTPException) as exc:
        web_server_files._agent_fs_path("/workspace/x.txt")
    assert exc.value.status_code == 404
    assert "agent sandbox" in exc.value.detail


@pytest.mark.platforms("posix")
def test_readable_host_path_wins_over_root_sandbox(client, monkeypatch, tmp_path):
    """Host-first on POSIX: a readable host file is served even when the docker
    sandbox has a file at the same container path."""
    sandbox = tmp_path / "sandboxes"
    host_dir = tmp_path / "host"
    host_dir.mkdir()
    (host_dir / "f.txt").write_text("HOST", encoding="utf-8")
    sandbox_ws = tmp_path / "mount"
    sandbox_ws.mkdir()
    (sandbox_ws / "f.txt").write_text("SANDBOX", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(sandbox))
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{sandbox_ws}:{host_dir}"]))

    response = client.get("/api/fs/read-text", params={"path": str(host_dir / "f.txt")})

    assert response.status_code == 200
    assert response.json()["text"] == "HOST"


def test_local_backend_ignores_leftover_docker_volumes(client, monkeypatch, tmp_path):
    """terminal.backend=local with docker_volumes still configured: no translation."""
    real = tmp_path / "real"
    alt = tmp_path / "alt"
    real.mkdir()
    alt.mkdir()
    (alt / "x.txt").write_text("MOUNTED-COPY", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{alt}:{real}"]))

    response = client.get("/api/fs/read-text", params={"path": str(real / "x.txt")})

    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"


def test_fs_download_profile_uses_that_profiles_sandbox(client, monkeypatch, tmp_path):
    """``profile=work`` must resolve /root/... in the work profile's sandbox, not the
    active profile's."""
    from hermes_cli import profiles

    default_home = tmp_path / "default-home"
    work_home = default_home / "profiles" / "work"
    for home in (default_home, work_home):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            "terminal:\n  backend: docker\n  container_persistent: true\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.delenv("TERMINAL_SANDBOX_DIR", raising=False)
    monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    from tools.environments.path_utils import sanitize_task_id_for_path

    (default_home / "sandboxes/docker/default/home").mkdir(parents=True)
    (default_home / "sandboxes/docker/default/home/f.txt").write_bytes(b"DEFAULT")
    work_dir = work_home / "sandboxes/docker" / sanitize_task_id_for_path("profile:work") / "home"
    work_dir.mkdir(parents=True)
    (work_dir / "f.txt").write_bytes(b"WORK")

    work = client.get("/api/fs/download", params={"path": "/root/f.txt", "profile": "work"})
    default = client.get("/api/fs/download", params={"path": "/root/f.txt"})

    assert work.status_code == 200, work.text
    assert work.content == b"WORK"
    assert default.content == b"DEFAULT"


def test_fs_reads_accept_profile_current_for_host_files(client, tmp_path):
    """``profile=current`` means the request's own profile, as elsewhere in the dashboard."""
    target = tmp_path / "plain.txt"
    target.write_text("host", encoding="utf-8")

    for selector in ("current", "", "CURRENT"):
        read = client.get("/api/fs/read-text", params={"path": str(target), "profile": selector})
        listing = client.get("/api/fs/list", params={"path": str(tmp_path), "profile": selector})
        assert read.status_code == 200, (selector, read.text)
        assert read.json()["text"] == "host"
        assert listing.status_code == 200, (selector, listing.text)


def test_fs_read_plain_host_path_logs_no_docker_media_warning(client, docker_volumes, tmp_path, caplog):
    """fs routes see ordinary host paths constantly; misses are not MEDIA warnings."""
    plain = tmp_path / "plain.txt"
    plain.write_text("x", encoding="utf-8")
    with caplog.at_level("WARNING", logger="gateway.platforms.base"):
        client.get("/api/fs/read-text", params={"path": str(plain)})
        client.get("/api/fs/read-text", params={"path": str(tmp_path / "missing.txt")})
        client.get("/api/fs/list", params={"path": str(tmp_path)})
    assert not [r for r in caplog.records if "Docker MEDIA path" in r.getMessage()]


def test_agent_fs_path_leaves_host_paths_to_fs_path(monkeypatch, tmp_path):
    from hermes_cli import web_server_files

    def translate(*_a, **_k):
        raise AssertionError("relative/UNC/drive paths are not container paths")

    monkeypatch.setattr("gateway.platforms.base._translate_docker_container_media_path", translate)
    target = tmp_path / "note.txt"
    target.write_text("x")

    assert web_server_files._agent_fs_path("note.txt", cwd=str(tmp_path)) == target.resolve()
    assert web_server_files._agent_container_path("//server/share/x") is None
    assert web_server_files._agent_container_path("C:/data/x") is None
