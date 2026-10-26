import base64
import os
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


def test_a_browser_download_reads_the_ssh_workspace_file_it_listed(client, monkeypatch, tmp_path):
    """The browser lists an SSH workspace through ``/api/fs/list``. Its Download checks the file
    with a HEAD, then hands the browser the URL a download ticket authorizes (an anchor cannot send
    the session header). Both read the file on the SSH machine, as native Desktop's download does,
    never a file at the same path on the gateway's own disk. The ticket is single-use and dies
    with its profile."""
    from hermes_cli import web_server_file_tickets as file_tickets
    from hermes_constants import mark_named_profile_deleted

    # The Webapp: the browser's file routes (download tickets) exist only there.
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "remote-dev"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    same_path = tmp_path / "srv" / "report.txt"
    same_path.parent.mkdir()
    same_path.write_bytes(b"gateway copy")
    remote_only = str(tmp_path / "srv" / "only-remote.txt")
    streamed = []

    class FakeSshFs:
        def inspect_file(self, path):
            return path, len(b"remote copy")

        def stream_file(self, path):
            streamed.append(path)
            yield b"remote copy"

    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: FakeSshFs())
    route = file_tickets.ROUTES["download"]
    anchor = TestClient(web_server.app)  # a download link sends no session header

    def ticketed(query):
        minted = client.post("/api/files/ticket", json={"route": "download", **query})
        assert minted.status_code == 200, (query, minted.text)
        return {**query, "ticket": minted.json()["ticket"]}

    try:
        for path in (str(same_path), remote_only):
            query = {"path": path, "profile": "remote-dev"}
            checked = client.head(route, params=query)
            assert (checked.status_code, checked.content) == (200, b""), (path, checked.text)
            params = ticketed(query)
            saved = anchor.get(route, params=params)
            assert (saved.status_code, saved.content) == (200, b"remote copy"), path
            assert saved.headers["content-disposition"].startswith("attachment;")
            assert anchor.get(route, params=params).status_code == 401
        assert streamed == [str(same_path), remote_only]  # the HEAD checks read nothing

        params = ticketed({"path": remote_only, "profile": "remote-dev"})
        mark_named_profile_deleted(profile)
        assert anchor.get(route, params=params).status_code in (401, 403, 404)
        assert streamed == [str(same_path), remote_only]
    finally:
        anchor.close()


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


def test_fs_default_cwd_leaves_a_placeholder_terminal_cwd_to_terminal_cwd(client, monkeypatch, tmp_path):
    """A placeholder terminal.cwd ("." is the default) counts as unset, so TERMINAL_CWD names the folder."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(file_routes, "load_config", lambda: {"terminal": {"cwd": "."}})
    monkeypatch.setenv("TERMINAL_CWD", str(home))

    response = client.get("/api/fs/default-cwd")

    assert response.status_code == 200
    assert response.json()["cwd"] == str(home.resolve())


def test_fs_default_cwd_answers_each_profile_with_its_own_start_folder(client, monkeypatch, tmp_path):
    """``/api/fs/default-cwd?profile=<name>`` answers with that profile's ``terminal.cwd``, so a
    gateway serving several profiles starts each one's sessions and browser terminals in its own
    folder. A profile without one starts in the home folder: never another profile's folder and
    never the folder the serving process was started from."""
    from hermes_cli import profiles

    default_home = tmp_path / "default-home"
    work_home = default_home / "profiles" / "work"
    plain_home = default_home / "profiles" / "plain"
    default_project = tmp_path / "default-project"
    work_project = tmp_path / "work-project"
    user_home = tmp_path / "user"
    launch_dir = tmp_path / "launch-dir"
    for folder in (work_home, plain_home, default_project, work_project, user_home, launch_dir):
        folder.mkdir(parents=True)
    (default_home / "config.yaml").write_text(
        f"terminal:\n  cwd: {default_project.as_posix()}\n", encoding="utf-8")
    (work_home / "config.yaml").write_text(
        f"terminal:\n  cwd: {work_project.as_posix()}\n", encoding="utf-8")
    (plain_home / "config.yaml").write_text("terminal:\n  cwd: .\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    # What the gateway's startup leaves in its own environment for its own profile.
    monkeypatch.setenv("TERMINAL_CWD", str(default_project))
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.chdir(launch_dir)

    def start_folder(**params):
        response = client.get("/api/fs/default-cwd", params=params)
        assert response.status_code == 200, response.text
        return response.json()["cwd"]

    assert start_folder(profile="work") == str(work_project.resolve())
    assert start_folder(profile="plain") == str(user_home.resolve())
    assert start_folder() == str(default_project.resolve())
    assert start_folder(profile="default") == str(default_project.resolve())


def test_fs_default_cwd_falls_back_to_the_home_folder(client, monkeypatch, tmp_path):
    """With neither ``terminal.cwd`` nor ``TERMINAL_CWD`` naming a folder, the serving profile's
    start folder is the home folder, not the folder the gateway was started from."""
    user_home = tmp_path / "user"
    launch_dir = tmp_path / "launch-dir"
    user_home.mkdir()
    launch_dir.mkdir()
    monkeypatch.setattr(file_routes, "load_config", lambda: {"terminal": {"cwd": "."}})
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.chdir(launch_dir)

    response = client.get("/api/fs/default-cwd")

    assert response.status_code == 200
    assert response.json()["cwd"] == str(user_home.resolve())


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


@pytest.mark.parametrize("name", ["..%252fsecret.md", "%252e%252e%252fsecret.md", "..%255csecret.md"])
def test_fs_read_text_decodes_a_file_url_once(client, tmp_path, name):
    """A ``file:`` URL is percent-decoded once. ``..%252f`` is then the name ``..%2f`` inside the
    folder the URL names, which does not exist; a second decode would turn it into ``../`` and read
    a file outside that folder."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (tmp_path / "secret.md").write_text("outside the named folder")

    response = client.get("/api/fs/read-text", params={"path": f"{folder.as_uri()}/{name}"})

    assert response.status_code == 404
    assert "outside the named folder" not in response.text


@pytest.mark.parametrize("endpoint", ["read-text", "read-data-url", "download"])
def test_a_file_url_is_decoded_once_by_every_read(client, tmp_path, endpoint):
    """A ``file:`` URL is percent-decoded once on every read, the regular-file check included.
    ``report%252fsecret.txt`` names the file ``report%2fsecret.txt`` in the folder the URL names;
    a second decode would read ``report/secret.txt``, a file the URL never named."""
    folder = tmp_path / "docs"
    (folder / "report").mkdir(parents=True)
    (folder / "report" / "secret.txt").write_text("a file the URL never named")
    url = f"{folder.as_uri()}/report%252fsecret.txt"

    unnamed = client.get(f"/api/fs/{endpoint}", params={"path": url})
    (folder / "report%2fsecret.txt").write_text("the file the URL names")
    named = client.get(f"/api/fs/{endpoint}", params={"path": url})

    assert unnamed.status_code == 404, unnamed.text
    assert named.status_code == 200, named.text
    if endpoint == "read-text":
        body = named.json()["text"]
    elif endpoint == "read-data-url":
        body = base64.b64decode(named.json()["dataUrl"].split(",", 1)[1]).decode()
    else:
        body = named.text
    assert body == "the file the URL names"


@pytest.mark.parametrize("name", ["..%2fsecret.md", "..%2Fsecret.md", "..%5csecret.md"])
@pytest.mark.parametrize("endpoint", ["read-text", "read-data-url", "download"])
def test_a_file_url_that_encodes_a_separator_is_refused(client, tmp_path, endpoint, name):
    """A ``file:`` URL that percent-encodes a separator (``%2f``; ``%5c`` where it decodes to one)
    is refused, as the shared parser in ``apps/shared/src/file-url.ts`` refuses it in the renderer
    and Electron: decoding it would add a segment, and ``..%2fsecret.md`` would read a file outside
    the folder the URL names."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (tmp_path / "secret.md").write_text("outside the named folder")

    response = client.get(f"/api/fs/{endpoint}", params={"path": f"{folder.as_uri()}/{name}"})

    assert response.status_code in {400, 404}, (endpoint, name, response.status_code)
    assert "outside the named folder" not in response.text


@pytest.mark.parametrize("name", ["..%2fsecret.md", "%2e%2e/secret.md", "%2E%2E%2Fsecret.md"])
def test_fs_read_text_decode_fallback_stays_in_the_named_folder(client, tmp_path, name):
    """The one-decode rescue for a doubly encoded path (issue #103425) never leaves the folder the
    path names: when decoding would add a ``..`` segment, the path is read as written."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (tmp_path / "secret.md").write_text("outside the named folder")

    response = client.get("/api/fs/read-text", params={"path": f"{folder}/{name}"})

    assert response.status_code == 404
    assert "outside the named folder" not in response.text


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


def test_fs_read_data_url_reads_a_browser_paste_from_the_host_for_an_ssh_profile(client, monkeypatch):
    """A browser paste is saved in the profile's host image cache even when its workspace is an
    SSH machine. Sending it reads it back through this route, so that one host-owned path must not
    go looking on the SSH machine; every other path still does."""

    class FakeSshFs:
        def read_bytes(self, path, *, max_bytes, read_limit=None):
            raise ssh_workspace_fs.SshWorkspaceFsError("ENOENT", f"{path} is not on the SSH machine")

        def inspect_file(self, path):
            raise ssh_workspace_fs.SshWorkspaceFsError("ENOENT", f"{path} is not on the SSH machine")

    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: FakeSshFs())
    pasted = base64.b64decode("iVBORw0KGgo=")
    upload = client.post(
        "/api/chat/image-upload",
        json={"data_url": "data:image/png;base64,iVBORw0KGgo=", "filename": "paste.png"},
    )
    assert upload.status_code == 200, upload.text
    host_path = Path(upload.json()["path"])

    # The same bytes on this host outside the image cache are still the SSH machine's to answer.
    elsewhere = host_path.parent.parent / "elsewhere.png"
    elsewhere.write_bytes(pasted)

    read_back = client.get("/api/fs/read-data-url", params={"path": str(host_path)})
    saved = client.get("/api/fs/download", params={"path": str(host_path)})
    not_host_owned = client.get("/api/fs/read-data-url", params={"path": str(elsewhere)})

    assert read_back.status_code == 200, read_back.text
    assert base64.b64decode(read_back.json()["dataUrl"].split(",", 1)[1]) == pasted
    assert (saved.status_code, saved.content) == (200, pasted)
    assert not_host_owned.status_code == 404


def test_a_browser_paste_reads_back_while_the_ssh_machine_is_unreachable(client, monkeypatch):
    """A paste in the host image cache is read on this host before anything dials the SSH machine,
    so it still sends while that machine is down. A path only the SSH machine has answers 503."""

    class UnreachableSsh:
        def __init__(self, **_options):
            raise ConnectionRefusedError("ssh: connect to host 127.0.0.1 port 1: Connection refused")

    monkeypatch.setattr(ssh_workspace_fs, "SSHEnvironment", UnreachableSsh)
    monkeypatch.setattr(ssh_workspace_fs, "_BACKENDS", {})
    monkeypatch.setattr(
        file_routes, "load_config",
        lambda: {"terminal": {"backend": "ssh", "ssh_host": "127.0.0.1", "ssh_user": "dev", "ssh_port": 1}},
    )
    monkeypatch.setattr(file_routes, "load_env", dict)
    upload = client.post(
        "/api/chat/image-upload",
        json={"data_url": "data:image/png;base64,iVBORw0KGgo=", "filename": "paste.png"},
    )
    assert upload.status_code == 200, upload.text
    host_path = upload.json()["path"]

    read_back = client.get("/api/fs/read-data-url", params={"path": host_path})
    saved = client.get("/api/fs/download", params={"path": host_path})
    remote_only = client.get("/api/fs/read-data-url", params={"path": "/srv/repos/shot.png"})
    remote_download = client.get("/api/fs/download", params={"path": "/srv/repos/shot.png"})

    assert read_back.status_code == 200, read_back.text
    assert read_back.json()["dataUrl"] == "data:image/png;base64,iVBORw0KGgo="
    assert (saved.status_code, saved.content) == (200, base64.b64decode("iVBORw0KGgo="))
    for response in (remote_only, remote_download):
        assert (response.status_code, response.json()) == (503, {"detail": "SSH workspace is unavailable"})


def test_a_down_ssh_host_is_unavailable_on_workspace_file_routes(client, monkeypatch, caplog):
    """The profile's first file request opens its SSH adapter, which dials the host. While the host
    refuses, list, read-text, write-text, git-root and the default folder answer 503, the error the
    image route and download already give, and the log names the refusal. Once the host answers,
    the next request dials again, and a folder that is not there or a file the SSH user cannot read
    keeps its own answer instead of that 503."""
    from tools.environments.base import EnvironmentConnectionError

    class SshHost:
        """The SSH machine: refuses the dial while ``down``; once up, it has no /srv/missing and
        its SSH user cannot read /srv/secret."""

        down = True
        cwd, timeout, _remote_home = "/srv/repos", 30, "/home/dev"

        def __init__(self, **_options):
            if SshHost.down:
                raise EnvironmentConnectionError(
                    "SSH connection failed: ssh: connect to host 127.0.0.1 port 1: Connection refused")

        def execute(self, command, **_options):
            code, rc = ("EACCES", 46) if "/srv/secret" in command else ("ENOENT", 44)
            return {"returncode": rc, "output": f"__HERMES_FS_ERROR__:{code}\n"}

        def cleanup(self):
            pass

    monkeypatch.setattr(ssh_workspace_fs, "SSHEnvironment", SshHost)
    monkeypatch.setattr(ssh_workspace_fs, "_BACKENDS", {})
    monkeypatch.setattr(
        file_routes, "load_config",
        lambda: {"terminal": {"backend": "ssh", "ssh_host": "127.0.0.1", "ssh_user": "dev", "ssh_port": 1}},
    )
    monkeypatch.setattr(file_routes, "load_env", dict)

    unavailable = (503, {"detail": "SSH workspace is unavailable"})
    with caplog.at_level("WARNING", logger=file_routes.__name__):
        for response in (
            client.get("/api/fs/list", params={"path": "/srv/repos"}),
            client.get("/api/fs/read-text", params={"path": "/srv/repos/README.md"}),
            client.post("/api/fs/write-text", json={"path": "/srv/repos/README.md", "content": "x"}),
            client.get("/api/fs/git-root", params={"path": "/srv/repos"}),
            client.get("/api/fs/default-cwd"),
        ):
            assert (response.status_code, response.json()) == unavailable, response.request.url
    assert "Connection refused" in caplog.text

    SshHost.down = False
    missing = client.get("/api/fs/list", params={"path": "/srv/missing"})
    denied = client.get("/api/fs/list", params={"path": "/srv/secret"})
    missing_text = client.get("/api/fs/read-text", params={"path": "/srv/missing/a.txt"})
    denied_text = client.get("/api/fs/read-text", params={"path": "/srv/secret"})
    assert (missing.status_code, missing.json()) == (200, {"entries": [], "error": "ENOENT"})
    assert (denied.status_code, denied.json()) == (200, {"entries": [], "error": "EACCES"})
    assert (missing_text.status_code, missing_text.json()) == (404, {"detail": "File not found"})
    assert (denied_text.status_code, denied_text.json()) == (403, {"detail": "File is not readable"})


class _NotOnTheSshMachine:
    """An SSH workspace that has none of the files a test asks it for; it records each ask."""

    def __init__(self):
        self.asked = []
        self.written = []

    def write_text(self, path, content, *, max_bytes):
        self.written.append((path, content))
        return path, len(content.encode("utf-8"))

    def read_bytes(self, path, *, max_bytes, read_limit=None):
        self.asked.append(path)
        raise ssh_workspace_fs.SshWorkspaceFsError("ENOENT", f"{path} is not on the SSH machine")

    def inspect_file(self, path):
        self.asked.append(path)
        raise ssh_workspace_fs.SshWorkspaceFsError("ENOENT", f"{path} is not on the SSH machine")


def _ssh_profiles(tmp_path, monkeypatch):
    """A default home and a named profile ``remote-dev``, both with an SSH machine as workspace."""
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "remote-dev"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    workspace = _NotOnTheSshMachine()
    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: workspace)
    return profile, workspace


def test_a_picked_image_reads_back_from_the_host_for_an_ssh_profile(client, monkeypatch, tmp_path):
    """The browser's file picker stages a file in the profile's uploads folder on this host, even
    when its workspace is an SSH machine. Previewing and sending a picked image read it back through
    /api/fs/read-data-url (saving it, through /api/fs/download), so a file the upload route staged
    for that profile is read on this host. Every other path still goes to the SSH machine: a staged
    file of another profile, a file in the uploads folder the upload route did not name, one in a
    folder below it, and a path that leaves the folder through ``..``."""
    profile, workspace = _ssh_profiles(tmp_path, monkeypatch)
    # The Webapp: the browser's file routes (the file picker's upload) exist only there.
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    image = base64.b64decode("iVBORw0KGgo=")

    def pick(query):
        staged = client.post("/api/chat/file-upload", params=query,
                             files={"file": ("shot.png", image, "image/png")})
        assert staged.status_code == 200, staged.text
        return Path(staged.json()["path"])

    remote_dev = {"profile": "remote-dev"}
    staged, default_staged = pick(remote_dev), pick({})
    unnamed = staged.parent / "shot.png"
    nested = staged.parent / "below" / "web-shot.png"
    nested.parent.mkdir()
    (profile / "web-shot.png").write_bytes(image)
    left_through_dots = f"{staged.parent}{os.sep}..{os.sep}web-shot.png"
    for path in (unnamed, nested):
        path.write_bytes(image)

    for query, path in ((remote_dev, staged), ({}, default_staged)):
        read_back = client.get("/api/fs/read-data-url", params={"path": str(path), **query})
        saved = client.get("/api/fs/download", params={"path": str(path), **query})
        assert read_back.status_code == 200, read_back.text
        assert read_back.json()["dataUrl"] == "data:image/png;base64,iVBORw0KGgo="
        assert (saved.status_code, saved.content) == (200, image)
    assert workspace.asked == []

    not_host_owned = [str(default_staged), str(unnamed), str(nested), left_through_dots]
    for path in not_host_owned:
        for route in ("read-data-url", "download"):
            response = client.get(f"/api/fs/{route}", params={"path": path, **remote_dev})
            assert response.status_code == 404, (route, path, response.text)
    assert workspace.asked == [path for path in not_host_owned for _route in range(2)]


def test_a_staged_text_preview_reads_back_from_the_host_for_an_ssh_profile(client, monkeypatch, tmp_path):
    """A long paste or a picked text file is staged in the profile's uploads folder on this host,
    even when its workspace is an SSH machine. Its chip previews it through /api/fs/read-text, so
    that one host-owned file is read here, as /api/fs/read-data-url reads a staged image. A file in
    the uploads folder the upload route did not name is still the SSH machine's to answer."""
    _profile, workspace = _ssh_profiles(tmp_path, monkeypatch)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    remote_dev = {"profile": "remote-dev"}

    staged = client.post("/api/chat/file-upload", params=remote_dev,
                         files={"file": ("pasted.txt", b"pasted words", "text/plain")})
    assert staged.status_code == 200, staged.text
    staged_path = Path(staged.json()["path"])
    unnamed = staged_path.parent / "pasted.txt"
    unnamed.write_bytes(b"not staged")

    preview = client.get("/api/fs/read-text", params={"path": str(staged_path), **remote_dev})
    assert preview.status_code == 200, preview.text
    assert preview.json()["text"] == "pasted words"
    assert workspace.asked == []

    not_host_owned = client.get("/api/fs/read-text", params={"path": str(unnamed), **remote_dev})
    assert not_host_owned.status_code == 404, not_host_owned.text
    assert workspace.asked == [str(unnamed)]


def test_a_staged_text_file_saves_where_its_preview_reads_it_for_an_ssh_profile(client, monkeypatch, tmp_path):
    """A staged text preview on this host can be edited, and the editor saves through
    /api/fs/write-text before it reads the file again. The save goes to the same host file the
    preview read, so the reread shows the edit; any other path is still saved on the SSH machine."""
    _profile, workspace = _ssh_profiles(tmp_path, monkeypatch)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    remote_dev = {"profile": "remote-dev"}

    staged = client.post("/api/chat/file-upload", params=remote_dev,
                         files={"file": ("pasted.txt", b"pasted words", "text/plain")})
    assert staged.status_code == 200, staged.text
    staged_path = Path(staged.json()["path"])

    def preview():
        response = client.get("/api/fs/read-text", params={"path": str(staged_path), **remote_dev})
        assert response.status_code == 200, response.text
        return response.json()["text"]

    assert preview() == "pasted words"
    saved = client.post("/api/fs/write-text", params=remote_dev,
                        json={"path": str(staged_path), "content": "edited words"})
    assert saved.status_code == 200, saved.text
    assert preview() == "edited words"
    assert staged_path.read_text(encoding="utf-8") == "edited words"
    assert (workspace.asked, workspace.written) == ([], [])

    remote = client.post("/api/fs/write-text", params=remote_dev,
                         json={"path": "/srv/repos/notes.txt", "content": "remote words"})
    assert remote.status_code == 200, remote.text
    assert workspace.written == [("/srv/repos/notes.txt", "remote words")]


def test_a_staged_text_preview_reads_back_under_a_locked_files_root(client, monkeypatch, tmp_path):
    """A locked files root outside the profile home still lets a staged paste or pick preview, as
    /api/fs/read-data-url already does, for an SSH profile and a host-local one. Every other host
    file outside the root stays refused: one in the profile home, and one in the uploads folder the
    upload route did not name."""
    profile, workspace = _ssh_profiles(tmp_path, monkeypatch)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    remote_dev = {"profile": "remote-dev"}

    def stage(query):
        staged = client.post("/api/chat/file-upload", params=query,
                             files={"file": ("pasted.txt", b"pasted words", "text/plain")})
        assert staged.status_code == 200, staged.text
        return Path(staged.json()["path"])

    staged, default_staged = stage(remote_dev), stage({})
    unrelated = profile / "notes.txt"
    unnamed = staged.parent / "pasted.txt"
    for path in (unrelated, unnamed):
        path.write_bytes(b"not staged")
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(managed))

    def read_text(path, query):
        return client.get("/api/fs/read-text", params={"path": str(path), **query})

    over_ssh = read_text(staged, remote_dev)
    assert (over_ssh.status_code, over_ssh.json().get("text")) == (200, "pasted words"), over_ssh.text

    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: None)
    for query, path in ((remote_dev, staged), ({}, default_staged)):
        response = read_text(path, query)
        assert (response.status_code, response.json().get("text")) == (200, "pasted words"), response.text
    for path in (unrelated, unnamed):
        response = read_text(path, remote_dev)
        assert response.status_code == 403, (path, response.text)
    assert workspace.asked == []


@pytest.mark.platforms("posix")
def test_a_link_staged_in_the_uploads_folder_is_not_read_from_the_host(client, monkeypatch, tmp_path):
    """A staged name that links to another host file resolves outside the uploads folder, so an
    SSH profile's read of it goes to the SSH machine and never returns the host file."""
    profile, workspace = _ssh_profiles(tmp_path, monkeypatch)
    host_file = tmp_path / "host-only.png"
    host_file.write_bytes(b"host only")
    uploads = profile / "uploads"
    uploads.mkdir(mode=0o700)
    link = uploads / "web-0123456789abcdef-shot.png"
    link.symlink_to(host_file)

    response = client.get("/api/fs/read-data-url", params={"path": str(link), "profile": "remote-dev"})

    assert response.status_code == 404, response.text
    assert workspace.asked == [str(link)]


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

    def translate(path, session_key="", workspace_key="", *, quiet=False, workspace_cwd=""):
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


def _workspace_sandbox(sandbox: Path, cwd: Path) -> Path:
    from tools.environments.path_utils import sanitize_task_id_for_path
    from tools.terminal_tool import workspace_key_for

    name = sanitize_task_id_for_path(f"default:ws:{workspace_key_for(str(cwd))}")
    workspace = sandbox / "docker" / name / "workspace"
    workspace.mkdir(parents=True)
    return workspace


def _session_detail(cwds: dict):
    async def fake_detail(session_id, profile=None):
        return {"cwd": cwds[session_id], "id": session_id}

    return fake_detail


def test_fs_download_maps_workspace_container_to_its_sandbox(client, monkeypatch, tmp_path):
    """A persistent per-workspace container stores ``/workspace/note.txt`` under
    ``sandboxes/docker/default_ws_<digest>-<hash>/workspace``, not the shared
    ``default`` sandbox. The download must serve that session's file, and must
    not scan every other workspace for the same relative path."""
    from tools.terminal_tool import register_task_env_overrides, workspace_key_for

    project = tmp_path / "project"
    project.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    sandbox = tmp_path / "sandboxes"
    own = _workspace_sandbox(sandbox, project)
    foreign = _workspace_sandbox(sandbox, other)
    shared = sandbox / "docker" / "default" / "workspace"
    shared.mkdir(parents=True)
    (own / "note.txt").write_bytes(b"session-file")
    (own / "only.txt").write_bytes(b"only-session")
    (foreign / "note.txt").write_bytes(b"other-file")
    (shared / "note.txt").write_bytes(b"shared-file")
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(sandbox))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)
    monkeypatch.delenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", raising=False)
    monkeypatch.setattr(
        "hermes_cli.web_routers.sessions.get_session_detail",
        _session_detail({
            "sess-ws": str(project),
            "sess-other": str(other),
            "sess-live": str(tmp_path / "missing-cwd"),
        }),
    )

    # No session id must not scan workspace sandboxes. The shared default copy
    # is the only match; a file that exists only under default_ws_* stays missing.
    bare = client.get("/api/fs/download", params={"path": "/workspace/note.txt"})
    assert bare.status_code == 200
    assert bare.content == b"shared-file"
    only = client.get("/api/fs/download", params={"path": "/workspace/only.txt"})
    assert only.status_code == 404

    response = client.get(
        "/api/fs/download", params={"path": "/workspace/note.txt", "session_id": "sess-ws"},
    )
    assert response.status_code == 200
    assert response.content == b"session-file"
    other_hit = client.get(
        "/api/fs/download", params={"path": "/workspace/note.txt", "session_id": "sess-other"},
    )
    assert other_hit.status_code == 200
    assert other_hit.content == b"other-file"

    digest = workspace_key_for(str(project))
    register_task_env_overrides("sess-live", {
        "cwd": str(tmp_path / "missing-cwd"),
        "cwd_source": "session",
        "workspace_key": digest,
    })
    try:
        live = client.get(
            "/api/fs/download", params={"path": "/workspace/note.txt", "session_id": "sess-live"},
        )
    finally:
        from gateway.session_policy import unbind_turn_workspace
        unbind_turn_workspace("sess-live")
    assert live.status_code == 200
    assert live.content == b"session-file"


def test_fs_download_cwd_mount_wins_then_falls_back_to_workspace_sandbox(
    client, monkeypatch, tmp_path,
):
    """With the host cwd mounted on ``/workspace``, the session's own cwd is that
    mount and wins when the file is there; the profile's ``terminal.cwd`` is another
    container's mount. When the container was still created as ``default:ws:`` and
    the file only exists in its sandbox workspace, the download serves that copy."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "note.txt").write_bytes(b"mounted-file")
    mounted = tmp_path / "mounted-cwd"
    mounted.mkdir()
    (mounted / "note.txt").write_bytes(b"profile-cwd-file")
    sandbox = tmp_path / "sandboxes"
    own = _workspace_sandbox(sandbox, project)
    (own / "note.txt").write_bytes(b"sandbox-file")
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(sandbox))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    monkeypatch.setenv("TERMINAL_CWD", str(mounted))
    monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)
    monkeypatch.setattr(
        "hermes_cli.web_routers.sessions.get_session_detail",
        _session_detail({"sess-ws": str(project)}),
    )

    mounted_hit = client.get(
        "/api/fs/download", params={"path": "/workspace/note.txt", "session_id": "sess-ws"},
    )
    assert mounted_hit.status_code == 200
    assert mounted_hit.content == b"mounted-file"

    (project / "note.txt").unlink()
    sandbox_hit = client.get(
        "/api/fs/download", params={"path": "/workspace/note.txt", "session_id": "sess-ws"},
    )
    assert sandbox_hit.status_code == 200
    assert sandbox_hit.content == b"sandbox-file"


@pytest.mark.platforms("windows")
def test_agent_fs_path_translates_before_host_normalization(monkeypatch, tmp_path):
    """On a Windows gateway ``_fs_path("/workspace/x")`` anchors to the current
    drive (``D:\\workspace\\x``) and could serve an unrelated host file; the
    container path must reach the translator verbatim, before normalization."""
    from hermes_cli import web_server_files

    mapped = tmp_path / "x.txt"
    seen = []

    def translate(candidate, session_key="", workspace_key="", *, quiet=False, workspace_cwd=""):
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


def _two_profile_homes(monkeypatch, tmp_path, *, active_backend, work_backend):
    """Default + ``work`` homes with their own ``terminal.backend``; returns (default, work)."""
    from hermes_cli import profiles

    default_home = tmp_path / "default-home"
    work_home = default_home / "profiles" / "work"
    for home, backend in ((default_home, active_backend), (work_home, work_backend)):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            f"terminal:\n  backend: {backend}\n  container_persistent: false\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("TERMINAL_ENV", active_backend)
    monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)
    return default_home, work_home


@pytest.mark.parametrize(("active", "work", "expected"), [
    ("local", "docker", True),   # requested docker profile: a miss is a sandbox path
    ("docker", "local", False),  # requested local profile: never the sandbox 404
])
def test_unmapped_classification_uses_requested_profiles_backend(
    monkeypatch, tmp_path, active, work, expected,
):
    """The Windows 404-or-drive decision and the sandbox hint read the same profile
    translation used, in both mixed-backend directions."""
    from hermes_cli import web_server_files

    _default, work_home = _two_profile_homes(
        monkeypatch, tmp_path, active_backend=active, work_backend=work)

    assert web_server_files._is_unmapped_container_path(
        "/workspace/x.txt", profile_home=work_home) is expected
    assert web_server_files._is_unmapped_container_path("/workspace/x.txt") is (active == "docker")


def test_native_home_override_translates_with_that_profiles_volumes(monkeypatch, tmp_path):
    """A native ticket binds the profile's home before the route runs; translation must
    install that profile's terminal policy, not read the launch env's volumes."""
    from hermes_cli import web_server_files
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _default, work_home = _two_profile_homes(
        monkeypatch, tmp_path, active_backend="docker", work_backend="docker")
    launch_mount = tmp_path / "launch-mount"
    work_mount = tmp_path / "work-mount"
    for mount, label in ((launch_mount, "LAUNCH"), (work_mount, "WORK")):
        mount.mkdir()
        (mount / "native.txt").write_text(label, encoding="utf-8")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{launch_mount}:/workspace"]))
    (work_home / "config.yaml").write_text(
        "terminal:\n  backend: docker\n  container_persistent: false\n"
        f"  docker_volumes:\n    - {work_mount}:/workspace\n", encoding="utf-8")

    token = set_hermes_home_override(str(work_home))
    try:
        translated = web_server_files._agent_container_path("/workspace/native.txt")
    finally:
        reset_hermes_home_override(token)

    assert translated == (work_mount / "native.txt").resolve()
    assert web_server_files._agent_container_path("/workspace/native.txt") == (
        launch_mount / "native.txt").resolve()


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
