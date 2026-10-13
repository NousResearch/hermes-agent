import base64
import os
import stat
import urllib.parse
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


def test_fs_create_file_and_directory(client, tmp_path):
    parent = tmp_path / "project"
    parent.mkdir()

    file_response = client.post("/api/fs/create", json={"path": str(parent / "notes.md"), "directory": False})
    dir_response = client.post("/api/fs/create", json={"path": str(parent / "subdir"), "directory": True})

    assert file_response.status_code == 200
    assert (parent / "notes.md").is_file()
    assert file_response.json()["isDirectory"] is False
    assert dir_response.status_code == 200
    assert (parent / "subdir").is_dir()
    assert dir_response.json()["isDirectory"] is True


def test_fs_create_refuses_existing_and_missing_parent(client, tmp_path):
    parent = tmp_path / "project"
    parent.mkdir()
    (parent / "taken.txt").write_text("x")

    existing = client.post("/api/fs/create", json={"path": str(parent / "taken.txt")})
    missing_parent = client.post("/api/fs/create", json={"path": str(parent / "no" / "such" / "file")})

    assert existing.status_code == 409
    assert missing_parent.status_code == 400
    assert not (parent / "no").exists()


def test_fs_rename_same_parent_collision_guard(client, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("a")
    (root / "b.txt").write_text("b")

    moved = client.post("/api/fs/rename", json={"path": str(root / "a.txt"), "name": "c.txt"})
    collision = client.post("/api/fs/rename", json={"path": str(root / "b.txt"), "name": "c.txt"})
    escape = client.post("/api/fs/rename", json={"path": str(root / "b.txt"), "name": "../escape.txt"})
    missing = client.post("/api/fs/rename", json={"path": str(root / "ghost.txt"), "name": "x.txt"})

    assert moved.status_code == 200
    assert moved.json()["path"] == str(root / "c.txt")
    assert (root / "c.txt").read_text() == "a"
    assert collision.status_code == 409
    assert escape.status_code == 400
    assert not (tmp_path / "escape.txt").exists()
    assert missing.status_code == 404


def test_fs_mutations_enforce_locked_managed_root_boundary(monkeypatch, client, tmp_path):
    from hermes_cli import web_server_files

    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "source.txt").write_text("source")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    monkeypatch.setattr(
        file_routes, "_managed_files_policy",
        lambda request, **kwargs: web_server_files.ManagedFilesPolicy(
            default_path=locked, locked_root=locked, can_change_path=False
        ),
    )

    create = client.post("/api/fs/create", json={"path": str(outside / "new.txt")})
    rename = client.post("/api/fs/rename", json={"path": str(outside / "keep.txt"), "name": "renamed.txt"})
    delete = client.request("DELETE", "/api/fs/delete", json={"path": str(outside / "keep.txt")})

    assert [create.status_code, rename.status_code, delete.status_code] == [403, 403, 403]
    assert (locked / "source.txt").read_text() == "source"
    assert (outside / "keep.txt").read_text() == "keep"
    assert not (outside / "new.txt").exists()


def test_fs_rename_noreplace_preserves_independent_destination(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "target.txt"
    source.write_text("source")
    destination.write_text("independent writer")

    with pytest.raises(FileExistsError):
        file_routes._rename_noreplace(source, destination)

    assert source.read_text() == "source"
    assert destination.read_text() == "independent writer"


def test_fs_rename_keeps_target_created_at_final_publication(client, tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    source = root / "source.txt"
    destination = root / "target.txt"
    source.write_text("source")
    original = file_routes._rename_noreplace

    def competing_writer(src, dst):
        dst.write_text("independent writer")
        return original(src, dst)

    monkeypatch.setattr(file_routes, "_rename_noreplace", competing_writer)
    response = client.post("/api/fs/rename", json={"path": str(source), "name": destination.name})

    assert response.status_code == 409
    assert source.read_text() == "source"
    assert destination.read_text() == "independent writer"


def test_fs_rename_unavailable_atomic_primitive_is_explicit(client, tmp_path, monkeypatch):
    import errno

    source = tmp_path / "source.txt"
    source.write_bytes(b"source")

    def unsupported(source, destination):
        raise OSError(errno.ENOTSUP, "Atomic no-replace rename is unsupported")

    monkeypatch.setattr(file_routes, "_rename_noreplace", unsupported)
    response = client.post("/api/fs/rename", json={"path": str(source), "name": "destination.txt"})

    assert response.status_code == 501
    assert source.read_bytes() == b"source"
    assert not (tmp_path / "destination.txt").exists()


def test_fs_rename_accepts_directory(client, tmp_path):
    root = tmp_path / "project"
    (root / "olddir").mkdir(parents=True)

    response = client.post("/api/fs/rename", json={"path": str(root / "olddir"), "name": "newdir"})

    assert response.status_code == 200
    assert (root / "newdir").is_dir()
    assert not (root / "olddir").exists()


def test_fs_delete_file_dir_and_recursive_guard(client, tmp_path):
    root = tmp_path / "project"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "deep.txt").write_text("d")
    (root / "leaf.txt").write_text("l")

    dir_without_recursive = client.request(
        "DELETE", "/api/fs/delete", json={"path": str(root / "sub"), "recursive": False}
    )
    file_delete = client.request("DELETE", "/api/fs/delete", json={"path": str(root / "leaf.txt")})
    dir_recursive = client.request(
        "DELETE", "/api/fs/delete", json={"path": str(root / "sub"), "recursive": True}
    )
    missing = client.request("DELETE", "/api/fs/delete", json={"path": str(root / "ghost.txt")})

    assert dir_without_recursive.status_code == 409
    assert file_delete.status_code == 200
    assert not (root / "leaf.txt").exists()
    assert dir_recursive.status_code == 200
    assert not (root / "sub").exists()
    assert missing.status_code == 404


def test_fs_delete_empty_directory_without_recursive(client, tmp_path):
    root = tmp_path / "project"
    (root / "empty").mkdir(parents=True)

    response = client.request("DELETE", "/api/fs/delete", json={"path": str(root / "empty"), "recursive": False})

    assert response.status_code == 200
    assert not (root / "empty").exists()


def test_fs_delete_refuses_managed_root(monkeypatch, client, tmp_path):
    from hermes_cli import web_server_files
    from hermes_cli.web_routers import files as files_router

    locked = tmp_path / "managed-root"
    locked.mkdir()

    monkeypatch.setattr(
        files_router,
        "_managed_files_policy",
        lambda request, **kwargs: web_server_files.ManagedFilesPolicy(
            default_path=locked, locked_root=locked, can_change_path=False
        ),
    )

    response = client.request("DELETE", "/api/fs/delete", json={"path": str(locked), "recursive": True})

    assert response.status_code == 400
    assert locked.exists()


def test_fs_mutation_refuses_symlinked_parent(client, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("keep")
    root = tmp_path / "project"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    create = client.post("/api/fs/create", json={"path": str(root / "link" / "evil.txt"), "directory": False})
    rename = client.post("/api/fs/rename", json={"path": str(root / "link" / "sentinel.txt"), "name": "moved.txt"})
    delete = client.request(
        "DELETE", "/api/fs/delete", json={"path": str(root / "link" / "sentinel.txt"), "recursive": False}
    )

    assert create.status_code == 400
    assert rename.status_code == 400
    assert delete.status_code == 400
    # Nothing was created/renamed/deleted outside the project tree.
    assert not (outside / "evil.txt").exists()
    assert (outside / "sentinel.txt").read_text() == "keep"
    assert not (outside / "moved.txt").exists()


def test_fs_delete_refuses_ancestor_of_managed_root(monkeypatch, client, tmp_path):
    from hermes_cli import web_server_files
    from hermes_cli.web_routers import files as files_router

    locked = tmp_path / "opt-data"
    locked.mkdir()
    (locked / "keep.txt").write_text("keep")

    monkeypatch.setattr(
        files_router,
        "_managed_files_policy",
        lambda request, **kwargs: web_server_files.ManagedFilesPolicy(
            default_path=locked, locked_root=locked, can_change_path=False
        ),
    )

    # Deleting an ANCESTOR of the locked root with recursive must be refused:
    # a naive `target == locked_root` guard would let rmtree(ancestor) remove
    # the locked /opt/data underneath it.
    response = client.request(
        "DELETE", "/api/fs/delete", json={"path": str(tmp_path), "recursive": True}
    )

    assert response.status_code == 400
    assert (locked / "keep.txt").exists()


def test_fs_create_file_atomically_refuses_existing(client, tmp_path):
    parent = tmp_path / "project"
    parent.mkdir()
    (parent / "taken.txt").write_text("precious")

    # The same-body call twice: os.open(O_CREAT|O_EXCL) makes this a hard 409 —
    # Path.touch() would have silently succeeded and clobbered the content.
    for _ in range(2):
        response = client.post("/api/fs/create", json={"path": str(parent / "taken.txt"), "directory": False})

    assert response.status_code == 409
    assert (parent / "taken.txt").read_text() == "precious"


def test_fs_mutation_refuses_dot_components(client, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "sub").mkdir()
    (root / "sub" / "keep.txt").write_text("keep")
    (root / "sentinel.txt").write_text("sentinel")

    # A trailing `..` would resolve the delete target to the PARENT directory —
    # `rmtree('/project/sub/..')` removes /project, escaping the selected tree.
    delete = client.request(
        "DELETE", "/api/fs/delete", json={"path": str(root / "sub" / ".."), "recursive": True}
    )
    create = client.post("/api/fs/create", json={"path": str(root / "sub" / ".." / "evil.txt")})
    rename = client.post("/api/fs/rename", json={"path": str(root / "sub" / "keep.txt"), "name": "../../escape.txt"})

    assert delete.status_code == 400
    assert create.status_code == 400
    assert rename.status_code == 400
    # Nothing outside the selected folder was created or removed.
    assert (root / "sentinel.txt").read_text() == "sentinel"
    assert (tmp_path / "evil.txt").exists() is False
    assert (root / "sub" / "keep.txt").read_text() == "keep"


def test_fs_delete_and_rename_handle_dangling_symlink(client, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "broken").symlink_to(root / "missing-target")

    # exists() follows the referent so a dangling link reports missing; the
    # mutation targets the LINK itself, so both must still succeed (200).
    renamed = client.post("/api/fs/rename", json={"path": str(root / "broken"), "name": "broken2"})

    assert renamed.status_code == 200
    assert renamed.json()["path"] == str(root / "broken2")
    assert (root / "broken2").is_symlink()
    assert not (root / "broken").exists()

    deleted = client.request("DELETE", "/api/fs/delete", json={"path": str(root / "broken2")})
    assert deleted.status_code == 200
    assert not (root / "broken2").exists()


@pytest.mark.platforms("windows")
def test_fs_mutations_fail_closed_when_junction_detection_is_unavailable(client, tmp_path, monkeypatch):
    """Mutation admission fails closed if pathlib cannot inspect junctions."""
    parent = tmp_path / "project"
    parent.mkdir()
    source = parent / "source.txt"
    source.write_text("preserve")

    monkeypatch.setattr(Path, "is_junction", None)

    create = client.post("/api/fs/create", json={"path": str(parent / "new.txt")})
    rename = client.post("/api/fs/rename", json={"path": str(source), "name": "renamed.txt"})
    delete = client.request("DELETE", "/api/fs/delete", json={"path": str(source)})

    assert [create.status_code, rename.status_code, delete.status_code] == [501, 501, 501]
    assert all("junction detection is unavailable" in response.text for response in (create, rename, delete))
    assert source.read_text() == "preserve"
    assert not (parent / "new.txt").exists()
    assert not (parent / "renamed.txt").exists()


@pytest.mark.platforms("windows")
def test_fs_mutations_reject_junction_parents_and_delete_junction_only(client, tmp_path):
    import subprocess

    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep")
    root = tmp_path / "project"
    root.mkdir()
    junction = root / "junction"
    created = subprocess.run(
        ["cmd.exe", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True, text=True, check=False,
    )
    assert created.returncode == 0, created.stdout + created.stderr

    blocked = client.post("/api/fs/create", json={"path": str(junction / "evil.txt")})
    removed = client.request("DELETE", "/api/fs/delete", json={"path": str(junction), "recursive": True})

    assert blocked.status_code == 400
    assert removed.status_code == 200
    assert not junction.exists()
    assert sentinel.read_text() == "keep"
    assert not (outside / "evil.txt").exists()


def test_fs_create_file_uses_0666_base_mode(client, tmp_path):
    parent = tmp_path / "project"
    parent.mkdir()
    current_umask = os.umask(0)
    os.umask(current_umask)

    response = client.post("/api/fs/create", json={"path": str(parent / "notes.md")})

    assert response.status_code == 200
    # 0o666 base masked by the umask → 0644 (never the 0755 an 0o777 default
    # open() would produce for new text files).
    mode = stat.S_IMODE((parent / "notes.md").stat().st_mode)
    assert mode == 0o666 & ~current_umask


def test_fs_delete_never_removes_filesystem_root(client, tmp_path):
    response = client.request("DELETE", "/api/fs/delete", json={"path": "/", "recursive": True})

    assert response.status_code == 400


def test_fs_mutations_require_auth(client, tmp_path):
    unauthenticated = TestClient(web_server.app)
    parent = tmp_path / "project"
    parent.mkdir()

    create_response = unauthenticated.post("/api/fs/create", json={"path": str(parent / "x.txt")})
    rename_response = unauthenticated.post("/api/fs/rename", json={"path": str(parent / "x.txt"), "name": "y.txt"})
    delete_response = unauthenticated.request("DELETE", "/api/fs/delete", json={"path": str(parent / "x.txt")})

    assert create_response.status_code == 401
    assert rename_response.status_code == 401
    assert delete_response.status_code == 401
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


def test_fs_mutations_route_to_selected_ssh_workspace_without_host_fallback(client, monkeypatch):
    calls = []

    class FakeSshFs:
        def create(self, path, *, directory):
            calls.append(("create", path, directory))
            return path

        def rename(self, path, name):
            calls.append(("rename", path, name))
            return "/srv/repos/project/renamed.txt"

        def delete(self, path, *, recursive):
            calls.append(("delete", path, recursive))
            return path

    backend = FakeSshFs()
    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: backend)
    monkeypatch.setattr(
        file_routes, "_fs_mutation_target",
        lambda *_args, **_kwargs: pytest.fail("SSH path must not be interpreted on gateway host"),
    )

    created = client.post(
        "/api/fs/create", params={"profile": "remote-dev"},
        json={"path": "/srv/repos/project/new.txt", "directory": False},
    )
    renamed = client.post(
        "/api/fs/rename", params={"profile": "remote-dev"},
        json={"path": "/srv/repos/project/new.txt", "name": "renamed.txt"},
    )
    deleted = client.request(
        "DELETE", "/api/fs/delete", params={"profile": "remote-dev"},
        json={"path": "/srv/repos/project/renamed.txt", "recursive": True},
    )

    assert created.status_code == 200, created.text
    assert renamed.status_code == 200, renamed.text
    assert deleted.status_code == 200, deleted.text
    assert calls == [
        ("create", "/srv/repos/project/new.txt", False),
        ("rename", "/srv/repos/project/new.txt", "renamed.txt"),
        ("delete", "/srv/repos/project/renamed.txt", True),
    ]


def test_fs_mutations_fail_closed_for_unimplemented_terminal_backend(client, monkeypatch):
    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: None)
    monkeypatch.setattr(file_routes, "_profile_scope", lambda _profile: nullcontext())
    monkeypatch.setattr(file_routes, "load_config", lambda: {"terminal": {"backend": "docker"}})
    monkeypatch.setattr(
        file_routes, "_fs_mutation_target",
        lambda *_args, **_kwargs: pytest.fail("unsupported backend must not fall back to gateway host"),
    )

    response = client.post(
        "/api/fs/create", params={"profile": "remote-dev"},
        json={"path": "/workspace/new.txt"},
    )

    assert response.status_code == 501
    assert "not supported" in response.text


def test_ssh_workspace_mutations_build_remote_only_operations():
    class FakeEnvironment:
        cwd = "/srv/repos"
        _remote_home = "/home/dev"
        timeout = 30

        def __init__(self):
            self.commands = []

        def execute(self, command, **kwargs):
            self.commands.append((command, kwargs))
            return {"returncode": 0, "output": ""}

    env = FakeEnvironment()
    backend = ssh_workspace_fs.SshWorkspaceFs(env)

    assert backend.create("project/new.txt", directory=False) == "/srv/repos/project/new.txt"
    assert backend.create("/srv/repos/project/new-dir", directory=True) == "/srv/repos/project/new-dir"
    assert backend.rename("/srv/repos/project/old.txt", "new.txt") == "/srv/repos/project/new.txt"
    assert backend.delete("/srv/repos/project/old-dir", recursive=True) == "/srv/repos/project/old-dir"
    assert "noclobber" in env.commands[0][0]
    assert "mkdir --" in env.commands[1][0]
    assert "python3 -c" in env.commands[2][0]
    assert "renameat2" in env.commands[2][0]
    assert "rm -rf --" in env.commands[3][0]
    assert "fi" in env.commands[3][0]
    assert all(kwargs.get("cwd") == "/srv/repos" for _, kwargs in env.commands)

    with pytest.raises(ssh_workspace_fs.SshWorkspaceFsError, match="Invalid path"):
        backend.create("/srv/repos/../outside.txt", directory=False)
    with pytest.raises(ssh_workspace_fs.SshWorkspaceFsError, match="workspace root"):
        backend.delete("/srv/repos", recursive=True)
    assert len(env.commands) == 4


def test_fs_mutations_require_explicit_profile_when_multiplexed(client, monkeypatch):
    from agent import secret_scope

    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: True)
    create = client.post("/api/fs/create", json={"path": "/srv/repos/new.txt"})
    rename = client.post("/api/fs/rename", json={"path": "/srv/repos/old.txt", "name": "new.txt"})
    delete = client.request("DELETE", "/api/fs/delete", json={"path": "/srv/repos/old.txt"})

    assert [create.status_code, rename.status_code, delete.status_code] == [400, 400, 400]
    assert all("explicit profile" in response.text for response in (create, rename, delete))
