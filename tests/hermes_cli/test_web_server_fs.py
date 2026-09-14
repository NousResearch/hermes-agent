import base64
import os
import stat
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
