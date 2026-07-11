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


def test_fs_list_accepts_relative_paths(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rel").mkdir()
    (tmp_path / "rel" / "file.txt").write_text("ok")

    response = client.get("/api/fs/list", params={"path": "rel"})

    assert response.status_code == 200
    assert response.json()["entries"] == [
        {"name": "file.txt", "path": str(tmp_path / "rel" / "file.txt"), "isDirectory": False}
    ]


def test_fs_list_missing_path_returns_structured_error(client, tmp_path):
    response = client.get("/api/fs/list", params={"path": str(tmp_path / "missing")})

    assert response.status_code == 200
    assert response.json() == {"entries": [], "error": "ENOENT"}


def test_fs_read_text_matches_preview_shape_and_truncates(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "_FS_TEXT_SOURCE_MAX_BYTES", 32)
    monkeypatch.setattr(web_server, "_FS_TEXT_PREVIEW_MAX_BYTES", 5)
    target = tmp_path / "sample.py"
    target.write_text("print('hello')")

    response = client.get("/api/fs/read-text", params={"path": str(target)})

    assert response.status_code == 200
    assert response.json() == {
        "binary": False,
        "byteSize": 14,
        "language": "python",
        "mimeType": "text/x-python",
        "path": str(target),
        "text": "print",
        "truncated": True,
    }


def test_fs_read_text_rejects_source_over_cap(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "_FS_TEXT_SOURCE_MAX_BYTES", 4)
    target = tmp_path / "large.txt"
    target.write_text("12345")

    response = client.get("/api/fs/read-text", params={"path": str(target)})

    assert response.status_code == 413


def test_fs_read_text_flags_binary(client, tmp_path):
    target = tmp_path / "blob.bin"
    target.write_bytes(b"hello\x00world")

    response = client.get("/api/fs/read-text", params={"path": str(target)})

    assert response.status_code == 200
    body = response.json()
    assert body["binary"] is True
    assert body["text"].startswith("hello")


def test_fs_read_data_url_returns_capped_data_url(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "_FS_DATA_URL_MAX_BYTES", 16)
    target = tmp_path / "image.png"
    target.write_bytes(b"pngbytes")

    response = client.get("/api/fs/read-data-url", params={"path": str(target)})

    assert response.status_code == 200
    assert response.json() == {"dataUrl": "data:image/png;base64," + base64.b64encode(b"pngbytes").decode("ascii")}


def test_fs_read_data_url_rejects_over_cap(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "_FS_DATA_URL_MAX_BYTES", 3)
    target = tmp_path / "image.png"
    target.write_bytes(b"1234")

    response = client.get("/api/fs/read-data-url", params={"path": str(target)})

    assert response.status_code == 413


def test_fs_git_root_for_nested_file(client, tmp_path):
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "pkg" / "mod"
    nested.mkdir(parents=True)
    target = nested / "file.py"
    target.write_text("x")

    response = client.get("/api/fs/git-root", params={"path": str(target)})

    assert response.status_code == 200
    assert response.json() == {"root": str(tmp_path)}


def test_fs_git_root_returns_null_outside_repo(client, tmp_path):
    response = client.get("/api/fs/git-root", params={"path": str(tmp_path)})

    assert response.status_code == 200
    assert response.json() == {"root": None}


def test_fs_default_cwd_prefers_existing_terminal_cwd(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "load_config", lambda: {"terminal": {"cwd": str(tmp_path)}})
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "env"))
    monkeypatch.setattr(web_server.Path, "cwd", lambda: tmp_path / "process")
    monkeypatch.setattr(web_server, "_fs_git_branch", lambda cwd: "main")

    response = client.get("/api/fs/default-cwd")

    assert response.status_code == 200
    assert response.json() == {"cwd": str(tmp_path), "branch": "main"}


def test_fs_default_cwd_falls_back_when_terminal_cwd_is_invalid(client, tmp_path, monkeypatch):
    fallback = tmp_path / "backend"
    fallback.mkdir()
    monkeypatch.setattr(web_server, "load_config", lambda: {"terminal": {"cwd": "/client/missing"}})
    monkeypatch.setenv("TERMINAL_CWD", "/client/missing")
    monkeypatch.setattr(web_server.Path, "cwd", lambda: fallback)
    monkeypatch.setattr(web_server, "_fs_git_branch", lambda cwd: "")

    response = client.get("/api/fs/default-cwd")

    assert response.status_code == 200
    assert response.json() == {"cwd": str(fallback), "branch": ""}


def test_fs_endpoints_require_auth(tmp_path):
    client = TestClient(web_server.app)
    target = tmp_path / "secret.txt"
    target.write_text("secret")

    list_response = client.get("/api/fs/list", params={"path": str(tmp_path)})
    read_response = client.get("/api/fs/read-text", params={"path": str(target)})
    default_response = client.get("/api/fs/default-cwd")
    mutation_response = client.post("/api/fs/mkdir", json={"parent": str(tmp_path), "name": "blocked"})

    assert list_response.status_code == 401
    assert read_response.status_code == 401
    assert default_response.status_code == 401
    assert mutation_response.status_code == 401
    assert not (tmp_path / "blocked").exists()


def test_fs_mutations_create_rename_move_and_delete(client, tmp_path):
    root = tmp_path / "project"
    archive = tmp_path / "archive"
    root.mkdir()
    archive.mkdir()

    created = client.post("/api/fs/mkdir", json={"parent": str(root), "name": "folder"})
    assert created.status_code == 200
    source = root / "note.txt"
    source.write_text("hello")
    renamed = client.post("/api/fs/rename", json={"path": str(source), "name": "renamed.txt"})
    assert renamed.status_code == 200
    moved = client.post(
        "/api/fs/move",
        json={"source": str(root / "renamed.txt"), "destination": str(archive), "browserRoot": str(root)},
    )
    assert moved.status_code == 200
    deleted = client.post(
        "/api/fs/delete", json={"path": str(archive / "renamed.txt"), "browserRoot": str(root)}
    )
    assert deleted.status_code == 200
    assert not (archive / "renamed.txt").exists()


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b", "bad\0name"])
def test_fs_mkdir_rejects_invalid_basename(client, tmp_path, name):
    response = client.post("/api/fs/mkdir", json={"parent": str(tmp_path), "name": name})
    assert response.status_code == 400


def test_fs_mutations_reject_roots_descendants_collisions_and_sensitive_paths(client, tmp_path):
    root = tmp_path / "root"
    source = root / "source"
    child = source / "child"
    target = root / "target"
    child.mkdir(parents=True)
    target.mkdir()
    (target / "source").mkdir()

    root_delete = client.post("/api/fs/delete", json={"path": str(root), "browserRoot": str(root)})
    descendant_move = client.post(
        "/api/fs/move", json={"source": str(source), "destination": str(child), "browserRoot": str(root)}
    )
    collision = client.post(
        "/api/fs/move", json={"source": str(source), "destination": str(target), "browserRoot": str(root)}
    )
    sensitive = root / ".env"
    sensitive.write_text("secret")
    sensitive_delete = client.post(
        "/api/fs/delete", json={"path": str(sensitive), "browserRoot": str(root)}
    )
    symlink = root / "symlink"
    symlink.symlink_to(source, target_is_directory=True)
    symlink_delete = client.post(
        "/api/fs/delete", json={"path": str(symlink), "browserRoot": str(root)}
    )
    broken_symlink = root / "broken"
    broken_symlink.symlink_to(root / "missing")
    broken_create = client.post(
        "/api/fs/create-file", json={"parent": str(root), "name": broken_symlink.name}
    )

    assert root_delete.status_code == 400
    assert descendant_move.status_code == 400
    assert collision.status_code == 409
    assert sensitive_delete.status_code == 403
    assert symlink_delete.status_code == 400
    assert source.is_dir()
    assert symlink.is_symlink()
    assert broken_create.status_code == 409
    assert broken_symlink.is_symlink()
