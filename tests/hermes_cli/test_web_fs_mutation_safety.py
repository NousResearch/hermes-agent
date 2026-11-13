"""Behavior contracts for browser entry mutation admission on synthetic state."""

import os
from pathlib import Path
import threading

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server, web_fs_mutations
from hermes_cli.sqlite_safe_read import connect_tracked, offline_file_access
from hermes_cli.web_routers import files


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    user = tmp_path / "user"
    home = user / ".hermes"
    home.mkdir(parents=True)
    root = home / "work"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: user)
    monkeypatch.setenv("HOME", str(user))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
    monkeypatch.setattr(files, "_fs_backend", lambda profile=None: None)
    monkeypatch.setattr(files, "load_config", lambda: {"terminal": {"backend": "local"}})
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    # Route contracts need no gateway/lifecycle startup or background services.
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield root, client
    finally:
        client.close()


def mutate(client, operation, path, **kwargs):
    method = "DELETE" if operation == "delete" else "POST"
    return client.request(method, f"/api/fs/{operation}", json={"path": str(path), **kwargs})


@pytest.mark.parametrize("name", [".env", "config.yaml", "auth.json", "vault", "browser-profile", "webapp"])
def test_create_refuses_protected_names(workspace, name):
    root, client = workspace
    target = root / name
    assert mutate(client, "create", target).status_code == 403
    assert not target.exists()


@pytest.mark.parametrize("sensitive_source", [True, False])
def test_rename_checks_both_names(workspace, sensitive_source):
    root, client = workspace
    source = root / (".env" if sensitive_source else "notes.txt")
    source.write_text("unchanged")
    name = "notes.txt" if sensitive_source else "config.yaml"
    assert mutate(client, "rename", source, name=name).status_code == 403
    assert source.read_text() == "unchanged"
    assert not (root / name).exists()


@pytest.mark.parametrize("operation", ["rename", "delete"])
@pytest.mark.parametrize("name", [".env", "vault", "browser-profile"])
def test_tree_mutation_checks_all_descendants(workspace, operation, name):
    root, client = workspace
    target = root / "tree"
    nested = target / "nested"
    nested.mkdir(parents=True)
    (nested / name).mkdir() if name != ".env" else (nested / name).write_text("unchanged")
    response = mutate(client, operation, target, name="renamed", recursive=True)
    assert response.status_code == 403
    assert (nested / name).exists()
    assert not (root / "renamed").exists()


@pytest.mark.parametrize("operation", ["rename", "delete"])
@pytest.mark.parametrize("tree", [True, False])
def test_live_database_mutations_are_refused(workspace, operation, tree):
    root, client = workspace
    directory = root / "tree"
    directory.mkdir()
    database = directory / "notes.sqlite"
    conn = connect_tracked(database)
    try:
        conn.execute("CREATE TABLE keep (value TEXT)")
        conn.execute("INSERT INTO keep VALUES ('unchanged')")
        conn.commit()
        target = directory if tree else database
        response = mutate(client, operation, target, name="renamed", recursive=True)
        assert response.status_code == 409
        assert database.exists()
        assert conn.execute("SELECT value FROM keep").fetchone() == ("unchanged",)
    finally:
        conn.close()


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_live_database_sidecars_are_refused(workspace, suffix):
    root, client = workspace
    database = root / "notes.sqlite"
    conn = connect_tracked(database)
    try:
        sidecar = Path(str(database) + suffix)
        sidecar.touch()
        assert mutate(client, "delete", sidecar).status_code == 409
        assert sidecar.exists()
    finally:
        conn.close()


@pytest.mark.parametrize("operation", ["rename", "delete"])
def test_unreadable_subtree_refuses_before_any_effect(workspace, monkeypatch, operation):
    root, client = workspace
    target = root / "tree"
    blocked = target / "blocked"
    blocked.mkdir(parents=True)
    good = target / "keep.txt"
    good.write_text("unchanged")
    original = os.scandir

    def scan(path):
        if Path(path) == blocked:
            raise PermissionError("synthetic inaccessible subtree")
        return original(path)

    monkeypatch.setattr(web_fs_mutations.os, "scandir", scan)
    assert mutate(client, operation, target, name="renamed", recursive=True).status_code == 403
    assert good.read_text() == "unchanged"
    assert blocked.exists()


def test_rename_locked_root_is_refused(workspace, monkeypatch):
    root, client = workspace
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    assert mutate(client, "rename", root, name="renamed").status_code == 400
    assert root.is_dir()


@pytest.mark.parametrize("operation", ["rename", "delete"])
@pytest.mark.platforms("posix")
def test_final_symlink_is_mutated_as_an_entry(workspace, operation):
    root, client = workspace
    referent = root / "referent"
    referent.mkdir()
    (referent / "keep.txt").write_text("unchanged")
    link = root / "link"
    link.symlink_to(referent, target_is_directory=True)
    assert mutate(client, operation, link, name="renamed", recursive=True).status_code == 200
    assert (referent / "keep.txt").read_text() == "unchanged"
    assert not os.path.lexists(link)
    if operation == "rename":
        assert (root / "renamed").is_symlink()


def test_rename_atomically_refuses_a_late_collision(workspace, monkeypatch):
    root, client = workspace
    source = root / "source.txt"
    source.write_text("source")
    destination = root / "destination.txt"
    real_rename = files._rename_noreplace

    def race(src, dst):
        dst.write_text("winner")
        return real_rename(src, dst)

    monkeypatch.setattr(files, "_rename_noreplace", race)
    assert mutate(client, "rename", source, name=destination.name).status_code == 409
    assert source.read_text() == "source"
    assert destination.read_text() == "winner"


def test_registry_admission_and_effect_hold_one_lifecycle_lock(workspace, monkeypatch):
    root, client = workspace
    source = root / "source.txt"
    source.write_text("source")
    entered = threading.Event()
    finished = threading.Event()
    failures = []
    threads = []
    real_rename = files._rename_noreplace

    def probe():
        entered.set()
        try:
            with offline_file_access(root / "unrelated.sqlite"):
                finished.set()
        except BaseException as exc:
            failures.append(exc)

    def rename(src, dst):
        thread = threading.Thread(target=probe)
        threads.append(thread)
        thread.start()
        assert entered.wait(5)
        assert not finished.is_set()
        return real_rename(src, dst)

    monkeypatch.setattr(files, "_rename_noreplace", rename)
    assert mutate(client, "rename", source, name="renamed.txt").status_code == 200
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert not failures
    assert finished.is_set()
