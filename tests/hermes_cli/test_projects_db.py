"""Tests for the per-profile Projects store (hermes_cli/projects_db)."""

from __future__ import annotations

import os
import sqlite3
import threading

import pytest

from hermes_cli import projects_db as pdb


@pytest.fixture
def conn(tmp_path):
    c = pdb.connect(db_path=tmp_path / "projects.db")
    try:
        yield c
    finally:
        c.close()


def test_record_and_list_discovered_repos(conn):
    n = pdb.record_discovered_repos(conn, [("/www/alpha", "alpha"), ("/www/beta", None)])
    assert n == 2

    rows = {r["root"]: r["label"] for r in pdb.list_discovered_repos(conn)}
    assert rows["/www/alpha"] == "alpha"
    # Label defaults to the basename when not given.
    assert rows["/www/beta"] == "beta"


def test_record_discovered_repos_upserts(conn):
    pdb.record_discovered_repos(conn, [("/www/alpha", "old")])
    pdb.record_discovered_repos(conn, [("/www/alpha", "new")])

    rows = pdb.list_discovered_repos(conn)
    assert len(rows) == 1
    assert rows[0]["label"] == "new"


def test_record_discovered_repos_replace_drops_stale_rows(conn):
    pdb.record_discovered_repos(conn, [("/www/alpha", "alpha"), ("/www/beta", "beta")])
    pdb.record_discovered_repos(conn, [("/www/alpha", "fresh")], replace=True)

    rows = {r["root"]: r["label"] for r in pdb.list_discovered_repos(conn)}
    assert rows == {"/www/alpha": "fresh"}


def test_create_get_list(conn):
    pid = pdb.create_project(conn, name="Hermes Agent", folders=["/tmp/hermes"])
    proj = pdb.get_project(conn, pid)

    assert proj is not None
    assert proj.slug == "hermes-agent"
    assert proj.name == "Hermes Agent"
    # First folder becomes primary.
    assert proj.primary_path == "/tmp/hermes"
    assert [f.path for f in proj.folders] == ["/tmp/hermes"]
    assert proj.folders[0].is_primary is True

    # Lookup by slug too.
    assert pdb.get_project(conn, "hermes-agent").id == pid
    assert len(pdb.list_projects(conn)) == 1


def test_slug_collision_disambiguates(conn):
    pdb.create_project(conn, name="Hermes Agent")
    pdb.create_project(conn, name="Hermes Agent")
    slugs = sorted(p.slug for p in pdb.list_projects(conn))

    assert slugs == ["hermes-agent", "hermes-agent-2"]


def test_empty_name_rejected(conn):
    with pytest.raises(ValueError):
        pdb.create_project(conn, name="   ")


def test_add_remove_folder_and_primary_repoint(conn):
    pid = pdb.create_project(conn, name="P", folders=["/a"])
    pdb.add_folder(conn, pid, "/b")
    pdb.add_folder(conn, pid, "/c", is_primary=True)

    proj = pdb.get_project(conn, pid)
    assert proj.primary_path == "/c"
    assert {f.path for f in proj.folders} == {"/a", "/b", "/c"}

    # Removing the primary repoints to the oldest remaining folder.
    pdb.remove_folder(conn, pid, "/c")
    proj = pdb.get_project(conn, pid)
    assert proj.primary_path == "/a"

    # Removing the last folder clears the primary.
    pdb.remove_folder(conn, pid, "/a")
    pdb.remove_folder(conn, pid, "/b")
    proj = pdb.get_project(conn, pid)
    assert proj.primary_path is None
    assert proj.folders == []


def test_set_primary_requires_existing_folder(conn):
    pid = pdb.create_project(conn, name="P", folders=["/a"])
    assert pdb.set_primary(conn, pid, "/nope") is False
    assert pdb.set_primary(conn, pid, "/a") is True


def test_paths_normalized(conn):
    pid = pdb.create_project(conn, name="P", folders=["/a/b/../c/"])
    proj = pdb.get_project(conn, pid)
    # Trailing slash stripped, .. collapsed.
    assert proj.primary_path == "/a/c"


def test_project_for_path_longest_prefix(conn):
    outer = pdb.create_project(conn, name="Outer", folders=["/www"])
    inner = pdb.create_project(conn, name="Inner", folders=["/www/app"])

    assert pdb.project_for_path(conn, "/www/app/src/x.py").id == inner
    assert pdb.project_for_path(conn, "/www/other").id == outer
    assert pdb.project_for_path(conn, "/elsewhere") is None
    # Segment-wise prefix only: /www/app must not match /www/application.
    assert pdb.project_for_path(conn, "/www/application").id == outer


def test_project_for_path_skips_archived(conn):
    pid = pdb.create_project(conn, name="P", folders=["/www/app"])
    pdb.archive_project(conn, pid)

    assert pdb.project_for_path(conn, "/www/app/src") is None
    # Archived hidden from the default list but visible with include_archived.
    assert pdb.list_projects(conn) == []
    assert len(pdb.list_projects(conn, include_archived=True)) == 1

    pdb.restore_project(conn, pid)
    assert pdb.project_for_path(conn, "/www/app/src").id == pid


def test_active_pointer(conn):
    pid = pdb.create_project(conn, name="P")
    assert pdb.get_active_id(conn) is None

    pdb.set_active(conn, pid)
    assert pdb.get_active_id(conn) == pid

    pdb.set_active(conn, None)
    assert pdb.get_active_id(conn) is None


def test_branch_name_for_is_deterministic():
    proj = pdb.Project(id="p_1", slug="web-app", name="Web App", created_at=0)

    assert pdb.branch_name_for(proj, "t_abc") == "web-app/t_abc"
    assert pdb.branch_name_for(proj, "t_abc", title="Add login!") == "web-app/t_abc-add-login"
    # Stable across calls.
    assert pdb.branch_name_for(proj, "t_abc") == pdb.branch_name_for(proj, "t_abc")


def test_per_profile_isolation(tmp_path):
    # Two distinct DB paths stand in for two profiles' HERMES_HOME.
    a = pdb.connect(db_path=tmp_path / "a" / "projects.db")
    b = pdb.connect(db_path=tmp_path / "b" / "projects.db")
    try:
        pdb.create_project(a, name="Only In A", folders=["/a"])

        assert [p.slug for p in pdb.list_projects(a)] == ["only-in-a"]
        assert pdb.list_projects(b) == []
    finally:
        a.close()
        b.close()


def test_db_path_under_hermes_home():
    # Resolves under HERMES_HOME (set by the autouse isolation fixture).
    assert pdb.projects_db_path().name == "projects.db"
    assert os.path.basename(str(pdb.projects_db_path().parent))  # non-empty parent


def test_concurrent_first_connect_does_not_race_wal_initialization(tmp_path):
    """Fresh stores must initialize cleanly even when desktop workers open them together."""
    db_path = tmp_path / "projects.db"
    barrier = threading.Barrier(12)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            with pdb.connect_closing(db_path) as db:
                db.execute("SELECT 1 FROM projects").fetchone()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []


def test_connect_reinitializes_replaced_database(tmp_path):
    """Replacing projects.db in-process must not leave its schema cache stale."""
    db_path = tmp_path / "projects.db"
    with pdb.connect_closing(db_path):
        pass

    replacement = tmp_path / "legacy-projects.db"
    legacy = sqlite3.connect(str(replacement))
    try:
        legacy.execute(
            "CREATE TABLE projects ("
            "id TEXT PRIMARY KEY, slug TEXT NOT NULL UNIQUE, name TEXT NOT NULL, "
            "description TEXT, created_at INTEGER NOT NULL, "
            "archived INTEGER NOT NULL DEFAULT 0)"
        )
        legacy.commit()
    finally:
        legacy.close()
    os.replace(replacement, db_path)

    with pdb.connect_closing(db_path) as db:
        project_id = pdb.create_project(
            db,
            name="Recovered",
            icon="folder",
            color="#2f855a",
            board_slug="work",
            primary_path="/tmp/recovered",
        )
        project = pdb.get_project(db, project_id)

    assert project is not None
    assert project.icon == "folder"
    assert project.board_slug == "work"
