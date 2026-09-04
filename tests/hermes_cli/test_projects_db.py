"""Tests for the per-profile Projects store (hermes_cli/projects_db)."""

from __future__ import annotations

import os

import pytest

from hermes_cli import projects_db as pdb


@pytest.fixture
def conn(tmp_path):
    c = pdb.connect(db_path=tmp_path / "projects.db")
    try:
        yield c
    finally:
        c.close()






def test_discovery_policy_change_clears_only_discovered_rows(conn):
    project_id = pdb.create_project(conn, name="Explicit", folders=["/www/explicit"])
    pdb.record_discovered_repos(
        conn, [("/www/scanned", "scanned")], policy_key="policy-a"
    )

    assert pdb.reconcile_discovered_repos_policy(conn, "policy-b") is True
    assert pdb.list_discovered_repos(conn) == []
    assert pdb.get_project(conn, project_id) is not None
    assert pdb.get_discovery_policy_key(conn) == "policy-b"






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












def test_project_for_path_skips_archived(conn):
    pid = pdb.create_project(conn, name="P", folders=["/www/app"])
    pdb.archive_project(conn, pid)

    assert pdb.project_for_path(conn, "/www/app/src") is None
    # Archived hidden from the default list but visible with include_archived.
    assert pdb.list_projects(conn) == []
    assert len(pdb.list_projects(conn, include_archived=True)) == 1

    pdb.restore_project(conn, pid)
    assert pdb.project_for_path(conn, "/www/app/src").id == pid


def test_create_dedups_by_primary_path(conn):
    pid = pdb.create_project(conn, name="GeoTrace", folders=["/www/geotrace"])

    # Same folder again (any name): refused, existing project named in error.
    with pytest.raises(ValueError, match="already belongs to project 'geotrace'"):
        pdb.create_project(conn, name="GeoTrace", folders=["/www/geotrace"])
    with pytest.raises(ValueError, match="already belongs"):
        pdb.create_project(conn, name="Other Name", primary_path="/www/geotrace")

    # Trailing-separator spelling of the same folder is still a duplicate.
    with pytest.raises(ValueError, match="already belongs"):
        pdb.create_project(conn, name="GeoTrace", primary_path="/www/geotrace/")

    # Deliberate duplicates stay possible.
    dup = pdb.create_project(
        conn, name="GeoTrace", folders=["/www/geotrace"], allow_duplicate_path=True
    )
    assert dup != pid
    assert len(pdb.list_projects(conn)) == 2


def test_create_dedup_ignores_archived_and_other_paths(conn):
    pid = pdb.create_project(conn, name="App", folders=["/www/app"])
    pdb.archive_project(conn, pid)

    # Archived project no longer blocks the path.
    fresh = pdb.create_project(conn, name="App", folders=["/www/app"])
    assert fresh != pid

    # Different folder is never a collision; folder-less projects don't match.
    pdb.create_project(conn, name="Elsewhere", folders=["/www/other"])
    pdb.create_project(conn, name="No Folder")


def test_find_by_primary_path(conn):
    pid = pdb.create_project(conn, name="App", folders=["/www/app"])

    assert pdb.find_by_primary_path(conn, "/www/app").id == pid
    assert pdb.find_by_primary_path(conn, "/www/app/").id == pid
    assert pdb.find_by_primary_path(conn, "/www/nope") is None
    assert pdb.find_by_primary_path(conn, "") is None






def test_delete_project_tombstones_its_folders(conn):
    # Deleting a project records its folders so the tree builder does not
    # re-promote them as auto projects (which would make the delete a no-op).
    pid = pdb.create_project(
        conn, name="Quarkus", folders=["/ws/main-quarkus", "/ws/extra"]
    )

    assert pdb.is_folder_tombstoned(conn, "/ws/main-quarkus") is False

    pdb.delete_project(conn, pid)

    assert pdb.is_folder_tombstoned(conn, "/ws/main-quarkus") is True
    assert pdb.is_folder_tombstoned(conn, "/ws/extra") is True
    assert pdb.is_folder_tombstoned(conn, "/ws/unrelated") is False


def test_tombstones_survive_a_reconnect(tmp_path):
    # The suppression has to outlive the process: the tree is rebuilt on every
    # app start, and an in-memory-only tombstone would let the row come back.
    db = tmp_path / "projects.db"
    first = pdb.connect(db_path=db)
    try:
        pid = pdb.create_project(first, name="Gone", folders=["/ws/gone"])
        pdb.delete_project(first, pid)
    finally:
        first.close()

    second = pdb.connect(db_path=db)
    try:
        assert pdb.is_folder_tombstoned(second, "/ws/gone") is True
    finally:
        second.close()


def test_creating_a_project_clears_the_tombstone(conn):
    # Re-creating a project on a deleted folder is newer intent than the delete.
    pid = pdb.create_project(conn, name="Again", folders=["/ws/again"])
    pdb.delete_project(conn, pid)
    assert pdb.is_folder_tombstoned(conn, "/ws/again") is True

    pdb.create_project(conn, name="Again 2", folders=["/ws/again"])

    assert pdb.is_folder_tombstoned(conn, "/ws/again") is False


def test_adding_a_folder_clears_its_tombstone(conn):
    # Same intent via the other write path: attaching a previously deleted
    # folder to a live project must un-suppress it.
    gone = pdb.create_project(conn, name="Gone", folders=["/ws/shared"])
    pdb.delete_project(conn, gone)
    keeper = pdb.create_project(conn, name="Keeper", folders=["/ws/keeper"])

    pdb.add_folder(conn, keeper, "/ws/shared")

    assert pdb.is_folder_tombstoned(conn, "/ws/shared") is False


def test_tombstone_lookup_normalizes_the_path(conn):
    # The builder passes git roots as reported by git; the store's own folders
    # are normalized, so both spellings must resolve to the same tombstone.
    pid = pdb.create_project(conn, name="Norm", folders=["/ws/norm"])
    pdb.delete_project(conn, pid)

    assert pdb.is_folder_tombstoned(conn, "/ws/norm/") is True
    assert pdb.is_folder_tombstoned(conn, "") is False


def test_tombstoned_folders_lists_every_suppressed_path(conn):
    # The gateway builds one predicate per tree read, so it needs the whole set
    # in a single query rather than a lookup per candidate folder.
    a = pdb.create_project(conn, name="A", folders=["/ws/a"])
    b = pdb.create_project(conn, name="B", folders=["/ws/b1", "/ws/b2"])
    pdb.create_project(conn, name="Live", folders=["/ws/live"])
    pdb.delete_project(conn, a)
    pdb.delete_project(conn, b)

    assert pdb.tombstoned_folders(conn) == {"/ws/a", "/ws/b1", "/ws/b2"}


def test_per_profile_isolation(tmp_path):
    # Two distinct DB paths stand in for two profiles' HERMES_HOME.
    a = pdb.connect(db_path=tmp_path / "a" / "projects.db")
    b = pdb.connect(db_path=tmp_path / "b" / "projects.db")
    try:
        pdb.create_project(a, name="Only In A", folders=["/a"])
        pdb.record_discovered_repos(a, [("/a/scanned", "scanned")])

        assert [p.slug for p in pdb.list_projects(a)] == ["only-in-a"]
        assert pdb.list_projects(b) == []
        assert [row["root"] for row in pdb.list_discovered_repos(a)] == [
            "/a/scanned"
        ]
        assert pdb.list_discovered_repos(b) == []
    finally:
        a.close()
        b.close()


