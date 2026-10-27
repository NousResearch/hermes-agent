"""Tests for the per-profile Projects store (hermes_cli/projects_db)."""

from __future__ import annotations

import os
import sqlite3

import pytest

from hermes_cli import projects_db as pdb


@pytest.fixture
def conn(tmp_path):
    c = pdb.connect(db_path=tmp_path / "projects.db")
    try:
        yield c
    finally:
        c.close()


def _row(connection, project_id: str) -> pdb.Project:
    """``get_project`` with the missing-row assertion folded in, so tests can read attributes."""
    project = pdb.get_project(connection, project_id)
    assert project is not None
    return project






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
    assert proj.primary_path == os.path.abspath("/tmp/hermes")
    assert [f.path for f in proj.folders] == [os.path.abspath("/tmp/hermes")]
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
            os.path.abspath("/a/scanned")
        ]
        assert pdb.list_discovered_repos(b) == []
    finally:
        a.close()
        b.close()


def test_create_project_with_a_parent_nests_it(conn):
    parent = pdb.create_project(conn, name="Dev", folders=["/www/dev"])
    child = pdb.create_project(
        conn, name="Align", folders=["/www/dev/m4l/align"], parent_id=parent
    )

    assert _row(conn, child).parent_id == parent
    assert _row(conn, parent).parent_id is None


def test_create_project_rejects_an_unknown_parent(conn):
    with pytest.raises(ValueError, match="no such parent"):
        pdb.create_project(conn, name="Orphan", folders=["/www/o"], parent_id="p_nope")


def test_set_project_parent_moves_a_project_in_and_out(conn):
    parent = pdb.create_project(conn, name="Dev", folders=["/www/dev"])
    child = pdb.create_project(conn, name="Align", folders=["/www/dev/m4l/align"])
    # Unset is the containment default, not "top level" — the stored value has to say which.
    assert _row(conn, child).parent_id is None

    assert pdb.set_project_parent(conn, child, parent) == parent
    assert _row(conn, child).parent_id == parent

    assert pdb.set_project_parent(conn, child, pdb.PARENT_TOP_LEVEL) == ""
    assert _row(conn, child).parent_id == ""

    assert pdb.set_project_parent(conn, child, None) is None
    assert _row(conn, child).parent_id is None


def test_set_project_parent_rejects_self_and_descendants(conn):
    top = pdb.create_project(conn, name="Top", folders=["/www/top"])
    mid = pdb.create_project(conn, name="Mid", folders=["/www/mid"], parent_id=top)
    leaf = pdb.create_project(conn, name="Leaf", folders=["/www/leaf"], parent_id=mid)

    with pytest.raises(ValueError, match="own parent"):
        pdb.set_project_parent(conn, top, top)
    with pytest.raises(ValueError, match="descendant"):
        pdb.set_project_parent(conn, top, leaf)
    with pytest.raises(ValueError, match="descendant"):
        pdb.set_project_parent(conn, mid, leaf)

    # A refused move writes nothing.
    assert _row(conn, top).parent_id is None
    assert _row(conn, mid).parent_id == top


def test_set_project_parent_rejects_an_unknown_project(conn):
    with pytest.raises(ValueError, match="no such project"):
        pdb.set_project_parent(conn, "p_nope", None)


def test_parent_id_round_trips_through_to_dict(conn):
    parent = pdb.create_project(conn, name="Dev", folders=["/www/dev"])
    child = pdb.create_project(conn, name="Align", folders=["/www/x"], parent_id=parent)

    payload = _row(conn, child).to_dict()

    assert payload["parent_id"] == parent


def test_legacy_db_without_the_parent_column_upgrades_in_place(tmp_path):
    """The column is additive: opening an old DB must grant it, not crash on the missing field."""
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE projects (id TEXT PRIMARY KEY, slug TEXT NOT NULL UNIQUE, name TEXT NOT NULL,"
        " created_at INTEGER NOT NULL, archived INTEGER NOT NULL DEFAULT 0)"
    )
    legacy.execute(
        "INSERT INTO projects (id, slug, name, created_at, archived)"
        " VALUES ('p_old', 'old', 'Old', 1, 0)"
    )
    legacy.commit()
    legacy.close()

    upgraded = pdb.connect(db_path=path)
    try:
        assert _row(upgraded, "p_old").parent_id is None
        pdb.set_project_parent(upgraded, "p_old", pdb.PARENT_TOP_LEVEL)
        assert _row(upgraded, "p_old").parent_id == ""
    finally:
        upgraded.close()


def test_a_move_under_a_project_this_one_contains_is_refused(conn, tmp_path):
    """Containment is a parent link too.

    B's folder sits inside A's, so the sidebar already nests B under A by folder containment. Moving A
    under B would leave each one nested under the other — a cycle every tree build then renders in both
    directions — and a walk over the explicit links alone does not see it.
    """
    outer = tmp_path / "a"
    (outer / "b").mkdir(parents=True)
    a = pdb.create_project(conn, name="A", folders=[str(outer)])
    b = pdb.create_project(conn, name="B", folders=[str(outer / "b")])

    with pytest.raises(ValueError):
        pdb.set_project_parent(conn, a, b)

    assert _row(conn, a).parent_id is None


def test_a_new_project_whose_folder_contains_its_parent_is_refused(conn, tmp_path):
    """The same loop from the other end: a subproject created into a folder that holds its parent."""
    outer = tmp_path / "a"
    (outer / "child").mkdir(parents=True)
    parent = pdb.create_project(conn, name="Child", folders=[str(outer / "child")])

    with pytest.raises(ValueError):
        pdb.create_project(conn, name="Outer", folders=[str(outer)], parent_id=parent)


