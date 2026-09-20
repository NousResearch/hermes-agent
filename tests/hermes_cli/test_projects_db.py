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






def test_create_get_list(conn, tmp_path):
    folder = tmp_path / "hermes"
    pid = pdb.create_project(conn, name="Hermes Agent", folders=[str(folder)])
    proj = pdb.get_project(conn, pid)

    assert proj is not None
    assert proj.slug == "hermes-agent"
    assert proj.name == "Hermes Agent"
    # First folder becomes primary.
    assert proj.primary_path == str(folder)
    assert [f.path for f in proj.folders] == [str(folder)]
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


def test_session_home_tristate_and_bulk_lookup(conn):
    project_id = pdb.create_project(conn, name="App", folders=["/www/app"])

    assert pdb.get_session_home_row(conn, "root-absent") is None
    assert pdb.session_home_overrides(conn) == {}

    pdb.set_session_home(conn, "root-project", project_id)
    pdb.set_session_home(conn, "root-unfiled", None)

    assert pdb.get_session_home_row(conn, "root-project")["project_id"] == project_id
    assert pdb.get_session_home_row(conn, "root-unfiled")["project_id"] is None
    assert pdb.session_home_overrides(conn) == {
        "root-project": project_id,
        "root-unfiled": None,
    }

    assert pdb.clear_session_home(conn, "root-project") is True
    assert pdb.clear_session_home(conn, "root-project") is False
    assert pdb.get_session_home_row(conn, "root-project") is None


def test_session_home_rejects_unknown_project_but_retains_archived_assignment(conn):
    with pytest.raises(ValueError, match="no such project"):
        pdb.set_session_home(conn, "root", "p_missing")

    project_id = pdb.create_project(conn, name="App", folders=["/www/app"])
    pdb.set_session_home(conn, "root", project_id)
    pdb.archive_project(conn, project_id)

    assert pdb.get_session_home_row(conn, "root")["project_id"] == project_id
    assert pdb.session_roots_for_project(conn, project_id) == ["root"]

    pdb.restore_project(conn, project_id)
    assert pdb.session_roots_for_project(conn, project_id) == ["root"]


def test_deleting_project_unfiles_assigned_lineages_without_touching_existing_unfiled(conn):
    project_id = pdb.create_project(conn, name="App", folders=["/www/app"])
    pdb.set_session_home(conn, "root-project", project_id)
    pdb.set_session_home(conn, "root-unfiled", None)

    assert pdb.delete_project(conn, project_id) is True

    assert pdb.get_session_home_row(conn, "root-project")["project_id"] is None
    assert pdb.get_session_home_row(conn, "root-unfiled")["project_id"] is None






def test_per_profile_isolation(tmp_path):
    # Two distinct DB paths stand in for two profiles' HERMES_HOME.
    a = pdb.connect(db_path=tmp_path / "a" / "projects.db")
    b = pdb.connect(db_path=tmp_path / "b" / "projects.db")
    project_root = tmp_path / "workspace-a"
    scanned_root = project_root / "scanned"
    try:
        pdb.create_project(a, name="Only In A", folders=[str(project_root)])
        pdb.record_discovered_repos(a, [(str(scanned_root), "scanned")])

        assert [p.slug for p in pdb.list_projects(a)] == ["only-in-a"]
        assert pdb.list_projects(b) == []
        assert [row["root"] for row in pdb.list_discovered_repos(a)] == [
            str(scanned_root)
        ]
        assert pdb.list_discovered_repos(b) == []
    finally:
        a.close()
        b.close()
