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


def test_find_by_folder_path_matches_any_folder_not_just_the_primary(conn):
    a = pdb.create_project(conn, name="A", folders=["/www/a", "/www/a-shared"])
    b = pdb.create_project(conn, name="B", folders=["/www/b"])

    assert pdb.find_by_folder_path(conn, "/www/a").id == a
    # A secondary folder counts as ownership, not just the primary one.
    assert pdb.find_by_folder_path(conn, "/www/a-shared").id == a
    assert pdb.find_by_folder_path(conn, "/www/b").id == b
    assert pdb.find_by_folder_path(conn, "/www/a/").id == a
    assert pdb.find_by_folder_path(conn, "/www/nope") is None
    assert pdb.find_by_folder_path(conn, "") is None
    # The project being edited is not its own collision.
    assert pdb.find_by_folder_path(conn, "/www/a", exclude_project_id=a) is None


def test_add_folder_refuses_a_folder_another_project_owns(conn):
    owner = pdb.create_project(conn, name="Inbox", folders=["/www/inbox"])
    other = pdb.create_project(conn, name="Crypto", folders=["/www/crypto"])

    with pytest.raises(ValueError, match="already belongs to project 'inbox'"):
        pdb.add_folder(conn, other, "/www/inbox")
    with pytest.raises(ValueError, match="already belongs to project"):
        pdb.add_folder(conn, other, "/www/inbox/")

    # Refused means untouched, on both sides.
    assert [f.path for f in pdb.get_project(conn, other).folders] == ["/www/crypto"]
    assert [f.path for f in pdb.get_project(conn, owner).folders] == ["/www/inbox"]

    # Its own folder, and a genuinely free one, still go through.
    pdb.add_folder(conn, other, "/www/extra")
    pdb.add_folder(conn, other, "/www/crypto")
    assert sorted(f.path for f in pdb.get_project(conn, other).folders) == [
        "/www/crypto", "/www/extra"]


def test_add_folder_ignores_an_archived_owner(conn):
    owner = pdb.create_project(conn, name="Old", folders=["/www/old"])
    other = pdb.create_project(conn, name="New", folders=["/www/new"])
    pdb.archive_project(conn, owner)

    pdb.add_folder(conn, other, "/www/old")
    assert "/www/old" in [f.path for f in pdb.get_project(conn, other).folders]


def test_create_dedups_a_secondary_folder_collision(conn):
    pdb.create_project(conn, name="A", folders=["/www/a"])

    # The collision is not the FIRST folder, so the primary-only guard would let it past.
    with pytest.raises(ValueError, match="already belongs to project 'a'"):
        pdb.create_project(conn, name="C", folders=["/www/c", "/www/a"])

    assert [p.slug for p in pdb.list_projects(conn)] == ["a"]






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


