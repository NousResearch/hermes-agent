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


def test_per_profile_isolation(tmp_path):
    # Two distinct DB paths stand in for two profiles' HERMES_HOME.
    a = pdb.connect(db_path=tmp_path / "a" / "projects.db")
    b = pdb.connect(db_path=tmp_path / "b" / "projects.db")
    try:
        pdb.create_project(a, name="Only In A", folders=["/a"])
        pdb.record_discovered_repos(a, [("/a/scanned", "scanned")])

        assert [p.slug for p in pdb.list_projects(a)] == ["only-in-a"]
        assert pdb.list_projects(b) == []
        assert [row["root"] for row in pdb.list_discovered_repos(a)] == ["/a/scanned"]
        assert pdb.list_discovered_repos(b) == []
    finally:
        a.close()
        b.close()


# -- Marker-file identity (survives rename/move) ---------------------------


def test_marker_written_on_create_when_dir_exists(tmp_path, conn):
    folder = tmp_path / "Foo"
    folder.mkdir()
    pid = pdb.create_project(conn, name="Foo", folders=[str(folder)])
    marker = folder / ".hermes" / "project.json"
    assert marker.exists(), "create should write marker when dir exists"
    data = __import__("json").loads(marker.read_text(encoding="utf-8"))
    assert data["id"] == pid
    assert data["name"] == "Foo"


def test_marker_survives_rename_and_heals_registry(tmp_path, conn):
    foo = tmp_path / "Foo"
    foo.mkdir()
    pid = pdb.create_project(conn, name="Foo", folders=[str(foo)])
    # marker written at Foo
    assert (foo / ".hermes" / "project.json").exists()
    foo_v2 = tmp_path / "Foo-v2"
    foo.rename(foo_v2)
    # new cwd inside renamed folder — marker walk must resolve same id
    sub = foo_v2 / "src"
    sub.mkdir()
    proj = pdb.project_for_path(conn, str(sub))
    assert proj is not None and proj.id == pid
    # heal: registry now contains new path as primary, old kept for history
    healed = pdb.get_project(conn, pid)
    assert healed is not None
    paths = {p.path for p in healed.folders}
    assert str(foo_v2) in paths
    assert str(foo) in paths  # old kept so old sessions stay grouped
    assert healed.primary_path == str(foo_v2)
    # old path still groups (session history)
    p_old = pdb.project_for_path(conn, str(foo / "x"))
    assert p_old is not None and p_old.id == pid
    # new sibling also groups
    p_new = pdb.project_for_path(conn, str(foo_v2 / "x"))
    assert p_new is not None and p_new.id == pid


def test_marker_corrupt_is_ignored_and_parent_wins(tmp_path, conn):
    foo = tmp_path / "Foo"
    foo.mkdir()
    pid = pdb.create_project(conn, name="Foo", folders=[str(foo)])
    sub = foo / "sub"
    sub.mkdir()
    bad = sub / ".hermes" / "project.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{ not json", encoding="utf-8")
    deeper = sub / "deep"
    deeper.mkdir()
    proj = pdb.project_for_path(conn, str(deeper))
    assert proj is not None and proj.id == pid


def test_marker_unknown_id_is_ignored(tmp_path, conn):
    other = tmp_path / "Other"
    other.mkdir()
    bad = other / ".hermes" / "project.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        __import__("json").dumps({"id": "p_00000000", "name": "ghost"}),
        encoding="utf-8",
    )
    assert pdb.project_for_path(conn, str(other / "src")) is None


def test_marker_fallback_to_registry_when_removed(tmp_path, conn):
    plain = tmp_path / "Plain"
    plain.mkdir()
    pid = pdb.create_project(conn, name="Plain", folders=[str(plain)])
    marker = plain / ".hermes" / "project.json"
    assert marker.exists()
    marker.unlink()
    # fallback to registry via folder prefix
    proj = pdb.project_for_path(conn, str(plain / "src"))
    assert proj is not None and proj.id == pid


def test_marker_nearest_wins_for_nested_projects(tmp_path, conn):
    outer = tmp_path / "outer"
    outer.mkdir()
    inner = outer / "inner"
    inner.mkdir()
    pid_outer = pdb.create_project(conn, name="Outer", folders=[str(outer)])
    pid_inner = pdb.create_project(conn, name="Inner", folders=[str(inner)])
    assert (outer / ".hermes" / "project.json").exists()
    assert (inner / ".hermes" / "project.json").exists()
    # cwd inside inner must resolve to inner (nearest marker)
    p_inner = pdb.project_for_path(conn, str(inner / "src"))
    assert p_inner is not None and p_inner.id == pid_inner
    # cwd inside outer but outside inner resolves to outer
    p_outer = pdb.project_for_path(conn, str(outer / "other"))
    assert p_outer is not None and p_outer.id == pid_outer


def test_set_primary_writes_new_marker_and_cleans_old(tmp_path, conn):
    a = tmp_path / "A"
    b = tmp_path / "B"
    a.mkdir()
    b.mkdir()
    pid = pdb.create_project(conn, name="Proj", folders=[str(a)])
    assert (a / ".hermes" / "project.json").exists()
    # add b as second folder then make it primary
    pdb.add_folder(conn, pid, str(b))
    assert pdb.set_primary(conn, pid, str(b)) is True
    assert (b / ".hermes" / "project.json").exists()
    # old primary marker removed
    assert not (a / ".hermes" / "project.json").exists()


def test_remove_folder_cleans_marker(tmp_path, conn):
    foo = tmp_path / "Foo"
    foo.mkdir()
    pid = pdb.create_project(conn, name="Foo", folders=[str(foo)])
    marker = foo / ".hermes" / "project.json"
    assert marker.exists()
    pdb.remove_folder(conn, pid, str(foo))
    assert not marker.exists()
