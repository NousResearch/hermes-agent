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


def test_create_rejects_duplicate_primary_path(conn, tmp_path):
    """Same primary folder must not seed a second active project (#75820)."""
    root = tmp_path / "geotrace"
    root.mkdir()
    first = pdb.create_project(conn, name="GeoTrace", folders=[str(root)])
    stored = pdb.get_project(conn, first).primary_path
    assert stored == pdb._normalize_path(str(root))

    with pytest.raises(ValueError, match=r"folder already belongs to project geotrace"):
        pdb.create_project(conn, name="GeoTrace Again", folders=[str(root)])

    # Equivalent spelling (trailing slash / abspath) still collides.
    with pytest.raises(ValueError, match=r"folder already belongs to project"):
        pdb.create_project(
            conn,
            name="GeoTrace Slash",
            primary_path=str(root) + os.sep,
        )

    assert len(pdb.list_projects(conn)) == 1


def test_create_allows_distinct_primary_paths(conn, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pid_a = pdb.create_project(conn, name="A", folders=[str(a)])
    pid_b = pdb.create_project(conn, name="B", folders=[str(b)])
    assert pid_a != pid_b
    assert len(pdb.list_projects(conn)) == 2


def test_create_allows_reusing_primary_path_after_archive(conn, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    old = pdb.create_project(conn, name="Old", folders=[str(root)])
    pdb.archive_project(conn, old)

    new = pdb.create_project(conn, name="New", folders=[str(root)])
    assert new != old
    assert pdb.get_project(conn, new).primary_path
    # Only the new active project is listed by default.
    assert [p.id for p in pdb.list_projects(conn)] == [new]


def test_create_pathless_projects_are_not_path_deduped(conn):
    """Projects with no primary folder remain free to multiply by name/slug."""
    a = pdb.create_project(conn, name="Scratch")
    b = pdb.create_project(conn, name="Scratch")
    assert a != b
    assert all(p.primary_path is None for p in pdb.list_projects(conn))


def test_path_identity_key_windows_case_and_separators():
    """Windows-style paths collide on case/separators; POSIX paths stay case-sensitive."""
    win_a = pdb._path_identity_key(r"E:\JUST_DO_IT\GeoTrace")
    win_b = pdb._path_identity_key(r"e:/just_do_it/geotrace/")
    assert win_a == win_b

    posix_a = pdb._path_identity_key("/tmp/GeoTrace")
    posix_b = pdb._path_identity_key("/tmp/geotrace")
    # On a Windows host every path is case-insensitive; elsewhere POSIX differs.
    if os.name == "nt":
        assert posix_a == posix_b
    else:
        assert posix_a != posix_b


def test_find_project_by_primary_path_uses_identity_key(conn, tmp_path, monkeypatch):
    """Lookup matches via identity key even when stored spelling differs."""
    root = tmp_path / "Repo"
    root.mkdir()
    pid = pdb.create_project(conn, name="Repo", folders=[str(root)])

    found = pdb.find_project_by_primary_path(conn, str(root) + os.sep)
    assert found is not None
    assert found.id == pid

    # Simulated Windows drive path identity (skip abspath host quirks).
    monkeypatch.setattr(
        pdb,
        "_normalize_path",
        lambda p: str(p).rstrip("/\\"),
    )
    assert pdb._path_identity_key(r"C:\Work\App") == pdb._path_identity_key(
        r"c:/work/app/"
    )


