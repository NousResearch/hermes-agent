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






def test_branch_name_for_sanitizes_git_refname():
    """branch_name_for must produce valid git refnames for all title inputs."""
    proj = pdb.Project(id="p_1", slug="myproj", name="My Project", created_at=0)

    # Double-dot collapsed to single dot
    branch = pdb.branch_name_for(proj, "t_1", title="fix 2.0..rc1")
    assert ".." not in branch
    assert branch == "myproj/t_1-fix-2.0.rc1"

    # Trailing .lock stripped
    branch = pdb.branch_name_for(proj, "t_2", title="update yarn.lock")
    assert not branch.endswith(".lock")
    assert branch == "myproj/t_2-update-yarn"

    # Triple dot collapsed
    branch = pdb.branch_name_for(proj, "t_3", title="v3...beta")
    assert ".." not in branch
    assert branch == "myproj/t_3-v3.beta"

    # Leading dot stripped
    branch = pdb.branch_name_for(proj, "t_4", title=".hidden config")
    assert branch == "myproj/t_4-hidden-config"

    # .lock in the middle is preserved (only trailing .lock is invalid)
    branch = pdb.branch_name_for(proj, "t_5", title="cleanup v2.0.lock.bak")
    assert "lock" in branch

    # Single dots preserved for version numbers
    branch = pdb.branch_name_for(proj, "t_6", title="v2.0.0 release")
    assert branch == "myproj/t_6-v2.0.0-release"

    # Degenerate title (only dots) → no slug appended
    branch = pdb.branch_name_for(proj, "t_7", title="....")
    assert branch == "myproj/t_7"

    # Just ".lock" → no slug appended
    branch = pdb.branch_name_for(proj, "t_8", title=".lock")
    assert branch == "myproj/t_8"


def test_branch_name_for_passes_git_check_ref_format():
    """Every branch name must pass git check-ref-format --branch."""
    import subprocess

    proj = pdb.Project(id="p_1", slug="app", name="App", created_at=0)
    titles = [
        "fix 2.0..rc1", "update yarn.lock", "v3...beta",
        ".hidden", "trailing.", "normal title", "v1.2.3",
        "....", ".lock", "", "a" * 100,
    ]
    for title in titles:
        branch = pdb.branch_name_for(proj, "t_x", title=title)
        result = subprocess.run(
            ["git", "check-ref-format", "--branch", branch],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"git check-ref-format rejected {branch!r} (title={title!r})"
        )


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


