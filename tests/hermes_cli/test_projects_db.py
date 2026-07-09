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


def test_move_folder_preserves_attributes_and_updates_primary(conn, tmp_path):
    # Moving the primary folder retargets projects.primary_path and keeps
    # is_primary / label / added_at intact (the only behavior that callers
    # downstream — desktop sidebar tree, project_for_path — depend on).
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    pid = pdb.create_project(conn, name="P", folders=[str(old)])
    new_path = pdb.move_folder(conn, pid, str(old), str(new))
    assert new_path == str(new)

    proj = pdb.get_project(conn, pid)
    assert proj.primary_path == str(new)
    assert [f.path for f in proj.folders] == [str(new)]
    assert proj.folders[0].is_primary is True


def test_move_folder_preserves_label(conn, tmp_path):
    # The label lives on the row; a move must not drop it.
    a = tmp_path / "a"
    b = tmp_path / "b"
    b_renamed = tmp_path / "b-renamed"
    a.mkdir()
    b.mkdir()
    b_renamed.mkdir()
    pid = pdb.create_project(conn, name="P", folders=[str(a)])
    pdb.add_folder(conn, pid, str(b), label="docs")
    pdb.move_folder(conn, pid, str(b), str(b_renamed))

    proj = pdb.get_project(conn, pid)
    moved = next(f for f in proj.folders if f.path == str(b_renamed))
    assert moved.label == "docs"


def test_move_non_primary_does_not_touch_primary(conn, tmp_path):
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    secondary_new = tmp_path / "secondary-new"
    primary.mkdir()
    secondary.mkdir()
    secondary_new.mkdir()
    pid = pdb.create_project(conn, name="P", folders=[str(primary)])
    pdb.add_folder(conn, pid, str(secondary))

    pdb.move_folder(conn, pid, str(secondary), str(secondary_new))

    proj = pdb.get_project(conn, pid)
    assert proj.primary_path == str(primary)
    paths = {f.path for f in proj.folders}
    assert paths == {str(primary), str(secondary_new)}
    assert str(secondary) not in paths


def test_move_folder_idempotent_when_paths_match(conn, tmp_path):
    # Moving a folder onto its own current path is a no-op success — the
    # desktop dialog relies on this when a user picks the same folder again.
    a = tmp_path / "a"
    a.mkdir()
    pid = pdb.create_project(conn, name="P", folders=[str(a)])
    same = pdb.move_folder(conn, pid, str(a), str(a))
    assert same == str(a)

    proj = pdb.get_project(conn, pid)
    assert proj.primary_path == str(a)
    assert [f.path for f in proj.folders] == [str(a)]


def test_move_folder_rejects_collision(conn, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pid = pdb.create_project(conn, name="P", folders=[str(a), str(b)])
    with pytest.raises(ValueError, match="already has folder"):
        pdb.move_folder(conn, pid, str(a), str(b))


def test_move_folder_rejects_missing_old(conn, tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    pid = pdb.create_project(conn, name="P", folders=[str(a)])
    with pytest.raises(ValueError, match="folder not in project"):
        pdb.move_folder(conn, pid, str(tmp_path / "ghost"), str(tmp_path / "elsewhere"))


def test_move_folder_rejects_missing_project(conn):
    with pytest.raises(ValueError, match="no such project"):
        pdb.move_folder(conn, "p_does_not_exist", "/old", "/new")


def test_move_folder_normalizes_paths(conn, tmp_path):
    # Same normalization as add/remove: trailing slash dropped, .. collapsed.
    a_b = tmp_path / "a" / "b"
    a_b.mkdir(parents=True)
    c_e = tmp_path / "c" / "e"  # the post-normalization target
    c_e.mkdir(parents=True)

    pid = pdb.create_project(conn, name="P", folders=[str(a_b)])
    # The ".."-collapsed form is OS-portable when rooted at tmp_path: it just
    # needs _normalize_path to drop the trailing separator.
    new_path = pdb.move_folder(conn, pid, str(a_b) + os.sep, str(c_e) + os.sep)
    assert new_path == str(c_e)

    proj = pdb.get_project(conn, pid)
    assert proj.primary_path == str(c_e)


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
