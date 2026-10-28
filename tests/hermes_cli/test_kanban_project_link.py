"""Kanban <-> Projects integration: project-linked tasks get a deterministic
worktree path + branch instead of the random ``wt/<task-id>`` fallback."""

from __future__ import annotations

import os

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import projects_db as pdb


@pytest.fixture
def kanban_conn(tmp_path):
    c = kbc.connect(db_path=tmp_path / "kanban.db")
    try:
        yield c
    finally:
        c.close()


def _make_project(name="Web App", repo="/tmp/webapp"):
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name=name, folders=[repo])
        return pdb.get_project(pc, pid)


def test_project_linked_task_gets_deterministic_worktree_and_branch(kanban_conn):
    proj = _make_project()
    tid = kb.create_task(kanban_conn, title="Add login", project_id=proj.slug)
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id == proj.id
    assert task.workspace_kind == "worktree"
    # Worktree dir anchored under the project's primary repo, keyed on task id.
    assert task.workspace_path == os.path.join(proj.primary_path, ".worktrees", tid)
    # Deterministic branch: <slug>/<task-id>-<title-slug>. NOT a random wt/...
    assert task.branch_name == f"{proj.slug}/{tid}-add-login"
    assert not task.branch_name.startswith("wt/")


def test_explicit_branch_overrides_project_default(kanban_conn):
    proj = _make_project()
    tid = kb.create_task(
        kanban_conn,
        title="x",
        project_id=proj.slug,
        workspace_kind="worktree",
        branch_name="feature/custom",
    )
    task = kb.get_task(kanban_conn, tid)
    assert task.branch_name == "feature/custom"


def test_unlinked_task_unchanged(kanban_conn):
    tid = kb.create_task(kanban_conn, title="plain")
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id is None
    assert task.workspace_kind == "scratch"
    # No branch is persisted — the worker still owns the wt/<id> fallback for
    # genuinely ad-hoc worktree tasks, but unlinked scratch tasks have none.
    assert task.branch_name is None


# ---------------------------------------------------------------------------
# A ``dir`` source task passes project identity on (PR2A)
#
# ``project_source_task_id`` is the cross-profile fallback: projects.db is
# per-profile while the board is shared, so the creating profile can hold no
# project at all. It recovered identity only from a canonical
# ``.worktrees/<id>`` parent; a parent pinned to a governed directory had no
# way to hand its project on. The child keeps the same directory — the
# workspace must not be turned into a worktree.
# ---------------------------------------------------------------------------

def _two_homes(tmp_path, monkeypatch):
    """``(conn, owner_home, filer_home)``: one board, two profile stores."""
    owner_home = tmp_path / "home-owner"
    filer_home = tmp_path / "home-filer"
    owner_home.mkdir()
    filer_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(owner_home))
    conn = kbc.connect(db_path=tmp_path / "kanban.db")
    return conn, owner_home, filer_home


def test_project_from_source_task_accepts_a_dir_parent(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    conn, _owner_home, filer_home = _two_homes(tmp_path, monkeypatch)
    try:
        with pdb.connect_closing() as pconn:
            project_id = pdb.create_project(pconn, name="Governed", primary_path=str(repo))
        parent = kb.create_task(
            conn, title="steward card", project_id=project_id,
            workspace_kind="dir", workspace_path=str(repo))
        # A relative dir parent is not a qualified source.
        relative_parent = kb.create_task(
            conn, title="relative card", project_id=project_id,
            workspace_kind="dir", workspace_path="relative/repo")

        # The filing profile's store has never carried the project.
        monkeypatch.setenv("HERMES_HOME", str(filer_home))

        child = kb.create_task(
            conn, title="builder child", project_id=project_id,
            project_source_task_id=parent,
            workspace_kind="dir", workspace_path=str(repo))
        task = kb.get_task(conn, child)
        assert task.project_id == project_id
        assert task.workspace_kind == "dir"
        assert task.workspace_path == str(repo)
        assert task.branch_name is None
        assert not (repo / ".worktrees").exists()

        refused = kb.create_task(
            conn, title="relative child", project_id=project_id,
            project_source_task_id=relative_parent,
            workspace_kind="dir", workspace_path="relative/repo")
        assert kb.get_task(conn, refused).project_id is None
    finally:
        conn.close()


def test_project_from_source_task_rejects_a_dir_parent_for_a_different_child(tmp_path, monkeypatch):
    """The DB boundary is authoritative: a direct caller cannot attach the
    parent's project to an unrelated directory."""
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    repo.mkdir()
    other.mkdir()
    conn, _owner_home, filer_home = _two_homes(tmp_path, monkeypatch)
    try:
        with pdb.connect_closing() as pconn:
            project_id = pdb.create_project(pconn, name="Governed", primary_path=str(repo))
        parent = kb.create_task(
            conn, title="steward card", project_id=project_id,
            workspace_kind="dir", workspace_path=str(repo))
        monkeypatch.setenv("HERMES_HOME", str(filer_home))

        child = kb.create_task(
            conn, title="builder child", project_id=project_id,
            project_source_task_id=parent,
            workspace_kind="dir", workspace_path=str(other))
        task = kb.get_task(conn, child)
        assert task.project_id is None
        assert task.workspace_kind == "dir"
        assert task.workspace_path == str(other)
    finally:
        conn.close()


def test_project_from_source_task_rejects_a_non_dir_child_for_a_dir_parent(tmp_path, monkeypatch):
    """A ``dir`` source only qualifies a ``dir`` child. A scratch or worktree
    child naming the same path must not be linked (and must not be silently
    upgraded into a worktree by the project branch)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    conn, _owner_home, filer_home = _two_homes(tmp_path, monkeypatch)
    try:
        with pdb.connect_closing() as pconn:
            project_id = pdb.create_project(pconn, name="Governed", primary_path=str(repo))
        parent = kb.create_task(
            conn, title="steward card", project_id=project_id,
            workspace_kind="dir", workspace_path=str(repo))
        monkeypatch.setenv("HERMES_HOME", str(filer_home))

        scratch = kb.create_task(
            conn, title="scratch child", project_id=project_id,
            project_source_task_id=parent,
            workspace_kind="scratch", workspace_path=str(repo))
        scratch_task = kb.get_task(conn, scratch)
        assert scratch_task.project_id is None
        assert scratch_task.workspace_kind == "scratch"
        assert scratch_task.workspace_path == str(repo)

        worktree = kb.create_task(
            conn, title="worktree child", project_id=project_id,
            project_source_task_id=parent,
            workspace_kind="worktree", workspace_path=str(repo))
        worktree_task = kb.get_task(conn, worktree)
        assert worktree_task.project_id is None
        assert worktree_task.workspace_kind == "worktree"
        assert worktree_task.workspace_path == str(repo)
        assert worktree_task.branch_name is None
    finally:
        conn.close()


def test_project_from_source_task_rejects_a_relative_child_path_for_a_dir_parent(tmp_path, monkeypatch):
    """Only an absolute child path can be canonical-equal to an absolute source."""
    repo = tmp_path / "repo"
    repo.mkdir()
    conn, _owner_home, filer_home = _two_homes(tmp_path, monkeypatch)
    try:
        with pdb.connect_closing() as pconn:
            project_id = pdb.create_project(pconn, name="Governed", primary_path=str(repo))
        parent = kb.create_task(
            conn, title="steward card", project_id=project_id,
            workspace_kind="dir", workspace_path=str(repo))
        monkeypatch.setenv("HERMES_HOME", str(filer_home))

        child = kb.create_task(
            conn, title="builder child", project_id=project_id,
            project_source_task_id=parent,
            workspace_kind="dir", workspace_path="relative/repo")
        task = kb.get_task(conn, child)
        assert task.project_id is None
        assert task.workspace_kind == "dir"
        assert task.workspace_path == "relative/repo"
    finally:
        conn.close()


def test_project_from_source_task_refuses_an_unsluggable_project_without_crashing(tmp_path, monkeypatch):
    """A project id that cannot be normalized to a slug is refused, not raised."""
    repo = tmp_path / "repo"
    repo.mkdir()
    conn, _owner_home, filer_home = _two_homes(tmp_path, monkeypatch)
    try:
        with pdb.connect_closing() as pconn:
            project_id = pdb.create_project(pconn, name="Governed", primary_path=str(repo))
        parent = kb.create_task(
            conn, title="steward card", project_id=project_id,
            workspace_kind="dir", workspace_path=str(repo))
        # Hand-seed an id the slug normalizer rejects, on the shared board row.
        bad_id = "Bad.Id"
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET project_id = ? WHERE id = ?", (bad_id, parent))
        monkeypatch.setenv("HERMES_HOME", str(filer_home))

        child = kb.create_task(
            conn, title="builder child", project_id=bad_id,
            project_source_task_id=parent,
            workspace_kind="dir", workspace_path=str(repo))
        task = kb.get_task(conn, child)
        assert task.project_id is None
        assert (task.workspace_kind, task.workspace_path) == ("dir", str(repo))
    finally:
        conn.close()


