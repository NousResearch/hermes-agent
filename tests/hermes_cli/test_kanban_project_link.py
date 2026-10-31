"""Kanban <-> Projects integration: project-linked tasks get a deterministic
worktree path + branch instead of the random ``wt/<task-id>`` fallback — but
only when the project's ``primary_path`` can actually host one.

Regression for the creation-time gap: a card that inherited a project whose
``primary_path`` was not inside a git repo was silently upgraded to a
``worktree`` anyway, so it was created, dispatched, and then died at spawn
(``workspace: ... is not inside a git repo``) — twice, then ``gave_up``. The
upgrade is a default, so it must never break card creation.
"""

from __future__ import annotations

import os
import subprocess

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


def _git_init(repo):
    """A real git work tree — ``git init`` alone is what a worktree anchor needs.

    No commit: the predicate is "is this path inside a work tree", which an
    unborn branch satisfies, and no worktree is materialized at creation (only
    the path is recorded; dispatch creates it).
    """
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


def _make_project(tmp_path, name="Web App"):
    repo = _git_init(tmp_path / "webapp")
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name=name, folders=[str(repo)])
        return pdb.get_project(pc, pid)


def _make_non_repo_project(tmp_path, name="Salvage"):
    """A project whose folder exists but is not inside a git work tree."""
    repo = tmp_path / "salvage-copy"
    repo.mkdir()
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name=name, folders=[str(repo)])
        return pdb.get_project(pc, pid)


def test_project_linked_task_gets_deterministic_worktree_and_branch(kanban_conn, tmp_path):
    proj = _make_project(tmp_path)
    tid = kb.create_task(kanban_conn, title="Add login", project_id=proj.slug)
    task = kb.get_task(kanban_conn, tid)

    assert task.project_id == proj.id
    assert task.workspace_kind == "worktree"
    # Worktree dir anchored under the project's primary repo, keyed on task id.
    assert task.workspace_path == os.path.join(proj.primary_path, ".worktrees", tid)
    # Deterministic branch: <slug>/<task-id>-<title-slug>. NOT a random wt/...
    assert task.branch_name == f"{proj.slug}/{tid}-add-login"
    assert not task.branch_name.startswith("wt/")


def test_explicit_branch_overrides_project_default(kanban_conn, tmp_path):
    proj = _make_project(tmp_path)
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


def test_project_whose_path_is_not_a_git_repo_falls_back_to_scratch(kanban_conn, tmp_path):
    """A worktree upgrade that cannot be materialized must not strand the card:
    it is created as scratch, keeps its project link, and records why."""
    proj = _make_non_repo_project(tmp_path)

    tid = kb.create_task(kanban_conn, title="no repo here", project_id=proj.id)
    task = kb.get_task(kanban_conn, tid)

    assert task.workspace_kind == "scratch"
    assert task.workspace_path is None
    assert task.branch_name is None
    assert task.project_id == proj.id

    skipped = [e.payload for e in kb.list_events(kanban_conn, tid)
               if e.kind == "project_worktree_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "primary_path_not_a_git_repo"
    assert proj.primary_path in skipped[0]["message"]
    assert "scratch" in skipped[0]["message"]


def test_project_inside_a_repo_subdir_still_gets_a_worktree(kanban_conn, tmp_path):
    """The check mirrors dispatch (nearest existing ancestor), so a project
    folder that does not exist yet but lives inside a repo is still a worktree."""
    repo = _git_init(tmp_path / "mono")
    folder = repo / "pkg" / "app"

    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name="Monorepo", folders=[str(folder)])
        proj = pdb.get_project(pc, pid)

    tid = kb.create_task(kanban_conn, title="subdir project", project_id=proj.id)
    task = kb.get_task(kanban_conn, tid)

    assert task.workspace_kind == "worktree"
    assert task.workspace_path == os.path.join(str(folder), ".worktrees", tid)


def test_explicit_worktree_on_non_repo_project_raises_at_creation(kanban_conn, tmp_path):
    """Asking for a worktree explicitly is not a default: refuse it at creation
    with the path named, instead of creating a card that cannot be dispatched."""
    proj = _make_non_repo_project(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        kb.create_task(
            kanban_conn, title="bad anchor", project_id=proj.id, workspace_kind="worktree",
        )

    message = str(excinfo.value)
    assert str(proj.primary_path) in message
    assert "workspace_kind='scratch'" in message
