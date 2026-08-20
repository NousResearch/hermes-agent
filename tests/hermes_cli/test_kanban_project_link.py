"""Kanban <-> Projects integration: project-linked tasks get a deterministic
worktree path + branch instead of the random ``wt/<task-id>`` fallback."""

from __future__ import annotations

import os

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb


@pytest.fixture
def kanban_conn(tmp_path):
    c = kb.connect(db_path=tmp_path / "kanban.db")
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


def test_unknown_project_id_fails_closed(kanban_conn):
    # An explicit project id/slug that doesn't resolve must fail closed so a
    # typo or per-profile registry mismatch cannot silently become a scratch
    # task (#70386). Omitting project_id keeps ordinary scratch behaviour.
    with pytest.raises(ValueError, match=r"project 'does-not-exist' not found"):
        kb.create_task(kanban_conn, title="x", project_id="does-not-exist")


def test_blank_project_id_treated_as_omitted(kanban_conn):
    tid = kb.create_task(kanban_conn, title="x", project_id="  ")
    task = kb.get_task(kanban_conn, tid)
    assert task.project_id is None
    assert task.workspace_kind == "scratch"
