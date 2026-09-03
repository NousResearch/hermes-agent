"""Kanban <-> Projects integration: project-linked tasks get a deterministic
worktree path + branch instead of the random ``wt/<task-id>`` fallback."""

from __future__ import annotations

import os
import subprocess

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


def _assert_valid_git_branch(branch: str) -> None:
    result = subprocess.run(
        ["git", "check-ref-format", "--branch", branch],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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


def test_project_linked_task_branch_survives_dot_at_title_truncation(kanban_conn):
    proj = _make_project(name="Reefmind Project")
    title = "3291 Integrate merge and prove Gemini 3.1 support"

    tid = kb.create_task(kanban_conn, title=title, project_id=proj.slug)
    branch = kb.get_task(kanban_conn, tid).branch_name

    # Regression: the 40-character title truncation used to produce
    # ``.../t_<id>-3291-integrate-merge-and-prove-gemini-3.``, which Git
    # rejects because a ref cannot end in a dot.
    assert branch == (
        f"reefmind-project/{tid}-3291-integrate-merge-and-prove-gemini-3"
    )
    _assert_valid_git_branch(branch)


@pytest.mark.parametrize(
    ("title", "suffix"),
    [
        ("release.", "release"),
        ("gemini..support", "gemini-support"),
        ("release.lock", "release-lock"),
    ],
)
def test_deterministic_project_branch_sanitizes_invalid_dot_forms(title, suffix):
    proj = _make_project()

    branch = pdb.branch_name_for(proj, "t_deadbeef", title=title)

    assert branch == f"web-app/t_deadbeef-{suffix}"
    _assert_valid_git_branch(branch)


def test_deterministic_project_branch_preserves_valid_single_dots():
    proj = _make_project()

    branch = pdb.branch_name_for(proj, "t_deadbeef", title="Gemini 3.1 support")

    assert branch == "web-app/t_deadbeef-gemini-3.1-support"
    _assert_valid_git_branch(branch)


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
