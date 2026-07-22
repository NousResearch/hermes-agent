"""Kanban <-> Projects integration: project-linked tasks get a deterministic
worktree path + branch instead of the random ``wt/<task-id>`` fallback."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

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


def _init_repo(repo: Path) -> Path:
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "kanban@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Kanban Test"],
        check=True,
    )
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )
    return repo


def _make_project(repo: Path, name="Web App"):
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name=name, folders=[str(repo)])
        return pdb.get_project(pc, pid)


def test_project_linked_task_gets_deterministic_worktree_and_branch(
    kanban_conn, tmp_path,
):
    proj = _make_project(_init_repo(tmp_path / "webapp"))
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
    proj = _make_project(_init_repo(tmp_path / "webapp"))
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


def test_unknown_project_id_falls_back_gracefully(kanban_conn):
    # A direct CLI/API project lookup is not a cross-profile source route.
    # Preserve the established behavior for this non-security-sensitive case.
    tid = kb.create_task(kanban_conn, title="x", project_id="does-not-exist")
    task = kb.get_task(kanban_conn, tid)
    assert task is not None
    assert task.workspace_kind == "scratch"
    assert task.project_id is None
    assert task.project_repo_root is None


def test_existing_project_rows_are_not_backfilled_with_an_unproven_anchor(
    kanban_conn,
    tmp_path,
):
    proj = _make_project(_init_repo(tmp_path / "webapp"))
    assert proj is not None
    tid = kb.create_task(kanban_conn, title="legacy", project_id=proj.id)
    with kb.write_txn(kanban_conn):
        kanban_conn.execute(
            "UPDATE tasks SET project_repo_root = NULL WHERE id = ?",
            (tid,),
        )
    kb._migrate_add_optional_columns(kanban_conn)
    legacy = kb.get_task(kanban_conn, tid)
    assert legacy is not None
    assert legacy.project_repo_root is None
    with pytest.raises(ValueError, match="trust anchor"):
        kb.resolve_workspace(legacy)
