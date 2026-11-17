"""t_716b221c: workspace-convention defaults at the DB layer.

The ``kanban-workspace-convention`` plugin (tool_request middleware) only
sees agent-loop ``kanban_create`` tool calls. CLI (``hermes kanban create``),
swarm, dashboard and Python creators insert rows straight through
``create_task`` / ``_insert_decomposed_child``, so the coding-worktree
default and the root-anchor inheritance live at the DB layer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli.kanban_db_workspace import (
    extract_body_repo_path,
    resolve_default_workspace,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "poc-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


def _coding_body(repo: Path) -> str:
    return f"type: coding\nrepo: {repo}"


# ---------------------------------------------------------------------------
# create_task: no explicit workspace
# ---------------------------------------------------------------------------


def test_coding_body_without_workspace_gets_worktree_anchor(kanban_home, git_repo):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Fix the thing", body=_coding_body(git_repo), assignee="operator",
        )
        task = kb.get_task(conn, tid)

    # A repo-root anchor: the dispatcher materializes <repo>/.worktrees/<id>.
    assert task.workspace_kind == "worktree"
    assert Path(task.workspace_path) == git_repo


def test_non_coding_task_keeps_scratch_default(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Research", body="Analysiere und schreibe Report.", assignee="operator",
        )
        task = kb.get_task(conn, tid)
    assert task.workspace_kind == "scratch"
    assert task.workspace_path is None


def test_coding_body_without_git_anchor_keeps_scratch_default(kanban_home, tmp_path):
    plain_dir = tmp_path / "not-a-repo"
    plain_dir.mkdir()
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Fix the thing",
            body=f"type: coding\nrepo: {plain_dir}", assignee="operator",
        )
        task = kb.get_task(conn, tid)
    assert task.workspace_kind == "scratch"


def test_explicit_scratch_is_not_overridden(kanban_home, git_repo):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Fix the thing", body=_coding_body(git_repo),
            assignee="operator", workspace_kind="scratch",
        )
        task = kb.get_task(conn, tid)
    assert task.workspace_kind == "scratch"


def test_explicit_dir_is_not_overridden(kanban_home, git_repo):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Fix the thing", body=_coding_body(git_repo),
            assignee="operator",
            workspace_kind="dir", workspace_path=str(kanban_home / "ws"),
        )
        task = kb.get_task(conn, tid)
    assert task.workspace_kind == "dir"


def test_path_without_kind_stays_legacy_scratch(kanban_home, git_repo):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Fix the thing", body=_coding_body(git_repo),
            assignee="operator", workspace_path=str(kanban_home / "ws"),
        )
        task = kb.get_task(conn, tid)
    assert task.workspace_kind == "scratch"


def test_coding_tag_in_title_alone_is_enough(kanban_home, git_repo):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="#bug fix the thing", body=f"repo: {git_repo}", assignee="operator",
        )
        task = kb.get_task(conn, tid)
    assert task.workspace_kind == "worktree"


# ---------------------------------------------------------------------------
# resolve_default_workspace / extract_body_repo_path (plugin-compatible syntax)
# ---------------------------------------------------------------------------


def test_extract_repo_path_syntaxes(git_repo, tmp_path):
    assert extract_body_repo_path(f"repo: {git_repo}") == str(git_repo)
    assert extract_body_repo_path(f"# repo: {git_repo}") == str(git_repo)
    assert extract_body_repo_path(f"repo={git_repo}") == str(git_repo)
    assert extract_body_repo_path("kein repo hier") is None
    assert extract_body_repo_path(None) is None


def test_resolve_default_workspace_returns_none_for_non_coding(git_repo):
    assert resolve_default_workspace("Recherche", f"repo: {git_repo}", "operator") is None


# ---------------------------------------------------------------------------
# decompose: worktree children inherit the root's REPO-ROOT anchor
# ---------------------------------------------------------------------------


def _decompose(conn, tid, children):
    return decompose_triage_task(conn, tid, root_assignee="orchestrator", children=children)


def test_worktree_children_inherit_root_repo_anchor(kanban_home, git_repo):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Parent coding epic", body=_coding_body(git_repo),
            assignee="operator", triage=True,
        )
        assert kb.get_task(conn, tid).workspace_kind == "worktree"

        child_ids = _decompose(conn, tid, [
            {"title": "child a", "body": "coding", "assignee": "operator"},
            {"title": "child b", "body": "coding", "assignee": "operator", "parents": [0]},
        ])
        child_a = kb.get_task(conn, child_ids[0])
        child_b = kb.get_task(conn, child_ids[1])

    for child in (child_a, child_b):
        assert child.workspace_kind == "worktree"
        # The anchor, NOT the root's own worktree path — the dispatcher
        # materializes a fresh <repo>/.worktrees/<child-id> per sibling.
        assert Path(child.workspace_path) == git_repo
        assert child.workspace_path != str(git_repo / ".worktrees" / child.id)


def test_decompose_child_explicit_workspace_override_wins(kanban_home, git_repo):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Parent coding epic", body=_coding_body(git_repo),
            assignee="operator", triage=True,
        )
        child_ids = _decompose(conn, tid, [
            {"title": "child dir", "assignee": "operator", "workspace_kind": "dir",
             "workspace_path": str(kanban_home / "child-ws")},
        ])
        child = kb.get_task(conn, child_ids[0])
    assert child.workspace_kind == "dir"
    assert child.workspace_path == str(kanban_home / "child-ws")


def test_worktree_root_without_anchor_leaves_children_unset(kanban_home, git_repo):
    # Legacy shape: worktree root with no path — children keep path=None so
    # the dispatcher falls back to the board's default_workdir anchor.
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Parent coding epic", body=_coding_body(git_repo),
            assignee="operator", triage=True, workspace_kind="worktree",
            workspace_path=None, branch_name=None,
        )
        child_ids = _decompose(conn, tid, [
            {"title": "child a", "assignee": "operator"},
        ])
        child = kb.get_task(conn, child_ids[0])
    assert child.workspace_kind == "worktree"
    assert child.workspace_path is None