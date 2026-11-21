"""Per-task worktree isolation for decompose siblings.

Decompose children used to inherit the root's literal ``workspace_path``,
so every sibling of a worktree-kind root pointed at the SAME checkout —
and ``_resolve_worktree_workspace``'s existing-checkout shortcut reused it
on whatever branch was there, letting sibling workers run concurrently in
one directory on one branch (cross-task provenance corruption, no lock).

Two-part fix under test:
- ``decompose_triage_task`` leaves worktree children's ``workspace_path``
  unset so each child materializes its own ``<repo>/.worktrees/<child-id>``.
- ``_resolve_worktree_workspace`` falls back to a fresh per-task worktree
  when the requested path is occupied by another task's branch (heals
  pre-existing rows that still carry a shared path).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _add_worktree(repo: Path, target: Path, branch: str) -> Path:
    _git(repo, "worktree", "add", str(target), "-b", branch, "HEAD")
    return target


def test_decompose_worktree_children_get_own_workspace(kanban_home):
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="build the feature", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', "
            "workspace_path='/repo/.worktrees/root' WHERE id = ?",
            (root,),
        )
        conn.commit()

        child_ids = decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[
                {"title": "spec it", "assignee": "alice", "parents": []},
                {"title": "implement it", "assignee": "bob", "parents": [0]},
            ],
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 2

        for cid in child_ids:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["workspace_kind"] == "worktree"
            # Each child resolves its own <repo>/.worktrees/<child-id> at
            # dispatch; the root's literal path must never be shared.
            assert row["workspace_path"] is None




def test_resolve_worktree_falls_back_when_path_occupied(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="second sibling",
            workspace_kind="worktree",
            workspace_path=str(occupied),  # inherited shared/stale path
        )
        task = kb.get_task(conn, tid)

    workspace, branch = kbw._resolve_worktree_workspace(task)
    assert workspace == (repo / ".worktrees" / tid).resolve()
    assert branch == f"wt/{tid}"
    # The sibling's checkout is untouched, still on its own branch.
    assert (occupied / "README.md").exists()
    head = subprocess.run(
        ["git", "-C", str(occupied), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == "wt/sibling"


def _rev_parse(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_board_project_subfolder_gives_each_task_its_own_worktree(kanban_home, tmp_path):
    """A board project dir that is a folder inside a checkout (a monorepo
    service) is inherited as the worktree path of every task; each task must
    still run in its own linked worktree, never in the user's checkout."""
    repo = _make_repo(tmp_path)
    service = repo / "services" / "api"
    service.mkdir(parents=True)
    (service / "app.py").write_text("print('hi')\n", encoding="utf-8")
    kb.create_board("mono", default_workdir=str(service))

    workspaces = []
    with kbc.connect(board="mono") as conn:
        for title in ("one", "two"):
            tid = kb.create_task(conn, title=title, workspace_kind="worktree", board="mono")
            workspace, branch = kbw._resolve_worktree_workspace(kb.get_task(conn, tid), board="mono")
            assert _rev_parse(workspace, "--absolute-git-dir") != str((repo / ".git").resolve())
            assert _rev_parse(workspace, "--abbrev-ref", "HEAD") == branch == f"wt/{tid}"
            workspaces.append(workspace)
    assert workspaces[0] != workspaces[1]
    assert _rev_parse(repo, "--abbrev-ref", "HEAD") == "main"


def test_pinned_empty_dir_inside_checkout_becomes_real_worktree(kanban_home, tmp_path):
    """An empty directory pinned inside the checkout is materialized in place
    as the task's worktree instead of being "reused" as the main checkout."""
    repo = _make_repo(tmp_path)
    pinned = repo / "pinned"
    pinned.mkdir()

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="pinned worktree", workspace_kind="worktree", workspace_path=str(pinned),
        )
        workspace = kbw.resolve_workspace(kb.get_task(conn, tid))

    assert Path(_rev_parse(workspace, "--show-toplevel")).resolve() == pinned.resolve()
    assert _rev_parse(workspace, "--abbrev-ref", "HEAD") == f"wt/{tid}"
    assert _rev_parse(repo, "--abbrev-ref", "HEAD") == "main"




