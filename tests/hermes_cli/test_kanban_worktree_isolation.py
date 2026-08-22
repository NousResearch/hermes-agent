"""Per-task worktree isolation and repository binding propagation.

Decompose children used to inherit the root's literal ``workspace_path``,
so every sibling of a worktree-kind root pointed at the SAME checkout —
and ``_resolve_worktree_workspace``'s existing-checkout shortcut reused it
on whatever branch was there, letting sibling workers run concurrently in
one directory on one branch (cross-task provenance corruption, no lock).

Two-part fix under test:
- ``decompose_triage_task`` stores the common repository root as each
  worktree child's binding, so each child materializes its own
  ``<repo>/.worktrees/<child-id>`` without depending on a board default.
- ``_resolve_worktree_workspace`` falls back to a fresh per-task worktree
  when the requested path is occupied by another task's branch (heals
  pre-existing rows that still carry a shared path).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


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


def test_decompose_repo_root_propagates_resolvable_binding(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    with kb.connect() as conn:
        root = kb.create_task(
            conn,
            title="build the feature",
            triage=True,
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        child_ids = kb.decompose_triage_task(
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
            child = kb.get_task(conn, cid)
            assert child is not None
            assert child.workspace_path is not None
            assert child.workspace_kind == "worktree"
            assert Path(child.workspace_path).resolve() == repo.resolve()
            workspace, branch = kb._resolve_worktree_workspace(child)
            assert workspace == (repo / ".worktrees" / cid).resolve()
            assert branch == f"wt/{cid}"


def test_decompose_linked_worktree_uses_common_repo_root_head(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    repo_head = _git(repo, "rev-parse", "HEAD")
    sibling = _add_worktree(repo, repo / ".worktrees" / "root", "wt/root")
    (sibling / "sibling.txt").write_text("sibling-only\n", encoding="utf-8")
    _git(sibling, "add", "sibling.txt")
    _git(sibling, "commit", "-m", "sibling commit")
    sibling_head = _git(sibling, "rev-parse", "HEAD")
    assert sibling_head != repo_head

    with kb.connect() as conn:
        root = kb.create_task(
            conn,
            title="correct sibling work",
            triage=True,
            workspace_kind="worktree",
            workspace_path=str(sibling),
        )
        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[{"title": "implement correction", "assignee": "alice"}],
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 1
        child = kb.get_task(conn, child_ids[0])

    assert child is not None
    assert child.workspace_path is not None
    assert Path(child.workspace_path).resolve() == repo.resolve()
    workspace, _branch = kb._resolve_worktree_workspace(child)
    child_head = _git(workspace, "rev-parse", "HEAD")
    assert child_head == repo_head
    assert child_head != sibling_head


def test_resolve_worktree_falls_back_when_path_occupied(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="second sibling",
            workspace_kind="worktree",
            workspace_path=str(occupied),  # inherited shared/stale path
        )
        task = kb.get_task(conn, tid)

    workspace, branch = kb._resolve_worktree_workspace(task)
    assert workspace == (repo / ".worktrees" / tid).resolve()
    assert branch == f"wt/{tid}"
    # The sibling's checkout is untouched, still on its own branch.
    assert (occupied / "README.md").exists()
    head = subprocess.run(
        ["git", "-C", str(occupied), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == "wt/sibling"
