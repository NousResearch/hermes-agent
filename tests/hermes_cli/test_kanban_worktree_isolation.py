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
    with kb.connect() as conn:
        root = kb.create_task(conn, title="build the feature", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', "
            "workspace_path='/repo/.worktrees/root' WHERE id = ?",
            (root,),
        )
        conn.commit()

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
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["workspace_kind"] == "worktree"
            # Each child resolves its own <repo>/.worktrees/<child-id> at
            # dispatch; the root's literal path must never be shared.
            assert row["workspace_path"] is None


def test_decompose_worktree_children_inherit_root_repo_anchor(kanban_home, tmp_path):
    """A worktree child with no explicit path gets the root's REPO as its
    anchor — not NULL.

    Dispatch resolves an anchorless worktree task against the board's
    ``default_workdir`` and fails the spawn outright when the board has
    none ("no default_workdir set"). Real decompose roots usually ARE
    dispatcher-materialized worktrees under ``<repo>/.worktrees/<id>``, so
    the repo is recoverable from the root row — children must inherit it
    explicitly so each spawns a fresh worktree in the same repository
    (live failure class: t_ab2a7ce8 / t_2dd0f5b7, 2026-09-02).
    """
    repo = _make_repo(tmp_path)
    root_wt = _add_worktree(repo, repo / ".worktrees" / "rootx", "wt/rootx")

    with kb.connect() as conn:
        root = kb.create_task(conn, title="build the feature", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', "
            "workspace_path=? WHERE id = ?",
            (str(root_wt), root),
        )
        conn.commit()

        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[
                {"title": "review it", "assignee": "alice", "parents": []},
                {"title": "qa it", "assignee": "bob", "parents": []},
                {"title": "verify live", "assignee": "carol", "parents": [0, 1]},
            ],
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 3

        for cid in child_ids:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["workspace_kind"] == "worktree"
            # The anchor is the root's REPO, never the root's checkout.
            assert row["workspace_path"] == str(repo.resolve())

        # Every child actually spawns: dispatch materializes a distinct
        # worktree per child inside the inherited repo (no default_workdir
        # configured on this board — the exact condition that used to
        # raise "no default_workdir set").
        import os

        os.environ.pop("HERMES_KANBAN_DB", None)
        for cid in child_ids:
            task = kb.get_task(conn, cid)
            assert task is not None
            workspace, branch = kb._resolve_worktree_workspace(task)
            assert workspace == (repo / ".worktrees" / cid).resolve()
            assert branch == f"wt/{cid}"
        # Sibling isolation still holds: distinct paths, distinct branches.
        resolved = {cid: str((repo / ".worktrees" / cid).resolve()) for cid in child_ids}
        assert len(set(resolved.values())) == len(child_ids)


def test_decompose_worktree_children_without_recoverable_root_stay_unset(kanban_home):
    """A root whose path points nowhere git-recognizable cannot supply an
    anchor; children keep the anchorless row (dispatch's board-default
    path) instead of storing a bogus path."""
    with kb.connect() as conn:
        root = kb.create_task(conn, title="build the feature", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', "
            "workspace_path='/nonexistent/repo/.worktrees/root' WHERE id = ?",
            (root,),
        )
        conn.commit()

        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[{"title": "spec it", "assignee": "alice", "parents": []}],
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 1
        row = conn.execute(
            "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
            (child_ids[0],),
        ).fetchone()
        assert row["workspace_kind"] == "worktree"
        assert row["workspace_path"] is None


def test_decompose_scratch_root_children_stay_scratch(kanban_home):
    """Scratch-only decomposition is unchanged: no worktree kind or repo
    path leaks into children of a scratch root."""
    with kb.connect() as conn:
        root = kb.create_task(conn, title="plan the offsite", triage=True)
        conn.commit()

        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[
                {"title": "book venue", "assignee": "alice", "parents": []},
                {"title": "write agenda", "assignee": "bob", "parents": []},
            ],
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 2
        for cid in child_ids:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["workspace_kind"] == "scratch"
            assert row["workspace_path"] is None




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




