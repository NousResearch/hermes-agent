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




def test_decompose_children_anchor_on_the_roots_repo(kanban_home, tmp_path):
    """A card filed by an outside consumer names its repository, so the anchor a
    decomposed child needs is already on the root row — no board default needed.

    This is the tix-agent case: ``--workspace worktree:<repo>`` lands on the
    root, and every child must land in its OWN ``<repo>/.worktrees/<child-id>``
    so siblings still cannot share a checkout.
    """
    repo = _make_repo(tmp_path)
    with kbc.connect() as conn:
        root = kb.create_task(
            conn,
            title="project view",
            triage=True,
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
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

        resolved = []
        for cid in child_ids:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["workspace_kind"] == "worktree"
            # The repo, never the root's checkout and never nothing at all.
            assert row["workspace_path"] == str(repo)
            workspace, branch = kbw._resolve_worktree_workspace(kb.get_task(conn, cid))
            assert workspace == (repo / ".worktrees" / cid).resolve()
            assert branch == f"wt/{cid}"
            resolved.append(workspace)

        # Distinct checkouts, one branch each: the isolation property is intact.
        assert len(set(resolved)) == 2
        heads = [
            subprocess.run(
                ["git", "-C", str(ws), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            for ws in resolved
        ]
        assert heads == [f"wt/{cid}" for cid in child_ids]


def test_decompose_children_of_a_worktree_root_stay_unanchored(kanban_home, tmp_path):
    """A root that already lives in a per-task checkout is NOT an anchor.

    ``git rev-parse --show-toplevel`` answers a linked worktree with itself, so
    a naive "is this a repo root?" test would inherit the root's checkout and
    put every sibling on that one branch. The guard has to ask the more
    specific question, and this test pins the answer on a REAL worktree.
    """
    repo = _make_repo(tmp_path)
    root_ws = _add_worktree(repo, repo / ".worktrees" / "root", "wt/root")

    with kbc.connect() as conn:
        root = kb.create_task(
            conn,
            title="build the feature",
            triage=True,
            workspace_kind="worktree",
            workspace_path=str(root_ws),
        )
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
                "SELECT workspace_path FROM tasks WHERE id = ?", (cid,),
            ).fetchone()
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




