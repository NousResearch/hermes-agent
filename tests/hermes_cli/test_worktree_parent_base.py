"""REQ-034 integration tests — child worktree branches from parent's surfaced head_sha.

When a kanban card has a parent whose completed run metadata contains
``worktree_integration`` facts, the child's worktree is created FROM the
parent's ``head_sha`` instead of ``HEAD``.  Falls back to current behavior
(HEAD) when no parent facts exist or the ref is missing — never errors the
claim path.

Follows the fixture patterns of tests/hermes_cli/test_kanban_worktree_surfacing.py.
"""

from __future__ import annotations

import json
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


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "kanban@example.com"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Kanban Test"], check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True, text=True)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _wt_commit(ws: Path, name: str, msg: str) -> str:
    (ws / name).write_text(f"{name}\n", encoding="utf-8")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-m", msg)
    return _git(ws, "rev-parse", "HEAD")


def _last_run_metadata(conn, task_id: str) -> dict | None:
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None or row["metadata"] is None:
        return None
    return json.loads(row["metadata"])


def _make_worktree_task(conn, board: str, repo: Path, title: str = "parent") -> tuple[str, Path]:
    """create_task -> materialize worktree -> claim: the pre-completion flow."""
    t = kb.create_task(conn, title=title, workspace_kind="worktree", board=board)
    task = kb.get_task(conn, t)
    assert task is not None
    assert task.workspace_kind == "worktree"
    ws = kb.resolve_workspace(task, board=board)
    assert ws == repo / ".worktrees" / t
    claimed = kb.claim_task(conn, t)
    assert claimed is not None
    return t, ws


# ---------------------------------------------------------------------------
# Test 1: parent with surfaced branch -> child worktree HEAD == parent head_sha
# ---------------------------------------------------------------------------

def test_child_worktree_branches_from_parent_head_sha(kanban_home, tmp_path):
    """When a parent worktree task completes with surfaced integration facts,
    a child worktree task's worktree is branched from the parent's head_sha."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("req034-parent", default_workdir=str(repo))

    with kb.connect(board="req034-parent") as conn:
        # Create and complete parent with work on its branch
        parent_id, parent_ws = _make_worktree_task(conn, "req034-parent", repo, "parent task")
        parent_head = _wt_commit(parent_ws, "parent_file.txt", "parent work")

        assert kb.complete_task(conn, parent_id, result="ok", summary="parent done")

        # Verify parent has worktree_integration facts
        parent_md = _last_run_metadata(conn, parent_id)
        assert parent_md is not None
        wt_facts = parent_md["worktree_integration"]
        assert wt_facts["head_sha"] == parent_head

        # Create child task linked to parent
        child_id = kb.create_task(
            conn, title="child task", workspace_kind="worktree",
            board="req034-parent", parents=[parent_id],
        )
        child_task = kb.get_task(conn, child_id)
        assert child_task is not None

        # Claim and resolve workspace — this is where the branching happens
        kb.claim_task(conn, child_id)
        child_task = kb.get_task(conn, child_id)
        child_ws = kb.resolve_workspace(child_task, board="req034-parent", conn=conn)

        # Verify child worktree HEAD == parent head_sha
        child_head = _git(child_ws, "rev-parse", "HEAD")
        assert child_head == parent_head, (
            f"child worktree HEAD {child_head} should equal parent head_sha {parent_head}"
        )

        # Verify child is on its own branch
        child_branch = _git(child_ws, "rev-parse", "--abbrev-ref", "HEAD")
        assert child_branch == f"wt/{child_id}"


# ---------------------------------------------------------------------------
# Test 2: no parent facts -> falls back to HEAD
# ---------------------------------------------------------------------------

def test_child_worktree_falls_back_to_head_when_no_parent_facts(kanban_home, tmp_path):
    """When a parent exists but has no worktree_integration metadata,
    child falls back to HEAD (current behavior)."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("req034-nofacts", default_workdir=str(repo))

    # Record the main HEAD before any worktree work
    main_head_before = _git(repo, "rev-parse", "HEAD")

    with kb.connect(board="req034-nofacts") as conn:
        # Create a non-worktree parent (scratch) — no worktree_integration facts
        parent_id = kb.create_task(conn, title="scratch parent")
        parent_task = kb.get_task(conn, parent_id)
        assert parent_task is not None
        kb.claim_task(conn, parent_id)
        kb.complete_task(conn, parent_id, result="ok")

        # Create child worktree task linked to the scratch parent
        child_id = kb.create_task(
            conn, title="child worktree", workspace_kind="worktree",
            board="req034-nofacts", parents=[parent_id],
        )
        child_task = kb.get_task(conn, child_id)
        assert child_task is not None

        kb.claim_task(conn, child_id)
        child_task = kb.get_task(conn, child_id)
        child_ws = kb.resolve_workspace(child_task, board="req034-nofacts", conn=conn)

        # Child should branch from HEAD (which is main_head_before since no
        # worktree commits touched main)
        child_head = _git(child_ws, "rev-parse", "HEAD")
        assert child_head == main_head_before, (
            f"child should fall back to HEAD ({main_head_before}), got {child_head}"
        )


# ---------------------------------------------------------------------------
# Test 3: missing/deleted parent branch -> falls back cleanly
# ---------------------------------------------------------------------------

def test_child_worktree_falls_back_when_parent_branch_deleted(kanban_home, tmp_path):
    """When parent had worktree_integration facts but the branch is gone,
    child falls back to HEAD without error."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("req034-deleted", default_workdir=str(repo))

    with kb.connect(board="req034-deleted") as conn:
        # Create and complete parent
        parent_id, parent_ws = _make_worktree_task(conn, "req034-deleted", repo)
        parent_head = _wt_commit(parent_ws, "parent.txt", "parent work")

        assert kb.complete_task(conn, parent_id, result="ok", summary="done")

        # Verify parent metadata has the head_sha
        parent_md = _last_run_metadata(conn, parent_id)
        assert parent_md["worktree_integration"]["head_sha"] == parent_head

        # Now remove the worktree and delete the parent branch from the repo
        parent_branch = f"wt/{parent_id}"
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(parent_ws)],
            capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "branch", "-D", parent_branch],
            capture_output=True, text=True,
        )
        assert not _git(repo, "branch", "--list", parent_branch)

        # Force GC so the parent's SHA is truly unreachable
        subprocess.run(
            ["git", "-C", str(repo), "reflog", "expire", "--expire-unreachable=0", "--all"],
            capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "gc", "--prune=now"],
            capture_output=True, text=True,
        )

        # Verify the parent SHA is now gone (use check=False since cat-file -e returns 1 when missing)
        result = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", parent_head],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode != 0, (
            "parent head_sha should be unreachable after GC"
        )

        # Record current HEAD (main, unchanged)
        main_head = _git(repo, "rev-parse", "HEAD")

        # Create child — should fall back to HEAD since parent branch is gone
        child_id = kb.create_task(
            conn, title="child after parent branch deleted",
            workspace_kind="worktree",
            board="req034-deleted", parents=[parent_id],
        )
        child_task = kb.get_task(conn, child_id)
        assert child_task is not None

        kb.claim_task(conn, child_id)
        child_task = kb.get_task(conn, child_id)

        # Should NOT raise — falls back to HEAD
        child_ws = kb.resolve_workspace(child_task, board="req034-deleted", conn=conn)

        # Child should have branched from HEAD (main) since parent SHA is
        # unreachable
        child_head = _git(child_ws, "rev-parse", "HEAD")
        assert child_head == main_head, (
            f"child should fall back to HEAD ({main_head}), got {child_head}"
        )


# ---------------------------------------------------------------------------
# Test 4: no parents at all -> falls back to HEAD (baseline)
# ---------------------------------------------------------------------------

def test_worktree_with_no_parents_uses_head(kanban_home, tmp_path):
    """A worktree task with no parents at all branches from HEAD."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("req034-noparent", default_workdir=str(repo))

    main_head = _git(repo, "rev-parse", "HEAD")

    with kb.connect(board="req034-noparent") as conn:
        child_id = kb.create_task(
            conn, title="orphan worktree", workspace_kind="worktree",
            board="req034-noparent",
        )
        child_task = kb.get_task(conn, child_id)
        assert child_task is not None

        kb.claim_task(conn, child_id)
        child_task = kb.get_task(conn, child_id)
        child_ws = kb.resolve_workspace(child_task, board="req034-noparent", conn=conn)

        child_head = _git(child_ws, "rev-parse", "HEAD")
        assert child_head == main_head


# ---------------------------------------------------------------------------
# Test 5: parent with worktree_integration error -> falls back to HEAD
# ---------------------------------------------------------------------------

def test_child_falls_back_when_parent_worktree_integration_has_error(kanban_home, tmp_path):
    """When parent's worktree_integration contains an 'error' key,
    child falls back to HEAD instead of trying the bad ref."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("req034-errparent", default_workdir=str(repo))

    main_head = _git(repo, "rev-parse", "HEAD")

    with kb.connect(board="req034-errparent") as conn:
        # Create a scratch parent with manually injected bad metadata
        parent_id = kb.create_task(conn, title="broken parent")
        parent_task = kb.get_task(conn, parent_id)
        assert parent_task is not None
        kb.claim_task(conn, parent_id)
        kb.complete_task(
            conn, parent_id, result="ok",
            metadata={"worktree_integration": {"error": "worktree not found"}},
        )

        # Create child
        child_id = kb.create_task(
            conn, title="child of broken parent", workspace_kind="worktree",
            board="req034-errparent", parents=[parent_id],
        )
        child_task = kb.get_task(conn, child_id)
        assert child_task is not None

        kb.claim_task(conn, child_id)
        child_task = kb.get_task(conn, child_id)
        child_ws = kb.resolve_workspace(child_task, board="req034-errparent", conn=conn)

        child_head = _git(child_ws, "rev-parse", "HEAD")
        assert child_head == main_head, (
            f"child should fall back to HEAD when parent has error, got {child_head}"
        )