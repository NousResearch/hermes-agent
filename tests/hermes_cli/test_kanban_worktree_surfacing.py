"""REQ-027 integration tests — complete_task surfaces worktree integration facts.

saga decision 0004 (surface-don't-merge, preserve-don't-stash): completing a
``workspace_kind='worktree'`` task must record integration facts into the
closing run's metadata (``worktree_integration``) and a ``worktree_surfaced``
task event — and a dirty tree is auto-preserved as a WIP commit on the task
branch, never stashed. Non-worktree tasks are completely unaffected.

Follows the fixture patterns of tests/hermes_cli/test_kanban_db.py (REQ-026
worktree board tests).
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


def _events(conn, task_id: str, kind: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (task_id, kind),
    ).fetchall()
    return [json.loads(r["payload"]) if r["payload"] else {} for r in rows]


def _make_worktree_task(conn, board: str, repo: Path, title: str = "ship") -> tuple[str, Path]:
    """create_task → materialize worktree → claim: the pre-completion flow."""
    t = kb.create_task(conn, title=title, workspace_kind="worktree", board=board)
    task = kb.get_task(conn, t)
    assert task is not None
    assert task.workspace_kind == "worktree"
    ws = kb.resolve_workspace(task, board=board)
    assert ws == repo / ".worktrees" / t
    claimed = kb.claim_task(conn, t)
    assert claimed is not None
    return t, ws


def test_complete_clean_worktree_surfaces_integration_facts(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("req027-clean", default_workdir=str(repo))
    with kb.connect(board="req027-clean") as conn:
        t, ws = _make_worktree_task(conn, "req027-clean", repo)
        _wt_commit(ws, "one.txt", "first")
        head = _wt_commit(ws, "two.txt", "second")

        assert kb.complete_task(conn, t, result="ok", summary="done")

        md = _last_run_metadata(conn, t)
        assert md is not None
        facts = md["worktree_integration"]
        assert "error" not in facts
        assert facts["branch"] == f"wt/{t}"
        assert facts["head_sha"] == head
        assert facts["base_branch"] == "main"
        assert facts["ahead_count"] == 2
        assert facts["dirty"] is False
        assert facts["wip_commit"] is None
        assert facts["repo_root"] == str(repo.resolve())
        assert facts["diffstat"]
        assert "one.txt" in facts["diffstat"]
        assert "two.txt" in facts["diffstat"]

        events = _events(conn, t, "worktree_surfaced")
        assert len(events) == 1
        assert events[0]["head_sha"] == head
        assert events[0]["ahead_count"] == 2


def test_complete_dirty_worktree_auto_preserves_wip_commit(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("req027-dirty", default_workdir=str(repo))
    with kb.connect(board="req027-dirty") as conn:
        t, ws = _make_worktree_task(conn, "req027-dirty", repo)
        _wt_commit(ws, "one.txt", "first")
        # Leave uncommitted work: one modified tracked file + one untracked.
        (ws / "one.txt").write_text("changed\n", encoding="utf-8")
        (ws / "loose.txt").write_text("wip\n", encoding="utf-8")

        assert kb.complete_task(conn, t, result="ok")

        md = _last_run_metadata(conn, t)
        facts = md["worktree_integration"]
        assert "error" not in facts
        assert facts["dirty"] is True
        wip = facts["wip_commit"]
        assert wip
        assert facts["head_sha"] == wip
        # The dirtiness was preserved INTO the branch (never stashed): the
        # worktree is clean afterwards and the WIP commit carries both files.
        assert _git(ws, "status", "--porcelain") == ""
        msg = _git(repo, "log", "-1", "--format=%s", wip)
        assert msg == f"wip({t}): auto-preserve at completion"
        changed = _git(repo, "show", "--stat", "--format=", wip)
        assert "one.txt" in changed
        assert "loose.txt" in changed
        # No stash entries were created.
        assert _git(repo, "stash", "list") == ""
        # WIP commit counts toward the surfaced ahead-count (first + wip).
        assert facts["ahead_count"] == 2

        events = _events(conn, t, "worktree_surfaced")
        assert len(events) == 1
        assert events[0]["wip_commit"] == wip


def test_complete_non_worktree_task_untouched(kanban_home, tmp_path):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="plain scratch card")
        task = kb.get_task(conn, t)
        assert task is not None
        assert task.workspace_kind == "scratch"
        kb.claim_task(conn, t)

        assert kb.complete_task(
            conn, t, result="ok", metadata={"tests_run": ["a"]}
        )

        md = _last_run_metadata(conn, t)
        assert md == {"tests_run": ["a"]}  # metadata untouched
        assert _events(conn, t, "worktree_surfaced") == []


def test_complete_worktree_task_missing_worktree_records_error_and_completes(
    kanban_home, tmp_path
):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("req027-missing", default_workdir=str(repo))
    with kb.connect(board="req027-missing") as conn:
        t, ws = _make_worktree_task(conn, "req027-missing", repo)
        # Simulate a lost checkout: remove the worktree before completion.
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(ws)],
            check=True, capture_output=True, text=True,
        )

        assert kb.complete_task(conn, t, result="ok")  # never blocked

        task = kb.get_task(conn, t)
        assert task is not None and task.status == "done"
        events = _events(conn, t, "worktree_surfaced")
        assert len(events) == 1
        assert "error" in events[0]
