from pathlib import Path
from types import SimpleNamespace

from hermes_cli import kanban_cleanup as kc
from hermes_cli.kanban_cleanup import (
    CleanupDecision,
    ProcessInfo,
    WorktreeInfo,
    classify_workspace_dir_for_cleanup,
    classify_worktree_for_cleanup,
)
from hermes_cli.kanban_policy import BoardPolicy


def test_classify_clean_secondary_worktree_safe(tmp_path):
    project = tmp_path / "Project"
    policy = BoardPolicy(board="demo", project_root=project, worktree_root=project / ".worktrees")
    info = WorktreeInfo(path=project / ".worktrees" / "task-1", main=False, dirty=False)

    decision = classify_worktree_for_cleanup(info, policy, [])

    assert decision.action == "safe_remove"


def test_classify_dirty_worktree_blocked(tmp_path):
    project = tmp_path / "Project"
    policy = BoardPolicy(board="demo", project_root=project, worktree_root=project / ".worktrees")
    info = WorktreeInfo(path=project / ".worktrees" / "task-1", main=False, dirty=True)

    decision = classify_worktree_for_cleanup(info, policy, [])

    assert decision.action == "blocked_dirty"


def test_classify_active_worktree_blocked(tmp_path):
    project = tmp_path / "Project"
    wt = project / ".worktrees" / "task-1"
    policy = BoardPolicy(board="demo", project_root=project, worktree_root=project / ".worktrees")
    info = WorktreeInfo(path=wt, main=False, dirty=False)

    decision = classify_worktree_for_cleanup(info, policy, [ProcessInfo(pid=123, command=f"pnpm dev {wt}")])

    assert decision.action == "blocked_active"


def test_classify_main_checkout_protected(tmp_path):
    project = tmp_path / "Project"
    policy = BoardPolicy(board="demo", project_root=project, worktree_root=project / ".worktrees")
    info = WorktreeInfo(path=project, main=True, dirty=False)

    decision = classify_worktree_for_cleanup(info, policy, [])

    assert decision.action == "protected_main"


def test_classify_scratch_dir_safe_when_inactive(tmp_path):
    project = tmp_path / "Project"
    scratch = tmp_path / ".hermes" / "kanban" / "boards" / "demo" / "workspaces" / "t1"
    policy = BoardPolicy(board="demo", project_root=project, worktree_root=project / ".worktrees")

    decision = classify_workspace_dir_for_cleanup(scratch, policy, [])

    assert decision.action == "safe_remove_scratch"


def test_teardown_reports_broad_matches_but_only_kills_db_workers(monkeypatch, tmp_path):
    project = tmp_path / "Project"
    policy = BoardPolicy(board="demo", project_root=project, worktree_root=project / ".worktrees")
    killed = []

    monkeypatch.setattr(kc, "load_policy", lambda board: policy)
    monkeypatch.setattr(kc, "find_killable_worker_processes", lambda board: [ProcessInfo(pid=111, command="hermes worker", source="worker_pid:t1")])
    monkeypatch.setattr(kc, "find_relevant_processes", lambda board, policy: [
        ProcessInfo(pid=111, command="hermes worker", source="worker_pid:t1"),
        ProcessInfo(pid=222, command="vim /projects/demo", source="broad_match"),
    ])
    monkeypatch.setattr(kc, "stop_processes", lambda processes: killed.extend([p.pid for p in processes]) or [p.pid for p in processes])
    monkeypatch.setattr(kc, "list_registered_worktrees", lambda root: [])
    monkeypatch.setattr(kc.kb, "board_dir", lambda board: tmp_path / "missing-board")

    result = kc.teardown_board("demo", remove_all_worktrees=True, delete_board=True, yes=True)

    assert killed == [111]
    assert [p["pid"] for p in result["ambiguous_processes_reported_only"]] == [222]


def test_cleanup_reports_failed_worktree_remove(monkeypatch, tmp_path):
    project = tmp_path / "Project"
    wt = project / ".worktrees" / "task-1"
    policy = BoardPolicy(board="demo", project_root=project, worktree_root=project / ".worktrees")
    monkeypatch.setattr(kc, "load_policy", lambda board: policy)
    monkeypatch.setattr(kc, "find_relevant_processes", lambda board, policy: [])
    monkeypatch.setattr(kc, "list_registered_worktrees", lambda root: [WorktreeInfo(path=wt, main=False, dirty=False)])
    monkeypatch.setattr(kc, "_run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="boom"))

    result = kc.cleanup_board("demo", dry_run=False)

    assert result["removed"] == []
    assert {e["path"] for e in result["errors"]} == {str(wt), str(project)}
    assert any(e["error"] == "boom" for e in result["errors"])
    assert result["verified"] is False
