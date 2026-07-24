"""Tests for kanban lifecycle plugin hooks.

Verifies that task transitions fire their kanban lifecycle plugin hooks AFTER
the board DB change is committed, with the documented kwargs, and that a
misbehaving hook callback never breaks the transition.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_hooks(monkeypatch):
    """Register capturing callbacks for the three kanban lifecycle hooks.

    Patches the plugin manager's _hooks dict directly (the same registry
    invoke_hook reads) and restores it afterward.
    """
    mgr = get_plugin_manager()
    events: list[tuple[str, dict]] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    for hook in (
        "kanban_task_claimed",
        "kanban_task_completed",
        "kanban_task_blocked",
        "kanban_task_crashed",
        "kanban_task_timed_out",
        "kanban_task_auto_blocked",
    ):
        mgr._hooks.setdefault(hook, []).append(
            lambda _h=hook, **kw: events.append((_h, kw))
        )
    try:
        yield events
    finally:
        mgr._hooks = saved


def test_hooks_are_registered_as_valid():
    """All task lifecycle hook names are part of VALID_HOOKS."""
    assert {
        "kanban_task_claimed",
        "kanban_task_completed",
        "kanban_task_blocked",
        "kanban_task_crashed",
        "kanban_task_timed_out",
        "kanban_task_auto_blocked",
    } <= VALID_HOOKS


def test_claim_fires_hook(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_claimed"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "worker"
    assert "profile_name" in kw
    assert kw["run_id"] is not None


def test_complete_fires_hook_with_summary(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.complete_task(conn, tid, summary="all done")
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_completed"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["summary"] == "all done"
    assert kw["assignee"] == "worker"


def test_block_fires_hook_with_reason(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.block_task(conn, tid, reason="needs human")
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_blocked"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["reason"] == "needs human"


def test_no_hook_on_failed_transition(kanban_home, captured_hooks):
    """complete_task on an unclaimed/nonexistent task fires no hook."""
    conn = kb.connect()
    try:
        # Completing a task that doesn't exist returns False without firing.
        assert kb.complete_task(conn, "t_doesnotexist", summary="x") is False
    finally:
        conn.close()
    assert [e for e in captured_hooks if e[0] == "kanban_task_completed"] == []


def test_misbehaving_hook_does_not_break_transition(kanban_home, monkeypatch):
    """A hook callback that raises must not break the board transition."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    mgr._hooks.setdefault("kanban_task_completed", []).append(_boom)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            kb.claim_task(conn, tid)
            # Despite the raising hook, completion succeeds and persists.
            assert kb.complete_task(conn, tid, summary="ok") is True
            assert kb.get_task(conn, tid).status == "done"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


def test_protocol_violation_fires_crash_then_auto_block_hooks(
    kanban_home, captured_hooks, monkeypatch, tmp_path
):
    """A clean worker exit exposes its cause before remediation is attempted."""
    conn = kb.connect()
    try:
        workspace = tmp_path / "repo"
        workspace.mkdir()
        tid = kb.create_task(
            conn,
            title="t",
            assignee="worker",
            max_retries=1,
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        host = kb._claimer_id().split(":", 1)[0]
        claimed = kb.claim_task(conn, tid, claimer=f"{host}:test")
        assert claimed is not None
        run_id = claimed.current_run_id
        pid = 987654
        kb._set_worker_pid(conn, tid, pid)
        kb._record_worker_exit(pid, 0)
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)

        assert kb.detect_crashed_workers(conn) == [tid]
    finally:
        conn.close()

    failures = [
        event for event in captured_hooks
        if event[0] in {"kanban_task_crashed", "kanban_task_auto_blocked"}
    ]
    assert [event[0] for event in failures] == [
        "kanban_task_crashed",
        "kanban_task_auto_blocked",
    ]
    crashed = failures[0][1]
    assert crashed["task_id"] == tid
    assert crashed["run_id"] == run_id
    assert crashed["outcome"] == "protocol_violation"
    assert crashed["status"] == "blocked"
    assert crashed["consecutive_failures"] == 1
    assert crashed["failure_limit"] == 1
    assert crashed["workspace"] == f"dir:{workspace}"
    assert crashed["error_fingerprint"]

    auto_blocked = failures[1][1]
    assert auto_blocked["outcome"] == "gave_up"
    assert auto_blocked["trigger_outcome"] == "protocol_violation"
    assert auto_blocked["status"] == "blocked"


def test_timeout_fires_post_commit_failure_hook(
    kanban_home, captured_hooks, monkeypatch
):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="t",
            assignee="worker",
            max_runtime_seconds=1,
        )
        host = kb._claimer_id().split(":", 1)[0]
        claimed = kb.claim_task(conn, tid, claimer=f"{host}:test")
        assert claimed is not None
        pid = 987653
        kb._set_worker_pid(conn, tid, pid)
        with kb.write_txn(conn):
            old = 1
            conn.execute("UPDATE tasks SET started_at=? WHERE id=?", (old, tid))
            conn.execute(
                "UPDATE task_runs SET started_at=? WHERE id=?",
                (old, claimed.current_run_id),
            )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

        assert kb.enforce_max_runtime(conn, signal_fn=lambda _pid, _sig: None) == [tid]
    finally:
        conn.close()

    fired = [e for e in captured_hooks if e[0] == "kanban_task_timed_out"]
    assert len(fired) == 1
    timeout = fired[0][1]
    assert timeout["task_id"] == tid
    assert timeout["run_id"] == claimed.current_run_id
    assert timeout["outcome"] == "timed_out"
    assert timeout["status"] == "ready"
    assert timeout["consecutive_failures"] == 1
    assert timeout["failure_limit"] == kb.DEFAULT_FAILURE_LIMIT
    assert timeout["elapsed_seconds"] > timeout["limit_seconds"]
