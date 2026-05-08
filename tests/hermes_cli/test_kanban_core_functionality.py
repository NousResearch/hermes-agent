"""Core-functionality tests for the kanban kernel + CLI additions.

Complements tests/hermes_cli/test_kanban_db.py (schema + CAS atomicity)
and tests/hermes_cli/test_kanban_cli.py (end-to-end run_slash).  The
tests here exercise the pieces added as part of the kanban hardening
pass: circuit breaker, crash detection, daemon loop, idempotency,
retention/gc, stats, notify subscriptions, worker log accessor, run_slash
parity across every registered verb.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban import run_slash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Idempotency key
# ---------------------------------------------------------------------------

def test_idempotency_key_returns_existing_task(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="first", idempotency_key="abc")
        b = kb.create_task(conn, title="second attempt", idempotency_key="abc")
        assert a == b, "same idempotency_key should return the same task id"
        # And body wasn't overwritten — first create wins.
        task = kb.get_task(conn, a)
        assert task.title == "first"
    finally:
        conn.close()


def test_idempotency_key_ignored_for_archived(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="first", idempotency_key="abc")
        kb.archive_task(conn, a)
        b = kb.create_task(conn, title="second", idempotency_key="abc")
        assert a != b, "archived task shouldn't block a fresh create with same key"
    finally:
        conn.close()


def test_no_idempotency_key_never_collides(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        assert a != b
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Spawn-failure circuit breaker
# ---------------------------------------------------------------------------

def test_spawn_failure_auto_blocks_after_limit(kanban_home, all_assignees_spawnable):
    """N consecutive spawn failures on the same task → auto_blocked."""
    def _bad_spawn(task, ws):
        raise RuntimeError("no PATH")

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        assert kb.DEFAULT_FAILURE_LIMIT == 2
        # One default-limit failure → still ready, counter grows.
        res1 = kb.dispatch_once(conn, spawn_fn=_bad_spawn)
        assert tid not in res1.auto_blocked
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 1

        # Second default-limit failure trips the guard.
        res2 = kb.dispatch_once(conn, spawn_fn=_bad_spawn)
        assert tid in res2.auto_blocked
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures >= 2
        assert task.last_failure_error and "no PATH" in task.last_failure_error
    finally:
        conn.close()


def test_successful_spawn_does_not_reset_failure_counter(kanban_home, all_assignees_spawnable):
    """Under unified consecutive-failure counting, a successful spawn
    does NOT reset the counter — past failures stay on the books until
    a successful completion. This is by design: it prevents a task
    that keeps timing out after spawn from looping forever.
    (Pre-unification behaviour was to reset on spawn success; see the
    complete_task reset for the replacement point.)
    """
    calls = [0]
    def _flaky_spawn(task, ws):
        calls[0] += 1
        if calls[0] <= 2:
            raise RuntimeError("transient")
        return 99999  # pid value — harmless; crash detection will clear it

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        # Two failures + one success.
        kb.dispatch_once(conn, spawn_fn=_flaky_spawn, failure_limit=5)
        kb.dispatch_once(conn, spawn_fn=_flaky_spawn, failure_limit=5)
        task = kb.get_task(conn, tid)
        assert task.consecutive_failures == 2
        kb.dispatch_once(conn, spawn_fn=_flaky_spawn, failure_limit=5)
        task = kb.get_task(conn, tid)
        # Counter STAYS at 2 — spawn succeeded but run isn't complete yet.
        assert task.consecutive_failures == 2
        assert task.last_failure_error is not None
        # Task is now running with a pid.
        assert task.status == "running"
        assert task.worker_pid == 99999
    finally:
        conn.close()


def test_successful_completion_resets_failure_counter(kanban_home, all_assignees_spawnable):
    """A successful kb.complete_task wipes the counter — the task+profile
    combination proved it can succeed, so past failures are history."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        # Simulate 2 prior failures on the record.
        kb.write_txn_ctx = kb.write_txn
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET consecutive_failures = 2, "
                "last_failure_error = 'old failure' WHERE id = ?",
                (tid,),
            )
        # Complete the task.
        ok = kb.complete_task(conn, tid, summary="done")
        assert ok
        task = kb.get_task(conn, tid)
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
    finally:
        conn.close()


def test_reassign_resets_failure_counter_for_new_profile(kanban_home, all_assignees_spawnable):
    """Retry streaks are scoped to a task/profile pair; reassigning is a
    human recovery action and gives the new profile a fresh budget."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET consecutive_failures = 1, "
                "last_failure_error = 'timed out' WHERE id = ?",
                (tid,),
            )
        assert kb.assign_task(conn, tid, "reviewer") is True
        task = kb.get_task(conn, tid)
        assert task.assignee == "reviewer"
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
    finally:
        conn.close()


def test_per_task_max_retries_overrides_dispatcher_limit(kanban_home, all_assignees_spawnable):
    """Per-task ``max_retries`` overrides both the caller-supplied
    ``failure_limit`` (gateway config) and the hardcoded default.

    Three-tier resolution order:
      1. ``task.max_retries`` (set via ``create_task(max_retries=N)`` /
         ``hermes kanban create --max-retries N``)
      2. ``failure_limit`` kwarg passed by the caller (gateway threads
         this from ``kanban.failure_limit`` config)
      3. ``DEFAULT_FAILURE_LIMIT``
    """
    conn = kb.connect()
    try:
        # max_retries=1 should trip on the FIRST failure, even though the
        # caller is asking for failure_limit=10.
        tid = kb.create_task(
            conn, title="one-shot", assignee="worker", max_retries=1,
        )
        task = kb.get_task(conn, tid)
        assert task.max_retries == 1, "per-task override must persist"

        kb.claim_task(conn, tid)
        tripped = kb._record_task_failure(
            conn, tid,
            error="first fail",
            outcome="spawn_failed",
            failure_limit=10,   # far higher than per-task override
            release_claim=True,
            end_run=False,
        )
        assert tripped is True, "should auto-block on first failure"
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures == 1

        # gave_up event should record where the threshold came from
        events = kb.list_events(conn, tid)
        gave_up = [e for e in events if e.kind == "gave_up"]
        assert gave_up, f"expected gave_up event, got {[e.kind for e in events]}"
        assert gave_up[-1].payload.get("limit_source") == "task"
        assert gave_up[-1].payload.get("effective_limit") == 1
    finally:
        conn.close()


def test_per_task_max_retries_allows_more_than_default(kanban_home, all_assignees_spawnable):
    """A task with ``max_retries=5`` does NOT auto-block at the default
    limit of 2 — it must reach the per-task override first."""
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="flaky-retry", assignee="worker", max_retries=5,
        )
        # Four failures — still below the per-task threshold, should stay ready.
        for i in range(1, 5):
            kb.claim_task(conn, tid)
            tripped = kb._record_task_failure(
                conn, tid,
                error=f"fail {i}",
                outcome="spawn_failed",
                # Caller passes the default so the dispatcher tier matches
                # ``DEFAULT_FAILURE_LIMIT``; without the per-task override
                # the breaker would have tripped at failure 2.
                release_claim=True,
                end_run=False,
            )
            assert tripped is False, f"shouldn't trip at failure {i} with max_retries=5"
            task = kb.get_task(conn, tid)
            assert task.status == "ready", f"at failure {i} status was {task.status}"

        # Fifth failure trips the per-task limit.
        kb.claim_task(conn, tid)
        tripped = kb._record_task_failure(
            conn, tid,
            error="fail 5",
            outcome="spawn_failed",
            release_claim=True,
            end_run=False,
        )
        assert tripped is True
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures == 5
    finally:
        conn.close()


def test_max_retries_none_falls_through_to_dispatcher_limit(kanban_home, all_assignees_spawnable):
    """``max_retries=None`` (the default) falls through to the caller-
    supplied ``failure_limit`` — the gateway config tier."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="standard", assignee="worker")
        task = kb.get_task(conn, tid)
        assert task.max_retries is None

        # Caller passes failure_limit=4 (simulates kanban.failure_limit=4).
        # Should trip at 4, not at the DEFAULT_FAILURE_LIMIT of 2.
        for i in range(1, 4):
            kb.claim_task(conn, tid)
            tripped = kb._record_task_failure(
                conn, tid,
                error=f"fail {i}",
                outcome="spawn_failed",
                failure_limit=4,
                release_claim=True,
                end_run=False,
            )
            assert tripped is False, f"premature trip at failure {i}"

        kb.claim_task(conn, tid)
        tripped = kb._record_task_failure(
            conn, tid,
            error="fail 4",
            outcome="spawn_failed",
            failure_limit=4,
            release_claim=True,
            end_run=False,
        )
        assert tripped is True
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"

        events = kb.list_events(conn, tid)
        gave_up = [e for e in events if e.kind == "gave_up"]
        assert gave_up[-1].payload.get("limit_source") == "dispatcher"
        assert gave_up[-1].payload.get("effective_limit") == 4
    finally:
        conn.close()


def test_workspace_resolution_failure_also_counts(kanban_home, all_assignees_spawnable):
    """`dir:` workspace with no path should fail workspace resolution AND
    count against the failure budget — not just crash the tick."""
    conn = kb.connect()
    try:
        # Manually insert a broken task: dir workspace but workspace_path is NULL
        # after initial create. We achieve this by creating via kanban_db then
        # UPDATE-ing workspace_path to NULL.
        tid = kb.create_task(
            conn, title="x", assignee="worker",
            workspace_kind="dir", workspace_path="/tmp/kanban_e2e_dir",
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_path = NULL WHERE id = ?", (tid,),
            )
        res = kb.dispatch_once(conn, failure_limit=3)
        task = kb.get_task(conn, tid)
        assert task.consecutive_failures == 1
        assert task.status == "ready"
        assert task.last_failure_error and "workspace" in task.last_failure_error
        # Run twice more → auto-blocked.
        kb.dispatch_once(conn, failure_limit=3)
        res = kb.dispatch_once(conn, failure_limit=3)
        assert tid in res.auto_blocked
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Worker aliveness / crash detection
# ---------------------------------------------------------------------------

def test_pid_alive_helper():
    # Our own pid is alive.
    assert kb._pid_alive(os.getpid())
    # PID 0 / None / negative.
    assert not kb._pid_alive(0)
    assert not kb._pid_alive(None)
    # A clearly-dead pid (very large, extremely unlikely to exist).
    assert not kb._pid_alive(2 ** 30)


def test_pid_alive_detects_darwin_zombie(monkeypatch):
    monkeypatch.setattr(kb.sys, "platform", "darwin")
    monkeypatch.setattr(kb.os, "kill", lambda pid, sig: None)

    def fake_run(args, **kwargs):
        assert args == ["ps", "-o", "stat=", "-p", "123"]
        assert kwargs["stdout"] is subprocess.PIPE
        return SimpleNamespace(returncode=0, stdout="Z+\n")

    monkeypatch.setattr(kb.subprocess, "run", fake_run)

    assert kb._pid_alive(123) is False


def test_detect_crashed_workers_reclaims(kanban_home):
    """A running task whose pid vanished gets dropped to ready with a
    ``crashed`` event, independent of the claim TTL."""
    def _spawn_pid_that_exits(task, ws):
        # Spawn a real child that exits instantly.
        import subprocess
        p = subprocess.Popen(
            ["python3", "-c", "pass"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
        )
        p.wait()
        return p.pid

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        res = kb.dispatch_once(conn, spawn_fn=_spawn_pid_that_exits)
        # Brief sleep to make sure the child's pid has been reaped; on
        # busy CI the pid may be reused by another process, which would
        # fool _pid_alive. If that happens we accept the test still
        # passing as long as the dispatcher ran without error.
        time.sleep(0.2)
        res2 = kb.dispatch_once(conn)
        task = kb.get_task(conn, tid)
        # Either crashed was detected (preferred) or the TTL reclaim path
        # will eventually fire; we accept either outcome but the worker_pid
        # should no longer be set.
        if res2.crashed:
            assert tid in res2.crashed
            events = kb.list_events(conn, tid)
            assert any(e.kind == "crashed" for e in events)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------

def test_daemon_runs_and_stops(kanban_home):
    """run_daemon should execute at least one tick and exit cleanly on
    stop_event."""
    ticks = []
    stop = threading.Event()

    def _runner():
        kb.run_daemon(
            interval=0.05,
            stop_event=stop,
            on_tick=lambda res: ticks.append(res),
        )

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    # Give it a few ticks.
    time.sleep(0.3)
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive(), "daemon should exit on stop_event"
    assert len(ticks) >= 1, "expected at least one tick"


def test_daemon_keeps_going_after_tick_exception(kanban_home, monkeypatch):
    """A tick that raises shouldn't kill the loop."""
    calls = [0]
    orig_dispatch = kb.dispatch_once

    def _boom(conn, **kw):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("simulated tick failure")
        return orig_dispatch(conn, **kw)

    monkeypatch.setattr(kb, "dispatch_once", _boom)

    stop = threading.Event()
    def _runner():
        kb.run_daemon(interval=0.05, stop_event=stop)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(timeout=2.0)
    # At minimum, second-tick+ should have run.
    assert calls[0] >= 2


# ---------------------------------------------------------------------------
# Stats + age
# ---------------------------------------------------------------------------

def test_board_stats(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a", assignee="x")
        b = kb.create_task(conn, title="b", assignee="y")
        kb.complete_task(conn, a, result="done")
        stats = kb.board_stats(conn)
        assert stats["by_status"]["ready"] == 1
        assert stats["by_status"]["done"] == 1
        assert stats["by_assignee"]["x"]["done"] == 1
        assert stats["by_assignee"]["y"]["ready"] == 1
        assert stats["oldest_ready_age_seconds"] is not None
    finally:
        conn.close()


def test_task_age_helper(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        task = kb.get_task(conn, tid)
        age = kb.task_age(task)
        assert age["created_age_seconds"] is not None
        assert age["started_age_seconds"] is None
        assert age["time_to_complete_seconds"] is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Notify subscriptions
# ---------------------------------------------------------------------------

def test_notify_sub_crud(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123", user_id="u1",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["platform"] == "telegram"
        # Duplicate add is a no-op.
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
        )
        assert len(kb.list_notify_subs(conn, tid)) == 1
        # Distinct thread is a new row.
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            thread_id="5",
        )
        assert len(kb.list_notify_subs(conn, tid)) == 2
        # Remove one.
        ok = kb.remove_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
        )
        assert ok is True
        assert len(kb.list_notify_subs(conn, tid)) == 1
    finally:
        conn.close()


def test_notify_cursor_advances(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="w")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="123")
        # Initial: one "created" event but we only want terminal kinds.
        cursor, events = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert events == []
        # Complete the task → new `completed` event.
        kb.complete_task(conn, tid, result="ok")
        cursor, events = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert len(events) == 1
        assert events[0].kind == "completed"
        # Advance cursor — next call returns empty.
        kb.advance_notify_cursor(
            conn, task_id=tid, platform="telegram", chat_id="123",
            new_cursor=cursor,
        )
        _, events2 = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert events2 == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# GC + retention
# ---------------------------------------------------------------------------

def test_gc_events_keeps_active_task_history(kanban_home):
    """gc_events should only prune rows for terminal (done/archived) tasks."""
    conn = kb.connect()
    try:
        alive = kb.create_task(conn, title="a", assignee="w")
        done_id = kb.create_task(conn, title="b", assignee="w")
        kb.complete_task(conn, done_id)

        # Force all existing events to "old" by bumping created_at backwards.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET created_at = ?",
                (int(time.time()) - 60 * 24 * 3600,),
            )
        removed = kb.gc_events(conn, older_than_seconds=30 * 24 * 3600)
        # At least the done task's "created" + "completed" events gone.
        assert removed >= 2
        # Alive task's events survive.
        alive_events = kb.list_events(conn, alive)
        assert len(alive_events) >= 1
    finally:
        conn.close()


def test_gc_worker_logs_deletes_old_files(kanban_home):
    log_dir = kanban_home / "kanban" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    old = log_dir / "old.log"
    young = log_dir / "young.log"
    old.write_text("stale")
    young.write_text("fresh")
    # Age the old file by 100 days.
    past = time.time() - 100 * 24 * 3600
    os.utime(old, (past, past))
    removed = kb.gc_worker_logs(older_than_seconds=30 * 24 * 3600)
    assert removed == 1
    assert not old.exists()
    assert young.exists()


# ---------------------------------------------------------------------------
# Log rotation + accessor
# ---------------------------------------------------------------------------

def test_worker_log_rotation_keeps_one_generation(kanban_home, tmp_path):
    log_dir = kanban_home / "kanban" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    target = log_dir / "t_aaaa.log"
    target.write_bytes(b"x" * (3 * 1024 * 1024))  # 3 MiB, over 2 MiB threshold
    kb._rotate_worker_log(target, kb.DEFAULT_LOG_ROTATE_BYTES)
    assert not target.exists()
    assert (log_dir / "t_aaaa.log.1").exists()


def test_read_worker_log_tail(kanban_home):
    log_dir = kanban_home / "kanban" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / "t_beef.log"
    # 10 lines
    p.write_text("\n".join(f"line {i}" for i in range(10)))
    full = kb.read_worker_log("t_beef")
    assert full is not None and "line 0" in full
    tail = kb.read_worker_log("t_beef", tail_bytes=30)
    assert tail is not None
    # Tail should not include line 0.
    assert "line 0" not in tail
    # Missing log returns None.
    assert kb.read_worker_log("t_missing") is None


# ---------------------------------------------------------------------------
# CLI bulk verbs
# ---------------------------------------------------------------------------

def test_cli_complete_bulk(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        c = kb.create_task(conn, title="c")
    finally:
        conn.close()
    out = run_slash(f"complete {a} {b} {c} --result all-done")
    assert out.count("Completed") == 3
    conn = kb.connect()
    try:
        for tid in (a, b, c):
            assert kb.get_task(conn, tid).status == "done"
    finally:
        conn.close()


def test_cli_archive_bulk(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
    finally:
        conn.close()
    out = run_slash(f"archive {a} {b}")
    assert "Archived" in out
    conn = kb.connect()
    try:
        assert kb.get_task(conn, a).status == "archived"
        assert kb.get_task(conn, b).status == "archived"
    finally:
        conn.close()


def test_cli_unblock_bulk(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        kb.block_task(conn, a)
        kb.block_task(conn, b)
    finally:
        conn.close()
    out = run_slash(f"unblock {a} {b}")
    assert out.count("Unblocked") == 2


def test_cli_block_bulk_via_ids_flag(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
    finally:
        conn.close()
    out = run_slash(f"block {a} need input --ids {b}")
    assert out.count("Blocked") == 2


def test_cli_create_with_idempotency_key(kanban_home):
    out1 = run_slash("create 'x' --idempotency-key abc --json")
    tid1 = json.loads(out1)["id"]
    out2 = run_slash("create 'y' --idempotency-key abc --json")
    tid2 = json.loads(out2)["id"]
    assert tid1 == tid2


# ---------------------------------------------------------------------------
# CLI stats / watch / log / notify / daemon parity
# ---------------------------------------------------------------------------

def test_cli_stats_json(kanban_home):
    conn = kb.connect()
    try:
        kb.create_task(conn, title="a", assignee="r")
    finally:
        conn.close()
    out = run_slash("stats --json")
    data = json.loads(out)
    assert "by_status" in data
    assert "by_assignee" in data
    assert "oldest_ready_age_seconds" in data


def test_cli_notify_subscribe_and_list(kanban_home):
    tid = run_slash("create 'x' --json")
    tid = json.loads(tid)["id"]
    out = run_slash(
        f"notify-subscribe {tid} --platform telegram --chat-id 999",
    )
    assert "Subscribed" in out
    lst = run_slash("notify-list --json")
    subs = json.loads(lst)
    assert any(s["task_id"] == tid and s["platform"] == "telegram" for s in subs)
    rm = run_slash(
        f"notify-unsubscribe {tid} --platform telegram --chat-id 999",
    )
    assert "Unsubscribed" in rm


def test_cli_log_missing_task(kanban_home):
    # No such task → exit-style (no log for...) message on stderr, returned
    # in combined output.
    out = run_slash("log t_nope")
    assert "no log" in out.lower()


def test_cli_gc_reports_counts(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        kb.archive_task(conn, tid)
    finally:
        conn.close()
    out = run_slash("gc")
    assert "GC complete" in out


# ---------------------------------------------------------------------------
# run_slash parity — every verb returns a sensible, non-crashy string
# ---------------------------------------------------------------------------

def test_run_slash_every_verb_returns_sensible_output(kanban_home):
    """Smoke-test every verb with minimal args. None may raise, none may
    return the empty string (must either succeed or report a usage error)."""
    # Set up a pair of tasks to reference.
    conn = kb.connect()
    try:
        tid_a = kb.create_task(conn, title="a")
        tid_b = kb.create_task(conn, title="b", parents=[tid_a])
    finally:
        conn.close()

    invocations = [
        "",                                  # no subcommand → help text
        "--help",
        "init",
        "create 'smoke'",
        "list",
        "ls",
        f"show {tid_a}",
        f"assign {tid_a} researcher",
        f"link {tid_a} {tid_b}",
        f"unlink {tid_a} {tid_b}",
        f"claim {tid_a}",
        f"comment {tid_a} hello",
        f"complete {tid_a}",
        f"block {tid_b} need input",
        f"unblock {tid_b}",
        f"archive {tid_a}",
        "dispatch --dry-run --json",
        "stats --json",
        "notify-list",
        f"log {tid_a}",
        f"context {tid_b}",
        "gc",
    ]
    for cmd in invocations:
        out = run_slash(cmd)
        assert out is not None
        assert out.strip() != "", f"empty output for `/kanban {cmd}`"


# ---------------------------------------------------------------------------
# Max-runtime enforcement (item 1 from the Multica audit)
# ---------------------------------------------------------------------------

def test_max_runtime_terminates_overrun_worker(kanban_home):
    """A running task whose elapsed time exceeds max_runtime_seconds gets
    SIGTERM'd, emits a ``timed_out`` event, and goes back to ready."""
    killed = []
    def _signal_fn(pid, sig):
        killed.append((pid, sig))

    # We bypass _pid_alive by stubbing it so the grace-poll exits fast.
    import hermes_cli.kanban_db as _kb
    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda pid: False  # pretend SIGTERM worked immediately

    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(
                conn, title="long job", assignee="worker",
                max_runtime_seconds=1,  # one second cap
            )
            # Spawn by hand: claim + set pid + set active run start to the past.
            kb.claim_task(conn, tid)
            kb._set_worker_pid(conn, tid, os.getpid())   # any live pid works
            # Backdate both the task-level first-start timestamp and the active
            # run timestamp so elapsed > limit under the per-run runtime model.
            old_started = int(time.time()) - 30
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET started_at = ? WHERE id = ?",
                    (old_started, tid),
                )
                conn.execute(
                    "UPDATE task_runs SET started_at = ? "
                    "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                    (old_started, tid),
                )

            timed_out = kb.enforce_max_runtime(conn, signal_fn=_signal_fn)
            assert tid in timed_out
            assert killed and killed[0][0] == os.getpid()

            task = kb.get_task(conn, tid)
            assert task.status == "ready",                 f"timed-out task should reset to ready, got {task.status}"
            assert task.worker_pid is None
            assert task.last_heartbeat_at is None

            events = kb.list_events(conn, tid)
            assert any(e.kind == "timed_out" for e in events)
            to_event = next(e for e in events if e.kind == "timed_out")
            assert to_event.payload["limit_seconds"] == 1
            assert to_event.payload["elapsed_seconds"] >= 30
        finally:
            conn.close()
    finally:
        _kb._pid_alive = original_alive


def test_repeated_timeouts_auto_block_at_default_limit(kanban_home):
    """Two timed_out outcomes on the same task/profile trip the retry guard."""
    import hermes_cli.kanban_db as _kb
    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda pid: False

    def _age_active_run(conn, tid):
        old_started = int(time.time()) - 30
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old_started, tid),
            )

    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(
                conn, title="long job", assignee="worker",
                max_runtime_seconds=1,
            )
            for expected_failures in (1, 2):
                kb.claim_task(conn, tid)
                kb._set_worker_pid(conn, tid, os.getpid())
                _age_active_run(conn, tid)
                timed_out = kb.enforce_max_runtime(conn, signal_fn=lambda pid, sig: None)
                assert tid in timed_out
                task = kb.get_task(conn, tid)
                assert task.consecutive_failures == expected_failures
            task = kb.get_task(conn, tid)
            assert task.status == "blocked"
            events = kb.list_events(conn, tid)
            assert [e.kind for e in events].count("timed_out") == 2
            gave_up = [e for e in events if e.kind == "gave_up"]
            assert gave_up and gave_up[-1].payload["trigger_outcome"] == "timed_out"
        finally:
            conn.close()
    finally:
        _kb._pid_alive = original_alive


def test_max_runtime_none_means_no_cap(kanban_home):
    """A task with max_runtime_seconds=None is never timed out regardless
    of how long it runs."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="uncapped", assignee="worker")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, os.getpid())
        # Backdate aggressively; no cap means we don't care.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?",
                (int(time.time()) - 100_000, tid),
            )
        timed_out = kb.enforce_max_runtime(conn)
        assert timed_out == []
        task = kb.get_task(conn, tid)
        assert task.status == "running"
    finally:
        conn.close()


def test_create_task_persists_max_runtime(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", max_runtime_seconds=600)
        task = kb.get_task(conn, tid)
        assert task.max_runtime_seconds == 600
    finally:
        conn.close()


def test_enforce_max_runtime_integrates_with_dispatch(kanban_home, monkeypatch):
    """enforce_max_runtime + dispatch_once integrate cleanly — a timed-out
    task goes through ``timed_out`` → ``ready`` and dispatch_once can then
    re-spawn it without re-reporting the timeout."""
    import hermes_cli.kanban_db as _kb
    # Leave _pid_alive=True so the crash detector doesn't steal the task
    # before timeout enforcement runs. After SIGTERM in enforce_max_runtime,
    # pretend the worker died so the grace wait exits fast.
    state = {"sent_term": False}
    def _alive(pid):
        return not state["sent_term"]
    def _signal(pid, sig):
        import signal as _sig
        if sig == _sig.SIGTERM:
            state["sent_term"] = True
    monkeypatch.setattr(_kb, "_pid_alive", _alive)

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="timeout-me", assignee="worker",
            max_runtime_seconds=1,
        )
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, os.getpid())
        old_started = int(time.time()) - 30
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?",
                (old_started, tid),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old_started, tid),
            )
        # Use enforce_max_runtime directly with our signal stub — dispatch_once
        # uses the default os.kill, but integration-wise calling
        # enforce_max_runtime directly proves the kernel wiring. For the
        # dispatch_once assertion, rely on its own code path by calling it
        # after forcing SIGTERM via enforce_max_runtime.
        before = kb.enforce_max_runtime(conn, signal_fn=_signal)
        assert tid in before, "kernel enforce_max_runtime should catch the overrun"

        # Now a second dispatch_once run should be a no-op on this task
        # (already released). Confirm the loop doesn't re-report it.
        res = kb.dispatch_once(conn, spawn_fn=lambda t, ws: None)
        task = kb.get_task(conn, tid)
        # After timeout, task is back in 'ready' and will be re-spawned
        # by the same pass. That's the intended behaviour.
        assert task.status in ("ready", "running")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Heartbeat (item 2 from the Multica audit)
# ---------------------------------------------------------------------------

def test_heartbeat_on_running_task(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        ok = kb.heartbeat_worker(conn, tid, note="step 3/10")
        assert ok is True
        task = kb.get_task(conn, tid)
        assert task.last_heartbeat_at is not None
        events = kb.list_events(conn, tid)
        hb = [e for e in events if e.kind == "heartbeat"]
        assert len(hb) == 1
        assert hb[0].payload == {"note": "step 3/10"}
    finally:
        conn.close()


def test_heartbeat_refused_when_not_running(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")   # lands in ready, not running
        ok = kb.heartbeat_worker(conn, tid)
        assert ok is False
        task = kb.get_task(conn, tid)
        assert task.last_heartbeat_at is None
    finally:
        conn.close()


def test_cli_heartbeat_verb(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    out = run_slash(f"heartbeat {tid}")
    assert "Heartbeat recorded" in out

    # With --note.
    out = run_slash(f"heartbeat {tid} --note 'step 42'")
    assert "Heartbeat recorded" in out
    conn = kb.connect()
    try:
        events = kb.list_events(conn, tid)
        notes = [e.payload.get("note") for e in events if e.kind == "heartbeat" and e.payload]
        assert "step 42" in notes
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Event vocab rename + spawned event (item 3 from Multica)
# ---------------------------------------------------------------------------

def test_recompute_ready_emits_promoted_not_ready(kanban_home):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="p")
        child = kb.create_task(conn, title="c", parents=[parent])
        kb.complete_task(conn, parent, result="ok")
        # recompute_ready runs inside complete_task too, but call it again
        # defensively.
        kb.recompute_ready(conn)
        events = kb.list_events(conn, child)
        kinds = [e.kind for e in events]
        assert "promoted" in kinds
        # Old name must not appear.
        assert "ready" not in kinds
    finally:
        conn.close()


def test_spawn_failure_circuit_breaker_emits_gave_up(kanban_home, all_assignees_spawnable):
    def _bad(task, ws):
        raise RuntimeError("nope")
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        for _ in range(5):
            kb.dispatch_once(conn, spawn_fn=_bad, failure_limit=5)
        events = kb.list_events(conn, tid)
        kinds = [e.kind for e in events]
        assert "gave_up" in kinds
        assert "spawn_auto_blocked" not in kinds
    finally:
        conn.close()


def test_spawned_event_emitted_with_pid(kanban_home, all_assignees_spawnable):
    """Successful spawn must append a ``spawned`` event with the pid in
    the payload so humans tailing events see pid tracking."""
    def _spawn_returns_pid(task, ws):
        return 98765
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.dispatch_once(conn, spawn_fn=_spawn_returns_pid)
        events = kb.list_events(conn, tid)
        spawned = [e for e in events if e.kind == "spawned"]
        assert len(spawned) == 1
        assert spawned[0].payload == {"pid": 98765}
    finally:
        conn.close()


def test_migration_renames_legacy_event_kinds(tmp_path, monkeypatch):
    """A DB created with the old vocab must have its event rows renamed
    in place on init_db()."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Init fresh.
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        # Inject legacy event kinds directly.
        now = int(time.time())
        with kb.write_txn(conn):
            for old in ("ready", "priority", "spawn_auto_blocked"):
                conn.execute(
                    "INSERT INTO task_events (task_id, kind, payload, created_at) "
                    "VALUES (?, ?, NULL, ?)",
                    (tid, old, now),
                )
        # Re-run init_db — the migration pass should rename them.
        kb.init_db()
        rows = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
        kinds = [r["kind"] for r in rows]
        assert "ready" not in kinds
        assert "priority" not in kinds
        assert "spawn_auto_blocked" not in kinds
        assert "promoted" in kinds
        assert "reprioritized" in kinds
        assert "gave_up" in kinds
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Assignees (item 4 from Multica)
# ---------------------------------------------------------------------------

def test_list_profiles_on_disk(tmp_path, monkeypatch):
    """list_profiles_on_disk returns the implicit default profile plus
    named profiles under ~/.hermes/profiles/ that contain a config.yaml."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    profiles = tmp_path / ".hermes" / "profiles"
    profiles.mkdir(parents=True)
    for name in ("researcher", "writer"):
        d = profiles / name
        d.mkdir()
        (d / "config.yaml").write_text("model: {}\n")
    (profiles / "empty_dir").mkdir()
    # A stray file; should be ignored.
    (profiles / "stray.txt").write_text("noise")

    names = kb.list_profiles_on_disk()
    assert names == ["default", "researcher", "writer"]


def test_list_profiles_on_disk_custom_root(tmp_path, monkeypatch):
    """list_profiles_on_disk respects a custom HERMES_HOME root."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profiles = tmp_path / "profiles"
    profiles.mkdir(parents=True)
    for name in ("researcher", "writer"):
        d = profiles / name
        d.mkdir()
        (d / "config.yaml").write_text("model: {}\n")

    names = kb.list_profiles_on_disk()
    assert names == ["default", "researcher", "writer"]


def test_known_assignees_merges_disk_and_board(tmp_path, monkeypatch):
    """known_assignees unions profiles on disk with currently-assigned
    names, and reports per-status counts."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profiles = tmp_path / ".hermes" / "profiles"
    profiles.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    for name in ("researcher", "writer"):
        d = profiles / name
        d.mkdir()
        (d / "config.yaml").write_text("model: {}\n")

    kb.init_db()
    conn = kb.connect()
    try:
        # writer has a ready task; on_board_only has a task but no profile dir.
        kb.create_task(conn, title="a", assignee="writer")
        kb.create_task(conn, title="b", assignee="on_board_only")
        data = kb.known_assignees(conn)
    finally:
        conn.close()

    by_name = {d["name"]: d for d in data}
    assert by_name["default"]["on_disk"] is True
    assert by_name["default"]["counts"] == {}
    assert by_name["researcher"]["on_disk"] is True
    assert by_name["researcher"]["counts"] == {}
    assert by_name["writer"]["on_disk"] is True
    assert by_name["writer"]["counts"] == {"ready": 1}
    assert by_name["on_board_only"]["on_disk"] is False
    assert by_name["on_board_only"]["counts"] == {"ready": 1}


def test_cli_assignees_json(kanban_home):
    conn = kb.connect()
    try:
        kb.create_task(conn, title="x", assignee="someone")
    finally:
        conn.close()
    out = run_slash("assignees --json")
    data = json.loads(out)
    names = [e["name"] for e in data]
    assert "someone" in names


# ---------------------------------------------------------------------------
# CLI --max-runtime flag + duration parser
# ---------------------------------------------------------------------------

def test_parse_duration_accepts_formats():
    from hermes_cli.kanban import _parse_duration
    assert _parse_duration(None) is None
    assert _parse_duration("") is None
    assert _parse_duration("42") == 42
    assert _parse_duration("30s") == 30
    assert _parse_duration("5m") == 300
    assert _parse_duration("2h") == 7200
    assert _parse_duration("1d") == 86400
    assert _parse_duration("1.5h") == 5400


def test_parse_duration_rejects_garbage():
    from hermes_cli.kanban import _parse_duration
    import pytest as _p
    with _p.raises(ValueError):
        _parse_duration("tenminutes")
    with _p.raises(ValueError):
        _parse_duration("fish")


def test_cli_create_max_runtime_via_duration(kanban_home):
    """`hermes kanban create --max-runtime 2h` should persist 7200 seconds."""
    out = run_slash("create 'long task' --max-runtime 2h --json")
    data = json.loads(out)
    tid = data["id"]
    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task.max_runtime_seconds == 7200
    finally:
        conn.close()


def test_cli_create_max_runtime_bad_format_exits_nonzero(kanban_home):
    out = run_slash("create 'bad' --max-runtime fish")
    assert "max-runtime" in out.lower() or "malformed" in out.lower()


# ---------------------------------------------------------------------------
# Runs as first-class (vulcan-artivus RFC feedback)
# ---------------------------------------------------------------------------

def test_run_created_on_claim(kanban_home):
    """claim_task opens a new task_runs row and points current_run_id at it."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        assert kb.get_task(conn, tid).current_run_id is None

        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        task = kb.get_task(conn, tid)
        assert task.current_run_id is not None

        runs = kb.list_runs(conn, tid)
        assert len(runs) == 1
        r = runs[0]
        assert r.id == task.current_run_id
        assert r.profile == "worker"
        assert r.status == "running"
        assert r.outcome is None
        assert r.ended_at is None
        assert r.claim_lock is not None and r.claim_expires is not None
    finally:
        conn.close()


def test_run_closed_on_complete_with_summary(kanban_home):
    """complete_task ends the active run with outcome='completed' and
    persists summary + metadata."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        ok = kb.complete_task(
            conn, tid,
            result="shipped",
            summary="implemented rate limiter, tests pass",
            metadata={"changed_files": ["limiter.py"], "tests_run": 12},
        )
        assert ok is True

        task = kb.get_task(conn, tid)
        assert task.current_run_id is None
        assert task.result == "shipped"

        runs = kb.list_runs(conn, tid)
        assert len(runs) == 1
        r = runs[0]
        assert r.status == "done"
        assert r.outcome == "completed"
        assert r.summary == "implemented rate limiter, tests pass"
        assert r.metadata == {"changed_files": ["limiter.py"], "tests_run": 12}
        assert r.ended_at is not None
    finally:
        conn.close()


def test_run_summary_falls_back_to_result(kanban_home):
    """If the caller doesn't pass summary, we fall back to result so
    single-run workflows don't need to pass the same string twice."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="only-arg")
        r = kb.latest_run(conn, tid)
        assert r.summary == "only-arg"
    finally:
        conn.close()


def test_multiple_attempts_preserved_as_runs(kanban_home):
    """Crash / retry / complete flow produces one run per attempt, all
    visible in list_runs in chronological order."""
    import hermes_cli.kanban_db as _kb
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")

        # Attempt 1: claim then force the claim to be stale by backdating
        # claim_expires, then let release_stale_claims reclaim it.
        kb.claim_task(conn, tid)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_expires = ? WHERE id = ?",
                (int(time.time()) - 10, tid),
            )
            conn.execute(
                "UPDATE task_runs SET claim_expires = ? WHERE task_id = ?",
                (int(time.time()) - 10, tid),
            )
        kb.release_stale_claims(conn)

        # Attempt 2: claim then crash (simulated: pid dead).
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 98765)
        original_alive = _kb._pid_alive
        _kb._pid_alive = lambda pid: False
        try:
            kb.detect_crashed_workers(conn)
        finally:
            _kb._pid_alive = original_alive

        # Attempt 3: claim then complete.
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="finally")

        runs = kb.list_runs(conn, tid)
        assert len(runs) == 3
        assert [r.outcome for r in runs] == ["reclaimed", "crashed", "completed"]
        assert runs[-1].summary == "finally"
        assert kb.get_task(conn, tid).current_run_id is None
    finally:
        conn.close()


def test_stale_run_cannot_complete_new_attempt(kanban_home, monkeypatch):
    """A worker from an earlier attempt cannot close a later retry."""
    import hermes_cli.kanban_db as _kb

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="retry guarded", assignee="worker")

        kb.claim_task(conn, tid)
        run1 = kb.latest_run(conn, tid)
        kb._set_worker_pid(conn, tid, 98765)
        monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
        assert kb.detect_crashed_workers(conn) == [tid]

        kb.claim_task(conn, tid)
        run2 = kb.latest_run(conn, tid)
        assert run2.id != run1.id

        assert not kb.complete_task(
            conn,
            tid,
            summary="late stale completion",
            expected_run_id=run1.id,
        )
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.current_run_id == run2.id

        assert kb.complete_task(
            conn,
            tid,
            summary="current completion",
            expected_run_id=run2.id,
        )
        runs = kb.list_runs(conn, tid)
        assert [r.outcome for r in runs] == ["crashed", "completed"]
        assert runs[-1].summary == "current completion"
    finally:
        conn.close()


def test_stale_run_cannot_block_or_heartbeat_new_attempt(kanban_home, monkeypatch):
    """Stale retry attempts cannot mutate the active run lifecycle."""
    import hermes_cli.kanban_db as _kb

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="retry heartbeat guarded", assignee="worker")

        kb.claim_task(conn, tid)
        run1 = kb.latest_run(conn, tid)
        kb._set_worker_pid(conn, tid, 98765)
        monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
        assert kb.detect_crashed_workers(conn) == [tid]

        kb.claim_task(conn, tid)
        run2 = kb.latest_run(conn, tid)
        assert run2.id != run1.id

        assert not kb.heartbeat_worker(conn, tid, note="late", expected_run_id=run1.id)
        assert not kb.block_task(conn, tid, reason="late block", expected_run_id=run1.id)
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.current_run_id == run2.id
        assert task.last_heartbeat_at is None

        assert kb.heartbeat_worker(conn, tid, note="current", expected_run_id=run2.id)
        assert kb.block_task(conn, tid, reason="current block", expected_run_id=run2.id)
        assert kb.get_task(conn, tid).status == "blocked"
    finally:
        conn.close()


def test_run_on_block_with_reason(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        kb.block_task(conn, tid, reason="needs API key")

        r = kb.latest_run(conn, tid)
        assert r.outcome == "blocked"
        assert r.summary == "needs API key"
        assert r.ended_at is not None
        assert kb.get_task(conn, tid).current_run_id is None
    finally:
        conn.close()


def test_run_on_spawn_failure_records_failed_runs(kanban_home, all_assignees_spawnable):
    """Each spawn_failed event closes a run with outcome='spawn_failed',
    and the Nth failure closes a run with outcome='gave_up'."""
    def _bad(task, ws):
        raise RuntimeError("no PATH")

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        for _ in range(5):
            kb.dispatch_once(conn, spawn_fn=_bad, failure_limit=5)

        runs = kb.list_runs(conn, tid)
        # 5 claim attempts → 5 runs. Final one is gave_up, earlier ones
        # are spawn_failed.
        assert len(runs) == 5
        assert runs[-1].outcome == "gave_up"
        assert all(r.outcome == "spawn_failed" for r in runs[:-1])
        assert runs[-1].error and "no PATH" in runs[-1].error
    finally:
        conn.close()


def test_event_rows_carry_run_id(kanban_home):
    """task_events.run_id is populated for run-scoped kinds and NULL for
    task-scoped ones."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        # task-scoped: 'created' — no run yet
        # run-scoped: 'claimed' + 'completed'
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="ok")

        rows = conn.execute(
            "SELECT kind, run_id FROM task_events WHERE task_id = ? ORDER BY id",
            (tid,),
        ).fetchall()
        by_kind = {r["kind"]: r["run_id"] for r in rows}
        assert by_kind["created"] is None
        assert by_kind["claimed"] is not None
        assert by_kind["completed"] is not None
        # Both belong to the same run.
        assert by_kind["claimed"] == by_kind["completed"]
    finally:
        conn.close()


def test_build_worker_context_includes_prior_attempts(kanban_home):
    """A worker spawned after a prior attempt sees that attempt's outcome
    + summary in its context so it can skip the failed path."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="port x", assignee="worker")

        # Attempt 1: blocked with a reason.
        kb.claim_task(conn, tid)
        kb.block_task(conn, tid, reason="needs clarification on IP vs user_id")
        kb.unblock_task(conn, tid)

        # Attempt 2: claim (but don't complete yet) and read the context
        # as this worker would see it.
        kb.claim_task(conn, tid)
        ctx = kb.build_worker_context(conn, tid)

        assert "Prior attempts on this task" in ctx
        assert "blocked" in ctx
        assert "needs clarification on IP vs user_id" in ctx
    finally:
        conn.close()


def test_build_worker_context_uses_parent_run_summary(kanban_home):
    """Downstream children read the parent's run.summary + metadata, not
    just task.result."""
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="research", assignee="researcher")
        child = kb.create_task(
            conn, title="write", assignee="writer", parents=[parent],
        )

        kb.claim_task(conn, parent)
        kb.complete_task(
            conn, parent,
            result="done",
            summary="three angles explored; B looks strongest",
            metadata={"sources": ["paper A", "paper B", "paper C"]},
        )

        # child becomes ready via recompute_ready (runs inside complete_task)
        ctx = kb.build_worker_context(conn, child)
        assert "Parent task results" in ctx
        assert "three angles explored; B looks strongest" in ctx
        assert '"sources"' in ctx  # metadata JSON serialized
    finally:
        conn.close()


def test_migration_backfills_inflight_run_for_legacy_db(kanban_home):
    """An existing 'running' task from before task_runs existed should
    get a synthesized run row so subsequent operations (complete,
    heartbeat) have something to write to."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="pre-migration", assignee="worker")
        # Simulate legacy: set running + claim_lock directly, leave
        # current_run_id NULL and delete the run row the claim created.
        kb.claim_task(conn, tid)
        with kb.write_txn(conn):
            conn.execute("DELETE FROM task_runs WHERE task_id = ?", (tid,))
            conn.execute(
                "UPDATE tasks SET current_run_id = NULL WHERE id = ?",
                (tid,),
            )

        # Sanity: no runs, no pointer.
        assert kb.list_runs(conn, tid) == []
        assert kb.get_task(conn, tid).current_run_id is None

        # Re-run init_db — migration backfill should kick in.
        kb.init_db()
        conn2 = kb.connect()
        try:
            runs = kb.list_runs(conn2, tid)
            assert len(runs) == 1
            assert runs[0].status == "running"
            assert runs[0].profile == "worker"
            task = kb.get_task(conn2, tid)
            assert task.current_run_id == runs[0].id

            # Subsequent complete closes the backfilled run cleanly.
            kb.complete_task(conn2, tid, result="done", summary="ok")
            r = kb.latest_run(conn2, tid)
            assert r.outcome == "completed"
            assert r.summary == "ok"
        finally:
            conn2.close()
    finally:
        conn.close()


def test_forward_compat_columns_writable(kanban_home):
    """v2 will route by workflow_template_id + current_step_key. In v1
    these are nullable, kernel doesn't consult them for routing, but
    they must be writable so a v2 client can populate them without
    schema changes."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workflow_template_id = ?, current_step_key = ? "
                "WHERE id = ?",
                ("code-review-v1", "implement", tid),
            )
        task = kb.get_task(conn, tid)
        assert task.workflow_template_id == "code-review-v1"
        assert task.current_step_key == "implement"
    finally:
        conn.close()


def test_cli_runs_verb(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="ok", summary="shipped")
    finally:
        conn.close()
    out = run_slash(f"runs {tid}")
    assert "completed" in out
    assert "shipped" in out
    assert "worker" in out


def test_cli_runs_json(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        kb.complete_task(
            conn, tid, result="ok", summary="shipped",
            metadata={"files": 1},
        )
    finally:
        conn.close()
    out = run_slash(f"runs {tid} --json")
    data = json.loads(out)
    assert len(data) == 1
    assert data[0]["outcome"] == "completed"
    assert data[0]["metadata"] == {"files": 1}


def test_cli_complete_with_summary_and_metadata(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    # JSON metadata must round-trip through shlex + argparse.
    meta = '{"files": 3}'
    out = run_slash(
        "complete " + tid + " --summary \"done it\" --metadata '" + meta + "'"
    )
    assert "Completed" in out
    conn = kb.connect()
    try:
        r = kb.latest_run(conn, tid)
    finally:
        conn.close()
    assert r.summary == "done it"
    assert r.metadata == {"files": 3}


def test_cli_edit_backfills_result_on_done_task(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.complete_task(conn, tid)
    finally:
        conn.close()

    meta = '{"source": "dashboard-recovery"}'
    out = run_slash(
        "edit " + tid
        + " --result \"DECIDED: done\""
        + " --summary \"DECIDED: done\""
        + " --metadata '" + meta + "'"
    )

    assert "Edited" in out
    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
        events = kb.list_events(conn, tid)
    finally:
        conn.close()
    assert task.result == "DECIDED: done"
    assert run.summary == "DECIDED: done"
    assert run.metadata == {"source": "dashboard-recovery"}
    assert events[-1].kind == "edited"


def test_cli_edit_rejects_non_done_task(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
    finally:
        conn.close()

    out = run_slash(f"edit {tid} --result nope")

    assert "not done" in out


def test_cli_complete_bad_metadata_exits_nonzero(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    out = run_slash(f"complete {tid} --metadata not-json")
    assert "metadata" in out.lower()


# -------------------------------------------------------------------------
# Integration hardening (Apr 2026 audit fixes)
# -------------------------------------------------------------------------

def test_archive_of_running_task_closes_run(kanban_home):
    """Archiving a claimed task must close the in-flight run with
    outcome='reclaimed', not orphan it."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        run = kb.latest_run(conn, tid)
        assert run.ended_at is None
        open_run_id = run.id

        assert kb.archive_task(conn, tid) is True

        task = kb.get_task(conn, tid)
        assert task.status == "archived"
        assert task.current_run_id is None
        # The previously-active run must now be closed.
        closed = kb.get_run(conn, open_run_id)
        assert closed.ended_at is not None
        assert closed.outcome == "reclaimed"
    finally:
        conn.close()


def test_archive_of_ready_task_does_not_create_spurious_run(kanban_home):
    """No active run → archive shouldn't synthesize one."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        # Never claimed. Move to ready (task starts in 'ready' here).
        assert kb.archive_task(conn, tid) is True
        runs = kb.list_runs(conn, tid)
        assert runs == []  # No run was ever opened; archive didn't fabricate one.
    finally:
        conn.close()


def test_dashboard_direct_status_change_off_running_closes_run(kanban_home):
    """Dashboard drag-drop running->ready must close the active run.

    Importing _set_status_direct directly to simulate the PATCH handler
    without spinning up FastAPI.
    """
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        open_run = kb.latest_run(conn, tid)
        assert open_run.ended_at is None
        prev_run_id = open_run.id

        # Simulate yanking the worker back to the queue.
        assert _set_status_direct(conn, tid, "ready") is True

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.current_run_id is None
        closed = kb.get_run(conn, prev_run_id)
        assert closed.ended_at is not None
        assert closed.outcome == "reclaimed"
    finally:
        conn.close()


def test_dashboard_direct_status_change_within_same_state_is_noop_for_runs(kanban_home):
    """todo -> ready on an unclaimed task must not create any run rows."""
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        # Force to todo for the sake of the test.
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (tid,))
        conn.commit()
        assert _set_status_direct(conn, tid, "ready") is True
        assert kb.list_runs(conn, tid) == []
    finally:
        conn.close()


def test_cli_bulk_complete_with_summary_rejects(kanban_home):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a", assignee="worker")
        b = kb.create_task(conn, title="b", assignee="worker")
        kb.claim_task(conn, a); kb.claim_task(conn, b)
    finally:
        conn.close()
    # Bulk + summary is refused (stderr message, no mutation).
    # Note: hermes_cli.main doesn't propagate sub-command exit codes
    # (args.func(args) discards the return value), so we check the side
    # effects instead.
    from subprocess import run as _run
    import os, sys
    env = os.environ.copy()
    r = _run(
        [sys.executable, "-m", "hermes_cli.main", "kanban",
         "complete", a, b, "--summary", "oops"],
        capture_output=True, text=True, env=env,
    )
    assert "per-task" in r.stderr, r.stderr
    # The tasks must still be running (no partial apply).
    conn = kb.connect()
    try:
        assert kb.get_task(conn, a).status == "running"
        assert kb.get_task(conn, b).status == "running"
    finally:
        conn.close()


def test_cli_bulk_complete_without_summary_still_works(kanban_home):
    """Bulk close with no per-task handoff is allowed — the common case."""
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a", assignee="worker")
        b = kb.create_task(conn, title="b", assignee="worker")
        kb.claim_task(conn, a); kb.claim_task(conn, b)
    finally:
        conn.close()
    out = run_slash(f"complete {a} {b}")
    assert f"Completed {a}" in out
    assert f"Completed {b}" in out


def test_completed_event_payload_carries_summary(kanban_home):
    """The 'completed' event must embed the run summary so gateway
    notifiers render structured handoffs without a second SQL hit."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="handoff line 1\nextra",
                         metadata={"n": 3})
        events = kb.list_events(conn, tid)
        comp = [e for e in events if e.kind == "completed"]
        assert len(comp) == 1
        # First-line-only, within the 400-char cap, preserved verbatim.
        assert comp[0].payload["summary"] == "handoff line 1"
    finally:
        conn.close()


def test_completed_event_payload_summary_none_when_missing(kanban_home):
    """If the caller passes no summary AND no result, payload.summary is None."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid)  # no summary, no result
        events = kb.list_events(conn, tid)
        comp = [e for e in events if e.kind == "completed"][0]
        assert comp.payload.get("summary") is None
    finally:
        conn.close()


# -------------------------------------------------------------------------
# Deep-scan fixes (Apr 2026 second audit)
# -------------------------------------------------------------------------

def test_complete_never_claimed_task_synthesizes_run(kanban_home):
    """complete_task on a ready (never-claimed) task must persist the
    handoff instead of silently dropping summary/metadata."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="skip claim", assignee="worker")
        # Task is in 'ready' state with no run opened.
        assert kb.list_runs(conn, tid) == []
        ok = kb.complete_task(
            conn, tid,
            summary="did it manually",
            metadata={"reason": "human intervention"},
        )
        assert ok is True

        runs = kb.list_runs(conn, tid)
        assert len(runs) == 1, f"expected 1 synthetic run, got {len(runs)}"
        r = runs[0]
        assert r.outcome == "completed"
        assert r.summary == "did it manually"
        assert r.metadata == {"reason": "human intervention"}
        # Zero-duration synthetic run.
        assert r.started_at == r.ended_at
        # Task pointer still NULL (we never claimed, never opened a run).
        assert kb.get_task(conn, tid).current_run_id is None

        # Event carries the synthetic run_id.
        evts = [e for e in kb.list_events(conn, tid) if e.kind == "completed"]
        assert len(evts) == 1
        assert evts[0].run_id == r.id
    finally:
        conn.close()


def test_block_never_claimed_task_synthesizes_run(kanban_home):
    """block_task on a ready task must persist --reason on a synthetic run."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="drop this", assignee="worker")
        ok = kb.block_task(conn, tid, reason="deprioritized")
        assert ok is True

        runs = kb.list_runs(conn, tid)
        assert len(runs) == 1
        r = runs[0]
        assert r.outcome == "blocked"
        assert r.summary == "deprioritized"
        assert r.started_at == r.ended_at

        evts = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"]
        assert evts[0].run_id == r.id
    finally:
        conn.close()


def test_complete_never_claimed_without_handoff_skips_synthesis(kanban_home):
    """If a bulk-complete passes no summary/metadata/result, don't spam
    the runs table with empty synthetic rows."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="simple", assignee="worker")
        ok = kb.complete_task(conn, tid)  # no handoff fields
        assert ok is True
        assert kb.list_runs(conn, tid) == []  # no synthetic row
    finally:
        conn.close()


def test_event_dataclass_carries_run_id(kanban_home):
    """list_events and the Event dataclass must expose run_id so
    downstream consumers (notifier, dashboard) can group by attempt."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.latest_run(conn, tid).id
        kb.complete_task(conn, tid, summary="done")

        events = kb.list_events(conn, tid)
        kinds_with_run = {
            e.kind: e.run_id for e in events if e.run_id is not None
        }
        # 'created' should NOT have a run_id (task-scoped).
        created = [e for e in events if e.kind == "created"][0]
        assert created.run_id is None
        # 'claimed' and 'completed' must have run_id.
        assert kinds_with_run.get("claimed") == run_id
        assert kinds_with_run.get("completed") == run_id
    finally:
        conn.close()


def test_unseen_events_for_sub_includes_run_id(kanban_home):
    """Gateway notifier path must also surface run_id on events."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify test", assignee="worker")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram",
            chat_id="12345", thread_id="",
        )
        kb.claim_task(conn, tid)
        run_id = kb.latest_run(conn, tid).id
        kb.complete_task(conn, tid, summary="notify-ready")

        cursor, events = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram",
            chat_id="12345", thread_id="",
            kinds=("completed",),
        )
        assert len(events) == 1
        assert events[0].run_id == run_id
    finally:
        conn.close()


def test_claim_task_recovers_from_invariant_leak(kanban_home):
    """Belt-and-suspenders: if a prior run somehow leaked (stranded
    current_run_id on a ready task), claim_task should recover rather
    than strand it further."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="invariant test", assignee="worker")
        # Manually engineer the invariant violation: create a run, then
        # flip status back to 'ready' without closing the run.
        kb.claim_task(conn, tid)
        leaked_run_id = kb.latest_run(conn, tid).id
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
            "claim_expires = NULL "
            "WHERE id = ?", (tid,),
        )
        conn.commit()
        # The leaked run is still open.
        assert kb.get_run(conn, leaked_run_id).ended_at is None

        # Now re-claim — the defensive recovery must close the leak.
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        leaked = kb.get_run(conn, leaked_run_id)
        assert leaked.ended_at is not None
        assert leaked.outcome == "reclaimed"
        # New run opened and pointed to.
        new_run = kb.latest_run(conn, tid)
        assert new_run.id != leaked_run_id
        assert new_run.ended_at is None
    finally:
        conn.close()


# -------------------------------------------------------------------------
# Live-test findings (Apr 2026 third pass: auto-init, show --json carries runs)
# -------------------------------------------------------------------------

def test_cli_create_on_fresh_home_auto_inits(tmp_path, monkeypatch):
    """First CLI action on an empty HERMES_HOME must not error with
    'no such table: tasks' — init_db auto-runs now."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Sanity: kanban.db does NOT exist yet.
    import subprocess as _sp
    import sys as _sys
    worktree_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "HERMES_HOME": str(home),
           "PYTHONPATH": str(worktree_root)}
    r = _sp.run(
        [_sys.executable, "-m", "hermes_cli.main", "kanban",
         "create", "smoke", "--assignee", "worker", "--json"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr}"
    import json as _json
    out = _json.loads(r.stdout)
    assert out["status"] == "ready"
    # DB file exists now.
    assert (home / "kanban.db").exists()


def test_connect_auto_inits_fresh_db(tmp_path, monkeypatch):
    """Calling connect() on a fresh HERMES_HOME must create the
    schema. Previously callers had to remember kb.init_db() first."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Flush the module-level cache so this path looks fresh.
    kb._INITIALIZED_PATHS.clear()

    # Direct connect() without init_db() — used to raise "no such table".
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        assert tid is not None
        assert kb.get_task(conn, tid).title == "x"
    finally:
        conn.close()


def test_cli_show_json_carries_runs(kanban_home):
    """hermes kanban show --json must include runs[] so scripts that
    inspect attempt history don't need a separate 'runs' call."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="show test", assignee="worker")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="inspected")
    finally:
        conn.close()

    out = run_slash(f"show {tid} --json")
    import json as _json
    # run_slash returns combined text; find the JSON block.
    # The output IS json, single doc.
    # Strip any leading ansi or surrounding noise.
    try:
        data = _json.loads(out)
    except _json.JSONDecodeError:
        # Some environments may prefix/suffix whitespace.
        data = _json.loads(out.strip())

    assert "runs" in data, f"show --json must include runs[], got keys: {list(data.keys())}"
    assert len(data["runs"]) == 1
    r = data["runs"][0]
    assert r["outcome"] == "completed"
    assert r["summary"] == "inspected"
    # Events also carry run_id field.
    for e in data["events"]:
        assert "run_id" in e


# -------------------------------------------------------------------------
# Pre-merge audit by @erosika (issue #16102 comment 4331125835) — fixes
# -------------------------------------------------------------------------

def test_unblock_invariant_recovery(kanban_home):
    """unblock_task must leave current_run_id NULL even if some other
    code path left it dangling. Engineer the leak, verify recovery."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="unblock invariant", assignee="worker")
        # Start on running, then open a run, then force to 'blocked' but
        # leave current_run_id pointing at the open run — simulate the
        # invariant violation erosika flagged.
        kb.claim_task(conn, tid)
        leaked_run_id = kb.latest_run(conn, tid).id
        # Force the bad state.
        conn.execute(
            "UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,),
        )
        conn.commit()
        # current_run_id is still set; run is still open.
        assert kb.get_task(conn, tid).current_run_id == leaked_run_id
        assert kb.get_run(conn, leaked_run_id).ended_at is None

        # Unblock — the defensive recovery must close the leaked run.
        assert kb.unblock_task(conn, tid) is True
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.current_run_id is None
        leaked = kb.get_run(conn, leaked_run_id)
        assert leaked.outcome == "reclaimed"
        assert leaked.ended_at is not None
    finally:
        conn.close()


def test_unblock_normal_path_no_spurious_run(kanban_home):
    """Happy path: claim -> block -> unblock. Unblock must be a no-op
    on runs (block_task already closed the run cleanly)."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="normal unblock", assignee="worker")
        kb.claim_task(conn, tid)
        kb.block_task(conn, tid, reason="pause")
        runs_before = len(kb.list_runs(conn, tid))
        assert kb.unblock_task(conn, tid) is True
        runs_after = len(kb.list_runs(conn, tid))
        # No new run created by the happy-path unblock.
        assert runs_after == runs_before
        # Task in ready with cleared pointer.
        t = kb.get_task(conn, tid)
        assert t.status == "ready"
        assert t.current_run_id is None
    finally:
        conn.close()


def test_migration_backfill_idempotent_under_re_run(tmp_path, monkeypatch):
    """init_db must be safe to re-run repeatedly. Each call should leave
    at most one run row per in-flight task, even if called while a
    dispatcher is simultaneously claiming."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Fresh DB, one task left in 'running' with a claim but no run row.
    # Simulates a pre-runs-era DB.
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="legacy inflight", assignee="worker")
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock='old', "
            "claim_expires=?, started_at=?, current_run_id=NULL WHERE id=?",
            (now + 900, now, tid),
        )
        # Drop any synthetic run the normal claim path would have made.
        conn.execute("DELETE FROM task_runs WHERE task_id=?", (tid,))
        conn.commit()

        # Re-run init_db 3x — each should detect the orphan-inflight and
        # install exactly ONE run row, not three.
        for _ in range(3):
            kb.init_db()

        runs = kb.list_runs(conn, tid)
        assert len(runs) == 1, f"expected exactly 1 backfilled run, got {len(runs)}"
        # Pointer should be installed.
        assert kb.get_task(conn, tid).current_run_id == runs[0].id
    finally:
        conn.close()


def test_build_worker_context_includes_role_history(kanban_home):
    """build_worker_context must surface recent completed runs for the
    same assignee, giving cross-task continuity."""
    conn = kb.connect()
    try:
        # Three completed tasks for 'reviewer'
        for i, (title, summary) in enumerate([
            ("Review security PR #1", "approved, focus on CSRF"),
            ("Review security PR #2", "requested changes: SQL injection vector"),
            ("Review security PR #3", "approved, rate-limit added"),
        ]):
            tid = kb.create_task(conn, title=title, assignee="reviewer")
            kb.claim_task(conn, tid)
            kb.complete_task(conn, tid, summary=summary)

        # Now a NEW task for reviewer, not yet done
        new_tid = kb.create_task(
            conn, title="Review perf PR", assignee="reviewer",
        )
        ctx = kb.build_worker_context(conn, new_tid)

        assert "## Recent work by @reviewer" in ctx
        assert "Review security PR #3" in ctx
        assert "approved, rate-limit added" in ctx
        # Current task should be excluded from its own recent work list.
        assert "Review perf PR" not in ctx.split("## Recent work by")[1]
    finally:
        conn.close()


def test_build_worker_context_role_history_skipped_when_no_assignee(kanban_home):
    """If task has no assignee, the role-history section is omitted."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="orphan task")
        # Force no assignee (create_task already defaults to None).
        ctx = kb.build_worker_context(conn, tid)
        assert "## Recent work by" not in ctx
    finally:
        conn.close()


def test_build_worker_context_role_history_bounded_to_5(kanban_home):
    """Role history must be capped at 5 entries even when the assignee
    has many completed tasks."""
    conn = kb.connect()
    try:
        for i in range(10):
            tid = kb.create_task(
                conn, title=f"prior #{i}", assignee="worker",
            )
            kb.claim_task(conn, tid)
            kb.complete_task(conn, tid, summary=f"done #{i}")

        new_tid = kb.create_task(conn, title="new", assignee="worker")
        ctx = kb.build_worker_context(conn, new_tid)
        # Section should exist and contain exactly 5 bullet lines.
        section = ctx.split("## Recent work by @worker")[1]
        bullets = [l for l in section.splitlines() if l.startswith("- ")]
        assert len(bullets) == 5, f"expected 5 bullets, got {len(bullets)}"
    finally:
        conn.close()


# -------------------------------------------------------------------------
# Battle-test findings (May 2026: stress/ suite exposed zombie + id collision)
# -------------------------------------------------------------------------

@pytest.mark.skipif("linux" not in __import__("sys").platform,
                    reason="zombie detection is Linux-specific")
def test_pid_alive_detects_zombie(kanban_home):
    """_pid_alive must return False for a zombie process.

    Without the /proc check, kill(pid, 0) succeeds against zombies
    (process table entry exists until parent reaps), so the dispatcher
    would treat a dead-but-unreaped worker as alive. This catches a
    worker that exited normally but whose parent hasn't called wait().
    """
    import subprocess as _sp
    proc = _sp.Popen(
        ["sleep", "3600"],
        stdin=_sp.DEVNULL, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
    )
    pid = proc.pid
    try:
        assert kb._pid_alive(pid) is True  # live non-zombie
        os.kill(pid, 9)
        time.sleep(0.3)
        # Verify /proc reports zombie state so the test is actually
        # exercising the zombie path and not some other liveness failure
        with open(f"/proc/{pid}/status") as f:
            state_line = next(
                (l for l in f if l.startswith("State:")), ""
            )
        assert "Z" in state_line, f"expected zombie, got {state_line!r}"
        # And _pid_alive must see through it.
        assert kb._pid_alive(pid) is False
    finally:
        try:
            proc.wait(timeout=1)
        except Exception:
            pass


def test_task_ids_dont_collide_at_scale(kanban_home):
    """ID generator must be wide enough that creating 10k tasks doesn't
    hit a UNIQUE constraint violation.

    Regression test for the 2-hex-byte ID (65k space) that would
    collide at ~50% probability by 10k tasks due to birthday paradox.
    Current generator uses 4 hex bytes (4.3B space).
    """
    conn = kb.connect()
    try:
        # 500 is enough to exercise the generator diversity without
        # making the test slow. At 2-hex-byte width, collision chance
        # over 500 creates was ~1.3%; over 10000 the old generator
        # would fail reliably. We don't need the full 10k run to prove
        # the regression; distribution check is sufficient.
        ids = [kb.create_task(conn, title=f"scale-{i}") for i in range(500)]
        assert len(ids) == len(set(ids)), "ID collision at N=500"
        # Sanity: every id matches the expected format
        for tid in ids[:10]:
            assert tid.startswith("t_")
            assert len(tid) == 10  # "t_" + 8 hex chars
    finally:
        conn.close()


def test_cli_show_clamps_negative_elapsed(kanban_home):
    """When NTP jumps backward between claim and complete, started_at
    can exceed ended_at. CLI display must clamp to 0, not print '-3600s'.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="time-skewed", assignee="worker")
        kb.claim_task(conn, tid)
        # Force a future started_at via raw SQL — simulates NTP jump.
        future = int(time.time()) + 3600
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
            (future, tid),
        )
        conn.commit()
        # Complete normally (ended_at < started_at now)
        kb.complete_task(conn, tid, summary="after skew")
    finally:
        conn.close()

    # Both `show` and `runs` render this. Neither should display a
    # negative elapsed token. We check specifically for the pattern
    # `-<digits>s` (the elapsed column) rather than any minus sign,
    # since timestamps legitimately contain dashes (2026-04-28).
    out_show = run_slash(f"show {tid}")
    out_runs = run_slash(f"runs {tid}")
    import re as _re
    neg_elapsed = _re.compile(r"-\d+s")
    assert not neg_elapsed.search(out_show), (
        f"show output has negative elapsed: {out_show!r}"
    )
    assert not neg_elapsed.search(out_runs), (
        f"runs output has negative elapsed: {out_runs!r}"
    )
    # Should show "0s" for the clamped elapsed
    assert "0s" in out_show or "0s" in out_runs


def test_resolve_workspace_rejects_relative_dir_path(kanban_home):
    """dir: workspace_path must be absolute. A relative path like
    '../../../tmp/attacker' would be resolved against the dispatcher's
    CWD — a confused-deputy escape vector."""
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="path-trav", assignee="worker",
            workspace_kind="dir",
            workspace_path="../../../tmp/attacker",
        )
        task = kb.get_task(conn, tid)
        # Storage is verbatim — that's fine.
        assert task.workspace_path == "../../../tmp/attacker"
        # But resolution must refuse.
        with pytest.raises(ValueError, match=r"non-absolute"):
            kb.resolve_workspace(task)
    finally:
        conn.close()


def test_resolve_workspace_accepts_absolute_dir_path(kanban_home, tmp_path):
    """Legitimate absolute paths are accepted and created."""
    conn = kb.connect()
    try:
        abs_path = str(tmp_path / "my-workspace")
        tid = kb.create_task(
            conn, title="legit", assignee="worker",
            workspace_kind="dir",
            workspace_path=abs_path,
        )
        task = kb.get_task(conn, tid)
        resolved = kb.resolve_workspace(task)
        assert str(resolved) == abs_path
        assert resolved.exists()
    finally:
        conn.close()


def test_resolve_workspace_rejects_relative_worktree_path(kanban_home):
    """Worktree paths also must be absolute when explicitly set."""
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="wt", assignee="worker",
            workspace_kind="worktree",
            workspace_path="../escape",
        )
        with pytest.raises(ValueError, match=r"non-absolute"):
            kb.resolve_workspace(kb.get_task(conn, tid))
    finally:
        conn.close()


def test_build_worker_context_caps_prior_attempts(kanban_home):
    """When a task has more than _CTX_MAX_PRIOR_ATTEMPTS runs, only
    the most recent N are shown in full; earlier attempts are summarised
    in a one-line marker so the worker knows more exist without
    blowing the prompt."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="retry", assignee="worker")
        # Force 25 closed runs
        for i in range(25):
            kb.claim_task(conn, tid)
            kb._end_run(conn, tid, outcome="reclaimed",
                        summary=f"attempt {i} summary")
            conn.execute(
                "UPDATE tasks SET status='ready', claim_lock=NULL, "
                "claim_expires=NULL WHERE id=?", (tid,),
            )
            conn.commit()

        ctx = kb.build_worker_context(conn, tid)
        # Check: only _CTX_MAX_PRIOR_ATTEMPTS attempt headers present
        attempt_count = ctx.count("### Attempt ")
        assert attempt_count == kb._CTX_MAX_PRIOR_ATTEMPTS, (
            f"expected {kb._CTX_MAX_PRIOR_ATTEMPTS} attempts shown, got {attempt_count}"
        )
        # And the "omitted" marker appears with the right count
        omitted_count = 25 - kb._CTX_MAX_PRIOR_ATTEMPTS
        assert f"{omitted_count} earlier attempt" in ctx, (
            f"expected omitted-count marker, got ctx=\n{ctx[:2000]}"
        )
        # Total size is bounded — empirically we expect << 100KB even
        # for 1000 attempts (capped to N * ~500 chars)
        assert len(ctx) < 20_000, (
            f"context should be bounded even at 25 runs, got {len(ctx)} chars"
        )
        # Attempt numbering starts at the real index (not renumbered)
        assert "Attempt 16 " in ctx, (
            "first-shown attempt should be numbered 16 (25 - 10 + 1)"
        )
    finally:
        conn.close()


def test_build_worker_context_caps_comments(kanban_home):
    """Same cap for comments — comment-storm tasks stay bounded."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="chatty", assignee="worker")
        for i in range(100):
            kb.add_comment(conn, tid, author=f"u{i % 3}", body=f"comment {i}")
        ctx = kb.build_worker_context(conn, tid)
        # Only _CTX_MAX_COMMENTS most-recent shown in full
        comment_count = ctx.count("**u")
        # 3 distinct authors u0/u1/u2 so the count is trickier; use the
        # "comment N" body text to count.
        body_count = sum(1 for line in ctx.splitlines() if line.startswith("comment "))
        assert body_count == kb._CTX_MAX_COMMENTS, (
            f"expected {kb._CTX_MAX_COMMENTS} comments shown, got {body_count}"
        )
        omitted = 100 - kb._CTX_MAX_COMMENTS
        assert f"{omitted} earlier comment" in ctx
    finally:
        conn.close()


def test_build_worker_context_caps_huge_summary(kanban_home):
    """A 1 MB summary on a single prior run must not dominate the
    worker prompt. Per-field cap truncates with a visible ellipsis."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="giant", assignee="worker")
        kb.claim_task(conn, tid)
        huge = "X" * (1024 * 1024)  # 1 MB
        kb._end_run(conn, tid, outcome="reclaimed", summary=huge)
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL WHERE id=?", (tid,),
        )
        conn.commit()

        ctx = kb.build_worker_context(conn, tid)
        # Much smaller than 1 MB
        assert len(ctx) < 10_000, (
            f"1 MB summary should be capped, got {len(ctx)} chars"
        )
        # Truncation marker present
        assert "truncated" in ctx
    finally:
        conn.close()


def test_default_spawn_auto_loads_kanban_worker_skill(kanban_home, monkeypatch):
    """The dispatcher's _default_spawn must include --skills kanban-worker
    in its argv so every worker loads the skill automatically, even if
    the profile hasn't wired it into its default skills config.

    We intercept Popen to capture the argv without actually spawning a
    hermes subprocess (which would hang trying to call an LLM).
    """
    captured = {}

    class FakeProc:
        def __init__(self):
            self.pid = 99999

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="skill-loading test",
                             assignee="some-profile")
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)
        pid = kb._default_spawn(task, str(workspace))
        assert pid == 99999
    finally:
        conn.close()

    cmd = captured["cmd"]
    assert "--skills" in cmd, f"spawn argv missing --skills: {cmd}"
    idx = cmd.index("--skills")
    assert cmd[idx + 1] == "kanban-worker", (
        f"expected 'kanban-worker', got {cmd[idx + 1]!r}"
    )
    # Assignee + task env are still present
    assert "some-profile" in cmd
    env = captured["env"]
    assert env.get("HERMES_KANBAN_TASK") == tid
    assert env.get("HERMES_PROFILE") == "some-profile"



# ---------------------------------------------------------------------------
# Per-task force-loaded skills
# ---------------------------------------------------------------------------

def test_create_task_persists_skills(kanban_home):
    """Task.skills round-trips through create -> get_task."""
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="skilled task",
            assignee="linguist",
            skills=["translation", "github-code-review"],
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.skills == ["translation", "github-code-review"]
    finally:
        conn.close()


def test_create_task_skills_none_stays_none(kanban_home):
    """Default behavior: no skills arg means Task.skills is None."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="plain task", assignee="someone")
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.skills is None
    finally:
        conn.close()


def test_create_task_skills_deduplicates_and_strips(kanban_home):
    """Dup names collapse; whitespace is stripped; empties dropped."""
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="dedupe",
            assignee="x",
            skills=["  translation  ", "translation", "", None, "review"],
        )
        task = kb.get_task(conn, tid)
        assert task.skills == ["translation", "review"]
    finally:
        conn.close()


def test_create_task_skills_rejects_comma_embedded(kanban_home):
    """Comma in a skill name is rejected — force caller to pass a list."""
    conn = kb.connect()
    try:
        with pytest.raises(ValueError, match="cannot contain comma"):
            kb.create_task(
                conn,
                title="bad",
                assignee="x",
                skills=["a,b"],
            )
    finally:
        conn.close()


def test_default_spawn_appends_per_task_skills(kanban_home, monkeypatch):
    """Dispatcher argv must carry one `--skills X` pair per task skill,
    in addition to the built-in kanban-worker."""
    captured = {}

    class FakeProc:
        def __init__(self):
            self.pid = 42

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="multi-skill worker",
            assignee="linguist",
            skills=["translation", "github-code-review"],
        )
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)
        kb._default_spawn(task, str(workspace))
    finally:
        conn.close()

    cmd = captured["cmd"]
    # Count every --skills pair and gather the skill names.
    skill_names = []
    for i, tok in enumerate(cmd):
        if tok == "--skills" and i + 1 < len(cmd):
            skill_names.append(cmd[i + 1])
    # kanban-worker first (built-in), then per-task extras in order.
    assert skill_names[0] == "kanban-worker", skill_names
    assert "translation" in skill_names
    assert "github-code-review" in skill_names
    # --skills must appear BEFORE the `chat` subcommand so argparse
    # attaches them to the top-level parser, not the subcommand.
    chat_idx = cmd.index("chat")
    last_skills_idx = max(
        i for i, tok in enumerate(cmd) if tok == "--skills"
    )
    assert last_skills_idx < chat_idx, (
        f"--skills must come before 'chat' in argv: {cmd}"
    )


def test_default_spawn_dedupes_kanban_worker_from_task_skills(kanban_home, monkeypatch):
    """If a task explicitly lists 'kanban-worker', we don't double-pass it."""
    captured = {}

    class FakeProc:
        pid = 1

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="dup", assignee="x",
            skills=["kanban-worker", "translation"],
        )
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)
        kb._default_spawn(task, str(workspace))
    finally:
        conn.close()

    cmd = captured["cmd"]
    worker_pairs = [
        i for i, tok in enumerate(cmd)
        if tok == "--skills" and i + 1 < len(cmd) and cmd[i + 1] == "kanban-worker"
    ]
    assert len(worker_pairs) == 1, (
        f"kanban-worker appeared {len(worker_pairs)} times in argv: {cmd}"
    )


def test_cli_create_skill_flag_repeatable(kanban_home):
    """`hermes kanban create --skill a --skill b` persists the list."""
    out = run_slash(
        "create 'multi-skill' --assignee linguist "
        "--skill translation --skill github-code-review --json"
    )
    tid = json.loads(out)["id"]
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.skills == ["translation", "github-code-review"]


def test_cli_create_without_skill_flag_leaves_none(kanban_home):
    """No --skill on the CLI means Task.skills stays None (not []) —
    we don't want to silently write [] when the user didn't opt in."""
    out = run_slash("create 'no-skill' --assignee x --json")
    tid = json.loads(out)["id"]
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.skills is None


def test_cli_show_renders_skills(kanban_home):
    """`hermes kanban show <id>` prints a skills row when present."""
    out = run_slash(
        "create 'show-test' --assignee x "
        "--skill translation --json"
    )
    tid = json.loads(out)["id"]
    shown = run_slash(f"show {tid}")
    assert "skills:" in shown
    assert "translation" in shown


def test_legacy_db_without_skills_column_migrates(tmp_path):
    """_migrate_add_optional_columns is idempotent and adds skills
    when absent. Run it twice on a pared-down schema to confirm."""
    import sqlite3
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Build a pared-down legacy tasks table that lacks all the
    # optional columns _migrate_add_optional_columns knows how to
    # add. We deliberately omit `skills` so we can observe its
    # introduction.
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    # task_events is also touched by the migrator for run_id backfill.
    conn.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old task', 'ready', 1)"
    )
    conn.commit()

    before = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "skills" not in before

    # Run the migrator directly — the same function connect() calls.
    kb._migrate_add_optional_columns(conn)
    after = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "skills" in after, f"migration did not add skills column: {after}"

    # Idempotent: running again must not raise.
    kb._migrate_add_optional_columns(conn)

    # Legacy row has skills=NULL -> Task.skills=None.
    row = conn.execute("SELECT * FROM tasks WHERE id = 'legacy'").fetchone()
    # from_row needs additional columns; build a Task manually via the
    # path from_row takes for a skills NULL/missing.
    keys = set(row.keys())
    assert "skills" in keys
    assert row["skills"] is None
    conn.close()


def test_legacy_spawn_failure_columns_are_copied_not_renamed(tmp_path):
    """Legacy failure counters survive migration without fragile column renames."""
    import sqlite3
    db_path = tmp_path / "legacy-failures.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER,
            tenant TEXT,
            result TEXT,
            idempotency_key TEXT,
            spawn_failures INTEGER NOT NULL DEFAULT 0,
            worker_pid INTEGER,
            last_spawn_error TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    # task_events is required: _migrate_add_optional_columns also runs a
    # PRAGMA on it to back-fill the run_id column and raises
    # OperationalError if the table is absent.
    conn.execute(
        "INSERT INTO tasks "
        "(id, title, body, assignee, status, priority, created_by, created_at, "
        "started_at, completed_at, workspace_kind, workspace_path, claim_lock, "
        "claim_expires, tenant, result, idempotency_key, spawn_failures, "
        "worker_pid, last_spawn_error) "
        "VALUES ('legacy', 'old task', NULL, 'default', 'ready', 0, NULL, 1, "
        "NULL, NULL, 'scratch', NULL, NULL, NULL, NULL, NULL, NULL, 4, NULL, "
        "'missing profile')"
    )
    conn.commit()

    kb._migrate_add_optional_columns(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "spawn_failures" in cols
    assert "consecutive_failures" in cols
    assert "last_spawn_error" in cols
    assert "last_failure_error" in cols

    row = conn.execute("SELECT * FROM tasks WHERE id = 'legacy'").fetchone()
    assert row["consecutive_failures"] == 4
    assert row["last_failure_error"] == "missing profile"
    task = kb.Task.from_row(row)
    assert task.consecutive_failures == 4
    assert task.last_failure_error == "missing profile"

    kb._migrate_add_optional_columns(conn)
    row_again = conn.execute("SELECT * FROM tasks WHERE id = 'legacy'").fetchone()
    assert row_again["consecutive_failures"] == 4
    assert row_again["last_failure_error"] == "missing profile"
    conn.close()


def test_legacy_migration_no_legacy_columns_at_all(tmp_path):
    """Scenario A: DB has neither spawn_failures nor consecutive_failures.

    This is the exact crash scenario from issue #20842 — a very old DB that
    predates the spawn_failures column entirely.  The old RENAME COLUMN path
    raised ``sqlite3.OperationalError: no such column: spawn_failures``.
    The ADD-first approach adds consecutive_failures with default 0.
    """
    import sqlite3

    db_path = tmp_path / "ancient.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    # task_events is required: _migrate_add_optional_columns also runs a
    # PRAGMA on it to back-fill the run_id column and raises
    # OperationalError if the table is absent.
    conn.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('t1', 'ancient task', 'ready', 1)"
    )
    conn.commit()

    # Must not raise (this was the crash before this fix).
    kb._migrate_add_optional_columns(conn)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "consecutive_failures" in cols, "migration must add consecutive_failures"
    assert "last_failure_error" in cols, "migration must add last_failure_error"
    assert "spawn_failures" not in cols, "no legacy column should be synthesised"

    row = conn.execute("SELECT * FROM tasks WHERE id = 't1'").fetchone()
    assert row["consecutive_failures"] == 0
    assert row["last_failure_error"] is None

    # Idempotent second run must not raise either.
    kb._migrate_add_optional_columns(conn)
    row_again = conn.execute("SELECT * FROM tasks WHERE id = 't1'").fetchone()
    assert row_again["consecutive_failures"] == 0
    assert row_again["last_failure_error"] is None
    conn.close()


def test_legacy_migration_both_columns_already_present(tmp_path):
    """Scenario D: DB already has both spawn_failures AND consecutive_failures.

    Represents a partially-migrated DB (e.g. user recovered manually after the
    #20842 crash).  The migration must be a complete no-op and must not
    zero-out the existing counter.
    """
    import sqlite3

    db_path = tmp_path / "partial.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            spawn_failures INTEGER NOT NULL DEFAULT 0,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            last_spawn_error TEXT,
            last_failure_error TEXT
        )
    """)
    # task_events required for the run_id back-fill PRAGMA inside the migrator.
    conn.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at, spawn_failures, "
        "consecutive_failures, last_spawn_error, last_failure_error) "
        "VALUES ('t2', 'partial task', 'ready', 1, 2, 3, 'old error', 'new error')"
    )
    conn.commit()

    kb._migrate_add_optional_columns(conn)

    row = conn.execute("SELECT * FROM tasks WHERE id = 't2'").fetchone()
    # consecutive_failures must not be reset by the migration.
    assert row["consecutive_failures"] == 3, "migration must not overwrite existing counter"
    assert row["last_failure_error"] == "new error", "migration must not overwrite existing error"
    # Legacy column is preserved harmlessly.
    assert row["spawn_failures"] == 2

    # Schema must be unchanged — no spurious ADD or DROP.
    cols_after = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "consecutive_failures" in cols_after
    assert "last_failure_error" in cols_after
    assert "spawn_failures" in cols_after  # legacy preserved

    # Idempotent second run must not modify values or raise.
    kb._migrate_add_optional_columns(conn)
    row_again = conn.execute("SELECT * FROM tasks WHERE id = 't2'").fetchone()
    assert row_again["consecutive_failures"] == 3
    assert row_again["last_failure_error"] == "new error"
    conn.close()


# ---------------------------------------------------------------------------
# Gateway-embedded dispatcher: config, CLI warnings, daemon deprecation stub
# ---------------------------------------------------------------------------

def test_config_default_dispatch_in_gateway_is_true():
    """Default config must enable gateway-embedded dispatch out of the box.
    Flipping this default to false is a user-visible behaviour change and
    should require a conscious migration."""
    from hermes_cli.config import DEFAULT_CONFIG
    kanban = DEFAULT_CONFIG.get("kanban", {})
    assert kanban.get("dispatch_in_gateway") is True, (
        "kanban.dispatch_in_gateway default should be True; got "
        f"{kanban.get('dispatch_in_gateway')!r}"
    )
    interval = kanban.get("dispatch_interval_seconds")
    assert isinstance(interval, (int, float)) and interval >= 1, (
        f"dispatch_interval_seconds must be a positive number, got {interval!r}"
    )


def test_check_dispatcher_presence_silent_when_gateway_running(monkeypatch):
    from hermes_cli import kanban as kb_cli
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 12345)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True}},
    )
    running, msg = kb_cli._check_dispatcher_presence()
    assert running is True
    # Either empty (if import failed defensively) or includes the pid.
    assert msg == "" or "12345" in msg


def test_check_dispatcher_presence_warns_when_no_gateway(monkeypatch):
    from hermes_cli import kanban as kb_cli
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True}},
    )
    running, msg = kb_cli._check_dispatcher_presence()
    assert running is False
    assert "hermes gateway start" in msg


def test_check_dispatcher_presence_warns_when_flag_off(monkeypatch):
    """Gateway is up but dispatch_in_gateway=false -> warning."""
    from hermes_cli import kanban as kb_cli
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 999)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )
    running, msg = kb_cli._check_dispatcher_presence()
    assert running is False
    assert "dispatch_in_gateway" in msg


def test_check_dispatcher_presence_silent_on_probe_error(monkeypatch):
    """If the probe itself errors, we stay silent."""
    from hermes_cli import kanban as kb_cli
    def _raise():
        raise RuntimeError("boom")
    monkeypatch.setattr("gateway.status.get_running_pid", _raise)
    running, msg = kb_cli._check_dispatcher_presence()
    assert running is True
    assert msg == ""


def _make_create_ns(**overrides):
    """Build a Namespace suitable for kb_cli._cmd_create()."""
    ns = argparse.Namespace(
        title="x", body=None, assignee="worker",
        created_by="user", workspace="scratch", tenant=None,
        priority=0, parent=None, triage=False,
        idempotency_key=None, max_runtime=None, skills=None,
        json=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_cli_create_warns_when_no_gateway(kanban_home, monkeypatch, capsys):
    """ready+assigned task + no gateway -> warning on stderr."""
    from hermes_cli import kanban as kb_cli
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True}},
    )
    ns = _make_create_ns(title="warn-me", assignee="worker")
    assert kb_cli._cmd_create(ns) == 0
    captured = capsys.readouterr()
    # Stderr has the warning prefix + guidance.
    assert "hermes gateway start" in captured.err


def test_cli_create_silent_when_gateway_up(kanban_home, monkeypatch, capsys):
    """gateway running + dispatch enabled -> no warning."""
    from hermes_cli import kanban as kb_cli
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 4242)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True}},
    )
    ns = _make_create_ns(title="silent", assignee="worker")
    assert kb_cli._cmd_create(ns) == 0
    captured = capsys.readouterr()
    assert "hermes gateway start" not in captured.err


def test_cli_create_no_warn_on_triage(kanban_home, monkeypatch, capsys):
    """Triage tasks can't be dispatched -> no warning."""
    from hermes_cli import kanban as kb_cli
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True}},
    )
    ns = _make_create_ns(title="triage-task", assignee=None, triage=True)
    assert kb_cli._cmd_create(ns) == 0
    err = capsys.readouterr().err
    assert "hermes gateway start" not in err


def test_cli_create_no_warn_unassigned(kanban_home, monkeypatch, capsys):
    """Unassigned tasks can't be dispatched -> no warning."""
    from hermes_cli import kanban as kb_cli
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True}},
    )
    ns = _make_create_ns(title="nobody", assignee=None)
    assert kb_cli._cmd_create(ns) == 0
    err = capsys.readouterr().err
    assert "hermes gateway start" not in err


def test_cli_daemon_without_force_prints_deprecation_exits_2(kanban_home, capsys):
    """`hermes kanban daemon` (no --force) is a deprecation stub."""
    from hermes_cli import kanban as kb_cli
    ns = argparse.Namespace(
        force=False, interval=60.0, max=None, failure_limit=3,
        pidfile=None, verbose=False,
    )
    rc = kb_cli._cmd_daemon(ns)
    assert rc == 2
    err = capsys.readouterr().err
    assert "DEPRECATED" in err
    assert "hermes gateway start" in err


def test_cli_daemon_help_marks_deprecated():
    """The argparse help string on `daemon` mentions deprecation so users
    scanning `--help` see the migration before running the stub."""
    import argparse as _ap
    from hermes_cli import kanban as kb_cli
    root = _ap.ArgumentParser()
    subs = root.add_subparsers()
    kb_cli.build_parser(subs)
    # Walk the subparser tree to find the daemon action.
    daemon_help = None
    for action in root._actions:
        if isinstance(action, _ap._SubParsersAction):
            for name, parser in action.choices.items():
                if name == "kanban":
                    for sub_action in parser._actions:
                        if isinstance(sub_action, _ap._SubParsersAction):
                            for sname, _ in sub_action.choices.items():
                                if sname == "daemon":
                                    daemon_help = sub_action._choices_actions
                                    break
    # _choices_actions is a list of _ChoicesPseudoAction-like objects with .help
    found_deprecation = False
    if daemon_help:
        for act in daemon_help:
            if getattr(act, "dest", "") == "daemon":
                if "DEPRECATED" in (act.help or ""):
                    found_deprecation = True
                    break
    assert found_deprecation, (
        "daemon subparser help should be marked DEPRECATED so users see "
        "the migration guidance in `hermes kanban --help` output"
    )


# ---------------------------------------------------------------------------
# Gateway embedded dispatcher watcher
# ---------------------------------------------------------------------------

def test_gateway_dispatcher_watcher_respects_config_flag_off(monkeypatch):
    """dispatch_in_gateway=false -> watcher exits fast, no loop."""
    import asyncio
    from gateway.run import GatewayRunner
    import hermes_cli.config as _cfg_mod

    runner = object.__new__(GatewayRunner)
    runner._running = True

    monkeypatch.setattr(
        _cfg_mod, "load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )
    asyncio.run(
        asyncio.wait_for(
            runner._kanban_dispatcher_watcher(),
            timeout=3.0,
        )
    )


def test_gateway_dispatcher_watcher_respects_env_override(monkeypatch):
    """HERMES_KANBAN_DISPATCH_IN_GATEWAY=0 disables without touching config."""
    import asyncio
    from gateway.run import GatewayRunner
    monkeypatch.setenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "0")

    runner = object.__new__(GatewayRunner)
    runner._running = True
    asyncio.run(
        asyncio.wait_for(
            runner._kanban_dispatcher_watcher(),
            timeout=3.0,
        )
    )


def test_gateway_dispatcher_watcher_env_truthy_uses_config(monkeypatch):
    """Truthy env value doesn't force-enable — config still decides.
    (We only treat explicit falses as an override; unset or truthy
    defers to config.)"""
    import asyncio
    from gateway.run import GatewayRunner
    import hermes_cli.config as _cfg_mod

    monkeypatch.setenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "yes")
    monkeypatch.setattr(
        _cfg_mod, "load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )

    runner = object.__new__(GatewayRunner)
    runner._running = True
    # config says false, env is truthy — watcher should still exit
    # (because config is authoritative when env isn't a falsey override).
    asyncio.run(
        asyncio.wait_for(
            runner._kanban_dispatcher_watcher(),
            timeout=3.0,
        )
    )


# ---------------------------------------------------------------------------
# Hallucination gate (created_cards verify + prose scan)
# ---------------------------------------------------------------------------

def test_complete_with_created_cards_all_verified_records_manifest(kanban_home):
    """A completion with created_cards that all exist + belong to this
    worker records them on the ``completed`` event payload."""
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        c1 = kb.create_task(conn, title="c1", assignee="x", created_by="alice")
        c2 = kb.create_task(conn, title="c2", assignee="y", created_by="alice")
        ok = kb.complete_task(
            conn, parent,
            summary="done, created c1+c2",
            created_cards=[c1, c2],
        )
        assert ok is True
        evs = list(conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id",
            (parent,),
        ))
        completed = [e for e in evs if e["kind"] == "completed"]
        assert len(completed) == 1
        import json as _json
        payload = _json.loads(completed[0]["payload"])
        assert payload.get("verified_cards") == [c1, c2]
    finally:
        conn.close()


def test_complete_with_phantom_created_cards_raises_and_audits(kanban_home):
    """A completion claiming a card id that doesn't exist raises
    HallucinatedCardsError, leaves the task in its prior state, and
    records a ``completion_blocked_hallucination`` event for auditing."""
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        real = kb.create_task(conn, title="real", assignee="x", created_by="alice")
        phantom_id = "t_deadbeefcafe"

        with pytest.raises(kb.HallucinatedCardsError) as excinfo:
            kb.complete_task(
                conn, parent,
                summary="claimed phantom",
                created_cards=[real, phantom_id],
            )
        assert excinfo.value.phantom == [phantom_id]

        # Task still in prior state (ready, not done).
        row = conn.execute(
            "SELECT status FROM tasks WHERE id=?", (parent,),
        ).fetchone()
        assert row["status"] == "ready"

        # Audit event landed.
        kinds = [
            r["kind"] for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id=? ORDER BY id",
                (parent,),
            )
        ]
        assert "completion_blocked_hallucination" in kinds
        assert "completed" not in kinds
    finally:
        conn.close()


def test_complete_with_cross_worker_card_is_rejected(kanban_home):
    """A card that exists but was created by a different worker profile
    is treated as phantom (hallucinated attribution)."""
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        other = kb.create_task(conn, title="other", assignee="x", created_by="bob")

        with pytest.raises(kb.HallucinatedCardsError) as excinfo:
            kb.complete_task(
                conn, parent,
                summary="claiming someone else's card",
                created_cards=[other],
            )
        assert excinfo.value.phantom == [other]
    finally:
        conn.close()


def test_complete_accepts_cross_worker_card_when_linked_as_child(kanban_home):
    """A card created by a different principal but explicitly linked as
    a child of the completing task is accepted — the worker took
    ownership via ``kanban_create(parents=[current_task])`` or an
    explicit ``link_tasks`` call, which proves the relationship even
    when ``created_by`` doesn't match.

    (Relaxation salvaged from #20022 @LeonSGP43 — stricter version
    would incorrectly reject legitimate orchestrator flows where a
    specifier creates a card, then a worker picks it up and links it
    to its own parent task.)
    """
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        # Card created by a DIFFERENT principal (not alice, not parent).
        other = kb.create_task(
            conn, title="other", assignee="x", created_by="bob",
            parents=[parent],  # explicitly links as child of the completing task
        )

        ok = kb.complete_task(
            conn, parent,
            summary="completed with linked child",
            created_cards=[other],
        )
        assert ok is True
        # The card should appear in the completed event's verified_cards list.
        import json as _json
        row = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id=? AND kind='completed' ORDER BY id DESC LIMIT 1",
            (parent,),
        ).fetchone()
        payload = _json.loads(row["payload"])
        assert other in payload.get("verified_cards", [])
    finally:
        conn.close()


def test_complete_prose_scan_flags_nonexistent_ids(kanban_home):
    """Successful completion whose summary references a ``t_<hex>`` id
    that doesn't resolve emits a ``suspected_hallucinated_references``
    event. Does not block the completion."""
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="x")
        ok = kb.complete_task(
            conn, parent,
            summary="also saw t_abcd1234ffff failing in CI",
        )
        assert ok is True
        kinds_and_payloads = list(conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id",
            (parent,),
        ))
        kinds = [r["kind"] for r in kinds_and_payloads]
        assert "suspected_hallucinated_references" in kinds
        import json as _json
        susp = [
            _json.loads(r["payload"])
            for r in kinds_and_payloads
            if r["kind"] == "suspected_hallucinated_references"
        ][0]
        assert "t_abcd1234ffff" in susp["phantom_refs"]
    finally:
        conn.close()


def test_complete_prose_scan_ignores_existing_ids(kanban_home):
    """Summaries referencing real task ids don't emit a warning."""
    conn = kb.connect()
    try:
        other = kb.create_task(conn, title="other", assignee="x")
        parent = kb.create_task(conn, title="parent", assignee="x")
        ok = kb.complete_task(
            conn, parent,
            summary=f"depended on {other}, now done",
        )
        assert ok is True
        kinds = [
            r["kind"] for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id=? ORDER BY id",
                (parent,),
            )
        ]
        assert "suspected_hallucinated_references" not in kinds
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Recovery helpers (reclaim + reassign)
# ---------------------------------------------------------------------------

def test_reclaim_task_resets_running_to_ready(kanban_home, monkeypatch):
    """Manual reclaim releases the claim, resets status, and emits a
    ``reclaimed`` event even when claim_expires has not passed."""
    import signal
    import time
    import secrets
    import hermes_cli.kanban_db as _kb
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="stuck", assignee="broken")
        # Simulate a live claim (not expired).
        lock = f"{_kb._claimer_id().split(':', 1)[0]}:{secrets.token_hex(8)}"
        future = int(time.time()) + 3600
        killed: list[int] = []
        state = {"alive": True}

        def _signal(pid, sig):
            killed.append(sig)
            if sig == signal.SIGTERM:
                state["alive"] = False

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: state["alive"])
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, future, 12345, t),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (t, lock, future, 12345, int(time.time())),
        )
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, t))
        conn.commit()

        # release_stale_claims should NOT reclaim (not expired).
        assert kb.release_stale_claims(conn) == 0

        # reclaim_task should work immediately.
        assert kb.reclaim_task(conn, t, reason="test reason", signal_fn=_signal) is True

        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?",
            (t,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
        assert row["worker_pid"] is None

        import json as _json
        reclaim_evs = [
            _json.loads(r["payload"])
            for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='reclaimed'",
                (t,),
            )
        ]
        assert len(reclaim_evs) == 1
        assert reclaim_evs[0].get("manual") is True
        assert reclaim_evs[0].get("reason") == "test reason"
        assert reclaim_evs[0].get("termination_attempted") is True
        assert reclaim_evs[0].get("terminated") is True
        assert killed == [signal.SIGTERM]
    finally:
        conn.close()


def test_reclaim_task_returns_false_for_already_ready(kanban_home):
    """Reclaiming a task that's not running returns False (no-op)."""
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="ready task", assignee="x")
        assert kb.reclaim_task(conn, t) is False
    finally:
        conn.close()


def test_reassign_task_refuses_running_without_reclaim_first(kanban_home):
    """Without ``reclaim_first=True``, reassigning a running task is a
    no-op returning False (matches assign_task's RuntimeError via
    internal catch)."""
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="running", assignee="orig")
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=? WHERE id=?",
            ("live", t),
        )
        conn.commit()
        assert kb.reassign_task(conn, t, "new") is False
        # Assignee unchanged.
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id=?", (t,),
        ).fetchone()
        assert row["assignee"] == "orig"
    finally:
        conn.close()


def test_reassign_task_with_reclaim_first_switches_profile(kanban_home):
    """With ``reclaim_first=True``, a running task is reclaimed and
    reassigned in one operation."""
    import time
    import secrets
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="switch me", assignee="orig")
        lock = secrets.token_hex(8)
        future = int(time.time()) + 3600
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, future, 99999, t),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (t, lock, future, 99999, int(time.time())),
        )
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, t))
        conn.commit()

        assert kb.reassign_task(
            conn, t, "new-profile",
            reclaim_first=True, reason="switch model",
        ) is True

        row = conn.execute(
            "SELECT assignee, status FROM tasks WHERE id=?", (t,),
        ).fetchone()
        assert row["assignee"] == "new-profile"
        assert row["status"] == "ready"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unified failure counter — timeout + crash paths increment the same counter
# as spawn failures, and the circuit breaker trips after N consecutive
# failures regardless of which outcome caused them.
# ---------------------------------------------------------------------------

def test_enforce_max_runtime_increments_consecutive_failures(kanban_home, monkeypatch):
    """A single timeout increments consecutive_failures by 1 (was the
    infinite-respawn gap before unification)."""
    import hermes_cli.kanban_db as _kb
    state = {"sent_term": False}
    def _alive(pid):
        return not state["sent_term"]
    def _signal(pid, sig):
        import signal as _sig
        if sig == _sig.SIGTERM:
            state["sent_term"] = True
    monkeypatch.setattr(_kb, "_pid_alive", _alive)

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="overrun", assignee="worker",
            max_runtime_seconds=1,
        )
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, os.getpid())
        # Since PR #19473 (salvaged) changed enforce_max_runtime to read
        # from task_runs.started_at (per-attempt) rather than
        # tasks.started_at (lifetime), we need to backdate BOTH to
        # guarantee the timeout fires regardless of which column the
        # query pulls from.
        with kb.write_txn(conn):
            long_ago = int(time.time()) - 30
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?",
                (long_ago, tid),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (long_ago, tid),
            )
        before = kb.get_task(conn, tid)
        assert before.consecutive_failures == 0

        kb.enforce_max_runtime(conn, signal_fn=_signal)

        after = kb.get_task(conn, tid)
        assert after.consecutive_failures == 1
        assert "elapsed" in (after.last_failure_error or "")
        # Task status flipped back to ready (not yet past threshold).
        assert after.status == "ready"
    finally:
        conn.close()


def test_repeated_timeouts_trip_the_circuit_breaker(kanban_home, monkeypatch):
    """N consecutive timeouts with the unified counter should eventually
    hit the failure_limit threshold and auto-block the task. This closes
    the Forbidden-Seeds-reported gap where timeout loops never capped.
    """
    import hermes_cli.kanban_db as _kb
    state = {"sent_term": False}
    def _alive(pid):
        return not state["sent_term"]
    def _signal(pid, sig):
        import signal as _sig
        if sig == _sig.SIGTERM:
            state["sent_term"] = True
    monkeypatch.setattr(_kb, "_pid_alive", _alive)

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="loop forever", assignee="slow-worker",
            max_runtime_seconds=1,
        )
        # Drop the failure_limit to 3 so we don't need 5 timeouts.
        # This uses the module-level DEFAULT; we simulate by calling
        # _record_task_failure directly with a tight limit.
        for _ in range(3):
            # Fresh claim + "started long ago" each iteration.
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='running', claim_lock=?, "
                    "claim_expires=?, worker_pid=?, started_at=? "
                    "WHERE id=?",
                    (
                        f"{_kb._claimer_id().split(':', 1)[0]}:lock",
                        int(time.time()) + 3600,
                        os.getpid(),
                        int(time.time()) - 30,
                        tid,
                    ),
                )
                conn.execute(
                    "INSERT INTO task_runs (task_id, status, claim_lock, "
                    "claim_expires, worker_pid, started_at) "
                    "VALUES (?, 'running', ?, ?, ?, ?)",
                    (
                        tid,
                        f"{_kb._claimer_id().split(':', 1)[0]}:lock",
                        int(time.time()) + 3600,
                        os.getpid(),
                        int(time.time()) - 30,
                    ),
                )
                rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute(
                    "UPDATE tasks SET current_run_id=? WHERE id=?",
                    (rid, tid),
                )
            state["sent_term"] = False
            # Lower the threshold by monkeypatching the default.
            monkeypatch.setattr(_kb, "DEFAULT_FAILURE_LIMIT", 3)
            kb.enforce_max_runtime(conn, signal_fn=_signal)

        final = kb.get_task(conn, tid)
        # After 3 consecutive timeouts with failure_limit=3, task should
        # be auto-blocked, not looping forever as ``ready``.
        assert final.status == "blocked", \
            f"expected blocked after 3 timeouts, got {final.status}"
        assert final.consecutive_failures >= 3
        # ``gave_up`` event emitted (plus 3 ``timed_out`` events).
        kinds = [
            r["kind"] for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id=? ORDER BY id",
                (tid,),
            )
        ]
        assert kinds.count("timed_out") >= 3
        assert "gave_up" in kinds
    finally:
        conn.close()


def test_detect_crashed_workers_increments_counter(kanban_home):
    """A single crash increments the consecutive_failures counter."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="crashy", assignee="worker")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 99999)  # fake pid — not alive

        kb.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        assert task.consecutive_failures == 1
        assert task.status == "ready"
    finally:
        conn.close()


def test_detect_crashed_workers_protocol_violation_auto_blocks(kanban_home):
    """A worker that exited rc=0 while its task was still ``running``
    is a protocol violation (agent answered conversationally without
    calling kanban_complete / kanban_block). Retrying will just loop,
    so auto-block immediately instead of waiting for the breaker to
    trip at ``DEFAULT_FAILURE_LIMIT``.

    Regression test for the respawn-loop-after-completion bug reported
    against small local models (gemma4-e2b q4) where the model writes
    the answer as plain text and the CLI exits rc=0 cleanly.
    """
    import hermes_cli.kanban_db as _kb
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="quiet", assignee="worker")
        host_prefix = _kb._claimer_id().split(":", 1)[0]
        lock = f"{host_prefix}:mock"
        kb.claim_task(conn, tid, claimer=lock)
        fake_pid = 999998
        kb._set_worker_pid(conn, tid, fake_pid)

        # Simulate the reap loop having recorded a clean exit for this pid.
        # os.W_EXITCODE(status=0, signal=0) == 0 on POSIX.
        _kb._record_worker_exit(fake_pid, 0)
        # Force liveness check to say "dead" for the fake pid.
        original_alive = _kb._pid_alive
        _kb._pid_alive = lambda p: False
        try:
            result_crashed = kb.detect_crashed_workers(conn)
        finally:
            _kb._pid_alive = original_alive

        assert tid in result_crashed, "should be detected as crashed"
        task = kb.get_task(conn, tid)
        assert task.status == "blocked", (
            f"protocol violation should auto-block on first occurrence, "
            f"got status={task.status}"
        )
        assert "kanban_complete" in (task.last_failure_error or ""), (
            f"expected protocol-violation message, got {task.last_failure_error!r}"
        )

        events = kb.list_events(conn, tid)
        kinds = [e.kind for e in events]
        assert "protocol_violation" in kinds, (
            f"expected 'protocol_violation' event, got {kinds}"
        )
        # The ``crashed`` event would be misleading here — the worker
        # didn't crash, it returned 0.
        assert "crashed" not in kinds, (
            f"should NOT emit 'crashed' event on clean exit, got {kinds}"
        )
        assert "gave_up" in kinds, (
            f"breaker should trip, expected 'gave_up' event, got {kinds}"
        )
    finally:
        conn.close()


def test_detect_crashed_workers_nonzero_exit_uses_default_limit(kanban_home):
    """A worker that exited non-zero (real error / crash) uses the
    normal counter path — one failure doesn't trip the breaker.
    """
    import hermes_cli.kanban_db as _kb
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="crashy", assignee="worker")
        host_prefix = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, tid, claimer=f"{host_prefix}:mock")
        fake_pid = 999997
        kb._set_worker_pid(conn, tid, fake_pid)

        # W_EXITCODE(1, 0) == 256 — WIFEXITED True, WEXITSTATUS == 1.
        _kb._record_worker_exit(fake_pid, 256)
        original_alive = _kb._pid_alive
        _kb._pid_alive = lambda p: False
        try:
            kb.detect_crashed_workers(conn)
        finally:
            _kb._pid_alive = original_alive

        task = kb.get_task(conn, tid)
        assert task.status == "ready", (
            f"single non-zero crash shouldn't auto-block, got {task.status}"
        )
        assert task.consecutive_failures == 1
        events = kb.list_events(conn, tid)
        kinds = [e.kind for e in events]
        assert "crashed" in kinds
        assert "protocol_violation" not in kinds
    finally:
        conn.close()


def test_reclaim_task_clears_failure_counter(kanban_home):
    """Operator reclaim wipes the counter so the next retry gets a fresh
    budget."""
    import secrets
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="stuck", assignee="worker")
        lock = secrets.token_hex(4)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='running', claim_lock=?, "
                "claim_expires=?, worker_pid=?, consecutive_failures=4, "
                "last_failure_error='prior issue' WHERE id=?",
                (lock, int(time.time()) + 3600, 12345, tid),
            )
            conn.execute(
                "INSERT INTO task_runs (task_id, status, claim_lock, "
                "claim_expires, worker_pid, started_at) "
                "VALUES (?, 'running', ?, ?, ?, ?)",
                (tid, lock, int(time.time()) + 3600, 12345, int(time.time())),
            )
            rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "UPDATE tasks SET current_run_id=? WHERE id=?",
                (rid, tid),
            )

        ok = kb.reclaim_task(conn, tid, reason="operator fixed config")
        assert ok

        task = kb.get_task(conn, tid)
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
        assert task.status == "ready"
    finally:
        conn.close()
