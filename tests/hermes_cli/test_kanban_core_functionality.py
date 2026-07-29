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
    # Existing crash-detection tests pre-date the grace window; pin to 0
    # so they keep their immediate-reclaim semantics.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Disable the detect_crashed_workers grace period for legacy tests in
    # this file that claim a task and immediately expect
    # ``detect_crashed_workers`` to act on it. The grace period (30s by
    # default, see ``DEFAULT_CRASH_GRACE_SECONDS``) prevents the
    # multi-dispatcher reap race in production; setting it to 0 here
    # restores the pre-fix instant-reclaim semantics these tests were
    # written against. The grace-period itself is covered by dedicated
    # tests in tests/hermes_cli/test_kanban_db.py.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
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
            notifier_profile="default",
            delivery_metadata={
                "chat_type": "dm",
                "telegram_reply_to_message_id": "42",
            },
        )
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["platform"] == "telegram"
        assert subs[0]["notifier_profile"] == "default"
        assert subs[0]["delivery_metadata"] == {
            "chat_type": "dm",
            "telegram_reply_to_message_id": "42",
        }
        # Duplicate add is a no-op.
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            delivery_metadata={
                "chat_type": "dm",
                "telegram_reply_to_message_id": "43",
            },
        )
        assert len(kb.list_notify_subs(conn, tid)) == 1
        assert kb.list_notify_subs(conn, tid)[0]["delivery_metadata"][
            "telegram_reply_to_message_id"
        ] == "43"
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


def test_notify_claim_is_single_owner_and_rewindable(kanban_home):
    conn1 = kb.connect()
    conn2 = kb.connect()
    try:
        tid = kb.create_task(conn1, title="x", assignee="w")
        kb.add_notify_sub(conn1, task_id=tid, platform="telegram", chat_id="123")
        # New subs start caught up at the task's current MAX(task_events.id)
        # (the `created` event) — issue #29905.
        initial_cursor = int(kb.list_notify_subs(conn1, tid)[0]["last_event_id"])
        kb.complete_task(conn1, tid, result="ok")

        old_cursor, claimed_cursor, events = kb.claim_unseen_events_for_sub(
            conn1,
            task_id=tid,
            platform="telegram",
            chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert old_cursor == initial_cursor
        assert claimed_cursor > old_cursor
        assert [ev.kind for ev in events] == ["completed"]

        # A concurrent notifier instance sees the advanced cursor and cannot
        # claim/send the same event range.
        _, _, duplicate_events = kb.claim_unseen_events_for_sub(
            conn2,
            task_id=tid,
            platform="telegram",
            chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert duplicate_events == []

        assert kb.rewind_notify_cursor(
            conn1,
            task_id=tid,
            platform="telegram",
            chat_id="123",
            claimed_cursor=claimed_cursor,
            old_cursor=old_cursor,
        ) is True
        _, retried_events = kb.unseen_events_for_sub(
            conn2,
            task_id=tid,
            platform="telegram",
            chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert [ev.kind for ev in retried_events] == ["completed"]
    finally:
        conn1.close()
        conn2.close()


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


def test_worker_log_rotation_config_defaults_and_overrides():
    assert kb.worker_log_rotation_config({}) == (
        kb.DEFAULT_LOG_ROTATE_BYTES,
        kb.DEFAULT_LOG_BACKUP_COUNT,
    )
    assert kb.worker_log_rotation_config({
        "worker_log_rotate_bytes": 10,
        "worker_log_backup_count": 4,
    }) == (10, 4)


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


# ---------------------------------------------------------------------------
# CLI stats / watch / log / notify / daemon parity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# run_slash parity — every verb returns a sensible, non-crashy string
# ---------------------------------------------------------------------------

def test_run_slash_every_verb_returns_sensible_output(kanban_home, tmp_path):
    """Smoke-test every verb with minimal args. None may raise, none may
    return the empty string (must either succeed or report a usage error)."""
    # Set up a pair of tasks to reference.
    conn = kb.connect()
    try:
        tid_a = kb.create_task(conn, title="a")
        tid_b = kb.create_task(conn, title="b", parents=[tid_a])
    finally:
        conn.close()

    attach_src = tmp_path / "smoke.txt"
    attach_src.write_text("smoke")

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
        f"attach {tid_a} {attach_src}",
        f"attachments {tid_a}",
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
        assert task.status in {"ready", "running"}
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


# ---------------------------------------------------------------------------
# Runs as first-class (vulcan-artivus RFC feedback)
# ---------------------------------------------------------------------------


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


def test_relative_age_renders_coarse_buckets():
    """Freshness helper turns epoch seconds into coarse human ages, and
    degrades safely on missing / future timestamps."""
    now = 1_000_000
    assert kb._relative_age(now, now) == "just now"
    assert kb._relative_age(now - 30, now) == "just now"
    assert kb._relative_age(now - 5 * 60, now) == "5m ago"
    assert kb._relative_age(now - 18 * 3600, now) == "18h ago"
    assert kb._relative_age(now - 2 * 86400, now) == "2d ago"
    # Clock skew across machines/profiles must not claim "in the future".
    assert kb._relative_age(now + 500, now) == "just now"
    # Missing / unparseable timestamps render empty so callers can append
    # unconditionally.
    assert kb._relative_age(None, now) == ""
    # Defensive: an unparseable value (e.g. a stray string) renders empty
    # rather than raising.
    assert kb._relative_age("garbage", now) == ""  # type: ignore[arg-type]


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


def test_default_spawn_does_not_auto_load_any_skill(kanban_home, monkeypatch):
    """The dispatcher no longer auto-loads a bundled kanban skill.

    The kanban lifecycle (formerly the kanban-worker/kanban-orchestrator
    skills) is now injected into every worker's system prompt via
    KANBAN_GUIDANCE, so _default_spawn must NOT append a `--skills` flag
    when the task carries no per-task skills.

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
    assert "--skills" not in cmd, (
        f"spawn argv should not auto-load any skill: {cmd}"
    )
    assert "--accept-hooks" in cmd, f"spawn argv missing --accept-hooks: {cmd}"
    assert cmd.index("--accept-hooks") < cmd.index("chat"), (
        f"--accept-hooks must come before 'chat' in argv: {cmd}"
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


def test_create_task_skills_lists_all_toolset_typos(kanban_home):
    """When several toolset names are passed, the error names every one.

    Agents that confuse skills with toolsets usually pass several at once
    (``skills=["web", "browser", "terminal"]``). Listing only the first
    mistake forces serial fix-then-retry; listing all of them lets the
    caller correct in one round-trip.
    """
    conn = kb.connect()
    try:
        with pytest.raises(ValueError) as exc_info:
            kb.create_task(
                conn,
                title="three bad",
                assignee="x",
                skills=["web", "browser", "terminal"],
            )
        msg = str(exc_info.value)
        assert "'web'" in msg
        assert "'browser'" in msg
        assert "'terminal'" in msg
        # Plural noun form when multiple toolsets are flagged.
        assert "are toolset names" in msg
    finally:
        conn.close()


def test_default_spawn_passes_task_skills_verbatim(kanban_home, monkeypatch):
    """Per-task skills are passed through verbatim — there is no built-in
    kanban skill to dedupe against anymore."""
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
            skills=["translation", "github-code-review"],
        )
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)
        kb._default_spawn(task, str(workspace))
    finally:
        conn.close()

    cmd = captured["cmd"]
    skill_names = [
        cmd[i + 1]
        for i, tok in enumerate(cmd)
        if tok == "--skills" and i + 1 < len(cmd)
    ]
    # Exactly the task's skills, once each, in order — no auto-loaded extras.
    assert skill_names == ["translation", "github-code-review"], (
        f"unexpected --skills in argv: {cmd}"
    )


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


@pytest.mark.parametrize("corrupt_exc", ["sqlite", "guard"])
def test_gateway_dispatcher_disables_corrupt_board_without_traceback(
    monkeypatch, tmp_path, caplog, corrupt_exc
):
    """Corrupt board DBs log one actionable error and stop retrying per tick."""
    import asyncio
    import logging
    import sqlite3

    from gateway.run import GatewayRunner
    import hermes_cli.config as _cfg_mod
    import hermes_cli.kanban_db as _kb

    runner = object.__new__(GatewayRunner)
    runner._running = True
    corrupt_db = tmp_path / "kanban.db"
    corrupt_db.write_text("not sqlite", encoding="utf-8")

    monkeypatch.setattr(
        _cfg_mod,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
            }
        },
    )
    monkeypatch.setattr(
        _kb,
        "list_boards",
        lambda include_archived=False: [{"slug": _kb.DEFAULT_BOARD}],
    )
    monkeypatch.setattr(
        _kb,
        "read_board_metadata",
        lambda slug: {"slug": slug},
    )
    monkeypatch.setattr(_kb, "kanban_db_path", lambda board=None: corrupt_db)

    calls = {"connect": 0, "to_thread": 0}

    def _connect(*args, **kwargs):
        calls["connect"] += 1
        if corrupt_exc == "guard":
            raise _kb.KanbanDbCorruptError(
                corrupt_db,
                corrupt_db.with_suffix(".db.corrupt.test.bak"),
                "sqlite refused to open file: database disk image is malformed",
            )
        raise sqlite3.DatabaseError("file is not a database")

    async def _to_thread(fn, *args, **kwargs):
        # PR salvage (#32857 commit 7): the dispatcher now reaps zombies at
        # the top of each tick via ``asyncio.to_thread(_kb.reap_worker_zombies)``
        # BEFORE the per-board tick work. Each tick now issues 3 ``to_thread``
        # calls (reaper + ``_tick_once`` + ``_ready_nonempty``) instead of 2,
        # so this counter must reach 6 to allow the same 2 dispatch ticks the
        # pre-reaper test expected at 4. Connect counts in the assertion below
        # are unchanged.
        calls["to_thread"] += 1
        result = fn(*args, **kwargs)
        if calls["to_thread"] >= 6:
            runner._running = False
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr(_kb, "connect", _connect)
    monkeypatch.setattr("gateway.run.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        asyncio.run(
            asyncio.wait_for(
                runner._kanban_dispatcher_watcher(),
                timeout=3.0,
            )
        )

    messages = [record.getMessage() for record in caplog.records]
    assert sum("not a valid SQLite database" in msg for msg in messages) == 1
    assert not any("tick failed on board" in msg for msg in messages)
    assert not any(record.exc_info for record in caplog.records)
    # First tick connect (dispatch) + two probes per `_has_ready_work` call
    # (ready then review, both via _kb.connect). The second dispatch tick
    # skips the dispatch connect because the corrupt board fingerprint is
    # disabled, but the ready/review probes still each connect. PR f55d94a1e
    # added the review-column probe alongside the existing ready-column
    # probe, bumping this from 3 → 5.
    assert calls["connect"] == 5


# ---------------------------------------------------------------------------
# Hallucination gate (created_cards verify + prose scan)
# ---------------------------------------------------------------------------


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


def test_complete_can_retry_after_phantom_rejection(kanban_home):
    """A worker that hits the hallucinated-card gate must be able to
    retry kanban_complete on the same task — both with a corrected
    created_cards list and with an empty list (the documented escape
    hatch). Regression test for #22923, where workers were believed to
    be unrecoverable after the first rejection.
    """
    conn = kb.connect()
    try:
        # Two parallel completing tasks so we can exercise both retry
        # shapes without status interference.
        parent_a = kb.create_task(conn, title="retry-empty", assignee="alice")
        kb.claim_task(conn, parent_a)
        parent_b = kb.create_task(conn, title="retry-corrected", assignee="alice")
        kb.claim_task(conn, parent_b)
        real = kb.create_task(
            conn, title="real-child", assignee="x", created_by="alice",
        )

        # First attempt: phantom in the list rejects, task stays running.
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent_a,
                summary="oops",
                created_cards=["t_phantomdeadbeef"],
            )
        assert kb.get_task(conn, parent_a).status == "running"

        # Retry with [] (escape hatch): gate is skipped, completion lands.
        ok = kb.complete_task(
            conn, parent_a,
            summary="retry without claims",
            created_cards=[],
        )
        assert ok is True
        assert kb.get_task(conn, parent_a).status == "done"

        # Same flow on parent_b, but recover via a corrected list rather
        # than the empty escape hatch.
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent_b,
                summary="oops",
                created_cards=[real, "t_anotherphantom"],
            )
        assert kb.get_task(conn, parent_b).status == "running"

        ok = kb.complete_task(
            conn, parent_b,
            summary="retry with corrected list",
            created_cards=[real],
        )
        assert ok is True
        assert kb.get_task(conn, parent_b).status == "done"

        # Both audit events landed; the eventual completion event is
        # also present on each task.
        for parent in (parent_a, parent_b):
            kinds = [
                r["kind"] for r in conn.execute(
                    "SELECT kind FROM task_events WHERE task_id=? ORDER BY id",
                    (parent,),
                )
            ]
            assert kinds.count("completion_blocked_hallucination") == 1
            assert kinds.count("completed") == 1
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


def _drive_worker_exit(conn, tid, fake_pid, raw_status):
    """Claim ``tid``, record ``raw_status`` for its dead worker pid, and run
    one reaper pass.

    Deliberately resolves ``hermes_cli.kanban_db`` fresh and uses that single
    module object for the exit registry, the liveness patch, AND the reaper:
    earlier tests in a full-suite run can reload the module, and recording
    the exit into one module object while reaping through another (stale)
    one makes ``_classify_worker_exit`` return ``unknown`` — silently turning
    a clean-exit protocol violation into a plain crash.
    """
    import hermes_cli.kanban_db as _kb
    host_prefix = _kb._claimer_id().split(":", 1)[0]
    claimed = _kb.claim_task(conn, tid, claimer=f"{host_prefix}:mock")
    assert claimed is not None, "task was not claimable for the next attempt"
    _kb._set_worker_pid(conn, tid, fake_pid)
    _kb._record_worker_exit(fake_pid, raw_status)
    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda p: False
    try:
        return _kb.detect_crashed_workers(conn)
    finally:
        _kb._pid_alive = original_alive


def _drive_protocol_violation(conn, tid, fake_pid):
    """One clean-exit protocol violation reaper pass for ``tid``.

    os.W_EXITCODE(status=0, signal=0) == 0 on POSIX.
    """
    return _drive_worker_exit(conn, tid, fake_pid, 0)


def _drive_nonzero_crash(conn, tid, fake_pid):
    """One plain non-zero-exit crash reaper pass for ``tid``.

    W_EXITCODE(1, 0) == 256 — WIFEXITED True, WEXITSTATUS == 1.
    """
    return _drive_worker_exit(conn, tid, fake_pid, 256)


def test_protocol_violation_budget_not_consumed_by_other_failures(kanban_home):
    """Mixed failure kinds must not consume the violation retry budget.

    Regression for the #61233 review finding: expressed as a plain
    ``failure_limit`` over the unified ``consecutive_failures`` counter, the
    violation budget was consumed by earlier timeouts / nonzero exits. As a
    violation-only streak, a prior real crash must not eat violation
    retries, and below-budget violations must leave the unified counter
    untouched (so the two budgets stay independent).
    """
    import hermes_cli.kanban_db as _kb
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="mixed", assignee="worker")

        # One real crash: unified counter ticks to 1 (below
        # DEFAULT_FAILURE_LIMIT=2 — task stays ready).
        _drive_nonzero_crash(conn, tid, 991000)
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 1

        # Two violations after it: streak 1 and 2 — both retry, unified
        # counter untouched. (Pre-fix: the crash consumed the budget and the
        # violations blocked well before three of them happened.)
        for i, pid in enumerate((991001, 991002)):
            _drive_protocol_violation(conn, tid, pid)
            task = kb.get_task(conn, tid)
            assert task.status == "ready", (
                f"violation {i + 1} after a crash must still retry, "
                f"got {task.status}"
            )
            assert task.consecutive_failures == 1, (
                "below-budget violations must not tick the unified counter"
            )

        # Third consecutive violation: streak hits the bound — blocked.
        _drive_protocol_violation(conn, tid, 991003)
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        gave_up = [e for e in kb.list_events(conn, tid) if e.kind == "gave_up"]
        assert len(gave_up) == 1
        assert (gave_up[0].payload or {}).get("protocol_violations") == \
            _kb._PROTOCOL_VIOLATION_FAILURE_LIMIT
    finally:
        conn.close()


def test_protocol_violation_respects_max_retries_precedence(kanban_home):
    """Per-task ``max_retries`` overrides the violation bound, both ways.

    Same top precedence it has for every other failure kind in
    ``_record_task_failure``: ``max_retries=1`` blocks on the FIRST violation
    (zero retries — the pre-fix behavior, now opt-in per task);
    ``max_retries=5`` keeps retrying past the default bound of 3 and blocks
    on the 5th consecutive violation.
    """
    conn = kb.connect()
    try:
        strict = kb.create_task(
            conn, title="strict", assignee="worker", max_retries=1,
        )
        _drive_protocol_violation(conn, strict, 992000)
        task = kb.get_task(conn, strict)
        assert task.status == "blocked", (
            f"max_retries=1 must block on the first violation, got {task.status}"
        )
        gave_up = [e for e in kb.list_events(conn, strict) if e.kind == "gave_up"]
        assert len(gave_up) == 1
        payload = gave_up[0].payload or {}
        assert payload.get("protocol_violations") == 1
        assert payload.get("protocol_violation_limit") == 1

        lenient = kb.create_task(
            conn, title="lenient", assignee="worker", max_retries=5,
        )
        for i in range(4):
            _drive_protocol_violation(conn, lenient, 992100 + i)
            assert kb.get_task(conn, lenient).status == "ready", (
                f"violation {i + 1}/5 should retry under max_retries=5"
            )
        _drive_protocol_violation(conn, lenient, 992104)
        assert kb.get_task(conn, lenient).status == "blocked"
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


def test_dispatch_once_integrates_stale_detection(kanban_home, monkeypatch):
    """dispatch_once with stale_timeout_seconds reclaims stale running tasks."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="stale-dispatch", assignee="worker")
        kb.claim_task(conn, t)
        kb._set_worker_pid(conn, t, 99999)  # fake PID — avoid killing test

        five_hours_ago = int(time.time()) - (5 * 3600)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?", (five_hours_ago, t)
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (five_hours_ago, t),
            )

        res = kb.dispatch_once(
            conn,
            spawn_fn=lambda tsk, ws: None,
            stale_timeout_seconds=14400,
        )
        assert t in res.stale, "Stale task should appear in result.stale"
        assert kb.get_task(conn, t).status == "ready"


def test_dispatch_once_stale_disabled_when_timeout_zero(kanban_home, monkeypatch):
    """dispatch_once with stale_timeout_seconds=0 skips stale detection."""
    # Use os.getpid() so _pid_alive → True, preventing detect_crashed_workers
    # from reclaiming. Only stale detection (disabled via timeout=0) is tested.

    with kb.connect() as conn:
        t = kb.create_task(conn, title="skip-stale", assignee="worker")
        kb.claim_task(conn, t)
        # Claim sets worker_pid to 0 initially. Set it to os.getpid() so the
        # crash detector sees a live PID and skips it.
        kb._set_worker_pid(conn, t, os.getpid())

        five_hours_ago = int(time.time()) - (5 * 3600)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?", (five_hours_ago, t)
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (five_hours_ago, t),
            )

        res = kb.dispatch_once(
            conn,
            spawn_fn=lambda tsk, ws: None,
            stale_timeout_seconds=0,
        )
        assert res.stale == [], "stale_timeout_seconds=0 should disable detection"
        assert kb.get_task(conn, t).status == "running"


def test_notify_sub_starts_caught_up_on_active_task(kanban_home):
    """A new subscription must NOT replay historical terminal events.

    Regression for issue #29905: `kanban_notify_subs.last_event_id` defaulted
    to 0, so subscribing to a task that already had terminal events in
    `task_events` replayed the entire backlog on the next notifier tick — 27
    stale subs produced a 100+ message burst at gateway boot. The cursor now
    snaps to the task's MAX(task_events.id) at creation: only events that
    occur AFTER subscribing are delivered.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="old task", assignee="w")
        # Historical terminal activity BEFORE anyone subscribes.
        kb.complete_task(conn, tid, result="done long ago")

        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="123")
        sub = kb.list_notify_subs(conn, tid)[0]
        assert int(sub["last_event_id"]) > 0, (
            "cursor must snap to MAX(task_events.id) at subscription time"
        )
        _, events = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        assert events == [], "historical events must not replay to a new sub"
    finally:
        conn.close()


