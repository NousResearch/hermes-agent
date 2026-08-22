"""Retry-accounting regressions for the Kanban dispatcher.

These tests stay separate from the broad core-functionality suite so retry
budget and transaction-boundary coverage survives pruning of unrelated Kanban
tests.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _drive_worker_exit(conn, task_id, fake_pid, raw_status):
    """Claim a task, record a dead worker status, and run one reaper pass."""
    import hermes_cli.kanban_db as current_kb

    host = current_kb._claimer_id().split(":", 1)[0]
    claimed = current_kb.claim_task(
        conn, task_id, claimer=f"{host}:retry-accounting-test",
    )
    assert claimed is not None, "task was not claimable for the next attempt"
    current_kb._set_worker_pid(conn, task_id, fake_pid)
    current_kb._record_worker_exit(fake_pid, raw_status)
    original_alive = current_kb._pid_alive
    current_kb._pid_alive = lambda _pid: False
    try:
        return current_kb.detect_crashed_workers(conn)
    finally:
        current_kb._pid_alive = original_alive


def _drive_protocol_violation(conn, task_id, fake_pid):
    return _drive_worker_exit(conn, task_id, fake_pid, 0)


def _drive_nonzero_crash(conn, task_id, fake_pid):
    # W_EXITCODE(1, 0) == 256: a normal exit with status 1.
    return _drive_worker_exit(conn, task_id, fake_pid, 256)


def test_explicit_max_retries_counts_mixed_failure_kinds(kanban_home):
    """Issue #72174: an explicit cap bounds all non-neutral failures."""
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="mixed-explicit-cap",
            assignee="worker",
            max_retries=3,
        )

        _drive_nonzero_crash(conn, task_id, 994000)
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.consecutive_failures == 1

        _drive_protocol_violation(conn, task_id, 994001)
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.consecutive_failures == 2

        _drive_protocol_violation(conn, task_id, 994002)
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.consecutive_failures == 3

        gave_up = [
            event
            for event in kb.list_events(conn, task_id)
            if event.kind == "gave_up"
        ]
        assert len(gave_up) == 1
        payload = gave_up[0].payload or {}
        assert payload.get("failures") == 3
        assert payload.get("effective_limit") == 3
        assert payload.get("limit_source") == "task"
        assert payload.get("protocol_violation") is True
        assert "protocol_violation_limit" not in payload
        assert kb.claim_task(conn, task_id) is None


def test_protocol_violation_accounting_is_atomic_with_requeue(
    kanban_home, monkeypatch,
):
    """A review contender cannot claim between requeue and give-up."""
    conn = kb.connect()
    contender_started = threading.Event()
    contender_done = threading.Event()
    contender_result = {}
    contender_errors = []
    try:
        task_id = kb.create_task(
            conn,
            title="atomic-protocol-accounting",
            assignee="worker",
            max_retries=1,
        )
        host = kb._claimer_id().split(":", 1)[0]
        implementation = kb.claim_task(
            conn, task_id, claimer=f"{host}:implementer",
        )
        assert implementation is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="ready for review",
            reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
        )
        assert kb.claim_review_task(
            conn, task_id, claimer=f"{host}:old",
        ) is not None
        kb._set_worker_pid(conn, task_id, 994100)
        kb._record_worker_exit(994100, 0)

        def contender():
            other = kb.connect()
            try:
                assert contender_started.wait(timeout=5)
                claimed = kb.claim_review_task(
                    other, task_id, claimer=f"{host}:contender",
                )
                contender_result["claimed"] = claimed is not None
                if claimed is not None:
                    kb._set_worker_pid(other, task_id, 994101)
            except BaseException as exc:
                contender_errors.append(exc)
            finally:
                other.close()
                contender_done.set()

        thread = threading.Thread(target=contender, daemon=True)
        thread.start()

        original_record_failure = kb._record_task_failure

        def widen_pre_fix_transaction_gap(*args, **kwargs):
            if not args[0].in_transaction:
                assert contender_done.wait(timeout=5), (
                    "contending claim did not finish in the transaction gap"
                )
            return original_record_failure(*args, **kwargs)

        monkeypatch.setattr(
            kb, "_record_task_failure", widen_pre_fix_transaction_gap,
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

        def trace(sql):
            if "UPDATE tasks SET status = 'review'" in sql:
                contender_started.set()

        conn.set_trace_callback(trace)
        kb.detect_crashed_workers(conn)
        conn.set_trace_callback(None)
        contender_started.set()
        thread.join(timeout=5)

        assert not thread.is_alive(), "contending dispatcher thread leaked"
        assert not contender_errors, contender_errors
        assert contender_result.get("claimed") is False
        _assert_blocked_and_fully_released(conn, task_id)
    finally:
        conn.set_trace_callback(None)
        contender_started.set()
        conn.close()


def test_timeout_accounting_is_atomic_with_requeue(kanban_home, monkeypatch):
    """A contender cannot claim between timeout requeue and give-up."""
    conn = kb.connect()
    contender_started = threading.Event()
    contender_done = threading.Event()
    contender_result = {}
    contender_errors = []
    try:
        task_id = kb.create_task(
            conn,
            title="atomic-timeout-accounting",
            assignee="worker",
            max_runtime_seconds=1,
            max_retries=1,
        )
        host = kb._claimer_id().split(":", 1)[0]
        assert kb.claim_task(conn, task_id, claimer=f"{host}:old") is not None
        kb._set_worker_pid(conn, task_id, 994300)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (int(time.time()) - 30, task_id),
            )

        def contender():
            other = kb.connect()
            try:
                assert contender_started.wait(timeout=5)
                claimed = kb.claim_task(
                    other, task_id, claimer=f"{host}:contender",
                )
                contender_result["claimed"] = claimed is not None
                if claimed is not None:
                    kb._set_worker_pid(other, task_id, 994301)
            except BaseException as exc:
                contender_errors.append(exc)
            finally:
                other.close()
                contender_done.set()

        thread = threading.Thread(target=contender, daemon=True)
        thread.start()

        original_record_failure = kb._record_task_failure

        def widen_pre_fix_transaction_gap(*args, **kwargs):
            if not args[0].in_transaction:
                assert contender_done.wait(timeout=5), (
                    "contending claim did not finish in the timeout transaction gap"
                )
            return original_record_failure(*args, **kwargs)

        monkeypatch.setattr(
            kb, "_record_task_failure", widen_pre_fix_transaction_gap,
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

        def trace(sql):
            if "UPDATE tasks SET status = 'ready'" in sql:
                contender_started.set()

        conn.set_trace_callback(trace)
        timed_out = kb.enforce_max_runtime(
            conn, signal_fn=lambda _pid, _sig: None,
        )
        conn.set_trace_callback(None)
        contender_started.set()
        thread.join(timeout=5)

        assert task_id in timed_out
        assert not thread.is_alive(), "contending dispatcher thread leaked"
        assert not contender_errors, contender_errors
        assert contender_result.get("claimed") is False
        _assert_blocked_and_fully_released(conn, task_id)
    finally:
        conn.set_trace_callback(None)
        contender_started.set()
        conn.close()


def _assert_blocked_and_fully_released(conn, task_id):
    row = conn.execute(
        "SELECT status, claim_lock, claim_expires, worker_pid, current_run_id "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    assert row["status"] == "blocked"
    assert row["claim_lock"] is None
    assert row["claim_expires"] is None
    assert row["worker_pid"] is None
    assert row["current_run_id"] is None
    open_runs = conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND ended_at IS NULL",
        (task_id,),
    ).fetchone()[0]
    assert open_runs == 0
    gave_up = [
        event
        for event in kb.list_events(conn, task_id)
        if event.kind == "gave_up"
    ]
    assert len(gave_up) == 1


def test_explicit_retry_payload_omits_capped_violation_streak(kanban_home):
    """Unified retry events must not report a truncated violation streak."""
    with kb.connect() as conn:
        limit = kb._PROTOCOL_VIOLATION_SCAN_LIMIT + 1
        task_id = kb.create_task(
            conn,
            title="large-explicit-cap",
            assignee="worker",
            max_retries=limit,
        )
        for index in range(limit):
            _drive_protocol_violation(conn, task_id, 994200 + index)

        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.consecutive_failures == limit
        gave_up = [
            event
            for event in kb.list_events(conn, task_id)
            if event.kind == "gave_up"
        ]
        assert len(gave_up) == 1
        payload = gave_up[0].payload or {}
        assert payload.get("failures") == limit
        assert payload.get("effective_limit") == limit
        assert payload.get("limit_source") == "task"
        assert payload.get("protocol_violation") is True
        assert "protocol_violations" not in payload
