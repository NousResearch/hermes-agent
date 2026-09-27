"""Regression: block_task/schedule_task on a RUNNING task must terminate the
live host-local worker after the transition commits.

Every other running→X reclaim path (reclaim_task, archive_task, reopen
invalidation) snapshots pid+claim inside the txn and signals the worker
post-commit (#76196: clearing ``worker_pid`` alone left the OS process running
and pushing work against an untracked card). ``block_task`` and
``schedule_task`` cleared the pid without signalling — same bug class.
"""

from __future__ import annotations

import json
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as c:
        yield c


def _make_running(conn, *, run_row=True):
    tid = kb.create_task(conn, title="live worker", assignee="p1")
    lock = kb._host_prefix() + ":claim-test"
    future = int(time.time()) + 3600
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, worker_pid=? WHERE id=?",
        (lock, future, 12345, tid),
    )
    run_id = None
    if run_row:
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, worker_pid, started_at)"
            " VALUES (?, 'running', ?, ?, ?, ?)",
            (tid, lock, future, 12345, int(time.time())),
        )
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, tid))
    conn.commit()
    return tid, run_id


def _signals(fn_calls):
    return [sig for _pid, sig in fn_calls]


def test_block_task_terminates_running_worker(conn):
    tid, _ = _make_running(conn)
    calls: list[tuple[int, int]] = []
    assert kb.block_task(conn, tid, reason="stop it", signal_fn=lambda pid, sig: calls.append((pid, sig)))
    assert 12345 in [pid for pid, _ in calls]
    evs = [
        json.loads(r["payload"])
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='block_worker_termination'", (tid,)
        )
    ]
    assert evs and evs[0]["prev_pid"] == 12345


def test_block_task_worker_self_call_does_not_signal(conn):
    tid, run_id = _make_running(conn)
    calls: list[tuple[int, int]] = []
    assert kb.block_task(
        conn, tid, reason="dependency handoff", kind="needs_input",
        expected_run_id=run_id, signal_fn=lambda pid, sig: calls.append((pid, sig)),
    )
    assert calls == []


def test_schedule_task_terminates_running_worker(conn):
    tid, _ = _make_running(conn)
    calls: list[tuple[int, int]] = []
    assert kb.schedule_task(conn, tid, reason="park", signal_fn=lambda pid, sig: calls.append((pid, sig)))
    assert 12345 in [pid for pid, _ in calls]
    evs = [
        json.loads(r["payload"])
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='schedule_worker_termination'", (tid,)
        )
    ]
    assert evs and evs[0]["prev_pid"] == 12345
