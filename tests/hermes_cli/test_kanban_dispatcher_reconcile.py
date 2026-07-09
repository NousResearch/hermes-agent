from __future__ import annotations

import json
import os
import signal
import time

import pytest


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _host_lock(kb) -> str:
    return f"{kb._claimer_id().split(':', 1)[0]}:phase7"


def _terminal_task_with_lingering_worker(
    conn,
    kb,
    *,
    status: str,
    pid: int,
    worker_session_id: str = "worker-session",
):
    tid = kb.create_task(conn, title=f"{status} lingering", assignee="worker")
    kb.claim_task(conn, tid)
    if status == "done":
        assert kb.complete_task(
            conn,
            tid,
            summary="done",
            metadata={"worker_session_id": worker_session_id},
        )
        run = kb.latest_run(conn, tid)
    else:
        assert kb.block_task(
            conn,
            tid,
            reason="blocked",
            kind="needs_input",
            metadata={"worker_session_id": worker_session_id},
        )
        run = kb.latest_run(conn, tid)
    lock = _host_lock(kb)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, claim_lock = ?, claim_expires = ? "
            "WHERE id = ?",
            (pid, lock, int(time.time()) + 3600, tid),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid = ?, claim_lock = ?, metadata = ? "
            "WHERE id = ?",
            (
                pid,
                lock,
                json.dumps({"worker_session_id": worker_session_id}),
                run.id,
            ),
        )
    return tid


def test_dispatcher_reconciles_done_task_with_live_worker(
    kanban_home,
    monkeypatch,
):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    calls = []
    alive = {"value": True}

    def signal_fn(pid, sig):
        calls.append((pid, sig))
        alive["value"] = False

    class FakeSessionDB:
        def __init__(self):
            self.ended = []

        def end_session(self, session_id, reason):
            self.ended.append((session_id, reason))

    try:
        tid = _terminal_task_with_lingering_worker(
            conn,
            kb,
            status="done",
            pid=12345,
            worker_session_id="worker-session-done",
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: alive["value"])
        session_db = FakeSessionDB()

        result = kb.dispatch_once(
            conn,
            signal_fn=signal_fn,
            session_db=session_db,
        )

        assert result.terminal_reconciled == [tid]
        assert calls == [(12345, signal.SIGTERM)]
        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert task.worker_pid is None
        assert task.claim_lock is None
        events = kb.list_events(conn, tid)
        assert events[-1].kind == "terminal_worker_reconciled"
        assert session_db.ended == [
            ("worker-session-done", "kanban_terminal_worker_reconciled")
        ]
    finally:
        conn.close()


def test_reconcile_blocked_task_with_live_worker(kanban_home, monkeypatch):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    calls = []
    alive = {"value": True}

    def signal_fn(pid, sig):
        calls.append((pid, sig))
        alive["value"] = False

    try:
        tid = _terminal_task_with_lingering_worker(
            conn,
            kb,
            status="blocked",
            pid=23456,
            worker_session_id="worker-session-blocked",
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: alive["value"])

        reconciled = kb.reconcile_terminal_worker_processes(
            conn,
            signal_fn=signal_fn,
        )

        assert reconciled == [tid]
        assert calls == [(23456, signal.SIGTERM)]
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.worker_pid is None
        assert task.claim_lock is None
    finally:
        conn.close()


def test_reconcile_does_not_kill_running_worker(kanban_home, monkeypatch):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    calls = []
    try:
        tid = kb.create_task(conn, title="running", assignee="worker")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 34567)
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)

        reconciled = kb.reconcile_terminal_worker_processes(
            conn,
            signal_fn=lambda pid, sig: calls.append((pid, sig)),
        )

        assert reconciled == []
        assert calls == []
        assert kb.get_task(conn, tid).status == "running"
        assert kb.get_task(conn, tid).worker_pid == 34567
    finally:
        conn.close()


def test_reconcile_does_not_kill_non_host_local_terminal_worker(
    kanban_home,
    monkeypatch,
):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    calls = []
    try:
        tid = _terminal_task_with_lingering_worker(
            conn,
            kb,
            status="done",
            pid=45678,
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_lock = ? WHERE id = ?",
                ("other-host:phase7", tid),
            )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)

        reconciled = kb.reconcile_terminal_worker_processes(
            conn,
            signal_fn=lambda pid, sig: calls.append((pid, sig)),
        )

        assert reconciled == []
        assert calls == []
        assert kb.get_task(conn, tid).worker_pid == 45678
    finally:
        conn.close()
