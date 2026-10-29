"""Behaviour pins for Kanban terminal-write fencing (hermes_cli/kanban_db.py).

PID-fingerprint reclaim is already pinned in tests/hermes_cli/test_kanban_worker_pid_fingerprint.py;
block/heartbeat stale-run fencing in test_kanban_core_functionality.py. This pins the remaining
terminal write: ``complete_task`` with a superseded ``expected_run_id``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as c:
        yield c


def test_complete_with_stale_expected_run_id_is_rejected(conn, monkeypatch):
    """A worker whose run was superseded by a reclaim cannot complete the card; the current run still can."""
    tid = kb.create_task(conn, title="fenced", assignee="worker")
    kb.claim_task(conn, tid)
    run1 = kb.latest_run(conn, tid).id
    kbd._set_worker_pid(conn, tid, 98765)
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    assert kbd.detect_crashed_workers(conn) == [tid]

    kb.claim_task(conn, tid)
    run2 = kb.latest_run(conn, tid).id
    assert run2 != run1

    assert kb.complete_task(conn, tid, result="late", expected_run_id=run1) is False
    task = kb.get_task(conn, tid)
    assert (task.status, task.current_run_id, task.result) == ("running", run2, None)
    assert conn.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run2,)).fetchone()["ended_at"] is None

    assert kb.complete_task(conn, tid, result="current", expected_run_id=run2) is True
    assert kb.get_task(conn, tid).status == "done"
    # A repeat terminal write is a CAS miss (False), not a replay returning the receipt.
    assert kb.complete_task(conn, tid, result="current", expected_run_id=run2) is False
