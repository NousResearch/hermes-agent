"""Tests: an expired claim left on a ``ready`` row is recovered by the sweep.

``claim_task`` only claims rows whose ``claim_lock IS NULL``, so a ``ready``
row that still carries an expired claim (worker gone) can never be claimed
again. ``release_stale_claims`` used to scan ``status = 'running'`` only, so
nothing ever released that claim and the task sat in the Ready lane forever.
"""

from __future__ import annotations

import subprocess
import time
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
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _leave_claim_on_ready_row(conn, task_id: str, *, expires: int, heartbeat: int | None = None) -> None:
    conn.execute(
        "UPDATE tasks SET status = 'ready', claim_expires = ?, last_heartbeat_at = ? WHERE id = ?",
        (expires, heartbeat, task_id),
    )
    conn.commit()


def test_expired_claim_on_ready_row_is_released_and_claimable(conn):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="ready row with stale claim", assignee="w")
    assert kb.claim_task(conn, tid, claimer=f"{host}:old") is not None
    dead = subprocess.Popen(["true"])
    dead.wait()
    kbd._set_worker_pid(conn, tid, dead.pid)
    _leave_claim_on_ready_row(conn, tid, expires=int(time.time()) - 10)
    assert kb.claim_task(conn, tid, claimer=f"{host}:new") is None  # wedged

    assert kb.release_stale_claims(conn) == 1

    row = conn.execute(
        "SELECT status, claim_lock, claim_expires, worker_pid FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    assert tuple(row) == ("ready", None, None, None)
    assert kb.claim_task(conn, tid, claimer=f"{host}:new") is not None


def test_expired_claim_on_ready_row_with_live_worker_is_kept(conn):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="ready row with live worker", assignee="w")
    assert kb.claim_task(conn, tid, claimer=f"{host}:live") is not None
    worker = subprocess.Popen(["sleep", "30"])
    try:
        kbd._set_worker_pid(conn, tid, worker.pid)
        old_expiry = int(time.time()) - 10
        _leave_claim_on_ready_row(conn, tid, expires=old_expiry, heartbeat=int(time.time()))

        assert kb.release_stale_claims(conn) == 0

        row = conn.execute(
            "SELECT claim_lock, claim_expires, worker_pid FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert row["claim_lock"] == f"{host}:live"
        assert row["claim_expires"] > old_expiry
        assert row["worker_pid"] == worker.pid
    finally:
        worker.terminate()
        worker.wait()
