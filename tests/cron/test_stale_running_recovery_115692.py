"""A live-but-deadlocked worker passes ``_owner_is_live()`` and never gets
reclaimed. The wall-clock guard catches that: a claimed/running execution
older than ``STALE_RUNNING_CLAIM_TIMEOUT_SECONDS`` is marked unknown even
though the owner PID exists (#115692)."""
import os
import sqlite3
import time
import uuid

import pytest

from cron.executions import (
    create_execution,
    recover_interrupted_executions,
    _transaction,
)


def _seed_running(job_id, claimed_at, pid=99999, proc="ext-proc"):
    with _transaction() as conn:
        eid = uuid.uuid4().hex
        conn.execute(
            """INSERT INTO executions
               (id, job_id, status, source, process_id, pid, process_started_at, claimed_at)
               VALUES (?, ?, 'running', 'cron', ?, ?, ?, ?)""",
            (eid, job_id, proc, pid, int(time.time()), claimed_at),
        )
        return eid


def test_live_owner_stale_claim_gets_recovered(tmp_path, monkeypatch):
    """A claimed/running row whose owner is alive but stale past the wall-clock
    guard must be marked unknown — a live-but-deadlocked worker otherwise
    blocks every future fire forever (#115692)."""
    monkeypatch.setattr("cron.executions._owner_is_live", lambda pid, sat: True)

    # claimed_at = 3h ago (well past the 2h stale-claim timeout)
    old = "2026-09-17T00:00:00"
    eid = _seed_running("job-stale", claimed_at=old, pid=99999)

    assert recover_interrupted_executions() >= 1

    with _transaction() as conn:
        row = conn.execute(
            "SELECT status FROM executions WHERE id=?", (eid,)
        ).fetchone()
        assert row["status"] == "unknown"


def test_live_owner_recent_claim_not_recovered(tmp_path, monkeypatch):
    """A live owner with a recent claim must NOT be recovered (normal running)."""
    monkeypatch.setattr("cron.executions._owner_is_live", lambda pid, sat: True)

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    eid = _seed_running("job-recent", claimed_at=now, pid=88888)

    assert recover_interrupted_executions() == 0

    with _transaction() as conn:
        row = conn.execute(
            "SELECT status FROM executions WHERE id=?", (eid,)
        ).fetchone()
        assert row["status"] == "running"
