"""Transient SQLite contention must not lose a handed-off occurrence (#135487)."""

import json
import sqlite3
import time
from unittest.mock import Mock

import pytest


def test_worker_retries_real_busy_before_ack_and_runs_once(tmp_path, monkeypatch):
    from cron import executions, scheduler

    db = tmp_path / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", db)
    record = executions.create_execution("busy-job", source="builtin")
    assert executions.mark_execution_handoff_pending(record["id"])
    real_connect = executions._connect

    def immediate_connection():
        conn = real_connect()
        conn.execute("PRAGMA busy_timeout=0")
        return conn

    monkeypatch.setattr(executions, "_connect", immediate_connection)
    holder = sqlite3.connect(db)
    holder.execute("BEGIN IMMEDIATE")
    payload, ack = tmp_path / "payload.json", tmp_path / "ready.json"
    payload.write_text(json.dumps({
        "job": {"id": "busy-job", "execution_id": record["id"]},
        "profile_home": str(tmp_path / "profile"),
        "adoption_deadline": time.monotonic() + 120,
    }), encoding="utf-8")
    real_adopt = executions.adopt_claimed_execution
    busy_errors = []

    def adopt(execution_id):
        try:
            return real_adopt(execution_id)
        except sqlite3.OperationalError as exc:
            busy_errors.append(exc.sqlite_errorcode)
            assert not ack.exists()
            holder.rollback()  # deterministic release only after a real failed UPDATE
            raise

    monkeypatch.setattr(executions, "adopt_claimed_execution", adopt)
    run = Mock(return_value=True)
    monkeypatch.setattr(scheduler, "run_one_job", run)
    try:
        try:
            result = scheduler._run_external_worker_payload(payload, ack)
        except sqlite3.OperationalError:
            result = False
    finally:
        holder.close()
    assert busy_errors == [sqlite3.SQLITE_BUSY]
    assert result is True, "worker abandoned occurrence on recoverable SQLite contention"
    run.assert_called_once()
    assert json.loads(ack.read_text(encoding="utf-8"))["execution_id"] == record["id"]
    assert executions.get_execution(record["id"])["status"] == "running"
    assert real_adopt(record["id"]) is None


@pytest.mark.parametrize("outcome", [None, {"status": "running"}, "busy", "locked", "io", "expired"])
def test_retry_preserves_cas_errors_and_budget(outcome, monkeypatch):
    from cron import executions, scheduler_adoption

    clock = [10.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(scheduler_adoption.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(scheduler_adoption.time, "sleep", sleep)
    busy = sqlite3.OperationalError("database is locked")
    busy.sqlite_errorcode = sqlite3.SQLITE_BUSY
    locked = sqlite3.OperationalError("database table is locked")
    locked.sqlite_errorcode = sqlite3.SQLITE_LOCKED | (1 << 8)
    io = sqlite3.OperationalError("disk I/O error")
    io.sqlite_errorcode = sqlite3.SQLITE_IOERR
    if outcome in ("busy", "locked", "io", "expired"):
        error = {"busy": busy, "locked": locked, "io": io, "expired": busy}[outcome]
        adopt = Mock(side_effect=error)
    else:
        adopt = Mock(side_effect=[busy, outcome])
    monkeypatch.setattr(executions, "adopt_claimed_execution", adopt)
    deadline = 10.0 if outcome == "expired" else 15.75
    if outcome in ("busy", "locked", "io", "expired"):
        with pytest.raises(sqlite3.OperationalError) as caught:
            scheduler_adoption.adopt_with_retry("exec", deadline)
        assert caught.value is error
        assert adopt.call_count == (3 if outcome in ("busy", "locked") else 1)
    else:
        assert scheduler_adoption.adopt_with_retry("exec", deadline) == outcome
        assert adopt.call_count == 2
    assert clock[0] <= deadline
