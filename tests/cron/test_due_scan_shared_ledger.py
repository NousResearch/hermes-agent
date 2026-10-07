"""The due scan must reuse one executions.db connection instead of opening one per job.

Every ``completed_occurrence()`` call in the due scan wraps its single indexed SELECT in its own
``_transaction()``, which opens AND closes ``executions.db``. Nothing else holds the ledger between
ticks, so every close is the last-connection case that checkpoints and deletes the ``-wal``/``-shm``
pair, and the next open recreates them and re-runs the pragmas and schema init. With 342 enabled
jobs that is 342 open/close cycles per minute and two unlinks per job per tick on the gateway's
largest single writer (#133883).

Regression for #133883: the whole scan opens the ledger once, behaviour outside the scan is
unchanged, the shared connection sees rows committed mid-scan, and it is closed when the scan ends.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import cron.executions as executions
import cron.occurrences as occurrences
from cron import store_health

FIXED_NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    with executions._transaction() as conn:  # create schema
        conn.execute("SELECT 1")
    return tmp_path / "executions.db"


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Redirect cron storage to a temp dir, pin the clock, start with no degraded store."""
    monkeypatch.setattr(store_health, "_degraded", {})
    monkeypatch.setattr(store_health, "_listener", None)
    monkeypatch.setattr(store_health, "_recovered", {})
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: FIXED_NOW)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return tmp_path


def _complete(job_id, instant):
    """Record one completed occurrence from a separate connection (as another process would)."""
    with executions._transaction() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(executions)")}
        row = {"id": f"x-{job_id}", "job_id": job_id, "source": "test", "process_id": "p",
               "pid": 1, "status": "completed", "scheduled_instant": instant,
               "claimed_at": instant, "finished_at": instant}
        row = {k: v for k, v in row.items() if k in cols}
        conn.execute(f"INSERT INTO executions ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                     tuple(row.values()))


def _count_connects(monkeypatch):
    calls = []
    real = executions._connect

    def counting():
        calls.append(1)
        return real()

    monkeypatch.setattr(executions, "_connect", counting)
    return calls


def test_shared_connection_opens_ledger_once(ledger, monkeypatch):
    instant = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc).isoformat()
    _complete("job-3", instant)
    calls = _count_connects(monkeypatch)
    with occurrences.shared_ledger_connection():
        results = [occurrences.completed_occurrence({"id": f"job-{i}"}, instant) for i in range(10)]
    assert results == [i == 3 for i in range(10)]
    assert len(calls) == 1


def test_without_scope_behaviour_is_unchanged(ledger, monkeypatch):
    instant = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc).isoformat()
    _complete("job-1", instant)
    calls = _count_connects(monkeypatch)
    assert occurrences.completed_occurrence({"id": "job-1"}, instant) is True
    assert occurrences.completed_occurrence({"id": "job-2"}, instant) is False
    assert len(calls) == 2


def test_shared_connection_sees_rows_committed_during_scan(ledger):
    instant = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    with occurrences.shared_ledger_connection():
        assert occurrences.completed_occurrence({"id": "job-1"}, instant) is False
        _complete("job-1", instant)  # another connection commits mid-scan
        assert occurrences.completed_occurrence({"id": "job-1"}, instant) is True


def test_shared_connection_is_closed_after_scope(ledger, monkeypatch):
    instant = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc).isoformat()
    with occurrences.shared_ledger_connection():
        occurrences.completed_occurrence({"id": "job-1"}, instant)
        conn = occurrences._scan_ledger.conn
    assert occurrences._scan_ledger.conn is None
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def _future_job(jid):
    """An enabled recurring job whose next run is an hour out — scanned, never fired."""
    return {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "cron", "expr": "* * * * *", "display": "every minute"},
        "next_run_at": (FIXED_NOW + timedelta(hours=1)).isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "active",
        "repeat": None,
        "deliver": "local",
    }


def test_due_scan_reuses_one_ledger_connection(cron_store, monkeypatch):
    """The whole scan opens the ledger once, not once per enabled job (#133883)."""
    from cron.jobs import get_due_jobs, save_jobs

    save_jobs([_future_job(f"job-{i}") for i in range(12)])
    calls = _count_connects(monkeypatch)
    assert get_due_jobs() == []
    assert len(calls) == 1
