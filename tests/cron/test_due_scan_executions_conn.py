"""Regression tests for Issue #133883: the due scan pays one open/close of
executions.db per enabled job per tick because completed_occurrence() runs its
own _transaction() per call, and every close is the last-connection case that
checkpoints and drops the WAL pair. The scan must share one connection across
the whole pass; the fire-claim path keeps its standalone transaction."""

import sqlite3
from unittest.mock import patch

import cron.jobs as jobs_mod
import cron.occurrences as occ_mod
from cron.occurrences import completed_occurrence


def _job(job_id="j1", kind="interval", minutes=60):
    return {
        "id": job_id,
        "name": f"job-{job_id}",
        "enabled": True,
        "schedule": {"kind": kind, "every_minutes": minutes},
        "next_run_at": "2026-01-01T00:00:00+00:00",
    }


def _seed_jobs(monkeypatch, count):
    monkeypatch.setattr(jobs_mod, "load_jobs", lambda: [_job(f"j{i}") for i in range(count)])
    monkeypatch.setattr(jobs_mod, "save_jobs", lambda *a, **k: None)


def test_due_scan_shares_one_executions_connection(monkeypatch):
    _seed_jobs(monkeypatch, 5)
    opens = []
    real_connect = sqlite3.connect

    def counting_connect(*a, **k):
        opens.append(1)
        return real_connect(":memory:")

    seen_conns = []
    real_occ = occ_mod.completed_occurrence

    def spy_occ(job, instant, *, conn=None):
        seen_conns.append(conn is not None)
        return real_occ(job, instant, conn=conn)

    monkeypatch.setattr("cron.executions._connect", counting_connect)
    monkeypatch.setattr(occ_mod, "completed_occurrence", spy_occ)
    monkeypatch.setattr(jobs_mod, "completed_occurrence", spy_occ, raising=False)

    jobs_mod.get_due_jobs()

    assert len(opens) == 1, f"expected 1 shared open, got {len(opens)}"
    assert seen_conns and all(seen_conns)


def test_due_scan_empty_store_opens_no_connection(monkeypatch):
    _seed_jobs(monkeypatch, 0)

    def boom():
        raise AssertionError("must not open executions.db for an empty scan")

    monkeypatch.setattr("cron.executions._connect", boom)
    assert jobs_mod.get_due_jobs() == []


def test_completed_occurrence_with_conn_skips_transaction(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE executions (id TEXT, job_id TEXT, finished_at TEXT, "
        "claimed_at TEXT, scheduled_instant TEXT, status TEXT)")

    def boom():
        raise AssertionError("shared-conn path must not open its own transaction")

    monkeypatch.setattr("cron.executions._transaction", boom)
    # A fresh connection has no matching row: not completed, but the read succeeded.
    assert completed_occurrence(_job(), "2026-01-01T00:00:00+00:00", conn=conn) is False


def test_connect_failure_falls_back_to_per_call_path(monkeypatch):
    _seed_jobs(monkeypatch, 2)

    def broken():
        raise sqlite3.OperationalError("cannot open")

    monkeypatch.setattr("cron.executions._connect", broken)
    with patch.object(occ_mod, "completed_occurrence", wraps=occ_mod.completed_occurrence):
        due = jobs_mod.get_due_jobs()
    assert isinstance(due, list)


def test_completed_occurrence_standalone_still_works():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE executions (id TEXT, job_id TEXT, finished_at TEXT, "
        "claimed_at TEXT, scheduled_instant TEXT, status TEXT)")
    conn.execute(
        "INSERT INTO executions VALUES ('e1', 'j1', '2026-01-01T00:05:00+00:00', "
        "'2026-01-01T00:00:10+00:00', '2026-01-01T00:00:00+00:00', 'completed')")
    conn.commit()
    with patch("cron.executions._transaction") as fake_txn:
        fake_txn.return_value.__enter__ = lambda s: conn
        fake_txn.return_value.__exit__ = lambda s, *a: False
        assert completed_occurrence(_job(), "2026-01-01T00:00:00+00:00") is True
