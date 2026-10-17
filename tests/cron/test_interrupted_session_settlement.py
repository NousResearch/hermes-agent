"""Regression for #138304: a dead cron owner must not strand its session transcript open."""

from __future__ import annotations

import time

from cron import executions, scheduler
from cron.session_settlement import close_interrupted_run_sessions
from hermes_state import SessionDB


def _seed_cron_session(db, job_id: str, *, age_seconds: float) -> str:
    session_id = f"cron_{job_id}_20261014_090000"
    db.create_session(session_id, source="cron")
    stale = time.time() - age_seconds
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?",
            (stale, stale, session_id),
        )
        db._conn.commit()
    return session_id


def _unknown_execution(monkeypatch, tmp_path, job_id: str) -> None:
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    monkeypatch.setattr(executions, "_PROCESS_ID", f"dead-owner-{job_id}")
    executions.create_execution(job_id, source="cron")
    monkeypatch.setattr(executions, "_PROCESS_ID", "recovery-owner")
    monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: False)
    assert executions.recover_interrupted_executions() == 1


def test_dead_owner_unknown_execution_closes_only_quiet_cron_session(monkeypatch, tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        old_job, fresh_job = "dead-owner", "fresh-owner"
        old_session = _seed_cron_session(db, old_job, age_seconds=301)
        fresh_session = _seed_cron_session(db, fresh_job, age_seconds=30)
        _unknown_execution(monkeypatch, tmp_path, old_job)
        _unknown_execution(monkeypatch, tmp_path, fresh_job)

        assert close_interrupted_run_sessions(db) == [old_session]
        assert db.get_session(old_session)["end_reason"] == "cron_incomplete_no_output"
        assert db.get_session(fresh_session)["ended_at"] is None
    finally:
        db.close()


def test_live_execution_keeps_its_quiet_cron_session_open(monkeypatch, tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
        job_id = "live-owner"
        session_id = _seed_cron_session(db, job_id, age_seconds=301)
        record = executions.create_execution(job_id, source="cron")
        executions.mark_execution_running(record["id"])

        assert close_interrupted_run_sessions(db) == []
        assert db.get_session(session_id)["ended_at"] is None
    finally:
        db.close()


def test_dead_owner_reap_runs_session_settlement(monkeypatch, tmp_path):
    """The scheduler invokes settlement after every bounded dead-owner ledger sweep."""
    from cron import session_settlement

    calls = []
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(scheduler, "_last_dead_owner_reap_at", {})
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)
    monkeypatch.setattr(
        session_settlement,
        "close_interrupted_run_sessions",
        lambda: calls.append("settled") or [],
    )

    scheduler._maybe_reap_dead_owners()

    assert calls == ["settled"]
