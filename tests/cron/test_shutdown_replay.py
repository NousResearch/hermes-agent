"""Bounded shutdown-replay tests.

The real cron.replay.replay_sweep() runs against a temp HERMES_HOME with the
executions ledger, jobs store and tombstone DB all real; only process-liveness
probes are stubbed (no gateway process exists in tests).
"""
import sqlite3

import pytest

from hermes_constants import get_hermes_home


@pytest.fixture
def replay_env(tmp_path, monkeypatch):
    """Temp HERMES_HOME with real executions ledger + jobs store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Reset module state so the throttle does not swallow sweeps between tests.
    import cron.replay as replay_mod

    monkeypatch.setattr(replay_mod, "_last_sweep_at", None)
    # Point the executions ledger at the temp home (production resolves at
    # transaction time, so the env var above is enough for new connections).
    monkeypatch.setattr(replay_mod, "REPLAY_SWEEP_INTERVAL_SECONDS", 0.0)
    return tmp_path


def _seed_job(tmp_path, job_id="testjob111111", name="replay-me", enabled=True):
    from cron.jobs import save_jobs, load_jobs

    jobs = load_jobs()
    jobs = [j for j in jobs if j.get("id") != job_id]
    jobs.append({
        "id": job_id,
        "name": name,
        "prompt": "do the thing",
        "schedule": {"kind": "cron", "expr": "0 6 * * *"},
        "enabled": enabled,
        "state": "scheduled",
    })
    save_jobs(jobs)


def _seed_interrupted_execution(job_id, error, *, status="failed", scheduled_instant=None):
    from cron.executions import (
        create_execution, finish_execution, _PROCESS_ID, _transaction)

    row = create_execution(job_id, source="test", scheduled_instant=scheduled_instant)
    # Re-fence the row to THIS process's ledger identity so the owner-fenced
    # finish_execution terminalizes it (the real flow has the dead owner's
    # identity — replay_sweep never re-probes liveness).
    with _transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id=? WHERE id=?",
            (_PROCESS_ID, row["id"]),
        )
    finish_execution(row["id"], success=(status == "completed"), error=error)
    return row["id"]


def test_replays_fresh_shutdown_interruption_once(replay_env, monkeypatch):
    job_id = "testjob222222"
    _seed_job(replay_env, job_id)
    exec_id = _seed_interrupted_execution(
        job_id, "Interrupted by gateway shutdown before terminal completion.")

    import cron.replay as replay_mod

    fired = []
    monkeypatch.setattr("cron.jobs.trigger_job", lambda jid, **kw: fired.append(jid) or {})

    result = replay_mod.replay_sweep()
    assert job_id in result
    assert fired == [job_id]

    # Second sweep: tombstoned — never replays again.
    result2 = replay_mod.replay_sweep()
    assert result2 == []
    assert fired == [job_id]


def test_does_not_replay_non_shutdown_failures(replay_env, monkeypatch):
    job_id = "testjob333333"
    _seed_job(replay_env, job_id)
    _seed_interrupted_execution(job_id, "Script exited with code 1")

    import cron.replay as replay_mod

    monkeypatch.setattr("cron.jobs.trigger_job", lambda jid, **kw: {})
    assert replay_mod.replay_sweep() == []


def test_does_not_replay_when_newer_execution_exists(replay_env, monkeypatch):
    job_id = "testjob444444"
    _seed_job(replay_env, job_id)
    _seed_interrupted_execution(
        job_id, "Interrupted by gateway shutdown before terminal completion.")
    # A newer failure (any terminal row) supersedes the interruption.
    _seed_interrupted_execution(job_id, "Script exited with code 1")

    import cron.replay as replay_mod

    monkeypatch.setattr("cron.jobs.trigger_job", lambda jid, **kw: {})
    assert replay_mod.replay_sweep() == []


def test_does_not_replay_disabled_jobs(replay_env, monkeypatch):
    job_id = "testjob555555"
    _seed_job(replay_env, job_id, enabled=False)
    _seed_interrupted_execution(
        job_id, "Interrupted by gateway shutdown before terminal completion.")

    import cron.replay as replay_mod

    monkeypatch.setattr("cron.jobs.trigger_job", lambda jid, **kw: {})
    assert replay_mod.replay_sweep() == []


def test_does_not_replay_when_scheduled_instant_already_completed(replay_env, monkeypatch):
    job_id = "testjob666666"
    _seed_job(replay_env, job_id)
    instant = "2026-09-11T06:00:00+01:00"
    _seed_interrupted_execution(
        job_id, "Interrupted by gateway shutdown before terminal completion.",
        scheduled_instant=instant)
    # Same instant already completed → the fire's side effects exist.
    _seed_interrupted_execution(job_id, None, status="completed", scheduled_instant=instant)

    import cron.replay as replay_mod

    monkeypatch.setattr("cron.jobs.trigger_job", lambda jid, **kw: {})
    # The newer completed row supersedes the interruption via the
    # latest-execution guard: never replayed.
    assert replay_mod.replay_sweep() == []


def test_stale_interruptions_are_not_replayed(replay_env, monkeypatch):
    job_id = "testjob777777"
    _seed_job(replay_env, job_id)
    exec_id = _seed_interrupted_execution(
        job_id, "Interrupted by gateway shutdown before terminal completion.")
    # Backdate the interruption beyond the freshness window.
    from cron.executions import _transaction

    with _transaction() as conn:
        conn.execute(
            "UPDATE executions SET finished_at=? WHERE id=?",
            ("2026-01-01T06:00:00+01:00", exec_id),
        )

    import cron.replay as replay_mod

    monkeypatch.setattr("cron.jobs.trigger_job", lambda jid, **kw: {})
    assert replay_mod.replay_sweep() == []
