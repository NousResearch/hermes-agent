"""Liveness fingerprints must tolerate cross-process probe drift (2026-09-19 RCA).

Observed live: one process read its own start time exactly +1.00s later than every
other probe of the same pid; ``_owner_is_live`` compared bit-exact and the reclaim
path marked live mid-flight executions ``unknown`` as "owner process died".
Guard doctrine (executions.py): inability to prove death must not rewrite state.
"""
import os
import sqlite3

import pytest

from cron import executions as X


def _probe(monkeypatch, current):
    monkeypatch.setattr(X, "_process_start_time", lambda pid: current)
    monkeypatch.setattr(
        "gateway.status._pid_exists", lambda pid: True, raising=True
    )
    return monkeypatch


def test_exact_match_is_live(tmp_path, monkeypatch):
    _probe(monkeypatch, 178978036015)
    assert X._owner_is_live(93220, 178978036015) is True


def test_observed_one_second_drift_is_live(tmp_path, monkeypatch):
    # The 2026-09-19 incident: recorded 178978036115, every other probe 178978036015.
    _probe(monkeypatch, 178978036015)
    assert X._owner_is_live(93220, 178978036115) is True


def test_drift_beyond_tolerance_is_dead_pid_reuse_guard(tmp_path, monkeypatch):
    # PID reuse shifts the start time by minutes-to-days; 3s+ drift is death/reuse.
    _probe(monkeypatch, 178978036015)
    assert X._owner_is_live(93220, 178978036015 - 301) is False


def test_unprobeable_existing_pid_is_not_proven_dead(tmp_path, monkeypatch):
    # Polarity vs the pre-fix code: None used to mean "dead" and reaped live owners.
    _probe(monkeypatch, None)
    assert X._owner_is_live(93220, 178978036115) is True


def test_missing_pid_is_dead(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False, raising=True)
    assert X._owner_is_live(93220, 178978036115) is False


def test_no_fingerprint_live_only_for_own_pid(tmp_path, monkeypatch):
    _probe(monkeypatch, 12345)
    assert X._owner_is_live(os.getpid(), None) is True
    assert X._owner_is_live(93220, None) is False


def test_recover_skips_drifted_live_owner(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(X, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    X.create_execution("j1", source="cron", scheduled_instant="2026-09-19T13:00:00+02:00")
    recorded = (X._process_start_time(os.getpid()) or 0) + 100  # the observed +1.00s drift
    with X._transaction() as conn:
        conn.execute(
            "UPDATE executions SET pid=?, process_started_at=? WHERE job_id='j1'",
            (os.getpid(), recorded),
        )
    monkeypatch.setattr(
        X, "_PROCESS_ID", "a-different-process-so-recovery-cannot-skip-own-rows"
    )
    assert X.recover_interrupted_executions() == 0
    with X._transaction() as conn:
        assert conn.execute("SELECT status FROM executions").fetchone()[0] == "claimed"


def test_recover_reaps_genuinely_dead_owner(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(X, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    X.create_execution("j1", source="cron", scheduled_instant="2026-09-19T13:00:00+02:00")
    with X._transaction() as conn:
        # A pid that does not exist: _pid_exists False → provably dead.
        conn.execute(
            "UPDATE executions SET pid=9999999, process_started_at=178978036015 WHERE job_id='j1'"
        )
    monkeypatch.setattr(
        X, "_PROCESS_ID", "a-different-process-so-recovery-cannot-skip-own-rows"
    )
    assert X.recover_interrupted_executions() == 1
    with X._transaction() as conn:
        assert conn.execute("SELECT status FROM executions").fetchone()[0] == "unknown"
