"""Per-job cron run-history pruning (``SessionDB.prune_cron_job_runs``).

Split out of ``tests/hermes_state/test_hermes_state.py`` (over its file ratchet):
the pruning contract — newest-N kept per job, keep=0 clearing, no-op avoiding a
write transaction, and the timestamp-shape scoping that keeps one job's prune
from eating an underscore-extension sibling's runs (#92133).
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    session_db = SessionDB(db_path=tmp_path / "test_state.db")
    yield session_db
    session_db.close()


def _seed_ts_run(db, job_id: str, run_stamp: str) -> str:
    sid = f"cron_{job_id}_{run_stamp}"
    db.create_session(session_id=sid, source="cron")
    db.end_session(sid, "completed")
    return sid


def test_prune_cron_job_runs_keeps_newest_per_job(db):
    stamps = [f"20260818_{h:02d}0000" for h in range(10, 18)]
    for stamp in stamps:
        _seed_ts_run(db, "alpha", stamp)
    _seed_ts_run(db, "beta", "20260818_090000")
    # A non-cron session must never be touched.
    db.create_session(session_id="user-session", source="desktop")

    deleted = db.prune_cron_job_runs("alpha", keep=5)

    assert deleted == 3
    remaining = {r["id"] for r in db.list_cron_job_runs("alpha", limit=20)}
    assert remaining == {f"cron_alpha_{s}" for s in stamps[3:]}
    # Other jobs and non-cron sessions untouched.
    assert len(db.list_cron_job_runs("beta", limit=20)) == 1
    assert db.get_session("user-session") is not None


def test_prune_cron_job_runs_keep_zero_clears_job(db):
    for h in range(12, 16):
        _seed_ts_run(db, "alpha", f"20260818_{h:02d}0000")

    assert db.prune_cron_job_runs("alpha", keep=0) == 4
    assert db.list_cron_job_runs("alpha", limit=20) == []


def test_prune_cron_job_runs_skips_write_when_within_retention(db):
    _seed_ts_run(db, "alpha", "20260818_120000")
    writes_before = db._write_count

    assert db.prune_cron_job_runs("alpha", keep=5) == 0
    assert db._write_count == writes_before


def test_prune_skips_underscore_extension_jobs(db):
    """#92133: job id `backup` must not prune `backup_weekly`'s runs —
    the bare prefix range leaks underscore-extensions; the timestamp-
    shaped remainder predicate scopes the prune to real run rows."""
    _seed_ts_run(db, "backup", "20260818_100000")
    _seed_ts_run(db, "backup_weekly", "20260818_110000")

    deleted = db.prune_cron_job_runs("backup_weekly", keep=1)

    assert deleted == 0
    assert len(db.list_cron_job_runs("backup_weekly", limit=20)) == 1
