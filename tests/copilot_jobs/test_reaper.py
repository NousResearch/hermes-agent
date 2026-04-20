"""Tests for copilot_jobs.reaper — dead-process detection and TTL enforcement."""

import time

import pytest

from hermes_state import SessionDB
from copilot_jobs.reaper import (
    _is_pid_alive,
    reap_dead_processes,
    reap_expired_idle,
    reap_stale_pending,
    reap,
)


@pytest.fixture()
def db(tmp_path):
    return SessionDB(db_path=tmp_path / "test.db")


def _make_job(db, job_id, **overrides):
    """Helper to create a copilot job with defaults."""
    kwargs = dict(
        job_id=job_id,
        repo_slug="test-repo",
        repo_path="/test",
        idle_ttl_seconds=60,
    )
    kwargs.update(overrides)
    db.create_copilot_job(**kwargs)
    return job_id


# ---------------------------------------------------------------------------
# _is_pid_alive
# ---------------------------------------------------------------------------

class TestIsPidAlive:
    def test_zero_pid(self):
        assert _is_pid_alive(0) is False

    def test_negative_pid(self):
        assert _is_pid_alive(-1) is False

    def test_current_process(self):
        import os
        assert _is_pid_alive(os.getpid()) is True

    def test_nonexistent_pid(self):
        # PID 4_000_000 is extremely unlikely to exist.
        assert _is_pid_alive(4_000_000) is False


# ---------------------------------------------------------------------------
# reap_dead_processes
# ---------------------------------------------------------------------------

class TestReapDeadProcesses:
    def test_reaps_dead_pid(self, db):
        """Running job with a dead PID should be transitioned to idle."""
        _make_job(db, "cj_dead")
        db.transition_copilot_job("cj_dead", "running", event_type="test")
        db.update_copilot_job_remote("cj_dead", pid=4_000_000)

        reaped = reap_dead_processes(db)
        assert len(reaped) == 1
        assert reaped[0]["id"] == "cj_dead"

        job = db.get_copilot_job("cj_dead")
        assert job["state"] == "idle"

    def test_skips_alive_pid(self, db):
        """Running job with a live PID should not be reaped."""
        import os
        _make_job(db, "cj_alive")
        db.transition_copilot_job("cj_alive", "running", event_type="test")
        db.update_copilot_job_remote("cj_alive", pid=os.getpid())

        reaped = reap_dead_processes(db)
        assert len(reaped) == 0

        job = db.get_copilot_job("cj_alive")
        assert job["state"] == "running"

    def test_skips_human_owned(self, db):
        """Human-owned running jobs should never be reaped."""
        _make_job(db, "cj_human")
        db.transition_copilot_job("cj_human", "running", event_type="test")
        db.update_copilot_job_remote("cj_human", pid=4_000_000)
        db.take_over_copilot_job("cj_human")

        reaped = reap_dead_processes(db)
        assert len(reaped) == 0

    def test_skips_no_pid(self, db):
        """Running job with no PID stored should not be reaped."""
        _make_job(db, "cj_nopid")
        db.transition_copilot_job("cj_nopid", "running", event_type="test")

        reaped = reap_dead_processes(db)
        assert len(reaped) == 0

    def test_records_event(self, db):
        """Reaping should record a reaper_process_dead event."""
        _make_job(db, "cj_evt")
        db.transition_copilot_job("cj_evt", "running", event_type="test")
        db.update_copilot_job_remote("cj_evt", pid=4_000_000)

        reap_dead_processes(db)

        events = db.get_copilot_job_events("cj_evt")
        event_types = [e["event_type"] for e in events]
        assert "reaper_process_dead" in event_types


# ---------------------------------------------------------------------------
# reap_expired_idle
# ---------------------------------------------------------------------------

class TestReapExpiredIdle:
    def test_closes_expired(self, db):
        """Idle job past its TTL should be closed."""
        _make_job(db, "cj_exp", idle_ttl_seconds=1)
        db.transition_copilot_job("cj_exp", "running", event_type="test")
        db.mark_copilot_job_idle("cj_exp")

        # Backdate idle_since so TTL is expired.
        db._conn.execute(
            "UPDATE copilot_jobs SET idle_since = ? WHERE id = ?",
            (time.time() - 10, "cj_exp"),
        )
        db._conn.commit()

        closed = reap_expired_idle(db)
        assert len(closed) == 1

        job = db.get_copilot_job("cj_exp")
        assert job["state"] == "closed"

    def test_skips_not_yet_expired(self, db):
        """Idle job within its TTL should not be closed."""
        _make_job(db, "cj_fresh", idle_ttl_seconds=3600)
        db.transition_copilot_job("cj_fresh", "running", event_type="test")
        db.mark_copilot_job_idle("cj_fresh")

        closed = reap_expired_idle(db)
        assert len(closed) == 0

    def test_skips_human_owned(self, db):
        """Human-owned idle jobs should never be auto-closed."""
        _make_job(db, "cj_hum", idle_ttl_seconds=1)
        db.transition_copilot_job("cj_hum", "running", event_type="test")
        db.take_over_copilot_job("cj_hum")
        db.mark_copilot_job_idle("cj_hum")

        db._conn.execute(
            "UPDATE copilot_jobs SET idle_since = ? WHERE id = ?",
            (time.time() - 10, "cj_hum"),
        )
        db._conn.commit()

        closed = reap_expired_idle(db)
        assert len(closed) == 0

    def test_records_event(self, db):
        """Closing via TTL should record a reaper_ttl_expired event."""
        _make_job(db, "cj_ttl", idle_ttl_seconds=1)
        db.transition_copilot_job("cj_ttl", "running", event_type="test")
        db.mark_copilot_job_idle("cj_ttl")

        db._conn.execute(
            "UPDATE copilot_jobs SET idle_since = ? WHERE id = ?",
            (time.time() - 10, "cj_ttl"),
        )
        db._conn.commit()

        reap_expired_idle(db)

        events = db.get_copilot_job_events("cj_ttl")
        event_types = [e["event_type"] for e in events]
        assert "reaper_ttl_expired" in event_types


# ---------------------------------------------------------------------------
# reap_stale_pending
# ---------------------------------------------------------------------------

class TestReapStalePending:
    def test_closes_old_pending(self, db):
        """Pending job older than max_age should be closed."""
        _make_job(db, "cj_old")

        db._conn.execute(
            "UPDATE copilot_jobs SET created_at = ? WHERE id = ?",
            (time.time() - 7200, "cj_old"),
        )
        db._conn.commit()

        closed = reap_stale_pending(db, max_age=3600)
        assert len(closed) == 1

        job = db.get_copilot_job("cj_old")
        assert job["state"] == "closed"

    def test_skips_recent_pending(self, db):
        """Recently created pending job should not be closed."""
        _make_job(db, "cj_new")

        closed = reap_stale_pending(db, max_age=3600)
        assert len(closed) == 0

    def test_skips_running(self, db):
        """Running jobs are not affected by stale-pending reaping."""
        _make_job(db, "cj_run")
        db.transition_copilot_job("cj_run", "running", event_type="test")

        db._conn.execute(
            "UPDATE copilot_jobs SET created_at = ? WHERE id = ?",
            (time.time() - 7200, "cj_run"),
        )
        db._conn.commit()

        closed = reap_stale_pending(db, max_age=3600)
        assert len(closed) == 0


# ---------------------------------------------------------------------------
# reap (combined)
# ---------------------------------------------------------------------------

class TestReapCombined:
    def test_full_reap_cycle(self, db):
        """reap() should handle dead processes, expired idles, and stale pending."""
        # Dead process job
        _make_job(db, "cj_d")
        db.transition_copilot_job("cj_d", "running", event_type="test")
        db.update_copilot_job_remote("cj_d", pid=4_000_000)

        # Expired idle job
        _make_job(db, "cj_i", idle_ttl_seconds=1, repo_slug="other-repo")
        db.transition_copilot_job("cj_i", "running", event_type="test")
        db.mark_copilot_job_idle("cj_i")
        db._conn.execute(
            "UPDATE copilot_jobs SET idle_since = ? WHERE id = ?",
            (time.time() - 10, "cj_i"),
        )

        # Stale pending job
        _make_job(db, "cj_p", repo_slug="third-repo")
        db._conn.execute(
            "UPDATE copilot_jobs SET created_at = ? WHERE id = ?",
            (time.time() - 7200, "cj_p"),
        )
        db._conn.commit()

        result = reap(db, pending_max_age=3600)
        assert len(result["dead_processes"]) == 1
        assert len(result["ttl_expired"]) == 1
        assert len(result["stale_pending"]) == 1

    def test_nothing_to_reap(self, db):
        """When all jobs are healthy, reap returns empty lists."""
        result = reap(db)
        assert result == {
            "dead_processes": [],
            "ttl_expired": [],
            "stale_pending": [],
        }
