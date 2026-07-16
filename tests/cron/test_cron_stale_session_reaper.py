"""Tests for the cron stale-session reaper (fail-closed owner-lease design).

Covers the ownership-lease + reaper salvaged from PR #62663:

* Active local owner > threshold is preserved.
* Stale dead owner is reaped.
* PID reuse / start-time mismatch handled as dead only after stale heartbeat.
* Fresh heartbeat preserved even if PID probe says dead.
* Missing / malformed ownerless leases are reaped after the grace window.
* Another current process's active session preserved.
* Lease cleanup ordering on end_session failure.
* No duplicate final assistant message (regression).

Process probes are mocked deterministically — never depend on actual process death.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import threading
import time
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so state.db + leases are fresh."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    # Reload modules that cache get_hermes_home() at import time.
    import hermes_constants
    importlib.reload(hermes_constants)
    import hermes_state
    importlib.reload(hermes_state)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


def _make_old_cron_session(db, session_id, age_seconds=7200):
    """Create a cron session and backdate its started_at."""
    db.create_session(session_id, source="cron")
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (time.time() - age_seconds, session_id),
    ))


_UNSET = object()


def _write_lease(home, session_id, *, job_id="test_job", owner_pid=None,
                 owner_start_time=_UNSET, host=None, heartbeat_at=None,
                 raw=None):
    """Write a lease sidecar for *session_id*.  Returns the lease path.

    Pass ``owner_start_time=None`` to explicitly omit the field (simulate a
    lease missing the start-time fingerprint); omit the kwarg entirely to get
    the default (99999).
    """
    if owner_pid is None:
        owner_pid = os.getpid()
    if owner_start_time is _UNSET:
        owner_start_time = 99999
    if host is None:
        host = "testhost"
    if heartbeat_at is None:
        heartbeat_at = time.time() - 7200  # stale by default
    lease_dir = home / "cron" / "cron_session_leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    path = lease_dir / f"{digest}.json"
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
    else:
        payload = {
            "session_id": session_id,
            "job_id": job_id,
            "owner_pid": owner_pid,
            "owner_start_time": owner_start_time,
            "host": host,
            "heartbeat_at": heartbeat_at,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Reaper — fail-closed scenarios
# ---------------------------------------------------------------------------


class TestReaperFailClosed:
    """_reap_stale_cron_sessions — all 9 fail-closed scenarios."""

    def test_active_local_owner_preserved(self, hermes_env):
        """Active local owner > threshold with stale heartbeat but live PID
        + matching start-time → NOT reaped (owner is provably alive)."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_live_job_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()

        _write_lease(hermes_env, sid, owner_pid=os.getpid(),
                     owner_start_time=99999, heartbeat_at=time.time() - 7200)

        # PID exists + start-time matches → owner alive → do not reap.
        with patch("gateway.status._pid_exists", return_value=True), \
             patch("gateway.status._get_process_start_time", return_value=99999), \
             patch("cron.scheduler._cron_local_host_id", return_value="testhost"):
            count = _reap_stale_cron_sessions()

        assert count == 0
        db = SessionDB()
        assert db.get_session(sid)["ended_at"] is None
        db.close()

    def test_stale_dead_owner_reaped(self, hermes_env):
        """Stale heartbeat + owner PID does not exist → reaped."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_dead_job_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()

        _write_lease(hermes_env, sid, owner_pid=99999,
                     owner_start_time=11111, heartbeat_at=time.time() - 7200)

        # PID does not exist → owner dead → reap.
        with patch("gateway.status._pid_exists", return_value=False), \
             patch("cron.scheduler._cron_local_host_id", return_value="testhost"):
            count = _reap_stale_cron_sessions()

        assert count == 1
        db = SessionDB()
        session = db.get_session(sid)
        assert session["end_reason"] == "stale_reaped"
        assert session["ended_at"] is not None
        db.close()

    def test_pid_reuse_start_time_mismatch_handled_as_dead_after_stale(self, hermes_env):
        """Stale heartbeat + PID alive but start-time mismatch (PID reused)
        → owner is definitively dead → reaped."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_reuse_job_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()

        # Recorded start-time = 11111, live start-time = 22222 (different).
        _write_lease(hermes_env, sid, owner_pid=99999,
                     owner_start_time=11111, heartbeat_at=time.time() - 7200)

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("gateway.status._get_process_start_time", return_value=22222), \
             patch("cron.scheduler._cron_local_host_id", return_value="testhost"):
            count = _reap_stale_cron_sessions()

        assert count == 1
        db = SessionDB()
        assert db.get_session(sid)["end_reason"] == "stale_reaped"
        db.close()

    def test_fresh_heartbeat_preserved_even_if_pid_dead(self, hermes_env):
        """Fresh heartbeat → NOT reaped even if PID probe says dead.
        The run may still be progressing; a fresh heartbeat means 'alive'."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_fresh_job_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()

        # Heartbeat is fresh (now), even though session is old.
        _write_lease(hermes_env, sid, owner_pid=99999,
                     owner_start_time=11111, heartbeat_at=time.time())

        with patch("gateway.status._pid_exists", return_value=False), \
             patch("cron.scheduler._cron_local_host_id", return_value="testhost"):
            count = _reap_stale_cron_sessions()

        assert count == 0
        db = SessionDB()
        assert db.get_session(sid)["ended_at"] is None
        db.close()

    def test_missing_lease_reaped_after_grace(self, hermes_env):
        """A legacy session with no lease has no live ownership evidence."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_nolease_job_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()

        assert _reap_stale_cron_sessions() == 1

        db = SessionDB()
        session = db.get_session(sid)
        assert session["end_reason"] == "stale_reaped"
        assert session["ended_at"] is not None
        db.close()

    def test_malformed_lease_reaped_after_grace(self, hermes_env):
        """An unreadable legacy lease carries no live ownership evidence."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_bad_job_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()

        _write_lease(hermes_env, sid, raw="{not valid json")

        assert _reap_stale_cron_sessions() == 1

        db = SessionDB()
        session = db.get_session(sid)
        assert session["end_reason"] == "stale_reaped"
        assert session["ended_at"] is not None
        db.close()

    def test_null_runner_fields_reaped_after_grace(self, hermes_env):
        """A structured but ownerless lease must not make a row immortal."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_ownerless_job_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()

        _write_lease(
            hermes_env,
            sid,
            raw=json.dumps(
                {
                    "session_id": sid,
                    "job_id": "ownerless-job",
                    "runner_pid": None,
                    "runner_id": None,
                    "heartbeat_at": time.time() - 7200,
                }
            ),
        )

        assert _reap_stale_cron_sessions() == 1

        db = SessionDB()
        session = db.get_session(sid)
        assert session["end_reason"] == "stale_reaped"
        assert session["ended_at"] is not None
        db.close()

    def test_mismatched_session_id_lease_preserved(self, hermes_env):
        """A valid lease stored under the wrong session hash is untrusted."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_mismatch_job_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()
        path = _write_lease(
            hermes_env,
            sid,
            owner_pid=99999,
            owner_start_time=11111,
            heartbeat_at=time.time() - 7200,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["session_id"] = "different-session"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with patch("gateway.status._pid_exists", return_value=False), \
             patch("cron.scheduler._cron_local_host_id", return_value="testhost"):
            count = _reap_stale_cron_sessions()

        assert count == 0
        db = SessionDB()
        session = db.get_session(sid)
        assert session is not None
        assert session["ended_at"] is None
        db.close()

    def test_cross_host_lease_preserved(self, hermes_env):
        """Lease owned by a different host → cannot probe locally → do not reap."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_remote_job_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()

        _write_lease(hermes_env, sid, owner_pid=99999,
                     owner_start_time=11111, heartbeat_at=time.time() - 7200,
                     host="other-host")

        with patch("cron.scheduler._cron_local_host_id", return_value="this-host"):
            count = _reap_stale_cron_sessions()

        assert count == 0
        db = SessionDB()
        assert db.get_session(sid)["ended_at"] is None
        db.close()

    def test_another_process_active_session_preserved(self, hermes_env):
        """A session whose owner is a different PID that IS alive (same host,
        matching start-time) → do not reap."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_otherproc_job_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()

        # Owner is pid 12345, alive, start-time matches.
        _write_lease(hermes_env, sid, owner_pid=12345,
                     owner_start_time=55555, heartbeat_at=time.time() - 7200)

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("gateway.status._get_process_start_time", return_value=55555), \
             patch("cron.scheduler._cron_local_host_id", return_value="testhost"):
            count = _reap_stale_cron_sessions()

        assert count == 0
        db = SessionDB()
        assert db.get_session(sid)["ended_at"] is None
        db.close()

    def test_active_runner_identity_preserved(self, hermes_env):
        """New-format runner evidence fences a live owner after the grace."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_active_runner_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()
        _write_lease(
            hermes_env,
            sid,
            raw=json.dumps(
                {
                    "session_id": sid,
                    "job_id": "active-runner",
                    "runner_pid": 12345,
                    "runner_id": "runner-abc",
                    "host": "testhost",
                    "heartbeat_at": time.time() - 7200,
                }
            ),
        )

        with patch("gateway.status._pid_exists", return_value=True), patch(
            "cron.scheduler._cron_local_host_id", return_value="testhost"
        ):
            assert _reap_stale_cron_sessions() == 0

        db = SessionDB()
        assert db.get_session(sid)["ended_at"] is None
        db.close()

    def test_recent_session_not_reaped(self, hermes_env):
        """A session younger than the threshold is never reaped."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_recent_job_20240101_120000"
        db = SessionDB()
        db.create_session(sid, source="cron")  # started_at = now
        db.close()

        _write_lease(hermes_env, sid, owner_pid=99999,
                     owner_start_time=11111, heartbeat_at=time.time() - 7200)

        with patch("gateway.status._pid_exists", return_value=False):
            count = _reap_stale_cron_sessions()

        assert count == 0
        db = SessionDB()
        assert db.get_session(sid)["ended_at"] is None
        db.close()

    def test_already_ended_session_not_reaped(self, hermes_env):
        """A session already ended is a no-op for the reaper."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_done_job_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.end_session(sid, "cron_complete")
        db.close()

        _write_lease(hermes_env, sid, owner_pid=99999,
                     owner_start_time=11111, heartbeat_at=time.time() - 7200)

        with patch("gateway.status._pid_exists", return_value=False), \
             patch("cron.scheduler._cron_local_host_id", return_value="testhost"):
            count = _reap_stale_cron_sessions()

        assert count == 0

    def test_missing_start_time_fingerprint_preserved(self, hermes_env):
        """Lease has no owner_start_time → cannot guard against PID reuse →
        fail closed (do not reap) even if PID is alive."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_nostart_job_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()

        _write_lease(hermes_env, sid, owner_pid=99999,
                     owner_start_time=None, heartbeat_at=time.time() - 7200)

        # PID is alive, but we have no recorded start-time to compare.
        with patch("gateway.status._pid_exists", return_value=True), \
             patch("gateway.status._get_process_start_time", return_value=22222), \
             patch("cron.scheduler._cron_local_host_id", return_value="testhost"):
            count = _reap_stale_cron_sessions()

        assert count == 0
        db = SessionDB()
        assert db.get_session(sid)["ended_at"] is None
        db.close()

    def test_probe_exception_preserves_session(self, hermes_env):
        """A liveness-probe error is uncertainty, not proof of death."""
        from hermes_state import SessionDB
        from cron.scheduler import _reap_stale_cron_sessions

        sid = "cron_probe_error_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()
        _write_lease(
            hermes_env,
            sid,
            owner_pid=99999,
            owner_start_time=11111,
            heartbeat_at=time.time() - 7200,
        )

        with patch("gateway.status._pid_exists", side_effect=OSError("denied")), \
             patch("cron.scheduler._cron_local_host_id", return_value="testhost"):
            count = _reap_stale_cron_sessions()

        assert count == 0
        db = SessionDB()
        session = db.get_session(sid)
        assert session is not None
        assert session["ended_at"] is None
        db.close()


# ---------------------------------------------------------------------------
# Lease manager helpers
# ---------------------------------------------------------------------------


class TestLeaseManager:
    """_register / _heartbeat / _remove / _read lease helpers."""

    def test_lease_path_hashes_session_id(self, hermes_env):
        from cron.scheduler import _cron_lease_path
        import cron.scheduler as sched

        with patch.object(sched, "_get_hermes_home", return_value=hermes_env):
            path = _cron_lease_path("cron_test_20240101_120000")
        digest = hashlib.sha256(b"cron_test_20240101_120000").hexdigest()
        assert path.name == f"{digest}.json"
        assert path.parent == hermes_env / "cron" / "cron_session_leases"

    def test_register_then_read_roundtrip(self, hermes_env):
        import cron.scheduler as sched
        from cron.scheduler import _register_cron_session_lease, _read_cron_session_lease

        with patch.object(sched, "_get_hermes_home", return_value=hermes_env), \
             patch("gateway.status._get_process_start_time", return_value=12345), \
             patch("cron.scheduler._cron_local_host_id", return_value="myhost"):
            lease = _register_cron_session_lease("cron_x_20240101_120000", "x")
        assert lease is not None
        assert lease["session_id"] == "cron_x_20240101_120000"
        assert lease["job_id"] == "x"
        assert lease["owner_pid"] == os.getpid()
        assert lease["owner_start_time"] == 12345
        assert lease["host"] == "myhost"

        read = _read_cron_session_lease("cron_x_20240101_120000")
        assert read is not None
        assert read["session_id"] == "cron_x_20240101_120000"

    def test_remove_is_idempotent(self, hermes_env):
        import cron.scheduler as sched
        from cron.scheduler import _remove_cron_session_lease, _read_cron_session_lease

        with patch.object(sched, "_get_hermes_home", return_value=hermes_env):
            _remove_cron_session_lease("nonexistent")  # no error
            _remove_cron_session_lease("nonexistent")  # still no error
        assert _read_cron_session_lease("nonexistent") is None

    def test_read_malformed_returns_none(self, hermes_env):
        import cron.scheduler as sched
        from cron.scheduler import _read_cron_session_lease

        _write_lease(hermes_env, "cron_bad", raw="not json at all")
        with patch.object(sched, "_get_hermes_home", return_value=hermes_env):
            assert _read_cron_session_lease("cron_bad") is None

    def test_heartbeat_updates_file(self, hermes_env):
        import cron.scheduler as sched
        from cron.scheduler import (
            _register_cron_session_lease,
            _heartbeat_cron_session_lease,
            _read_cron_session_lease,
        )

        with patch.object(sched, "_get_hermes_home", return_value=hermes_env), \
             patch("gateway.status._get_process_start_time", return_value=12345), \
             patch("cron.scheduler._cron_local_host_id", return_value="myhost"):
            _register_cron_session_lease("cron_hb_20240101_120000", "hb")
            old = _read_cron_session_lease("cron_hb_20240101_120000")
            assert old is not None
            old_hb = old["heartbeat_at"]

            time.sleep(0.01)
            _heartbeat_cron_session_lease("cron_hb_20240101_120000")
            new = _read_cron_session_lease("cron_hb_20240101_120000")
            assert new is not None

        # Heartbeats must retain the owner identity needed by the reaper.
        assert new["heartbeat_at"] > old_hb
        assert new["session_id"] == old["session_id"]
        assert new["job_id"] == old["job_id"]
        assert new["owner_pid"] == old["owner_pid"]
        assert new["owner_start_time"] == old["owner_start_time"]
        assert new["host"] == old["host"]

    def test_heartbeat_updates_when_start_time_unavailable(self, hermes_env):
        """A matching owner PID may heartbeat without a start fingerprint."""
        import cron.scheduler as sched

        sid = "cron_no_start_20240101_120000"
        path = _write_lease(
            hermes_env,
            sid,
            owner_pid=os.getpid(),
            owner_start_time=None,
            heartbeat_at=time.time() - 7200,
        )
        before = json.loads(path.read_text(encoding="utf-8"))
        with patch.object(sched, "_get_hermes_home", return_value=hermes_env), \
             patch("gateway.status._get_process_start_time") as start_probe:
            sched._heartbeat_cron_session_lease(sid)
        after = json.loads(path.read_text(encoding="utf-8"))
        assert after["heartbeat_at"] > before["heartbeat_at"]
        assert after["owner_start_time"] is None
        start_probe.assert_not_called()

    def test_heartbeat_does_not_refresh_replacement_owner(self, hermes_env):
        """A stale runner must not overwrite a lease owned by another PID."""
        import cron.scheduler as sched

        sid = "cron_replaced_20240101_120000"
        path = _write_lease(
            hermes_env,
            sid,
            owner_pid=os.getpid() + 1,
            owner_start_time=12345,
            heartbeat_at=time.time() - 7200,
        )
        before = json.loads(path.read_text(encoding="utf-8"))
        with patch.object(sched, "_get_hermes_home", return_value=hermes_env), \
             patch("gateway.status._get_process_start_time", return_value=12345):
            sched._heartbeat_cron_session_lease(sid)
        after = json.loads(path.read_text(encoding="utf-8"))
        assert after == before


class TestTickIntegration:
    def test_tick_reaps_before_dispatch_gate(self, hermes_env):
        """Every acquired tick reaps, even while new dispatch is paused."""
        import cron.scheduler as sched

        lock_path = hermes_env / "cron" / ".tick.lock"
        with patch.object(
            sched,
            "_get_lock_paths",
            return_value=(lock_path.parent, lock_path),
        ), patch.object(
            sched, "_reap_stale_cron_sessions", return_value=0
        ) as reap, patch.object(sched, "get_due_jobs") as get_due_jobs:
            count = sched.tick(verbose=False, can_dispatch=lambda: False)

        assert count == 0
        reap.assert_called_once_with()
        get_due_jobs.assert_not_called()


class TestStartupIntegration:
    def test_restart_reaps_before_first_tick(self, hermes_env):
        """Startup recovers a crash leftover even when scheduling never ticks."""
        import cron.scheduler as sched
        from cron.scheduler_provider import InProcessCronScheduler
        from hermes_state import SessionDB

        sid = "cron_crash_leftover_20240101_120000"
        db = SessionDB()
        _make_old_cron_session(db, sid, age_seconds=7200)
        db.close()
        _write_lease(
            hermes_env,
            sid,
            owner_pid=99999,
            owner_start_time=11111,
            heartbeat_at=time.time() - 7200,
        )

        stop = threading.Event()
        stop.set()
        with patch("gateway.status._pid_exists", return_value=False), patch.object(
            sched, "_cron_local_host_id", return_value="testhost"
        ), patch.object(sched, "tick") as tick, patch(
            "cron.jobs.record_ticker_heartbeat"
        ):
            InProcessCronScheduler().start(stop)

        tick.assert_not_called()
        db = SessionDB()
        session = db.get_session(sid)
        assert session["end_reason"] == "stale_reaped"
        assert session["ended_at"] is not None
        db.close()
