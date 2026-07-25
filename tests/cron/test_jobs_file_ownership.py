"""Regression tests for issue #68483.

Running a state-writing ``hermes cron`` CLI command as root (the default for
``docker exec``) rewrote ``jobs.json`` as ``root:root`` mode 600 via the
mkstemp + atomic_replace pattern, silently locking out the unprivileged
gateway ticker — which then failed every tick with PermissionError while the
liveness heartbeat stayed fresh, so nothing surfaced the outage.

Two behavior contracts are pinned here:

1. Ownership preservation: a privileged (euid 0) writer must hand ownership
   of the rewritten ``jobs.json`` back to its previous owner; unprivileged
   writers must not attempt a chown at all.
2. Zombie-ticker surfacing: a failing tick must persist the failure reason
   (``ticker_last_error``) where ``hermes cron status`` can show it, and a
   subsequent clean tick must clear it.
"""

import os
import sys
import threading

import pytest

import cron.jobs as jobs


pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only: uid/gid ownership semantics"
)


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Route the cron store to an isolated temp dir."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    return cron_dir


# =========================================================================
# 1. Ownership preservation on save (root writer)
# =========================================================================


class TestSaveJobsOwnershipPreservation:
    def test_root_writer_restores_previous_owner(self, cron_store, monkeypatch):
        """When euid==0 and jobs.json was owned by another uid/gid, the
        rewrite must chown the new file back to that owner."""
        jobs.save_jobs([{"id": "seed", "prompt": "hello"}])
        jobs_file = cron_store / "jobs.json"
        assert jobs_file.exists()

        chown_calls = []

        # Pretend the existing file is owned by the gateway user (uid 1000)
        # and that WE are root.
        real_stat = os.stat

        class _FakeStat:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self.st_uid = 1000
                self.st_gid = 1000

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        def fake_stat(path, *a, **k):
            result = real_stat(path, *a, **k)
            if str(path) == str(jobs_file):
                return _FakeStat(result)
            return result

        monkeypatch.setattr(jobs.os, "stat", fake_stat)
        monkeypatch.setattr(jobs.os, "geteuid", lambda: 0)
        monkeypatch.setattr(jobs.os, "getegid", lambda: 0)
        monkeypatch.setattr(
            jobs.os, "chown", lambda path, uid, gid: chown_calls.append((str(path), uid, gid))
        )

        jobs.save_jobs([{"id": "seed", "prompt": "updated"}])

        assert chown_calls == [(str(jobs_file), 1000, 1000)], (
            "root rewrite must hand jobs.json back to the previous owner "
            "(uid/gid 1000) instead of leaving it root:600 (#68483)"
        )

    def test_root_first_write_inherits_cron_dir_owner(self, cron_store, monkeypatch):
        """Creating jobs.json for the first time as root must inherit the
        cron dir's owner (the gateway user in the Docker image)."""
        chown_calls = []
        jobs_file = cron_store / "jobs.json"

        real_stat = os.stat

        class _FakeStat:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self.st_uid = 1000
                self.st_gid = 1000

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        def fake_stat(path, *a, **k):
            result = real_stat(path, *a, **k)
            if str(path) == str(cron_store):
                return _FakeStat(result)
            return result

        monkeypatch.setattr(jobs.os, "stat", fake_stat)
        monkeypatch.setattr(jobs.os, "geteuid", lambda: 0)
        monkeypatch.setattr(jobs.os, "getegid", lambda: 0)
        monkeypatch.setattr(
            jobs.os, "chown", lambda path, uid, gid: chown_calls.append((str(path), uid, gid))
        )

        assert not jobs_file.exists()
        jobs.save_jobs([{"id": "new", "prompt": "hello"}])

        assert chown_calls == [(str(jobs_file), 1000, 1000)]

    def test_unprivileged_writer_never_chowns(self, cron_store, monkeypatch):
        """A same-uid (non-root) writer must not attempt chown at all —
        it would raise EPERM for foreign-owned files anyway."""
        jobs.save_jobs([{"id": "seed", "prompt": "hello"}])

        def _fail_chown(*a, **k):
            raise AssertionError("unprivileged save must not call os.chown")

        monkeypatch.setattr(jobs.os, "chown", _fail_chown)
        jobs.save_jobs([{"id": "seed", "prompt": "updated"}])  # must not raise

    def test_chown_failure_never_breaks_save(self, cron_store, monkeypatch):
        """A chown failure is logged, but the save itself must succeed."""
        jobs.save_jobs([{"id": "seed", "prompt": "hello"}])
        jobs_file = cron_store / "jobs.json"

        real_stat = os.stat

        class _FakeStat:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self.st_uid = 1000
                self.st_gid = 1000

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        def fake_stat(path, *a, **k):
            result = real_stat(path, *a, **k)
            if str(path) == str(jobs_file):
                return _FakeStat(result)
            return result

        def _broken_chown(*a, **k):
            raise PermissionError("simulated chown failure")

        monkeypatch.setattr(jobs.os, "stat", fake_stat)
        monkeypatch.setattr(jobs.os, "geteuid", lambda: 0)
        monkeypatch.setattr(jobs.os, "getegid", lambda: 0)
        monkeypatch.setattr(jobs.os, "chown", _broken_chown)

        jobs.save_jobs([{"id": "seed", "prompt": "updated"}])  # must not raise
        assert jobs.load_jobs()[0]["prompt"] == "updated"

    def test_save_still_enforces_0600(self, cron_store):
        """The ownership fix must not regress the 0600 hardening."""
        import stat

        jobs.save_jobs([{"id": "seed", "prompt": "hello"}])
        mode = stat.S_IMODE(os.stat(cron_store / "jobs.json").st_mode)
        assert mode == 0o600


# =========================================================================
# 2. Zombie-ticker surfacing (ticker_last_error marker)
# =========================================================================


class TestTickerErrorMarker:
    def test_record_and_get_roundtrip(self, cron_store):
        assert jobs.get_ticker_last_error() is None
        jobs.record_ticker_error(
            "RuntimeError: Failed to read cron database: "
            "[Errno 13] Permission denied: '/opt/data/cron/jobs.json'"
        )
        msg = jobs.get_ticker_last_error()
        assert msg is not None
        assert "Permission denied" in msg

    def test_clear_removes_marker(self, cron_store):
        jobs.record_ticker_error("RuntimeError: boom")
        assert jobs.get_ticker_last_error() is not None
        jobs.clear_ticker_error()
        assert jobs.get_ticker_last_error() is None

    def test_clear_when_absent_is_noop(self, cron_store):
        jobs.clear_ticker_error()  # must not raise
        assert jobs.get_ticker_last_error() is None

    def test_record_failure_is_silent(self, tmp_path, monkeypatch):
        """Marker write failure must never disrupt the tick loop."""
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("i am a file")
        bad_dir = blocker / "cron"
        monkeypatch.setattr(jobs, "CRON_DIR", bad_dir)
        monkeypatch.setattr(jobs, "JOBS_FILE", bad_dir / "jobs.json")
        monkeypatch.setattr(jobs, "OUTPUT_DIR", bad_dir / "output")

        jobs.record_ticker_error("RuntimeError: boom")  # must not raise
        assert jobs.get_ticker_last_error() is None


class TestTickerLoopRecordsErrors:
    def _run_one_tick(self, monkeypatch, tick_fn):
        """Run one iteration of the built-in ticker loop with a stubbed tick."""
        from cron.scheduler_provider import InProcessCronScheduler

        provider = InProcessCronScheduler()
        monkeypatch.setattr(provider, "recover_interrupted", lambda: 0)
        monkeypatch.setattr("cron.scheduler.tick", tick_fn)

        stop_event = threading.Event()
        original_wait = stop_event.wait

        def _stop_after_first_tick(timeout=None):
            stop_event.set()
            return original_wait(0)

        stop_event.wait = _stop_after_first_tick
        provider.start(stop_event)

    def test_failing_tick_persists_error(self, cron_store, monkeypatch):
        def _boom(**kwargs):
            raise RuntimeError(
                "Failed to read cron database: [Errno 13] Permission denied"
            )

        self._run_one_tick(monkeypatch, _boom)

        msg = jobs.get_ticker_last_error()
        assert msg is not None, (
            "a failing tick must persist its reason for `hermes cron status` "
            "to surface (#68483)"
        )
        assert "Permission denied" in msg

    def test_successful_tick_clears_error(self, cron_store, monkeypatch):
        jobs.record_ticker_error("RuntimeError: stale failure")

        self._run_one_tick(monkeypatch, lambda **kwargs: None)

        assert jobs.get_ticker_last_error() is None, (
            "a clean tick must clear the stale error marker"
        )


# =========================================================================
# 3. `hermes cron status` surfaces the failure reason
# =========================================================================


class TestCronStatusSurfacesError:
    def test_status_shows_last_error_and_permission_hint(self, monkeypatch, capsys):
        from hermes_cli import cron as cron_cli

        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [4321])
        monkeypatch.setattr(jobs, "get_ticker_heartbeat_age", lambda: 5.0)   # alive
        monkeypatch.setattr(jobs, "get_ticker_success_age", lambda: 9_999.0)  # failing
        monkeypatch.setattr(
            jobs,
            "get_ticker_last_error",
            lambda: (
                "RuntimeError: Failed to read cron database: "
                "[Errno 13] Permission denied: '/opt/data/cron/jobs.json'"
            ),
        )
        monkeypatch.setattr("cron.jobs.list_jobs", lambda **k: [])

        cron_cli.cron_status()
        out = capsys.readouterr().out
        assert "Last tick error:" in out
        assert "Permission denied" in out
        # The permission-specific hint must point at the ownership fix.
        assert "docker exec -u" in out

    def test_status_without_marker_keeps_generic_message(self, monkeypatch, capsys):
        from hermes_cli import cron as cron_cli

        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [4321])
        monkeypatch.setattr(jobs, "get_ticker_heartbeat_age", lambda: 5.0)
        monkeypatch.setattr(jobs, "get_ticker_success_age", lambda: 9_999.0)
        monkeypatch.setattr(jobs, "get_ticker_last_error", lambda: None)
        monkeypatch.setattr("cron.jobs.list_jobs", lambda **k: [])

        cron_cli.cron_status()
        out = capsys.readouterr().out
        assert "no tick has succeeded" in out
        assert "Last tick error:" not in out
