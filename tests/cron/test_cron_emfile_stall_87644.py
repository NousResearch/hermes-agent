"""Regression tests for #87644 — cron scheduler silently stalls after EMFILE.

Three fixes under test:

1. ``tick()`` no longer swallows a real ``OSError`` at tick-lock acquisition as
   "another instance holds the lock".  An EMFILE/ENFILE (fd exhaustion) on
   ``open(.tick.lock)`` previously made ``tick()`` return 0 — which the ticker
   loop recorded as a *successful* tick and cleared the error — so the
   scheduler looked perfectly healthy (heartbeat + success markers fresh)
   while no job ever ran again.  Now the failure propagates to the ticker
   loop, which records it and backs off.

2. The ticker loop detects fd exhaustion (EMFILE/ENFILE/"Too many open
   files"), attempts best-effort fd reclamation (``gc.collect()`` + raising
   the soft nofile limit), and applies exponential backoff so an exhausted
   process stops hammering the store every 60s.  Once FDs free up, the next
   tick succeeds and the backoff resets — the scheduler self-heals without a
   gateway restart.

3. Genuine lock contention (EWOULDBLOCK/EAGAIN on flock) still skips the tick
   silently — that behavior is preserved.
"""
import errno
import threading
import time
from unittest.mock import patch

import pytest

import cron.scheduler as scheduler_mod

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

pytestmark = pytest.mark.skipif(fcntl is None, reason="flock semantics are POSIX-only")


def _wait_until(predicate, timeout=10.0, interval=0.005):
    """Block until ``predicate()`` is truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


# ── Fix 1: tick() must not swallow EMFILE as "another instance holds the lock" ─


class TestTickLockEmfileNotSwallowed:
    def test_lock_open_emfile_raises_instead_of_silent_skip(self, monkeypatch):
        """open(.tick.lock) raising EMFILE must propagate — not return 0 as if
        another ticker held the lock (the pre-fix silent stall)."""
        real_open = open
        calls = []

        def flaky_open(path, *args, **kwargs):
            if str(path).endswith(".tick.lock"):
                calls.append(("emfile", str(path)))
                raise OSError(errno.EMFILE, "Too many open files")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", flaky_open)
        with pytest.raises(OSError) as excinfo:
            scheduler_mod.tick(verbose=False)
        assert excinfo.value.errno == errno.EMFILE
        assert calls, "the lock file open must have been attempted"

    def test_lock_flock_emfile_raises(self, monkeypatch):
        """flock() raising EMFILE (fd exhaustion at lock syscall) must also
        propagate — only EWOULDBLOCK/EAGAIN/EACCES mean contention."""
        with patch.object(fcntl, "flock", side_effect=OSError(errno.EMFILE, "Too many open files")):
            with pytest.raises(OSError) as excinfo:
                scheduler_mod.tick(verbose=False)
        assert excinfo.value.errno == errno.EMFILE

    def test_lock_contention_ewouldblock_still_skips_silently(self, monkeypatch):
        """Genuine contention (another ticker holds the lock) still returns 0
        silently — the cross-process mutual-exclusion contract is preserved."""
        with patch.object(fcntl, "flock", side_effect=OSError(errno.EWOULDBLOCK, "Resource temporarily unavailable")):
            assert scheduler_mod.tick(verbose=False) == 0

    def test_lock_contention_eagain_still_skips(self):
        with patch.object(fcntl, "flock", side_effect=OSError(errno.EAGAIN, "Resource temporarily unavailable")):
            assert scheduler_mod.tick(verbose=False) == 0


# ── Fix 2: ticker loop survives EMFILE, reclaims fds, backs off, self-heals ──


class TestTickerEmfileBackoff:
    def test_emfile_tick_records_error_and_keeps_looping(self, monkeypatch):
        """An EMFILE tick must be recorded as a FAILED tick (not success) and
        the loop must keep running — the pre-fix state where the scheduler
        stalled silently while the heartbeat kept reporting healthy."""
        from cron.scheduler_provider import InProcessCronScheduler

        beats = []
        errors = []
        stop = threading.Event()
        prov = InProcessCronScheduler()

        with patch(
            "cron.scheduler.tick",
            side_effect=OSError(errno.EMFILE, "Too many open files"),
        ), patch(
            "cron.jobs.record_ticker_heartbeat",
            side_effect=lambda success=False: beats.append(success),
        ), patch(
            "cron.jobs.record_ticker_error",
            side_effect=lambda msg: errors.append(msg),
        ), patch("cron.jobs.clear_ticker_error") as clear:
            t = threading.Thread(target=prov.start, args=(stop,), kwargs={"interval": 0}, daemon=True)
            t.start()
            assert _wait_until(lambda: len(beats) >= 3), "ticker did not keep beating"
            stop.set()
            t.join(timeout=5)

        assert not t.is_alive(), "ticker must survive EMFILE ticks"
        assert errors, "ticker error must be persisted so `hermes cron status` can show it"
        assert "Too many open files" in errors[0]
        # Every post-failure beat must be success=False: liveness yes, success no.
        assert beats[-1] is False
        clear.assert_not_called()

    def test_emfile_triggers_fd_reclamation(self, monkeypatch):
        """The ticker must attempt best-effort fd reclamation on EMFILE so the
        next tick can succeed once descriptors free up."""
        from cron.scheduler_provider import InProcessCronScheduler

        reclaimed = []
        stop = threading.Event()
        prov = InProcessCronScheduler()

        with patch(
            "cron.scheduler.tick",
            side_effect=OSError(errno.EMFILE, "Too many open files"),
        ), patch(
            "cron.scheduler._reclaim_fds_best_effort",
            side_effect=lambda: reclaimed.append(1),
        ), patch("cron.jobs.record_ticker_heartbeat"), patch("cron.jobs.record_ticker_error"):
            t = threading.Thread(target=prov.start, args=(stop,), kwargs={"interval": 0}, daemon=True)
            t.start()
            assert _wait_until(lambda: len(reclaimed) >= 1), "fd reclamation never ran"
            stop.set()
            t.join(timeout=5)

        assert len(reclaimed) >= 1

    def test_ticker_recovers_after_emfile_clears(self, monkeypatch):
        """Once ticks stop raising EMFILE, the next tick succeeds, the error is
        cleared and the success marker is bumped again — self-healing without a
        gateway restart."""
        from cron.scheduler_provider import InProcessCronScheduler

        attempts = {"n": 0}
        beats = []
        errors = []
        stop = threading.Event()
        prov = InProcessCronScheduler()

        def flaky_tick(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise OSError(errno.EMFILE, "Too many open files")
            return 0

        with patch("cron.scheduler.tick", side_effect=flaky_tick), patch(
            "cron.jobs.record_ticker_heartbeat",
            side_effect=lambda success=False: beats.append(success),
        ), patch(
            "cron.jobs.record_ticker_error",
            side_effect=lambda msg: errors.append(msg),
        ), patch("cron.jobs.clear_ticker_error") as clear, patch(
            "cron.scheduler._reclaim_fds_best_effort"
        ):
            t = threading.Thread(target=prov.start, args=(stop,), kwargs={"interval": 0}, daemon=True)
            t.start()
            # Wait until we've seen at least one success beat AFTER the failures.
            assert _wait_until(lambda: len(beats) >= 4), "ticker never recovered"
            stop.set()
            t.join(timeout=5)

        assert not t.is_alive()
        assert attempts["n"] >= 3
        assert errors, "failures must be recorded"
        assert True in beats, "a successful tick must bump the success marker"
        clear.assert_called(), "a successful tick must clear the recorded error"


# ── Helpers ────────────────────────────────────────────────────────────────────


class TestEmfileHelpers:
    def test_is_fd_exhaustion_errno(self):
        assert scheduler_mod._is_fd_exhaustion(OSError(errno.EMFILE, "Too many open files"))
        assert scheduler_mod._is_fd_exhaustion(OSError(errno.ENFILE, "file table overflow"))
        assert not scheduler_mod._is_fd_exhaustion(OSError(errno.EACCES, "Permission denied"))
        assert not scheduler_mod._is_fd_exhaustion(OSError(errno.EWOULDBLOCK, "Resource temporarily unavailable"))

    def test_is_fd_exhaustion_wrapped_message(self):
        """load_jobs wraps the raw OSError in a RuntimeError whose message
        carries the errno text — the ticker must classify that too."""
        assert scheduler_mod._is_fd_exhaustion(
            RuntimeError("Failed to read cron database: [Errno 24] Too many open files")
        )

    def test_reclaim_fds_best_effort_never_raises(self):
        # gc.collect + apply_nofile_soft_limit are both best-effort; the helper
        # must tolerate missing/refusing platforms without raising.
        assert scheduler_mod._reclaim_fds_best_effort() is None

    def test_lock_contention_errno_classification(self):
        assert scheduler_mod._is_lock_contention_errno(OSError(errno.EWOULDBLOCK, "x"))
        assert scheduler_mod._is_lock_contention_errno(OSError(errno.EAGAIN, "x"))
        assert not scheduler_mod._is_lock_contention_errno(OSError(errno.EMFILE, "x"))
        assert not scheduler_mod._is_lock_contention_errno(OSError(errno.ENFILE, "x"))
