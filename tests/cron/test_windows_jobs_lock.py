"""Regression tests for the Windows (msvcrt) _jobs_lock path.

The POSIX bounded-lock fix (#60703, #60855) added a polling loop with
30-second timeout and graceful degradation for fcntl/flock, but the
msvcrt/Windows path was not updated — it used a blocking LK_LOCK call
with ~10s internal retry and a generic WARNING on failure.

These tests verify that the msvcrt path now uses LK_NBLCK with the same
30-second bounded polling, ERROR-level timeout logging, and degraded
critical-section entry as the POSIX path.  They run on any platform by
faking an msvcrt module (since real msvcrt is only available on Windows).
"""

import time

import cron.jobs as jobs_mod


def _fake_msvcrt(*, fail_count: int = 0):
    """Build a fake ``msvcrt`` module for ``_jobs_lock``.

    *fail_count*: how many ``locking()`` calls raise ``OSError`` before
    succeeding.  ``-1`` means always fail.
    """
    attempts = [0]  # mutable closure

    class FakeMsvcrt:
        LK_NBLCK = 0x0004
        LK_UNLCK = 0x0000

        @staticmethod
        def locking(fd, mode, nbytes):
            if fail_count == -1:
                raise OSError(33, "lock held by another process")
            attempts[0] += 1
            if attempts[0] <= fail_count:
                raise OSError(33, "lock held by another process")

    return FakeMsvcrt


class TestWindowsBoundedJobsLock:
    def test_lock_times_out_and_degrades(self, monkeypatch, caplog):
        """LK_NBLCK contention must not block _jobs_lock forever on Windows."""
        monkeypatch.setattr(jobs_mod, "fcntl", None)
        monkeypatch.setattr(jobs_mod, "msvcrt", _fake_msvcrt(fail_count=-1))
        monkeypatch.setattr(jobs_mod, "_JOBS_LOCK_TIMEOUT_SECONDS", 1.0)
        jobs_mod.ensure_dirs()
        _ = jobs_mod._jobs_lock_file().touch()

        start = time.monotonic()
        entered = False
        with caplog.at_level("ERROR", logger="cron.jobs"):
            with jobs_mod._jobs_lock():
                entered = True
        elapsed = time.monotonic() - start

        assert entered, "critical section must still run in degraded mode"
        assert elapsed < 10, f"lock wait was not bounded (took {elapsed:.1f}s)"
        assert any("Timed out" in r.message for r in caplog.records), (
            "degraded-mode fallback must be logged at ERROR"
        )

    def test_uncontended_lock_is_fast_and_silent(self, monkeypatch, caplog):
        """Uncontested LK_NBLCK acquisition should succeed immediately."""
        monkeypatch.setattr(jobs_mod, "fcntl", None)
        monkeypatch.setattr(jobs_mod, "msvcrt", _fake_msvcrt(fail_count=0))
        jobs_mod.ensure_dirs()

        start = time.monotonic()
        with caplog.at_level("ERROR", logger="cron.jobs"):
            with jobs_mod._jobs_lock():
                pass
        assert time.monotonic() - start < 5
        assert not [r for r in caplog.records if "Timed out" in r.message]

    def test_lock_recovers_after_transient_contention(self, monkeypatch, caplog):
        """A brief LK_NBLCK conflict (1 failure, then success) must not degrade."""
        monkeypatch.setattr(jobs_mod, "fcntl", None)
        monkeypatch.setattr(jobs_mod, "msvcrt", _fake_msvcrt(fail_count=1))
        monkeypatch.setattr(jobs_mod, "_JOBS_LOCK_TIMEOUT_SECONDS", 3.0)
        jobs_mod.ensure_dirs()

        with caplog.at_level("ERROR", logger="cron.jobs"):
            with jobs_mod._jobs_lock():
                pass
        assert not [r for r in caplog.records if "Timed out" in r.message]
