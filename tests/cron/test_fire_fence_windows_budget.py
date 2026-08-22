"""Regression tests for the Windows fire-fence bounded-polling fix.

Historical behaviour: ``_fire_job_lock``/``_jobs_lock`` used
``msvcrt.locking(LK_LOCK)`` on Windows, which retries internally for only
~10 seconds before raising ``OSError(EDEADLK)``. The POSIX branches poll
``LOCK_NB`` against a 30-second budget (#60703). Windows therefore failed
closed EARLY: a second gateway legitimately holding the per-job fire fence
for 10-30 seconds (long delivery) made ``mark_job_run`` return False, which
``run_one_job`` interprets as "fire claim ownership lost" — the successful
completion was silently discarded.

The fix routes both branches through
``cron.cross_process_lock.lock_exclusive_bounded`` so every backend waits
the same ``_JOBS_LOCK_TIMEOUT_SECONDS`` budget.

These tests exercise the real store against a temp HERMES_HOME (E2E over
mocks, per project discipline). The cross-process contention tests are
Windows-only (msvcrt); POSIX is covered by the equivalent fcntl polling
already tested in test_ticker_stall_60703.py.
"""
import sys
import time
from pathlib import Path

import pytest

msvcrt = pytest.importorskip("msvcrt", reason="Windows-only lock backend")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _hold_raw_lock(lock_path_str, ready, hold):
    """Child process: hold a raw msvcrt lock on ``lock_path_str`` for ``hold`` s."""
    fd = open(lock_path_str, "a+")
    fd.seek(0)
    msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
    ready.set()
    time.sleep(hold)
    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
    fd.close()


def _hold_fire_fence(store_str, job_id, ready, hold):
    """Child process: hold the per-job fire fence via the real public API."""
    sys.path.insert(0, str(_REPO_ROOT))
    from cron.jobs import use_cron_store, _fire_job_lock
    with use_cron_store(Path(store_str)):
        with _fire_job_lock(job_id):
            ready.set()
            time.sleep(hold)


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json doesn't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def test_lock_exclusive_bounded_polls_past_msvcrt_internal_limit(temp_home):
    """Direct probe: a lock held for >10s (<30s budget) by another PROCESS
    must still be acquired once released — not fail at msvcrt's ~10s
    internal LK_LOCK retry ceiling."""
    import multiprocessing as mp
    from cron.cross_process_lock import lock_exclusive_bounded, unlock_quietly

    lock_path = temp_home / "probe.lock"
    lock_path.touch()

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(target=_hold_raw_lock, args=(str(lock_path), ready, 12))
    proc.start()
    try:
        assert ready.wait(10), "holder never acquired the lock"

        fd = open(lock_path, "a+")
        fd.seek(0)
        t0 = time.monotonic()
        acquired = lock_exclusive_bounded(fd, 30.0)
        waited = time.monotonic() - t0
        assert acquired is True, (
            "fence acquisition failed before the 30s budget elapsed — "
            "the msvcrt ~10s internal retry ceiling is leaking through"
        )
        assert waited >= 10.0, f"acquired after only {waited:.1f}s; contention not exercised"
        unlock_quietly(fd)
        fd.close()
    finally:
        proc.terminate()
        proc.join(timeout=10)


def test_mark_job_run_survives_cross_process_fence_contention(temp_home):
    """End-to-end: while another PROCESS holds the job's fire fence for 12s
    (a legitimate long delivery window), mark_job_run must block past the
    old ~10s failure point and record the successful completion."""
    import multiprocessing as mp
    from cron.jobs import use_cron_store, save_jobs, mark_job_run

    with use_cron_store(temp_home):
        save_jobs([{
            "id": "fence-e2e",
            "name": "e2e",
            "schedule": {"kind": "once", "at": "2030-01-01T00:00:00"},
        }])

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(target=_hold_fire_fence, args=(str(temp_home), "fence-e2e", ready, 12))
    proc.start()
    try:
        assert ready.wait(10), "holder never acquired the fence"

        with use_cron_store(temp_home):
            ok = mark_job_run("fence-e2e", True, None)
        assert ok is True, (
            "mark_job_run failed closed under cross-process contention "
            "before the 30s budget elapsed (old msvcrt LK_LOCK ~10s ceiling)"
        )
    finally:
        proc.terminate()
        proc.join(timeout=10)
