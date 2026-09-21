"""A worker that DIES raising a timeout-class error must not be mistaken for a live one.

On Python 3.11+ ``concurrent.futures.TimeoutError`` IS the builtin ``TimeoutError``, so a poll
loop's ``except concurrent.futures.TimeoutError`` cannot tell "wait slice expired" (worker alive)
from "worker raised TimeoutError" (worker dead). The aux client raises bare ``TimeoutError`` on
a stalled summary stream, so without a ``future.done()`` guard the host re-waited on a settled
future at ~2k iterations/sec until the idle budget burned (or forever, for the ceiling-less
commit wait). #117261 / #63892.

Budgets here are tiny so the red-on-base run FAILS fast instead of hanging. The commit-wait
loop has no ceiling, so on base it never returns; the runner's per-file timeout
(``scripts/run_tests.sh``, ``HERMES_TEST_FILE_TIMEOUT``) is what bounds that hang —
pytest-timeout is not a project dependency, so a ``pytest.mark.timeout`` marker would be inert.
"""

import concurrent.futures
import time

import pytest

from agent.conversation_compression import (
    CompressionCommitFence,
    _await_in_flight_commit,
    _await_worker_within_budget,
    _join_cancelled_worker,
)


def _settled_dead_future(pool, exc):
    future = pool.submit(lambda: (_ for _ in ()).throw(exc))
    concurrent.futures.wait([future], timeout=5)
    assert future.done()
    return future


def test_compression_wait_returns_promptly_when_worker_dies():
    """_await_worker_within_budget takes the stall path at once; the teardown join reports the
    dead worker as exited (so its lease is released) rather than as an orphan."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = _settled_dead_future(pool, TimeoutError("aux stream stalled"))
        fence = CompressionCommitFence()
        fence.touch_progress()
        started = time.monotonic()
        settled, result = _await_worker_within_budget(
            future, fence, idle=1.0, ceiling=2.0, wait_started=started
        )
        elapsed = time.monotonic() - started

        # Pre-fix this hot-spun for the whole idle window (1.0s here) before returning.
        assert elapsed < 0.5, f"host waited {elapsed:.2f}s on an already-dead worker"
        assert (settled, result) == (False, None)
        assert _join_cancelled_worker(future, 0.5) is True


def test_in_flight_commit_surfaces_worker_timeout_instead_of_looping():
    """_await_in_flight_commit has no ceiling on this path — pre-fix a dead worker spun forever."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = _settled_dead_future(pool, TimeoutError("commit stream stalled"))
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="commit stream stalled"):
            _await_in_flight_commit(future, ceiling=1.0, wait_started=started, on_commit_overrun=None)
        elapsed = time.monotonic() - started

    assert elapsed < 0.5, f"commit wait hung {elapsed:.2f}s on a dead worker"
