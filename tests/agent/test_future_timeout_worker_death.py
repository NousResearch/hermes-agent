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
from types import SimpleNamespace

import pytest

from agent.conversation_compression import (
    CompressionCommitFence,
    _await_in_flight_commit,
    _await_worker_within_budget,
    _join_cancelled_worker,
)
from agent.tool_executor import _poll_sequential_future


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

    # Success race: the worker finishes in the window between ``result(timeout=)`` expiring and
    # the ``done()`` check. The guard must hand back the completed compression, not discard it
    # onto the stall/fallback path (which would cost a second LLM call).
    raced = _RacedSuccessFuture()
    raced.set_result("summary")
    fence = CompressionCommitFence()
    fence.touch_progress()
    started = time.monotonic()
    assert _await_worker_within_budget(raced, fence, idle=1.0, ceiling=2.0, wait_started=started) == (
        True,
        "summary",
    )


class _RacedSuccessFuture(concurrent.futures.Future):
    """Already-settled future whose FIRST timed ``result()`` still raises TimeoutError, modelling
    a worker that completed just after the wait slice expired."""

    def __init__(self):
        super().__init__()
        self._timed_out_once = False

    def result(self, timeout=None):
        if timeout is not None and not self._timed_out_once:
            self._timed_out_once = True
            raise concurrent.futures.TimeoutError()
        return super().result(timeout)


def test_in_flight_commit_surfaces_worker_timeout_instead_of_looping():
    """_await_in_flight_commit has no ceiling on this path — pre-fix a dead worker spun forever."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = _settled_dead_future(pool, TimeoutError("commit stream stalled"))
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="commit stream stalled"):
            _await_in_flight_commit(future, ceiling=1.0, wait_started=started, on_commit_overrun=None)
        elapsed = time.monotonic() - started

    assert elapsed < 0.5, f"commit wait hung {elapsed:.2f}s on a dead worker"


def test_sequential_tool_poll_surfaces_worker_timeout_instead_of_looping():
    """_poll_sequential_future re-waits on the future after each slice; without the done() guard
    a worker that died with TimeoutError kept it spinning until the deadline. With a 2s deadline
    on base this returns ("timeout", None) after 2s — post-fix the worker's error propagates at
    once."""
    agent = SimpleNamespace(_interrupt_requested=False, _touch_activity=lambda msg: None)
    gate = SimpleNamespace(excluded_seconds=lambda: 0.0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = _settled_dead_future(pool, TimeoutError("tool stream stalled"))
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="tool stream stalled"):
            _poll_sequential_future(agent, future, "some_tool", deadline=started + 2.0, started=started, authorization_gate=gate)
        elapsed = time.monotonic() - started

    assert elapsed < 0.5, f"sequential poll hung {elapsed:.2f}s on a dead worker"
