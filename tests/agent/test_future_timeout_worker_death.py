"""A worker that DIES raising a timeout-class error must not be mistaken for a live one.

On Python 3.11+ ``concurrent.futures.TimeoutError`` IS the builtin ``TimeoutError``
(and ``asyncio.TimeoutError`` and ``socket.timeout`` alias it too). Every poll loop
shaped like::

    try:
        return future.result(timeout=slice)
    except concurrent.futures.TimeoutError:
        ...keep waiting...

therefore cannot tell "the wait slice expired" (worker alive) from "the worker
raised TimeoutError" (worker dead). The aux client raises bare ``TimeoutError``
when a summary stream stalls, so this is reachable in production: the host re-waits
on a settled future, each ``result()`` returns instantly, and the loop spins at
thousands of iterations/sec — logging "still streaming" about a corpse — until the
whole idle budget burns. That wedged a real session for 535s and flooded the agent
log with ~90k duplicate lines.

The guard is ``if future.done()``: a settled future never becomes unsettled.
"""

import concurrent.futures
import time

import pytest


def test_timeout_error_aliases_are_indistinguishable():
    """The premise of the bug — if this ever stops holding, the guards can be revisited."""
    import asyncio
    import socket

    assert concurrent.futures.TimeoutError is TimeoutError
    assert asyncio.TimeoutError is TimeoutError
    assert socket.timeout is TimeoutError


@pytest.mark.parametrize(
    "exc",
    [TimeoutError("aux stream stalled"), __import__("asyncio").TimeoutError("aux stream stalled")],
    ids=["builtin-TimeoutError", "asyncio-TimeoutError"],
)
def test_compression_wait_returns_promptly_when_worker_dies(exc):
    """_await_worker_within_budget must take the stall path at once, not burn the idle budget."""
    from agent.conversation_compression import (
        CompressionCommitFence,
        _await_worker_within_budget,
        _join_cancelled_worker,
    )

    def worker():
        raise exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        concurrent.futures.wait([future], timeout=5)
        fence = CompressionCommitFence()
        fence.touch_progress()
        started = time.monotonic()
        settled, result = _await_worker_within_budget(
            future, fence, idle=30.0, ceiling=60.0, wait_started=started
        )
        elapsed = time.monotonic() - started

    # Pre-fix this consumed the full 30s idle window in a hot spin.
    assert elapsed < 5.0, f"host waited {elapsed:.1f}s on an already-dead worker"
    assert settled is False
    assert result is None
    # Sibling guard: the teardown join must report the dead worker as EXITED (its lease is then
    # released) rather than as a still-running orphan.
    assert _join_cancelled_worker(future, 0.5) is True


def test_compression_wait_still_polls_a_live_worker():
    """The guard must not collapse the normal case: an unfinished future keeps waiting."""
    from agent.conversation_compression import (
        CompressionCommitFence,
        _await_worker_within_budget,
    )

    release = __import__("threading").Event()

    def worker():
        release.wait(timeout=10)
        return "compressed"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        fence = CompressionCommitFence()

        def keep_alive():
            for _ in range(40):
                fence.touch_progress()
                time.sleep(0.01)
            release.set()

        __import__("threading").Thread(target=keep_alive, daemon=True).start()
        settled, result = _await_worker_within_budget(
            future, fence, idle=5.0, ceiling=10.0, wait_started=time.monotonic()
        )

    assert settled is True
    assert result == "compressed"


def test_compression_wait_propagates_non_timeout_worker_errors():
    """A non-timeout worker exception must still surface, exactly as before the fix."""
    from agent.conversation_compression import (
        CompressionCommitFence,
        _await_worker_within_budget,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: (_ for _ in ()).throw(ValueError("boom")))
        concurrent.futures.wait([future], timeout=5)
        fence = CompressionCommitFence()
        fence.touch_progress()
        with pytest.raises(ValueError, match="boom"):
            _await_worker_within_budget(
                future, fence, idle=5.0, ceiling=10.0, wait_started=time.monotonic()
            )


def test_in_flight_commit_surfaces_worker_timeout_instead_of_looping():
    """_await_in_flight_commit had no ceiling on this path — a dead worker spun forever."""
    from agent.conversation_compression import _await_in_flight_commit

    def worker():
        raise TimeoutError("commit stream stalled")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        concurrent.futures.wait([future], timeout=5)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            _await_in_flight_commit(
                future, ceiling=30.0, wait_started=started, on_commit_overrun=None
            )
        elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"commit wait hung {elapsed:.1f}s on a dead worker"


def test_sequential_tool_poll_surfaces_worker_timeout():
    """_poll_sequential_future with deadline=None would otherwise spin indefinitely."""
    from unittest.mock import MagicMock

    from agent.tool_executor import _poll_sequential_future

    agent = MagicMock()
    agent._interrupt_requested = False

    gate = MagicMock()
    gate.excluded_seconds = MagicMock(return_value=0.0)

    def worker():
        raise TimeoutError("tool call timed out")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        concurrent.futures.wait([future], timeout=5)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            _poll_sequential_future(agent, future, "some_tool", None, started, gate)
        elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"tool poll hung {elapsed:.1f}s on a dead worker"
