"""Best-effort session housekeeping must not be able to starve agent turn bodies.

``_finalize_session_off_loop``, ``_cleanup_agent_resources_off_loop`` and
``_cleanup_old_agent_for_reset`` each bound their await with ``asyncio.wait_for`` and, on
timeout, log "the worker thread is left to finish on its own" and proceed.

``asyncio.wait_for`` bounds the AWAIT but not the OCCUPANCY: a ``concurrent.futures`` work item
that has already begun executing is not cancellable, so the abandoned worker keeps its pool slot
until its blocking call returns. While those callers shared the turn pool, N abandonments retired
N turn slots for ANY N, and a saturated pool then delayed the turn body of every subsequent
message with nothing in the logs naming the wait. The turn pool is now unbounded, so housekeeping
keeps its own bounded pool to cap its thread growth; the pin is that it never lands on the turn pool.

The runner here is a real ``GatewayRunner`` instance (``__new__``, no ``__init__``) carrying only
the executor attributes.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from gateway.run import GatewayRunner


def _runner(cleanup=None, *, cleanup_timeout=0.5):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._executor_lock = threading.Lock()
    runner._executor = None
    runner._housekeeping_executor = None
    runner._executor_closing = False
    runner._CLEANUP_TIMEOUT_S = cleanup_timeout
    if cleanup is not None:
        runner._cleanup_agent_resources = cleanup
    return runner


async def _abandon_housekeeping(runner, count, timeout):
    """Submit ``count`` wedged housekeeping items, abandoning each like the real callers do."""

    async def one():
        try:
            await asyncio.wait_for(
                runner._run_housekeeping_in_executor(runner._cleanup_agent_resources, object()),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            pass  # exactly what the production callers do

    await asyncio.gather(*(one() for _ in range(count)))


def test_abandoned_housekeeping_stays_off_the_turn_pool():
    """The regression pin: wedged, abandoned housekeeping occupies the housekeeping pool only."""
    wedge = threading.Event()
    ran_on: list[str] = []

    def wedged_cleanup(agent):
        ran_on.append(threading.current_thread().name)
        assert wedge.wait(60), "wedge never released"

    runner = _runner(wedged_cleanup)
    try:
        asyncio.run(_abandon_housekeeping(runner, 3, runner._CLEANUP_TIMEOUT_S))
        assert ran_on, "no housekeeping item started"
        assert all(name.startswith("hermes-gateway-hk") for name in ran_on), ran_on
        assert runner._executor is None, "housekeeping created or used the turn pool"
    finally:
        wedge.set()
        runner._shutdown_executor()


def test_shutdown_stops_the_housekeeping_pool():
    """A pool nobody drains must still be shut down, or its threads outlive the runner — and a
    wedged housekeeping worker must be counted as live, or the #101093 skip-close heuristic in
    ``_stop_quiesce_and_close_session_dbs`` closes SessionDB under a mid-write worker."""
    wedge = threading.Event()
    entered = threading.Event()

    def wedged(_agent):
        entered.set()
        assert wedge.wait(60), "wedge never released"

    runner = _runner()
    hk = runner._get_housekeeping_executor()
    assert not hk._shutdown
    hk.submit(wedged, object())
    assert entered.wait(5), "wedged housekeeping item never started"

    try:
        started = time.monotonic()
        live = runner._shutdown_executor(drain_timeout=0.2)
        elapsed = time.monotonic() - started
    finally:
        wedge.set()

    assert live == 1, f"wedged housekeeping worker not reported as live (got {live})"
    assert elapsed < 5, f"drain deadline not honoured for the housekeeping pool ({elapsed:.2f}s)"
    assert hk._shutdown, "housekeeping pool was left running at shutdown"
    assert runner._housekeeping_executor is None
    with pytest.raises(RuntimeError):
        runner._get_housekeeping_executor()
