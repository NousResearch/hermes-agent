"""Best-effort session housekeeping must not be able to starve agent turn bodies.

``_finalize_session_off_loop``, ``_cleanup_agent_resources_off_loop`` and
``_cleanup_old_agent_for_reset`` each bound their await with ``asyncio.wait_for`` and, on
timeout, log "the worker thread is left to finish on its own" and proceed.

``asyncio.wait_for`` bounds the AWAIT but not the OCCUPANCY: a ``concurrent.futures`` work item
that has already begun executing is not cancellable, so the abandoned worker keeps its pool slot
until its blocking call returns. While those callers shared the turn pool, N abandonments retired
N turn slots for ANY N — the failure is scale-invariant, so raising ``max_workers`` does not fix
it (pinned below). A saturated pool then delays the turn body of every subsequent message,
including automatic session resumes, with nothing in the logs naming the wait.

These tests bind the REAL runner methods onto a minimal object holding only the attributes they
read, so they cannot pass by mirroring the implementation.
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import logging
import threading
import time
import types
from pathlib import Path

import pytest

from gateway.run import GatewayRunner


def _runner(cleanup=None, *, cleanup_timeout=1.0):
    obj = types.SimpleNamespace()
    obj._executor_lock = threading.Lock()
    obj._executor = None
    obj._housekeeping_executor = None
    obj._executor_closing = False
    obj._CLEANUP_TIMEOUT_S = cleanup_timeout
    obj._FINALIZE_TIMEOUT_S = cleanup_timeout
    if cleanup is not None:
        obj._cleanup_agent_resources = cleanup
    for name in (
        "_get_executor",
        "_get_housekeeping_executor",
        "_submit_with_context",
        "_run_in_executor_with_context",
        "_run_housekeeping_in_executor",
        "_shutdown_executor",
    ):
        setattr(obj, name, types.MethodType(getattr(GatewayRunner, name), obj))
    return obj


def _stop(runner):
    for attr in ("_executor", "_housekeeping_executor"):
        pool = getattr(runner, attr, None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


async def _abandon_housekeeping(runner, count, timeout):
    """Submit ``count`` wedged housekeeping items, abandoning each like the real callers do."""

    async def one():
        try:
            await asyncio.wait_for(
                runner._run_housekeeping_in_executor(
                    "cleanup", runner._cleanup_agent_resources, object()
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            pass  # exactly what the production callers do

    await asyncio.gather(*(one() for _ in range(count)))


def test_housekeeping_and_turns_use_separate_pools():
    runner = _runner()
    try:
        assert runner._get_executor() is not runner._get_housekeeping_executor()
    finally:
        _stop(runner)


def test_abandoned_housekeeping_cannot_delay_a_turn_body():
    """The regression pin: wedged, abandoned housekeeping must leave turn slots free."""
    wedge = threading.Event()
    entered = threading.Semaphore(0)

    def wedged_cleanup(agent):
        entered.release()
        assert wedge.wait(60), "wedge never released"

    runner = _runner(wedged_cleanup)
    turn_pool = runner._get_executor()
    hk_pool = runner._get_housekeeping_executor()
    hk_workers = hk_pool._max_workers
    # More than the housekeeping pool holds, so it is saturated AND backlogged.
    abandoned = turn_pool._max_workers + hk_workers

    async def exercise():
        await _abandon_housekeeping(runner, abandoned, runner._CLEANUP_TIMEOUT_S)
        for _ in range(hk_workers):
            assert await asyncio.to_thread(entered.acquire, True, 30)

        assert len(hk_pool._threads or ()) == hk_workers
        assert hk_pool._work_queue.qsize() > 0
        # ...and it consumed nothing from the turn pool.
        assert len(turn_pool._threads or ()) == 0

        started: dict[str, float] = {}

        def turn_body():
            started["at"] = time.monotonic()

        submitted = time.monotonic()
        await asyncio.wait_for(
            runner._run_in_executor_with_context(turn_body), timeout=30
        )
        return started["at"] - submitted

    try:
        latency = asyncio.run(exercise())
    finally:
        wedge.set()
        _stop(runner)

    assert latency < 5.0, f"turn body waited {latency:.2f}s behind abandoned housekeeping"


def test_turn_pool_still_queues_a_full_pool():
    """Turn-vs-turn concurrency is unchanged; only housekeeping was moved off."""
    release = threading.Event()
    occupied = threading.Semaphore(0)
    runner = _runner()
    max_workers = runner._get_executor()._max_workers

    def long_turn():
        occupied.release()
        assert release.wait(60)

    async def exercise():
        turns = [
            asyncio.ensure_future(runner._run_in_executor_with_context(long_turn))
            for _ in range(max_workers)
        ]
        for _ in range(max_workers):
            assert await asyncio.to_thread(occupied.acquire, True, 30)
        started = threading.Event()
        queued = asyncio.ensure_future(
            runner._run_in_executor_with_context(started.set)
        )
        await asyncio.sleep(0.5)
        was_queued = not started.is_set()
        release.set()
        await asyncio.gather(queued, *turns)
        return was_queued

    try:
        assert asyncio.run(exercise()), "a full turn pool must queue the next turn"
    finally:
        release.set()
        _stop(runner)


def test_executor_wait_logs_pool_depth(caplog, monkeypatch):
    """A queued submit must say so, naming pool, key, wait and saturation."""
    monkeypatch.setenv("HERMES_GATEWAY_EXECUTOR_WAIT_WARN", "0.2")
    monkeypatch.setenv("HERMES_GATEWAY_EXECUTOR_MAX_WORKERS", "1")
    release = threading.Event()
    occupied = threading.Semaphore(0)
    runner = _runner()

    def long_turn():
        occupied.release()
        assert release.wait(60)

    def queued_turn():
        return "ran"

    async def exercise():
        first = asyncio.ensure_future(runner._run_in_executor_with_context(long_turn))
        assert await asyncio.to_thread(occupied.acquire, True, 30)
        second = asyncio.ensure_future(
            runner._run_in_executor_with_context(queued_turn)
        )
        await asyncio.sleep(0.5)
        release.set()
        await asyncio.gather(first, second)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        try:
            asyncio.run(exercise())
        finally:
            release.set()
            _stop(runner)

    waits = [
        r.getMessage() for r in caplog.records if "PHASE=executor_wait" in r.getMessage()
    ]
    assert waits, "no executor_wait line emitted"
    assert "pool=turn" in waits[0]
    assert "key=queued_turn" in waits[0]
    assert "max_workers=1" in waits[0]


def test_fast_submit_is_not_logged(caplog, monkeypatch):
    """The line must be a saturation signal, not per-submit noise."""
    monkeypatch.setenv("HERMES_GATEWAY_EXECUTOR_WAIT_WARN", "5")
    runner = _runner()
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        try:
            asyncio.run(runner._run_in_executor_with_context(lambda: "quick"))
        finally:
            _stop(runner)
    assert not [r for r in caplog.records if "PHASE=executor_wait" in r.getMessage()]


def test_housekeeping_backlog_is_attributed_to_its_own_pool(caplog, monkeypatch):
    """A housekeeping backlog must be distinguishable from a turn backlog."""
    monkeypatch.setenv("HERMES_GATEWAY_EXECUTOR_WAIT_WARN", "0.2")
    monkeypatch.setenv("HERMES_GATEWAY_HOUSEKEEPING_MAX_WORKERS", "1")
    wedge = threading.Event()
    entered = threading.Semaphore(0)

    def wedged_cleanup(agent):
        entered.release()
        assert wedge.wait(60)

    runner = _runner(wedged_cleanup, cleanup_timeout=0.3)

    async def exercise():
        await _abandon_housekeeping(runner, 1, 0.3)
        assert await asyncio.to_thread(entered.acquire, True, 30)
        second = asyncio.ensure_future(
            runner._run_housekeeping_in_executor("cleanup", lambda: "hk")
        )
        await asyncio.sleep(0.5)
        wedge.set()
        await second

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        try:
            asyncio.run(exercise())
        finally:
            wedge.set()
            _stop(runner)

    waits = [
        r.getMessage() for r in caplog.records if "PHASE=executor_wait" in r.getMessage()
    ]
    assert waits, "housekeeping backlog emitted no executor_wait line"
    assert any("pool=cleanup" in line for line in waits), waits


def test_shutdown_stops_the_housekeeping_pool():
    """Non-daemon housekeeping workers must not survive shutdown.

    concurrent.futures' atexit hook joins non-daemon workers; a pool left running would strand
    the process "down but not exited".
    """
    runner = _runner()
    hk = runner._get_housekeeping_executor()
    assert not hk._shutdown

    runner._shutdown_executor()

    assert hk._shutdown, "housekeeping pool was left running at shutdown"
    assert runner._housekeeping_executor is None
    with pytest.raises(RuntimeError):
        runner._get_housekeeping_executor()


def test_get_executor_works_on_a_bare_duck_typed_double():
    """_get_executor is called UNBOUND against minimal doubles; keep it self-contained.

    Existing suites do ``GatewayRunner._get_executor(fake)`` where ``fake`` implements only a
    handful of attributes. Routing the body through a sibling METHOD broke those doubles with
    AttributeError, so the get-or-create helper is a module-level function and this pins it.
    """
    double = types.SimpleNamespace(_executor=None, _executor_closing=False)
    pool = GatewayRunner._get_executor(double)
    try:
        assert isinstance(pool, concurrent.futures.ThreadPoolExecutor)
        assert double._executor is pool
        # Same for the housekeeping pool, on a double that has never heard of it.
        hk_double = types.SimpleNamespace(_executor_closing=False)
        hk = GatewayRunner._get_housekeeping_executor(hk_double)
        try:
            assert hk is not pool
        finally:
            hk.shutdown(wait=False)
    finally:
        pool.shutdown(wait=False)


def test_pool_sizes_are_configurable(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_EXECUTOR_MAX_WORKERS", "17")
    monkeypatch.setenv("HERMES_GATEWAY_HOUSEKEEPING_MAX_WORKERS", "3")
    runner = _runner()
    try:
        assert runner._get_executor()._max_workers == 17
        assert runner._get_housekeeping_executor()._max_workers == 3
    finally:
        _stop(runner)


@pytest.mark.parametrize("bad", ["0", "-4", "not-a-number", ""])
def test_invalid_pool_size_falls_back_to_default(monkeypatch, bad):
    """A typo in an ops knob must not create a zero-width or crashing pool."""
    monkeypatch.setenv("HERMES_GATEWAY_EXECUTOR_MAX_WORKERS", bad)
    runner = _runner()
    try:
        pool = runner._get_executor()
        assert isinstance(pool, concurrent.futures.ThreadPoolExecutor)
        assert pool._max_workers == 10
    finally:
        _stop(runner)


def test_contextvars_survive_both_pools():
    """The context hop is why these helpers exist; it must not regress."""
    from contextvars import ContextVar

    probe: ContextVar[str] = ContextVar("probe", default="unset")
    runner = _runner()

    async def exercise():
        probe.set("scoped")
        turn = await runner._run_in_executor_with_context(probe.get)
        hk = await runner._run_housekeeping_in_executor("cleanup", probe.get)
        return turn, hk

    try:
        assert asyncio.run(exercise()) == ("scoped", "scoped")
    finally:
        _stop(runner)


def test_no_abandonment_site_uses_the_turn_pool():
    """Class gate: wait_for must never abandon work on the turn pool."""
    gateway_dir = Path(__file__).resolve().parents[2] / "gateway"
    offenders = []

    for path in sorted(gateway_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "wait_for"
            ):
                continue
            submitted = node.args[0]
            if (
                isinstance(submitted, ast.Call)
                and isinstance(submitted.func, ast.Attribute)
                and submitted.func.attr == "_run_in_executor_with_context"
            ):
                offenders.append(f"{path.relative_to(gateway_dir.parent)}:{node.lineno}")

    assert offenders == [], (
        "wait_for abandons started executor workers; route these best-effort "
        f"calls through _run_housekeeping_in_executor instead: {offenders}"
    )


def test_args_results_and_errors_round_trip():
    """The timing wrapper must be transparent to args, results and exceptions."""
    runner = _runner()

    async def exercise():
        assert await runner._run_in_executor_with_context(lambda a, b: a + b, 2, 3) == 5
        with pytest.raises(ValueError, match="boom"):
            await runner._run_housekeeping_in_executor("cleanup", _raise_boom)

    try:
        asyncio.run(exercise())
    finally:
        _stop(runner)


def _raise_boom():
    raise ValueError("boom")
