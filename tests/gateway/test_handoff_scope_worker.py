"""A watcher tick must never start a thread on the event loop.

``_resolve_handoff_watch_scopes`` hands the multiplex scope walk to an executor, but
``loop.run_in_executor`` calls ``executor.submit()`` on the LOOP thread, and a pool that has to grow
at submit time starts a thread there. ``Thread.start()`` returns only once the new thread is running,
so on a host under memory-commit pressure it holds the loop past the three missed liveness probes:
the shutdown watchdog then exits the gateway with code 75, killing every Kanban worker mid-flight
(2026-10-31 01:00 incident, stack ``_resolve_handoff_watch_scopes`` -> ``_run_in_executor_with_context``
-> ``turn_executor.submit`` -> ``Thread.start``).

``gateway.handoff_scope`` keeps one worker parked, so a tick is a queue put. These tests pin the two
halves of that invariant: the tick starts no thread on the loop, and the loop keeps serving callbacks
(the liveness probe's job) while a slow walk is in progress.
"""

import asyncio
import contextvars
import logging
import threading
import types

import pytest

from gateway import handoff_scope, run
from gateway.turn_executor import _UnboundedThreadExecutor


class _Multiplex:
    multiplex_profiles = True


def _production_hop(pool):
    """The hop a real runner's ``_run_in_executor_with_context`` is: loop.run_in_executor over a pool."""
    async def _hop(func, *args):
        ctx = contextvars.copy_context()
        return await asyncio.get_running_loop().run_in_executor(
            pool, lambda: ctx.run(func, *args))

    return _hop


@pytest.mark.asyncio
async def test_watcher_tick_starts_no_thread_on_the_loop(monkeypatch):
    """After the boot warm-up, a tick must not call Thread.start() from the loop thread."""
    starts: list[str] = []
    real_start = threading.Thread.start

    def _record_start(thread):
        starts.append(threading.current_thread().name)
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", _record_start)
    # Boot starts the scope worker off the loop; every later submit is a queue put.
    assert await asyncio.to_thread(handoff_scope.warm)
    starts.clear()

    walked_on: list[str] = []

    def _scopes(_runner):
        walked_on.append(threading.current_thread().name)
        return [(None, None)]

    monkeypatch.setattr(run, "_handoff_watch_scopes", _scopes)
    # A cold turn pool is the production shape on a booting gateway and after its 60 s idle timeout.
    pool = _UnboundedThreadExecutor(thread_name_prefix="hermes-gateway")
    try:
        runner = types.SimpleNamespace(
            config=_Multiplex(), _run_in_executor_with_context=_production_hop(pool))
        scopes = await asyncio.wait_for(run._resolve_handoff_watch_scopes(runner), timeout=10)
    finally:
        pool.shutdown(wait=False)

    assert scopes == [(None, None)]
    assert starts == [], f"a watcher tick started a thread on the loop thread: {starts}"
    assert walked_on and walked_on[0].startswith("hermes-handoff-scope"), (
        f"the scope walk ran on {walked_on}, not on the warm scope worker")


@pytest.mark.asyncio
async def test_probe_tick_fires_while_scopes_resolve(monkeypatch):
    """The loop keeps running its callbacks while a slow scope walk is in progress.

    The liveness probe is one such callback: if the tick held the loop, the probe below could not fire
    and the walk's bounded wait on it would expire — a failure, never a hang.
    """
    walk_started = threading.Event()
    probe_tick = threading.Event()
    ticked_while_walking: list[bool] = []
    ticks = [0]

    def _slow_scopes(_runner):
        walk_started.set()
        ticked_while_walking.append(probe_tick.wait(timeout=5))
        return [(None, None)]

    monkeypatch.setattr(run, "_handoff_watch_scopes", _slow_scopes)
    assert await asyncio.to_thread(handoff_scope.warm)

    async def _probe():
        while True:
            ticks[0] += 1
            probe_tick.set()
            await asyncio.sleep(0)

    probe = asyncio.ensure_future(_probe())
    pool = _UnboundedThreadExecutor(thread_name_prefix="hermes-gateway")
    try:
        runner = types.SimpleNamespace(
            config=_Multiplex(), _run_in_executor_with_context=_production_hop(pool))
        scopes = await asyncio.wait_for(run._resolve_handoff_watch_scopes(runner), timeout=10)
    finally:
        probe.cancel()
        pool.shutdown(wait=False)

    assert scopes == [(None, None)]
    assert walk_started.is_set()
    assert ticked_while_walking == [True], "the loop stopped ticking while the scope walk ran"
    assert ticks[0] > 1, "the probe ran once and was starved while the walk resolved"


@pytest.mark.asyncio
async def test_stalled_scope_walk_warns_and_still_returns(monkeypatch, caplog):
    """A walk past the stall threshold logs one WARNING and is still awaited to its own result.

    The watchdog is log-only: it must neither cancel the walk nor change what the tick returns.
    """
    monkeypatch.setattr(handoff_scope, "_STALL_WARN_S", 0.05)
    release = threading.Event()

    def _wedged_scopes(_runner):
        release.wait(timeout=5)
        return [("profile-a", "/homes/a")]

    monkeypatch.setattr(run, "_handoff_watch_scopes", _wedged_scopes)
    assert await asyncio.to_thread(handoff_scope.warm)

    pool = _UnboundedThreadExecutor(thread_name_prefix="hermes-gateway")
    try:
        runner = types.SimpleNamespace(
            config=_Multiplex(), _run_in_executor_with_context=_production_hop(pool))
        with caplog.at_level(logging.WARNING, logger=handoff_scope.__name__):
            tick = asyncio.ensure_future(run._resolve_handoff_watch_scopes(runner))
            deadline = asyncio.get_running_loop().time() + 5
            while not caplog.records and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert not tick.done(), "the stall watchdog ended the tick instead of only warning"
            release.set()
            scopes = await asyncio.wait_for(tick, timeout=10)
    finally:
        release.set()
        pool.shutdown(wait=False)

    assert scopes == [("profile-a", "/homes/a")]
    stalls = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(stalls) == 1, [r.getMessage() for r in caplog.records]
    assert "_wedged_scopes" in stalls[0].getMessage()
    assert "still running after" in stalls[0].getMessage()


@pytest.mark.asyncio
async def test_fast_scope_walk_does_not_warn(monkeypatch, caplog):
    """A walk that beats the threshold leaves no stall warning behind: the timer is cancelled."""
    monkeypatch.setattr(handoff_scope, "_STALL_WARN_S", 0.2)
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _runner: [(None, None)])
    assert await asyncio.to_thread(handoff_scope.warm)

    pool = _UnboundedThreadExecutor(thread_name_prefix="hermes-gateway")
    try:
        runner = types.SimpleNamespace(
            config=_Multiplex(), _run_in_executor_with_context=_production_hop(pool))
        with caplog.at_level(logging.WARNING, logger=handoff_scope.__name__):
            assert await run._resolve_handoff_watch_scopes(runner) == [(None, None)]
            await asyncio.sleep(0.3)
    finally:
        pool.shutdown(wait=False)

    assert not caplog.records, [r.getMessage() for r in caplog.records]
