"""Off-loop ACP session construction (#78205).

Building an ``AIAgent`` is slow and fully blocking. When ``session/new`` called
``SessionManager.create_session`` straight from a coroutine, that build ran on
the event loop serving every JSON-RPC request, so the host saw an agent that
initialized, accepted requests, and answered none, with no log output either.

These tests pin the three properties the async path has to hold: the loop stays
free during a build, concurrent requests for one session id build a single
agent, and a caller that gives up does not destroy the build others are waiting
on.
"""

import asyncio
import threading
import time

import pytest

from acp_adapter.session import SessionManager, SessionState

BUILD_SECONDS = 0.4


class SlowAgent:
    """Stand-in for AIAgent whose construction blocks, like memory-provider init."""

    def __init__(self, session_id, build_log):
        time.sleep(BUILD_SECONDS)
        build_log.append(session_id)
        self.session_id = session_id
        self.model = "test-model"
        self.messages = []


@pytest.fixture
def build_log():
    return []


@pytest.fixture
def manager(build_log):
    def agent_factory(**kwargs):
        return SlowAgent(kwargs.get("session_id"), build_log)

    return SessionManager(agent_factory=agent_factory, db=None)


def _restoring(manager, build_log):
    """Point ``_restore`` at a real (slow) agent build, as a DB restore does."""

    def fake_restore(session_id):
        agent = manager._make_agent(session_id=session_id, cwd=".")
        state = SessionState(
            session_id=session_id,
            agent=agent,
            cwd=".",
            model="test-model",
            cancel_event=threading.Event(),
        )
        with manager._lock:
            manager._sessions[session_id] = state
        return state

    manager._restore = fake_restore
    return manager


async def _heartbeat(stop_at, ticks):
    while time.monotonic() < stop_at:
        ticks.append(time.monotonic())
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_create_session_async_keeps_the_loop_responsive(manager):
    """A slow build must not stall other handlers on the same loop."""
    ticks: list[float] = []
    task = asyncio.create_task(
        _heartbeat(time.monotonic() + BUILD_SECONDS + 0.3, ticks)
    )
    await asyncio.sleep(0.05)

    before = len(ticks)
    state = await manager.create_session_async(cwd=".")
    during = len(ticks) - before
    await task

    assert state.session_id
    assert during > 5, (
        f"loop served only {during} heartbeats during a {BUILD_SECONDS}s build — "
        "session construction is still running on the event loop"
    )
    worst = max((b - a for a, b in zip(ticks, ticks[1:])), default=0.0)
    assert worst < BUILD_SECONDS / 2, f"loop stalled {worst:.2f}s during the build"


@pytest.mark.asyncio
async def test_concurrent_restore_builds_one_agent(manager, build_log):
    """Five lifecycle requests for one absent session must not build five agents.

    The blocked loop used to serialize these by accident. Off-loop, each caller
    would otherwise build its own agent and the last writer would win, leaving
    live agents nobody can reach and callers holding divergent state.
    """
    _restoring(manager, build_log)

    results = await asyncio.gather(
        *[manager.get_session_async("shared-id") for _ in range(5)]
    )

    assert len(build_log) == 1, f"built {len(build_log)} agents for one session id"
    assert len({id(r) for r in results}) == 1, "callers received divergent state objects"
    assert all(r is not None for r in results)


@pytest.mark.asyncio
async def test_cancelled_waiter_leaves_the_build_intact(manager, build_log):
    """A host that times out and retries must not destroy the shared build.

    Cancelling a waiter cancels only that wait. The owner finishes and publishes
    the session, so a retry finds it instead of starting a second build.
    """
    _restoring(manager, build_log)

    owner = asyncio.create_task(manager.get_session_async("cancel-id"))
    await asyncio.sleep(0.05)
    waiter = asyncio.create_task(manager.get_session_async("cancel-id"))
    await asyncio.sleep(0.05)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    state = await owner
    assert state is not None, "cancelling a waiter killed the owner's build"
    assert len(build_log) == 1
    assert manager.get_session("cancel-id") is not None


@pytest.mark.asyncio
async def test_single_flight_map_is_emptied(manager, build_log):
    """The in-flight map must not leak entries, on success or failure."""
    _restoring(manager, build_log)
    await manager.get_session_async("done-id")
    assert manager._building == {}

    def boom(session_id):
        raise RuntimeError("restore failed")

    manager._restore = boom
    with pytest.raises(RuntimeError):
        await manager.get_session_async("boom-id")
    assert manager._building == {}


@pytest.mark.asyncio
async def test_memory_hit_does_not_go_to_a_thread(manager):
    """An already-loaded session is answered inline, with no build."""
    state = await manager.create_session_async(cwd=".")
    again = await manager.get_session_async(state.session_id)
    assert again is state
