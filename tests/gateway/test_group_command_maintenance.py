"""Group commands must not hide work from a maintenance drain."""

import asyncio
from dataclasses import replace
import threading

import pytest
import pytest_asyncio

from tests.gateway.test_group_home_consent import command, home


@pytest_asyncio.fixture
async def authorized_home(home):
    await command(home, "/group")
    await command(home, "/group confirm")
    home.runner._running_agents = {}
    return home


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["_external_drain_active", "_draining"])
@pytest.mark.parametrize("action", ["send hello", "retry"])
async def test_new_group_work_refused_during_maintenance(authorized_home, flag, action):
    state = authorized_home
    state.service.room_status["pending_actions"] = [{"kind": "retry", "task_id": "retry-me"}]
    setattr(state.runner, flag, True)

    result = await command(state, f"/group 1 {action}")

    assert "maintenance" in str(result).lower()
    assert state.service.sent == []
    assert state.service.retried == []
    assert state.runner._active_deferred_agent_worker_count() == 0


@pytest.mark.asyncio
async def test_existing_stop_still_available_during_drain(authorized_home):
    state = authorized_home
    state.runner._external_drain_active = True
    result = await command(state, "/group 1 stop")
    assert "Stop" in result
    assert state.service.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_admitted_group_worker_stays_counted_until_actual_exit(
    authorized_home, monkeypatch, cancel_waiter
):
    state = authorized_home
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = state.service.send

    def held_send(**kwargs):
        entered.set()
        try:
            if not release.wait(5):
                raise AssertionError("test worker was not released")
            return original(**kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(state.service, "send", held_send)
    event = replace(state.event, text="/group 1 send first", message_id="admitted-first")
    task = asyncio.create_task(state.runner._handle_rooms_command(event))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert state.runner._active_work_count() == 1
        state.runner._external_drain_active = True
        denied = await command(state, "/group 1 send second")
        assert "maintenance" in str(denied).lower()
        if cancel_waiter:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert state.runner._active_work_count() == 1
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        if not cancel_waiter:
            await task
        pending = tuple(getattr(state.runner, "_deferred_agent_workers", {}))
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 2)
        assert state.runner._active_work_count() == 0
        assert len(state.service.sent) == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if entered.is_set():
            assert await asyncio.to_thread(finished.wait, 2)
