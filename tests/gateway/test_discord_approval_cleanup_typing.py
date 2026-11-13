"""One session's typing cleanup must preserve another session's approval wait."""
import asyncio

import pytest

import gateway.platforms.base as base_module
from tests.gateway.test_discord_typing_pause import TypingClock, typing as _typing_fixture
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


typing_fixture = _typing_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["outer-finally", "turn-stop"])
async def test_cleanup_preserves_other_sessions_pending_pause(typing_fixture, monkeypatch, cleanup):
    adapter, request, inner_clock = typing_fixture
    outer_clock = TypingClock(2)
    monkeypatch.setattr(base_module, "asyncio", outer_clock)
    loop = asyncio.get_running_loop()
    keys = ["cleanup-user-a", "cleanup-user-b"]
    ready = [asyncio.Event(), asyncio.Event()]
    workers = []
    outer = []
    try:
        for key, event in zip(keys, ready):
            def notify(data, event=event):
                adapter.pause_typing_for_chat("77")
                loop.call_soon_threadsafe(event.set)
            approval.register_gateway_notify(key, notify, wait_group=(adapter, "77"))
            workers.append(asyncio.create_task(asyncio.to_thread(
                _await_gateway_decision, key, notify, {"command": key, "pattern_key": key})))
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), 5)
        resume = lambda: adapter.resume_typing_for_chat("77")
        assert approval.resolve_gateway_approval(keys[0], "deny", before_wake=resume) == 1
        await asyncio.wait_for(workers[0], 5)
        assert "77" in adapter._typing_paused
        outer.extend(asyncio.create_task(adapter._keep_typing("77")) for _ in keys)
        await outer_clock.parked()
        b_tick = await outer_clock.parked()
        if cleanup == "outer-finally":
            outer[0].cancel()
            await asyncio.gather(outer[0], return_exceptions=True)
        else:
            await adapter._stop_typing_refresh("77", outer[0])
        assert approval.has_blocking_approval(keys[1])
        assert "77" in adapter._typing_paused, "A cleanup erased B's pending approval pause"
        b_tick.set_result(None)
        await outer_clock.parked()
        await adapter.send_typing("77")
        request.assert_not_awaited()
        assert approval.resolve_gateway_approval(keys[1], "deny", before_wake=resume) == 1
        await asyncio.wait_for(workers[1], 5)
        assert "77" not in adapter._typing_paused
        await adapter.send_typing("77")
        await inner_clock.parked()
        assert request.await_count == 1
    finally:
        for key in keys:
            approval.unregister_gateway_notify(key)
        for task in outer:
            task.cancel()
        await asyncio.gather(*outer, return_exceptions=True)
        await asyncio.wait_for(asyncio.gather(*workers), 5)
        await adapter.stop_typing("77")
