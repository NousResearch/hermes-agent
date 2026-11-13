"""Foreground typing ownership is independent of retained background approval leases."""
import asyncio
from types import SimpleNamespace

import pytest

import gateway.run  # Bootstrap before worker deadlines and the per-test home guard.
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.helpers import typing_pause_key
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision
from tools.approval_notify_lease import acquire, run


@pytest.mark.asyncio
@pytest.mark.parametrize("background_first", [False, True])
async def test_background_wait_neither_holds_nor_releases_foreground_pause(background_first):
    key, chat = "typing-lease-owner", "77"
    paused = set()
    adapter = SimpleNamespace(_typing_paused=paused)
    loop = asyncio.get_running_loop()
    prompts = asyncio.Queue()
    wakes = []

    def foreground(data):
        paused.add(chat)
        loop.call_soon_threadsafe(prompts.put_nowait, data)

    def background(data):
        loop.call_soon_threadsafe(prompts.put_nowait, data)

    def resume():
        wakes.append("resume")
        paused.discard(chat)

    foreground.background_notify = background
    approval.register_gateway_notify(key, foreground, wait_group=(id(adapter), chat), before_wake=resume)
    lease = acquire(key)
    workers = []
    try:
        workers.append(asyncio.create_task(asyncio.to_thread(
            run, lease, lambda: _await_gateway_decision(key, lease.notify, {"command": "same"}))))
        bg = await asyncio.wait_for(prompts.get(), 5)
        assert not paused
        workers.append(asyncio.create_task(asyncio.to_thread(
            _await_gateway_decision, key, foreground, {"command": "same"})))
        fg = await asyncio.wait_for(prompts.get(), 5)
        assert bg["request_id"] != fg["request_id"], "different owners must not coalesce"
        assert paused == {chat}
        first, second = (bg, fg) if background_first else (fg, bg)
        assert approval.resolve_gateway_approval(key, "deny", request_id=first["request_id"], before_wake=resume) == 1
        assert paused == ({chat} if background_first else set())
        assert wakes == ([] if background_first else ["resume"])
        assert approval.resolve_gateway_approval(key, "deny", request_id=second["request_id"], before_wake=resume) == 1
        await asyncio.wait_for(asyncio.gather(*workers), 5)
        assert wakes == ["resume"]
        assert not paused
    finally:
        approval.resolve_gateway_approval(key, "deny", resolve_all=True)
        approval.unregister_gateway_notify(key)
        lease.release()
        await asyncio.wait_for(asyncio.gather(*workers), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("threaded", [False, True])
async def test_cleanup_clears_own_silence_but_keeps_other_sessions_approval(threaded):
    chat, key = "77", "other-session-typing-owner"
    metadata = {"thread_id": "own-topic"} if threaded else None
    own = typing_pause_key(chat, metadata)
    other = typing_pause_key(chat, {"thread_id": "other-topic"})

    class Adapter:
        _stop_typing_refresh = BasePlatformAdapter._stop_typing_refresh

        def __init__(self):
            self._typing_paused = {own, other}

        async def _stop_typing_quietly(self, *args):
            pass

    adapter = Adapter()
    ready = asyncio.Event()
    loop = asyncio.get_running_loop()

    def notify(data):
        adapter._typing_paused.add(chat)
        loop.call_soon_threadsafe(ready.set)

    approval.register_gateway_notify(key, notify, wait_group=(adapter, chat))
    worker = asyncio.create_task(asyncio.to_thread(_await_gateway_decision, key, notify, {"command": "other"}))
    try:
        await asyncio.wait_for(ready.wait(), 5)
        await adapter._stop_typing_refresh(chat, metadata=metadata, stop_attempts=1)
        assert chat in adapter._typing_paused
        assert other in adapter._typing_paused
        if threaded:
            assert own not in adapter._typing_paused
        approval.resolve_gateway_approval(key, "deny", before_wake=lambda: adapter._typing_paused.discard(chat))
        await asyncio.wait_for(worker, 5)
        assert adapter._typing_paused == {other}
    finally:
        approval.unregister_gateway_notify(key)
        await asyncio.wait_for(worker, 5)
