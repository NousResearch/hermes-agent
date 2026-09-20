"""Durability and generation ownership of requester restart notices (PR #70859)."""

import asyncio
import json
import threading
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
import gateway.run_restart_notifications as notices
import gateway.slash_commands as commands
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source
from utils import atomic_json_write


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["missing", "degraded", "disconnected", "refused", "exhausted", "ambiguous", "timeout", "cancelled"])
async def test_restart_notice_retries_only_known_unsent_and_keeps_exhausted_work(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(notices, "_RESTART_NOTICE_TIMEOUT", 0.1, raising=False)
    path = tmp_path / ".restart_notify.json"
    atomic_json_write(path, {"platform": "telegram", "chat_id": "42", "thread_id": "77", "request_id": "boot"})
    payload = path.read_text(encoding="utf-8")
    runner, adapter = make_restart_runner()
    calls, waits = [], []
    entered, release = asyncio.Event(), asyncio.Event()
    degraded = outcome == "degraded"
    monkeypatch.setattr(type(adapter), "send_path_degraded", property(lambda _: degraded))
    if outcome == "missing":
        runner.adapters = {}
    if outcome == "disconnected":
        adapter._running = False

    async def send(chat_id, content, **kwargs):
        calls.append((chat_id, content, kwargs))
        entered.set()
        if outcome in {"timeout", "cancelled"}:
            try:
                await release.wait()
            except asyncio.CancelledError:
                # A provider may resist cancellation; the caller must still release its gate.
                await release.wait()
        if outcome == "ambiguous":
            raise ConnectionError("read timed out after request was transmitted")
        if outcome == "exhausted" or (outcome == "refused" and len(calls) == 1):
            return SendResult(success=False, retryable=True, known_unsent=True, retry_after=60 if outcome == "exhausted" else 0.007)
        return SendResult(success=True, message_id="delivered")

    adapter.send = send

    async def sleep(delay):
        nonlocal degraded
        waits.append(delay)
        runner.adapters = {adapter.platform: adapter}
        adapter._running = True
        degraded = False
        await asyncio.sleep(delay if outcome == "exhausted" else 0)

    # Local clock/sleep seam only: do not monkeypatch asyncio for the gateway/test runner.
    class AsyncioProxy:
        def __getattr__(self, name):
            return sleep if name == "sleep" else getattr(asyncio, name)
    monkeypatch.setattr(notices, "asyncio", AsyncioProxy(), raising=False)
    task = asyncio.create_task(runner._send_restart_notification())
    try:
        if outcome == "cancelled":
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            target = None
        else:
            target = await asyncio.wait_for(task, 3)
        if outcome == "exhausted":
            assert path.read_text(encoding="utf-8") == payload
            assert len(calls) == 1  # retry_after exceeds the budget: never retry early
            fresh, fresh_adapter = make_restart_runner()
            assert await fresh._send_restart_notification() == ("telegram", "42", "77")
            assert len(fresh_adapter.sent) == 1
        else:
            assert target == (("telegram", "42", "77") if outcome in {"missing", "degraded", "disconnected", "refused"} else None)
            assert len(calls) == (2 if outcome == "refused" else 1)
        assert not path.exists()
        assert await runner._send_restart_notification() is None
        if outcome == "refused":
            assert waits == [0.007]
        if outcome in {"missing", "degraded", "disconnected"}:
            assert waits  # no send while transport was absent/unproven
        assert calls[-1][2]["metadata"]["thread_id"] == "77"
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["publish_cancel", "boot_replaced", "boot_empty", "send_replaced"])
async def test_restart_publication_and_cleanup_belong_to_exact_generation(tmp_path, monkeypatch, race):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    path = tmp_path / ".restart_notify.json"
    runner, adapter = make_restart_runner()
    requested, requested_by = [], []

    def request(**kwargs):
        requested_by.append(asyncio.current_task())
        requested.append(json.loads(path.read_text(encoding="utf-8")))
        runner._restart_requested = True
        return True

    runner.request_restart = request
    event = MessageEvent(text="/restart", source=make_restart_source(chat_id="new"), message_id="m-new")
    if race == "publish_cancel":
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def write(path, data, **kwargs):
            if path.name == ".restart_notify.json":
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5), "test failed to release publisher"
            atomic_json_write(path, data, **kwargs)

        monkeypatch.setattr(commands, "atomic_json_write", write)
        first = asyncio.create_task(runner._handle_restart_command(event))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()  # repeated cancellation cannot release the publication lock
            competing = MessageEvent(text="/restart", source=make_restart_source(chat_id="competing"), message_id="m-other")
            second = asyncio.create_task(runner._handle_restart_command(competing))
            await asyncio.sleep(0)
        finally:
            release.set()
        results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 3)
        assert isinstance(results[0], asyncio.CancelledError)
        assert requested_by == [first]  # publication's owner must drive restart even after cancellation
        assert len(requested) == 1
        assert requested[0]["chat_id"] == "new"  # cancelled first writer, not the competing request
        assert requested[0]["request_id"]
        assert json.loads(path.read_text(encoding="utf-8")) == requested[0]
        assert (tmp_path / ".restart_last_processed.json").exists()
        # Completing a cancelled publication still requested restart, not a false success marker.
        fresh, fresh_adapter = make_restart_runner()
        assert await fresh._send_restart_notification() == ("telegram", "new", None)
        assert len(fresh_adapter.sent) == 1
        return

    if race != "boot_empty":
        atomic_json_write(path, {"platform": "telegram", "chat_id": "old", "request_id": "old"})

    class BootPaused(Exception):
        pass

    def pause_boot():
        raise BootPaused

    # Exercise the real start() entry, stopping before unrelated gateway/services are started.
    runner._start_install_faulthandler = pause_boot
    with pytest.raises(BootPaused):
        await runner.start()
    if race == "send_replaced":
        async def send(*args, **kwargs):
            await runner._handle_restart_command(event)
            return SendResult(success=True)
        adapter.send = AsyncMock(side_effect=send)
        assert await runner._send_restart_notification() == ("telegram", "old", None)
        adapter.send.assert_awaited_once()
    else:
        # Another process/request can replace the file without changing this runner's flags.
        atomic_json_write(path, {"platform": "telegram", "chat_id": "new", "request_id": "new"})
        assert await runner._send_restart_notification() is None
        assert not adapter.sent
    assert json.loads(path.read_text(encoding="utf-8"))["chat_id"] == "new"
    fresh, fresh_adapter = make_restart_runner()
    assert await fresh._send_restart_notification() == ("telegram", "new", None)
    assert len(fresh_adapter.sent) == 1
    assert not path.exists()
