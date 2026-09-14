"""Real adapter admission is the completion acknowledgement boundary."""
import asyncio
import logging
import time

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from hermes_state import SessionDB
from plugins.platforms.discord.adapter import DiscordAdapter
from tools import async_delegation as delegation


def pending(key, name):
    evt = {"type": "async_delegation", "session_key": key, "delegation_id": name,
           "summary": name, "status": "completed", "dispatched_at": time.time()}
    delegation._persist_dispatch(evt)
    delegation._persist_completion(evt, {"status": "completed", "summary": name})
    return evt


async def drain(adapter):
    while adapter._background_tasks:
        await asyncio.gather(*list(adapter._background_tasks))


@pytest.mark.asyncio
async def test_completion_ack_requires_admission_and_replay_never_repeats(tmp_path):
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    source = SessionSource(platform=Platform.DISCORD, chat_type="dm", chat_id="42", user_id="42")
    key = build_session_key(source)
    events = [pending(key, f"admission-{i}") for i in range(2)]
    received = []
    release, started = asyncio.Event(), asyncio.Event()

    async def handler(event):
        received.append(event.text)
        started.set()
        await release.wait()
        if key not in adapter._pending_messages:
            queued = runner._promote_queued_event(key, adapter, None)
            if queued is not None:
                adapter._pending_messages[key] = queued

    try:
        # Missing handler must not acknowledge either durable sibling.
        for _ in range(10):
            assert await runner._deliver_async_delegation_group(events) is False
        for event in events:
            row = delegation.get_durable_delegation(event["delegation_id"])
            assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
        assert not runner._completion_deliveries_delivered
        adapter.set_message_handler(handler)
        await adapter.handle_message(MessageEvent(text="human-active", source=source))
        await asyncio.wait_for(started.wait(), 2)
        await adapter.handle_message(MessageEvent(text="human-pending", source=source))
        adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
        runner._BUSY_QUEUE_MAX_PENDING = 1
        for _ in range(10):
            assert await runner._deliver_async_delegation_group(events) is False
        assert adapter._pending_messages[key].text == "human-pending"
        assert not runner._completion_deliveries_delivered
        for event in events:
            row = delegation.get_durable_delegation(event["delegation_id"])
            assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
        # An explicitly mismatched adapter key must fail closed too.
        wrong = dict(events[0], session_key="agent:main:discord:dm:other",
                     platform="discord", chat_type="dm", chat_id="42")
        assert await runner._inject_watch_notification("wrong-route", wrong) is False
        runner._BUSY_QUEUE_MAX_PENDING = 4
        assert await runner._deliver_async_delegation_group(events) is True
        assert await runner._deliver_async_delegation_group(events) is None
        release.set()
        await drain(adapter)
        assert received[:2] == ["human-active", "human-pending"]
        assert len(received) == 3 and all(event["summary"] in received[-1] for event in events)
        for event in events:
            assert delegation.get_durable_delegation(event["delegation_id"])["delivery_state"] == "delivered"
        idle = pending(key, "idle-admitted")
        assert await runner._deliver_async_delegation_group([idle]) is True
        await drain(adapter)
        assert len(received) == 4 and "idle-admitted" in received[-1]
    finally:
        release.set()
        await drain(adapter)
        await runner._cancel_process_completion_batch_tasks()


@pytest.mark.asyncio
async def test_unavailable_raw_route_is_quiet_without_hiding_invalid_routes(tmp_path, caplog):
    runner = GatewayRunner(GatewayConfig())
    runner.adapters = {}
    evt = pending("opaque-client-session", "raw-admission")
    caplog.set_level(logging.WARNING, logger="gateway.run")
    for _ in range(3):
        assert await runner._deliver_async_delegation_group([evt]) is False
    assert not caplog.records
    row = delegation.get_durable_delegation(evt["delegation_id"])
    assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
    assert await runner._inject_watch_notification("watch", {"type": "watch_match", "session_key": "agent:broken"}) is None
    assert any("unresolvable" in record.message for record in caplog.records)
    # API recovery writes only the delivery row, never starts a model turn.
    from gateway.platforms.api_server import APIServerAdapter
    api = APIServerAdapter(PlatformConfig())
    db = SessionDB(tmp_path / "api.db")
    db.create_session(evt["session_key"], "api_server")
    api._ensure_session_db = lambda: db
    runner.adapters = {Platform.API_SERVER: api}
    try:
        caplog.clear()
        assert await runner._deliver_async_delegation_group([evt]) is True
        assert await runner._deliver_async_delegation_group([evt]) is None
        rows = db.get_messages(evt["session_key"])
        assert len(rows) == 1 and rows[0]["display_kind"] == "async_delegation_complete"
        assert not api._background_tasks and not caplog.records
    finally:
        db.close()
