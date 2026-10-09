"""Durable completion acknowledgement survives asynchronous owner resolution."""

import asyncio
import time

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.discord.adapter import DiscordAdapter
from tools import async_delegation as delegation


@pytest.fixture
def owner(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "person")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "voice.json")
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, typing_indicator=False))
    runner.adapters[Platform.DISCORD] = adapter
    source = SessionSource(Platform.DISCORD, chat_id="room", chat_type="thread",
                           thread_id="topic", user_id="person")
    entry = runner.session_store.get_or_create_session(source)
    return runner, adapter, source, entry, runner.session_store._db_for_key(entry.session_key)


def batch(entry):
    events = []
    for name in ("first-result", "second-result"):
        event = dict(type="async_delegation", session_key=entry.session_key,
                     parent_session_id=entry.session_id, delegation_id=name, summary=name,
                     status="completed", dispatched_at=time.time())
        delegation._persist_dispatch(event)
        delegation._persist_completion(event, {"status": "completed", "summary": name})
        events.append(event)
    return events


async def drain(adapter):
    while adapter._background_tasks:
        await asyncio.gather(*list(adapter._background_tasks))
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
async def test_admission_proves_owner_before_ack_and_turn_needs_no_second_lookup(owner, monkeypatch, busy):
    runner, adapter, source, entry, db = owner
    events = batch(entry)
    delivered = []
    release, started = asyncio.Event(), asyncio.Event()
    unavailable = False
    read = db.get_session

    def lookup(session_id):
        if unavailable:
            raise OSError("temporary owner lookup failure")
        return read(session_id)

    monkeypatch.setattr(db, "get_session", lookup)

    async def handler(event):
        nonlocal unavailable
        if not event.internal:
            started.set()
            await release.wait()
            return
        # The adapter has already accepted this event. An independent owner lookup here is
        # precisely the old loss window: a successful preflight was acknowledged, then discarded.
        unavailable = True
        try:
            resolved = await runner._hmwa_resolve_session(event, event.source)
        finally:
            unavailable = False
        if resolved is not None:
            delivered.append((event.text, resolved[1].session_id))

    adapter.set_message_handler(handler)
    try:
        if busy:
            await adapter.handle_message(MessageEvent(text="human", source=source))
            await asyncio.wait_for(started.wait(), 2)
        assert await asyncio.wait_for(runner._deliver_async_delegation_group(events), 2) is True
        assert all(delegation.get_durable_delegation(e["delegation_id"])["delivery_state"] == "delivered"
                   for e in events)
        release.set()
        await asyncio.wait_for(drain(adapter), 2)
        assert len(delivered) == 1
        assert delivered[0][1] == entry.session_id
        assert all(e["summary"] in delivered[0][0] for e in events)
    finally:
        release.set()
        await drain(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["owner_preparation", "claimed_preflight"])
async def test_pre_admission_storage_outage_refunds_every_sibling_for_retry(owner, monkeypatch, failure_phase):
    runner, adapter, _source, entry, db = owner
    events = batch(entry)
    received = []

    async def handler(event):
        resolved = await runner._hmwa_resolve_session(event, event.source)
        if resolved is not None:
            received.append(event.text)

    adapter.set_message_handler(handler)
    inject, read = runner._inject_watch_notification, db.get_session
    claim = delegation.claim_completion_delivery

    def unavailable(_session_id):
        raise OSError("temporary owner lookup failure")

    async def fail_after_preflight(*args, **kwargs):
        monkeypatch.setattr(db, "get_session", unavailable)
        try:
            return await inject(*args, **kwargs)
        finally:
            monkeypatch.setattr(db, "get_session", read)

    def fail_after_claim(delegation_id, claim_id):
        acquired = claim(delegation_id, claim_id)
        if acquired and delegation_id == events[0]["delegation_id"]:
            monkeypatch.setattr(db, "get_session", unavailable)
        return acquired

    if failure_phase == "owner_preparation":
        monkeypatch.setattr(runner, "_inject_watch_notification", fail_after_preflight)
    else:
        monkeypatch.setattr(delegation, "claim_completion_delivery", fail_after_claim)
    for _ in range(delegation._MAX_DELIVERY_ATTEMPTS + 1):
        try:
            assert await runner._deliver_async_delegation_group(events) is False
        finally:
            monkeypatch.setattr(db, "get_session", read)
    assert not adapter._background_tasks and not received
    assert not runner._completion_deliveries_delivered
    for event in events:
        row = delegation.get_durable_delegation(event["delegation_id"])
        assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
    monkeypatch.setattr(runner, "_inject_watch_notification", inject)
    monkeypatch.setattr(delegation, "claim_completion_delivery", claim)
    assert await runner._deliver_async_delegation_group(events) is True
    await drain(adapter)
    assert len(received) == 1 and all(e["summary"] in received[0] for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["compression", "new"])
async def test_queued_owner_change_preserves_compression_retry_and_new_boundary(owner, monkeypatch, transition):
    runner, adapter, source, entry, db = owner
    events, original_id, key = batch(entry), entry.session_id, entry.session_key
    release, started = asyncio.Event(), asyncio.Event()
    received, resolved_ids = [], []
    lookup_failed = False
    read = db.get_session

    def fail_once(session_id):
        nonlocal lookup_failed
        if not lookup_failed:
            lookup_failed = True
            raise OSError("temporary continuation lookup failure")
        return read(session_id)

    async def handler(event):
        if event.text == "human-active":
            started.set()
            await release.wait()
        elif event.internal:
            resolved = await runner._hmwa_resolve_session(event, event.source)
            if resolved is not None:
                received.append(event.text)
                resolved_ids.append(resolved[1].session_id)
        else:
            received.append(event.text)
        if key not in adapter._pending_messages:
            next_event = runner._promote_queued_event(key, adapter, None)
            if next_event is not None:
                adapter._pending_messages[key] = next_event

    adapter.set_message_handler(handler)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    try:
        await adapter.handle_message(MessageEvent(text="human-active", source=source))
        await asyncio.wait_for(started.wait(), 2)
        await adapter.handle_message(MessageEvent(text="human-followup", source=source))
        assert await asyncio.wait_for(runner._deliver_async_delegation_group(events), 2) is True
        if transition == "compression":
            db.end_session(original_id, "compression")
            db.create_session("receipt-tip", source="discord", parent_session_id=original_id,
                              session_key=key)
            runner.session_store.advance_compression_session(key, original_id, "receipt-tip")
        else:
            runner.session_store.reset_session(key)
        monkeypatch.setattr(db, "get_session", fail_once)
        release.set()
        await asyncio.wait_for(drain(adapter), 3)
        assert lookup_failed
        assert received[0] == "human-followup"
        if transition == "compression":
            assert len(received) == 2 and all(e["summary"] in received[1] for e in events)
            assert resolved_ids == ["receipt-tip"]
        else:
            assert received == ["human-followup"] and not resolved_ids
    finally:
        release.set()
        await drain(adapter)
