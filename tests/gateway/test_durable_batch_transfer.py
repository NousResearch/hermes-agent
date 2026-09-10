"""Disk-backed incident regression: one output owns every original batch input."""
from datetime import datetime
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl, shutdown_flush as spool
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult, merge_pending_message_event
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.run import GatewayRunner
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(dl, "ledger_enabled", lambda *a, **k: True)
    monkeypatch.setattr(dl, "_owner_alive", lambda *a: False)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fixture", extra={}))
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id="out"))
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="out"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="fixture-chat", user_id="fixture-owner")
    return adapter, source, "agent:main:telegram:dm:fixture-chat"


def inputs(source, key, count=5):
    events = [MessageEvent(text="same fresh question", source=source, message_id=f"member-{i}",
                           platform_update_id=i, timestamp=datetime(2026, 1, 1, 0, 0, i),
                           reply_to_message_id=f"prior-{i}") for i in range(count)]
    pending = {}
    for event in events:
        assert spool.record_durable_inbound_event(key, event)
        merge_pending_message_event(pending, key, event, merge_text=True)
    return events, pending[key]


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["before_record", "after_record", "after_send", "failed_send"])
async def test_batch_transfer_restart(world, monkeypatch, boundary):
    adapter, source, key = world
    events, batch = inputs(source, key)
    ids = [spool.durable_inbound_obligation_id(e) for e in events]
    if boundary == "before_record":
        assert len(spool.recover_durable_inbound_events()) == len(ids)
        assert dl.sweep_recoverable() == []
        return
    if boundary == "after_record":
        monkeypatch.setattr(spool, "acknowledge_durable_inbound_event", lambda e: None)
        await adapter._record_delivery_obligation(batch, key, "authoritative response", adapter, False)
    else:
        if boundary == "failed_send":
            adapter._send_with_retry.return_value = SendResult(success=False, error="rejected")
        await adapter.send_final_ledgered(batch, key, "authoritative response", {}, reply_to=None)
    assert spool.recover_durable_inbound_events() == [], "answered batch members must not execute again"
    recovered = dl.sweep_recoverable()
    assert len(recovered) == (0 if boundary == "after_send" else 1)
    with dl._transaction() as conn:
        assert conn.execute("SELECT count(*) FROM delivery_obligations").fetchone()[0] == 1
    # A fresh same-text identity is not a replay, even after a successful batch.
    fresh = MessageEvent(text=events[0].text, source=source, message_id="fresh")
    assert spool.record_durable_inbound_event(key, fresh)
    assert [e.message_id for e in spool.recover_durable_inbound_events()] == ["fresh"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
async def test_queued_transfer_keeps_original_envelope(world, streamed):
    adapter, source, key = world
    events, batch = inputs(source, key)
    runner = object.__new__(GatewayRunner)
    runner._deliver_media_from_response = AsyncMock()
    envelope = ({"inbound_event": batch} if "inbound_event" in
                inspect.signature(runner._deliver_queued_first_response).parameters else {})
    await runner._deliver_queued_first_response(
        "authoritative response", source, adapter, session_key=key,
        inbound_message_id=batch.message_id, **envelope,
        text_already_delivered=streamed, deliver_media=False)
    assert spool.recover_durable_inbound_events() == []
    assert dl.sweep_recoverable() == []
    assert adapter._send_with_retry.await_count == (0 if streamed else 1)


@pytest.mark.asyncio
async def test_restored_provenance_and_same_identity_admission(world):
    import json

    adapter, source, key = world
    events, batch = inputs(source, key)
    restored = spool._deserialise_inbound_event(spool._serialise_inbound_event(batch))
    runner = object.__new__(GatewayRunner)
    runner._pending_native_image_paths = {}
    runner.config = SimpleNamespace()
    text = await runner._prepare_inbound_message_text(
        event=restored, source=source, history=[], session_key=key)
    assert text is not None
    provenance = json.loads(text.split("\n", 2)[1])
    assert [m["message_id"] for m in provenance] == [e.message_id for e in events]
    assert [m["reply_to_message_id"] for m in provenance] == [e.reply_to_message_id for e in events]
    assert all(m["delivery_state"] is None for m in provenance)
    await adapter.send_final_ledgered(batch, key, "answer", {}, reply_to=None)
    assert await runner._prepare_inbound_message_text(
        event=restored, source=source, history=[], session_key=key) is None
    fresh = MessageEvent(text=events[0].text, source=source, message_id="new-followup")
    assert spool.record_durable_inbound_event(key, fresh)
    assert await runner._prepare_inbound_message_text(
        event=fresh, source=source, history=[], session_key=key)


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [0, 2])
async def test_recursive_chain_transfers_terminal_batch(world, depth):
    from unittest.mock import MagicMock
    from gateway.turn_context import TurnContext

    adapter, source, key = world
    events, batch = inputs(source, key)
    opening = MessageEvent(text="opening", source=source, message_id="opening")
    assert spool.record_durable_inbound_event(key, opening)
    runner = object.__new__(GatewayRunner)
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value=key)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="followup")
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._deliver_media_from_response = AsyncMock()
    runner._pop_post_delivery_callback = MagicMock(return_value=None)
    terminal = {"final_response": "terminal", "messages": []}
    if depth:
        terminal["queued_terminal_event"] = batch
        terminal["queued_terminal_inbound_id"] = batch.message_id
    runner._run_agent = AsyncMock(return_value=terminal)
    ctx = TurnContext(source=source, session_id="sid", session_key=key,
                      inbound_event=opening, inbound_message_id=opening.message_id,
                      _interrupt_depth=depth, history=[])
    result = await runner._run_agent_queued_followup(
        ctx, adapter, "pending", batch,
        {"final_response": "first"}, {"messages": []}, None)
    assert runner._run_agent.await_args.kwargs["inbound_event"] is batch
    opening.terminal_event = result["queued_terminal_event"]
    await adapter.send_final_ledgered(opening, key, result["final_response"], {}, reply_to=None)
    assert spool.recover_durable_inbound_events() == []
    with dl._transaction() as conn:
        assert conn.execute("SELECT count(*) FROM delivery_obligations").fetchone()[0] == 2
