"""Replay filtering must not discard the fresh tail of a partially answered batch."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl, shutdown_flush as spool
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult, merge_pending_message_event
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(dl, "ledger_enabled", lambda *a, **k: True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="synthetic", user_id="owner")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fixture", extra={}))
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id="receipt"))
    runner = object.__new__(GatewayRunner)
    runner._pending_native_image_paths = {}
    runner.config = SimpleNamespace()
    return source, adapter, runner, "agent:main:telegram:dm:synthetic"


def event(source, identity, text):
    return MessageEvent(text=text, source=source, message_id=identity,
                        timestamp=datetime(2026, 1, 1), reply_to_message_id="reply-" + identity)


@pytest.mark.asyncio
@pytest.mark.parametrize("attachments", [False, True])
async def test_partial_batch_executes_and_transfers_only_fresh_members(world, tmp_path, attachments):
    source, adapter, runner, key = world
    old = event(source, "old", "obsolete instruction")
    fresh = event(source, "fresh", "current question")
    if attachments:
        for item in (old, fresh):
            path = tmp_path / (item.message_id + ".txt")
            path.write_text("synthetic attachment", encoding="utf-8")
            item.media_urls = [str(path)]
            item.media_types = ["text/plain"]
            item.media_text_inlined = [True]
    for item in (old, fresh):
        assert spool.record_durable_inbound_event(key, item)
    await adapter.send_final_ledgered(old, key, "previous answer", {}, reply_to=None)
    old_id = spool.durable_inbound_obligation_id(old)
    pending = {key: spool._deserialise_inbound_event(spool._serialise_inbound_event(old))}
    merge_pending_message_event(pending, key, fresh, merge_text=True)
    batch = pending[key]
    text = await runner._prepare_inbound_message_text(
        event=batch, source=source, history=[], session_key=key)
    assert text is not None
    assert [m["message_id"] for m in batch.original_inputs()] == [fresh.message_id]
    assert batch.reply_to_message_id == fresh.reply_to_message_id
    assert batch.text == fresh.text
    assert batch.media_urls == fresh.media_urls
    assert batch.media_text_inlined == fresh.media_text_inlined
    await adapter.send_final_ledgered(batch, key, "current answer", {}, reply_to=None)
    assert spool.recover_durable_inbound_events() == []
    assert dl.inbound_transfer_state(key, old_id) == "delivered"
    with dl._transaction() as conn:
        assert conn.execute("SELECT count(*) FROM delivery_obligations").fetchone()[0] == 2
    assert adapter._send_with_retry.await_count == 2


def test_repeated_pending_identity_is_not_appended(world):
    source, _, _, key = world
    original = event(source, "one", "same text")
    original_text = original.text
    assert spool.record_durable_inbound_event(key, original)
    duplicate = spool._deserialise_inbound_event(spool._serialise_inbound_event(original))
    pending = {key: original}
    merge_pending_message_event(pending, key, duplicate, merge_text=True)
    assert pending[key].text == original_text
    assert len(pending[key].original_inputs()) == 1
    fresh = event(source, "two", original.text)
    assert spool.record_durable_inbound_event(key, fresh)
    merge_pending_message_event(pending, key, fresh, merge_text=True)
    assert len(pending[key].original_inputs()) == 2
