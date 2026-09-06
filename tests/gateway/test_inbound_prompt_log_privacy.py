"""Turn and queued-message logs use metadata while live inputs stay exact."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.WHATSAPP_CLOUD, Platform.TELEGRAM])
async def test_inbound_log_does_not_copy_identity_message_or_reply(caplog, platform):
    body = "private question\nprivate continuation"
    source = SessionSource(platform=platform, chat_id="15551234567", user_id="15551234567", user_name="Private Name")
    event = SimpleNamespace(text=body, source=source, reply_to_message_id="private reply id", reply_to_text="private reply body")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._hmwa_resolve_session = AsyncMock(return_value=None)
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await runner._handle_message_with_agent(event, source, "raw quick key", 1)
    runner._hmwa_resolve_session.assert_awaited_once_with(event, source)
    assert event.text == body
    assert event.reply_to_text == "private reply body"
    assert source.user_name == "Private Name"
    assert source.chat_id == source.user_id == "15551234567"
    record = next(record.message for record in caplog.records if record.message.startswith("inbound message:"))
    assert "private" not in record.lower()
    assert "15551234567" not in record
    assert f"msg_len={len(body)}" in record
    assert "reply_to_id_present=True" in record


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [True, False])
async def test_pending_text_is_delivered_without_log_excerpt(caplog, queued):
    body = "private followup\nprivate continuation"
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._draining = False
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="15551234567")
    pending_event = SimpleNamespace(text=body) if queued else None
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter) if queued else None
    if adapter is not None:
        adapter._pending_messages = {"raw-key": pending_event}
    runner._promote_queued_event = lambda key, adapter, event: event
    runner._pending_event_audio_paths = lambda event: []
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        event, text = await runner._run_agent_drain_pending(
            {"pending_steer": body}, adapter, source, "raw-key"
        )
    assert text == body
    assert event is pending_event
    if adapter is not None:
        assert adapter._pending_messages == {}
    assert "private" not in caplog.text
    assert f"msg_len={len(body)}" in caplog.text
