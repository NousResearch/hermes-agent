"""Thinking/status drafts are previews, never durable assistant answers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
async def test_thinking_preview_does_not_become_final_answer():
    tg = TelegramAdapter(PlatformConfig(enabled=True, token="123:test", extra={"rich_messages": True, "rich_drafts": True}))
    tg._bot = MagicMock()
    tg._bot.send_message_draft = AsyncMock(return_value=True)
    tg._bot.do_api_request = AsyncMock(return_value=True)
    tg.send = AsyncMock(return_value=MagicMock(success=True, message_id="18"))
    sc = GatewayStreamConsumer(tg, "123", StreamConsumerConfig(transport="auto", chat_type="dm", edit_interval=0.01, buffer_threshold=1), on_generation_stop=AsyncMock())
    await sc._start_transports()
    assert sc.accepts_tool_progress
    sc.on_tool_progress("Searching documentation")
    sc._drain_queue()
    await sc._send_draft_frame("")
    payload = tg._bot.do_api_request.call_args.kwargs["api_kwargs"]
    assert "<tg-thinking>" in payload["rich_message"]["markdown"]
    assert "Searching documentation" in payload["rich_message"]["markdown"]
    assert not sc.final_content_delivered
    sc.on_delta("Actual answer")
    sc.finish("Actual answer")
    await sc.run()
    assert tg.send.call_args.kwargs["content"] == "Actual answer"
    assert not tg._generation_controls
