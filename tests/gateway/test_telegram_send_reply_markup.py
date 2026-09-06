"""Tests for Telegram adapter.send() reply_markup and per-send preview metadata."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(extra=None) -> TelegramAdapter:
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestTelegramSendMarkupAndPreview:
    @pytest.mark.asyncio
    async def test_send_attaches_structured_reply_markup(self):
        adapter = _make_adapter()
        adapter._rich_send_disabled = True
        mock_msg = MagicMock()
        mock_msg.message_id = 7
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        rows = [[{"text": "Do it", "callback_data": "task:go"}]]
        result = await adapter.send(
            "12345",
            "hello",
            metadata={"reply_markup": rows},
        )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args.kwargs
        assert kwargs["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_send_passes_through_existing_markup(self):
        adapter = _make_adapter()
        adapter._rich_send_disabled = True
        mock_msg = MagicMock()
        mock_msg.message_id = 8
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)
        sentinel = object()

        result = await adapter.send(
            "12345",
            "hello",
            metadata={"reply_markup": sentinel},
        )

        assert result.success is True
        assert adapter._bot.send_message.call_args.kwargs["reply_markup"] is sentinel

    @pytest.mark.asyncio
    async def test_send_keyboard_on_first_chunk_only(self):
        adapter = _make_adapter()
        adapter._rich_send_disabled = True
        adapter.MAX_MESSAGE_LENGTH = 80
        adapter._bot.send_message = AsyncMock(
            side_effect=lambda **kwargs: MagicMock(message_id=1)
        )

        content = ("chunk content " * 20).strip()
        result = await adapter.send(
            "123",
            content,
            metadata={
                "reply_markup": [[{"text": "Go", "callback_data": "task:go"}]],
            },
        )

        assert result.success is True
        assert adapter._bot.send_message.await_count > 1
        first = adapter._bot.send_message.await_args_list[0].kwargs
        last = adapter._bot.send_message.await_args_list[-1].kwargs
        assert first.get("reply_markup") is not None
        assert "reply_markup" not in last

    @pytest.mark.asyncio
    async def test_send_honors_per_send_preview_disable(self):
        adapter = _make_adapter()
        adapter._rich_send_disabled = True
        adapter._disable_link_previews = False
        mock_msg = MagicMock()
        mock_msg.message_id = 9
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        await adapter.send(
            "12345",
            "see https://example.com",
            metadata={"disable_link_preview": True},
        )

        kwargs = adapter._bot.send_message.call_args.kwargs
        assert "link_preview_options" in kwargs or kwargs.get(
            "disable_web_page_preview"
        ) is True
        assert adapter._link_preview_disabled({"disable_link_preview": True}) is True
        assert adapter._link_preview_kwargs({"disable_link_preview": True})

    @pytest.mark.asyncio
    async def test_rich_path_skipped_when_markup_present(self):
        adapter = _make_adapter(extra={"rich_messages": True})
        adapter._bot.do_api_request = AsyncMock(
            return_value=SimpleNamespace(message_id=123)
        )
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
        adapter._bot.send_chat_action = AsyncMock()

        table = (
            "| Case | Status |\n"
            "|---|---|\n"
            "| rich | ok |\n"
        )
        result = await adapter.send(
            "12345",
            table,
            metadata={
                "reply_markup": [[{"text": "Go", "callback_data": "task:go"}]],
            },
        )

        assert result.success is True
        adapter._bot.do_api_request.assert_not_called()
        assert adapter._bot.send_message.call_args.kwargs.get("reply_markup") is not None
