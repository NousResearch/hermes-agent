"""Tests for /commands handler Telegram inline-menu behavior."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionSource


def _make_event(text="/commands", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._running_agents = {}
    return runner


class TestHandleCommandsCommand:
    @pytest.mark.asyncio
    async def test_telegram_uses_inline_menu_when_available(self):
        runner = _make_runner()
        event = _make_event(text="/commands")
        tg_adapter = MagicMock()
        tg_adapter.send_commands_menu = AsyncMock(return_value=SendResult(success=True, message_id="42"))
        runner.adapters[Platform.TELEGRAM] = tg_adapter

        result = await runner._handle_commands_command(event)

        assert result == ""
        tg_adapter.send_commands_menu.assert_called_once()
        kwargs = tg_adapter.send_commands_menu.call_args.kwargs
        assert kwargs["chat_id"] == "67890"
        assert kwargs["page"] == 0
        assert kwargs["metadata"] == {"thread_id": None}
        assert any("`/help`" in entry for entry in kwargs["entries"])

    @pytest.mark.asyncio
    async def test_non_telegram_keeps_text_fallback(self):
        runner = _make_runner()
        event = _make_event(text="/commands", platform=Platform.DISCORD)

        result = await runner._handle_commands_command(event)

        assert "Commands" in result
        assert "`/new`" in result
