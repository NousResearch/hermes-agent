"""The full-command send ahead of a truncated approval card refuses a command that splits past
Telegram's ``MAX_SPLIT_MESSAGES`` (#152012: past it a chat floods into a ban). Telegram splits in UTF-16
code units, so an astral-heavy command can fit the cap counted with ``len`` yet split past it on the
wire; the check must count the way ``send()`` does."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.gateway.platforms.base import utf16_len
from hermes_agent.gateway.platforms.base_exec_approval import send_full_command
from plugins.platforms.telegram.adapter import TelegramAdapter


def test_astral_command_capped_only_in_utf16_is_not_sent_as_whole():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._rich_send_disabled = True
    bot = adapter._bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    cap, limit = adapter.MAX_SPLIT_MESSAGES, adapter.MAX_MESSAGE_LENGTH
    # Each emoji is 1 ``len`` unit and 2 UTF-16 units: ~0.75 * cap chunks by ``len``, ~1.5 * cap on the wire.
    command = "\n".join("\U0001F600" * 60 for _ in range(limit * cap * 3 // 4 // 61))
    formatted = adapter.format_message(f"```\n{command}\n```")
    assert len(adapter.truncate_message(formatted, limit)) <= cap
    assert len(adapter.truncate_message(formatted, limit, len_fn=utf16_len)) > cap

    shown = asyncio.run(send_full_command(adapter, "4242", command, None))

    assert shown is False
    bot.send_message.assert_not_awaited()
