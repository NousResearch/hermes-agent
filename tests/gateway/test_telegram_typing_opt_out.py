"""``typing_indicator: false`` stops every Telegram ``sendChatAction``, not only the refresh loop (#40885).

The flag gated ``_keep_typing`` and the post-send re-arm, but the turn's progress path, the proxy
turn and the queued follow-up call ``send_typing`` directly, so a user who switched the indicator
off to stop the chat from jumping still got a ``sendChatAction`` after every progress edit.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter(**cfg):
    adapter = TelegramAdapter(PlatformConfig.from_dict({"enabled": True, "token": "t", **cfg}))
    adapter._bot = AsyncMock()
    adapter._bot.send_chat_action = AsyncMock(return_value=None)
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("cfg", [{"typing_indicator": False}, {"extra": {"typing_indicator": False}}])
async def test_disabled_indicator_sends_no_chat_action(cfg):
    adapter = _adapter(**cfg)
    await adapter.send_typing("123", metadata={"thread_id": "7"})
    adapter._bot.send_chat_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_indicator_still_sends_chat_action():
    adapter = _adapter()
    await adapter.send_typing("123")
    adapter._bot.send_chat_action.assert_awaited_once()
