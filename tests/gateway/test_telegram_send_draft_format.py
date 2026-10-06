"""Native drafts preserve text prefixes; final replies retain Markdown formatting."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from telegram.constants import ParseMode


@pytest.mark.asyncio
async def test_plain_preview_keeps_markdown_quality_in_persistent_final():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    adapter._bot.send_message_draft = AsyncMock(return_value=True)
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=9))
    adapter._bot.send_chat_action = AsyncMock()
    content = "**A bold answer** with `inline_code` and _underscores_"

    preview = await adapter.send_draft("123", 7, content)
    final = await adapter.send("123", content)

    assert preview.success and final.success
    adapter._bot.send_message_draft.assert_awaited_once_with(
        chat_id=123, draft_id=7, text=content,
    )
    final_kwargs = adapter._bot.send_message.call_args.kwargs
    assert final_kwargs["parse_mode"] == ParseMode.MARKDOWN_V2
    assert final_kwargs["text"] == adapter.format_message(content)
