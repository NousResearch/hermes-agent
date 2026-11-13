"""Streaming previews edit in the same parse mode as the final edit (#49536).

A plain preview finalized as MarkdownV2 flips the message's parse mode in place, and Telegram
clients then draw the old layout over the new text until a refresh.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_agent.gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = MagicMock()
    adapter._bot.edit_message_text = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_preview_and_final_edits_share_markdown_parse_mode():
    adapter = _adapter()
    await adapter.edit_message("123", "456", "Here is **bold** and", finalize=False)
    adapter._chat_outbound_slot_remaining = lambda _chat: 0.0  # the next edit fires, not paced
    await adapter.edit_message("123", "456", "Here is **bold** and more", finalize=True)

    calls = adapter._bot.edit_message_text.await_args_list
    assert [c.kwargs.get("parse_mode") for c in calls] == [tg.ParseMode.MARKDOWN_V2] * 2
    assert calls[0].kwargs["text"] == adapter.format_message("Here is **bold** and")


@pytest.mark.asyncio
async def test_preview_rejected_by_markdown_parser_retries_raw_text_once():
    adapter = _adapter()
    partial = "Code:\n```python\nprint('x')\nhalf **bold"
    adapter._bot.edit_message_text.side_effect = [
        Exception("Bad Request: can't parse entities: Can't find end of Pre entity"), None]

    result = await adapter.edit_message("123", "456", partial, finalize=False)

    assert result.success is True
    first, retry = adapter._bot.edit_message_text.await_args_list
    assert first.kwargs["parse_mode"] == tg.ParseMode.MARKDOWN_V2
    assert "parse_mode" not in retry.kwargs and retry.kwargs["text"] == partial  # nothing lost


@pytest.mark.asyncio
async def test_final_edit_after_short_flood_wait_retries_in_markdown(monkeypatch):
    adapter = _adapter()
    flood = Exception("Flood control exceeded. Retry after 2 seconds")
    flood.retry_after = 2.0
    adapter._bot.edit_message_text.side_effect = [flood, None]
    sleep = AsyncMock()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", sleep)

    result = await adapter.edit_message("123", "456", "Done: **bold**", finalize=True)

    assert result.success is True
    sleep.assert_awaited_once_with(2.0)
    retry = adapter._bot.edit_message_text.await_args_list[-1]
    assert retry.kwargs["parse_mode"] == tg.ParseMode.MARKDOWN_V2
    assert retry.kwargs["text"] == adapter.format_message("Done: **bold**")
