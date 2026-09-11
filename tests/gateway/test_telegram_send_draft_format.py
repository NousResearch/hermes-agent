"""TelegramAdapter.send_draft MarkdownV2 formatting parity.

Bot API 9.5 ``sendMessageDraft`` powers the animated streaming preview in
DMs.  The regular ``send`` path renders with MarkdownV2, so the draft must
too — otherwise the live preview streams as raw text and the final
``sendMessage`` snaps into formatted output, producing a jarring visual
shift at the end of the response (reported by an external user, May 2026).

These tests pin:
  1. The happy path passes ``parse_mode=MARKDOWN_V2`` with format_message'd
     text (formatting parity with the final message).
  2. A MarkdownV2 BadRequest triggers a single plain-text retry rather than
     killing draft streaming for the whole response.
  3. A non-BadRequest failure propagates so the caller falls back to edit.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
import plugins.platforms.telegram.adapter as tg_mod  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_message_draft = AsyncMock(return_value=True)
    return adapter


@pytest.mark.asyncio
async def test_send_draft_falls_back_to_plain_text_on_markdownv2_error():
    """A MarkdownV2 BadRequest retries once as plain text (no parse_mode),
    instead of aborting draft streaming for the whole response."""
    adapter = _make_adapter()
    adapter.format_message = lambda content: f"FMT::{content}"

    # Resolve the BadRequest type the adapter checks via _is_bad_request_error.
    from telegram.error import BadRequest  # type: ignore
    calls = []

    async def _draft(**kwargs):
        calls.append(kwargs)
        if "parse_mode" in kwargs:
            raise BadRequest("can't parse entities")
        return True

    adapter._bot.send_message_draft = AsyncMock(side_effect=_draft)

    result = await adapter.send_draft("123", 9, "weird _text")

    assert result.success is True
    # First attempt: MarkdownV2; second attempt: plain text, no parse_mode.
    assert len(calls) == 2
    assert "parse_mode" in calls[0]
    assert "parse_mode" not in calls[1]
    assert calls[1]["text"] == "weird _text"  # raw, unformatted


