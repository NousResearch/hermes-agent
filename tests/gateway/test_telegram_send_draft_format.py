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
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from gateway.platforms import telegram as tg_mod  # noqa: E402
from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_message_draft = AsyncMock(return_value=True)
    return adapter


@pytest.mark.asyncio
async def test_send_draft_passes_markdownv2_parse_mode():
    """Happy path: draft is sent with parse_mode set and format_message'd text."""
    adapter = _make_adapter()
    # Make format_message observable and deterministic.
    adapter.format_message = lambda c: f"FMT::{c}"

    result = await adapter.send_draft("123", 7, "**bold** body")

    assert result.success is True
    adapter._bot.send_message_draft.assert_awaited_once()
    kwargs = adapter._bot.send_message_draft.await_args.kwargs
    assert kwargs["text"] == "FMT::**bold** body"
    assert kwargs["parse_mode"] is tg_mod.ParseMode.MARKDOWN_V2
    assert kwargs["chat_id"] == 123
    assert kwargs["draft_id"] == 7


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


@pytest.mark.asyncio
async def test_send_draft_non_badrequest_propagates_without_retry():
    """A non-BadRequest failure (e.g. drafts not allowed) returns failure
    immediately so the caller falls back to the edit transport."""
    adapter = _make_adapter()
    adapter.format_message = lambda c: f"FMT::{c}"

    calls = []

    async def _draft(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("drafts disabled for this chat")

    adapter._bot.send_message_draft = AsyncMock(side_effect=_draft)

    result = await adapter.send_draft("123", 11, "hi")

    assert result.success is False
    assert len(calls) == 1  # no plain-text retry on non-BadRequest
