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
  4. A RetryAfter on sendRichMessageDraft returns a retryable SendResult so
     send_draft does NOT fall through to legacy sendMessageDraft and burn
     another call in the same rate-limit bucket (issue #54275).
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

import plugins.platforms.telegram.adapter as tg_mod  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


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
async def test_send_draft_non_badrequest_is_suppressed():
    """A non-BadRequest failure (e.g. expired typing action, unsupported chat)
    is suppressed with success=True so the caller stays in draft mode and never
    cascades to rapid editMessageText calls that exhaust the rate-limit quota."""
    adapter = _make_adapter()
    adapter.format_message = lambda c: f"FMT::{c}"

    calls = []

    async def _draft(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("drafts disabled for this chat")

    adapter._bot.send_message_draft = AsyncMock(side_effect=_draft)

    result = await adapter.send_draft("123", 11, "hi")

    assert result.success is True
    assert result.message_id is None
    assert len(calls) == 1  # no plain-text retry on non-BadRequest


def _make_rich_draft_adapter() -> TelegramAdapter:
    """Adapter with rich draft path enabled and do_api_request wired."""
    adapter = _make_adapter()
    adapter.format_message = lambda c: f"FMT::{c}"
    adapter._rich_messages_enabled = True
    adapter._rich_drafts_enabled = True
    adapter._rich_send_disabled = False
    adapter._rich_draft_disabled = False
    adapter._bot.do_api_request = AsyncMock(return_value=True)
    return adapter


@pytest.mark.asyncio
async def test_rich_draft_retry_after_returns_retryable_not_legacy_fallback():
    """When sendRichMessageDraft raises a flood-control (RetryAfter) exception,
    send_draft must return a retryable SendResult immediately — it must NOT fall
    through to legacy sendMessageDraft and burn another API call in the same
    rate-limit bucket.

    Regression for: https://github.com/NousResearch/hermes-agent/issues/54275
    """
    adapter = _make_rich_draft_adapter()

    flood_exc = Exception("Too Many Requests: retry after 280")
    flood_exc.retry_after = 280  # type: ignore[attr-defined]

    async def _rich_api(method, **kwargs):
        if method == "sendRichMessageDraft":
            raise flood_exc
        return True

    adapter._bot.do_api_request = AsyncMock(side_effect=_rich_api)
    legacy_calls = []

    async def _legacy(**kwargs):
        legacy_calls.append(kwargs)
        return True

    adapter._bot.send_message_draft = AsyncMock(side_effect=_legacy)

    result = await adapter.send_draft("123", 42, "streaming text")

    # Must return retryable failure — not fall through to legacy.
    assert result.success is False
    assert result.retryable is True
    assert getattr(result, "retry_after", None) == 280.0
    # Legacy sendMessageDraft must NOT have been called.
    assert len(legacy_calls) == 0, (
        "RetryAfter on rich draft must not burn a legacy sendMessageDraft call"
    )


@pytest.mark.asyncio
async def test_rich_draft_capability_error_falls_back_to_legacy():
    """When sendRichMessageDraft raises a capability error (not RetryAfter),
    send_draft falls through to legacy sendMessageDraft — this is correct and
    must not regress."""
    adapter = _make_rich_draft_adapter()

    cap_exc = Exception("Bad Request: method not found")

    async def _rich_api(method, **kwargs):
        if method == "sendRichMessageDraft":
            raise cap_exc
        return True

    legacy_calls = []

    async def _legacy(**kwargs):
        legacy_calls.append(kwargs)
        return True

    adapter._bot.do_api_request = AsyncMock(side_effect=_rich_api)
    adapter._bot.send_message_draft = AsyncMock(side_effect=_legacy)

    result = await adapter.send_draft("123", 42, "text")

    # Capability failure → success via legacy fallback.
    assert result.success is True
    assert len(legacy_calls) >= 1
