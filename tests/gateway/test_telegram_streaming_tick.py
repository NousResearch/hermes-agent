"""TelegramAdapter streaming-tick formatting tests.

``edit_message(finalize=False)`` — the path called on every streaming
update — previously sent plain text so MarkdownV2 markup was visible
as raw characters during streaming.  It now attempts MarkdownV2 first and
falls back to plain text only when Telegram returns a BadRequest.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


# ── minimal Telegram mock (mirrors test_telegram_send_draft_flood_control) ─


def _ensure_telegram_mock() -> None:
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
    adapter._bot.edit_message_text = AsyncMock(return_value=None)
    # Disable rich-message path so tests focus on the MarkdownV2 tick logic
    adapter._rich_messages_enabled = False
    # format_message should be a no-op for these unit tests unless overridden
    adapter.format_message = lambda c: c
    return adapter


def _make_bad_request(msg: str = "can't parse entities") -> Exception:
    """Return an exception that _is_bad_request_error will recognise."""
    # Import the mocked BadRequest type
    err_cls = sys.modules["telegram.error"].BadRequest
    exc = err_cls(msg)
    return exc


# ── edit_message(finalize=False) — MarkdownV2 streaming tick ─────────────


class TestStreamingTickMarkdownV2:
    """edit_message with finalize=False tries MarkdownV2, falls back on BadRequest."""

    @pytest.mark.asyncio
    async def test_streaming_tick_uses_markdownv2(self):
        """A plain streaming tick (finalize=False) sends the formatted text
        with ParseMode.MARKDOWN_V2, not plain text."""
        from telegram.constants import ParseMode

        adapter = _make_adapter()
        result = await adapter.edit_message(
            chat_id="42",
            message_id="100",
            content="**bold** text",
            finalize=False,
        )

        assert result.success
        call_kwargs = adapter._bot.edit_message_text.call_args.kwargs
        assert call_kwargs.get("parse_mode") == ParseMode.MARKDOWN_V2

    @pytest.mark.asyncio
    async def test_streaming_tick_falls_back_on_bad_request(self):
        """When Telegram rejects the MarkdownV2 frame with a BadRequest,
        the tick retries as plain text and still returns success."""
        bad_request = _make_bad_request()
        plain_call_kwargs = {}

        async def _edit(**kwargs):
            if kwargs.get("parse_mode") is not None:
                raise bad_request
            plain_call_kwargs.update(kwargs)
            return None

        adapter = _make_adapter()
        adapter._bot.edit_message_text = _edit

        result = await adapter.edit_message(
            chat_id="42",
            message_id="100",
            content="raw ** unclosed",
            finalize=False,
        )

        assert result.success, "Streaming tick must succeed even after MarkdownV2 BadRequest"
        # Plain-text fallback must have been attempted
        assert plain_call_kwargs, "Plain-text fallback call must have happened"
        assert plain_call_kwargs.get("parse_mode") is None

    @pytest.mark.asyncio
    async def test_streaming_tick_not_modified_is_success(self):
        """'Message is not modified' from the MarkdownV2 attempt is treated
        as success, not an error."""
        async def _edit(**kwargs):
            raise Exception("Message is not modified")

        adapter = _make_adapter()
        adapter._bot.edit_message_text = _edit

        result = await adapter.edit_message(
            chat_id="42",
            message_id="100",
            content="same content",
            finalize=False,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_streaming_tick_flood_propagates(self):
        """Flood-control errors from the MarkdownV2 attempt propagate to the
        outer handler (not silently swallowed) so the consumer can back off."""
        flood_exc = Exception("flood control: retry in 2 seconds")
        flood_exc.retry_after = 2

        async def _edit(**kwargs):
            if kwargs.get("parse_mode") is not None:
                raise flood_exc

        adapter = _make_adapter()
        adapter._bot.edit_message_text = _edit

        # The flood error should reach the outer except block
        # (which handles retry_after and returns a failure result)
        result = await adapter.edit_message(
            chat_id="42",
            message_id="100",
            content="some content",
            finalize=False,
        )

        # Outer handler converts flood to a SendResult with success=False
        assert not result.success

    @pytest.mark.asyncio
    async def test_finalize_true_still_uses_own_path(self):
        """finalize=True must NOT use the streaming-tick path — it goes
        through the separate MarkdownV2 + rich-edit finalize flow."""
        from telegram.constants import ParseMode

        mdv2_calls = []

        async def _edit(**kwargs):
            mdv2_calls.append(kwargs)
            return None

        adapter = _make_adapter()
        adapter._bot.edit_message_text = _edit

        await adapter.edit_message(
            chat_id="42",
            message_id="100",
            content="final answer",
            finalize=True,
        )

        # Should have been called with MarkdownV2 (finalize path)
        assert any(
            c.get("parse_mode") == ParseMode.MARKDOWN_V2 for c in mdv2_calls
        ), "finalize=True path must use MarkdownV2"
