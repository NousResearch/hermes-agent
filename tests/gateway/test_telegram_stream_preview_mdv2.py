"""Streamed edit-transport previews must render MarkdownV2 mid-stream.

Related: #54817

With ``streaming.transport: edit`` (the stable alternative to draft streaming,
whose per-frame re-render makes the reply bubble collapse/flicker on current
clients — #54817), ``edit_message`` used to apply MarkdownV2 formatting only on
the ``finalize=True`` edit: every mid-stream frame was sent as raw text, so
users watched literal ``##`` / ``**`` / ``|`` markdown until the last frame
snapped into formatted output.

``send_draft`` already converts every frame with the regular ``format_message``
+ ``ParseMode.MARKDOWN_V2`` pipeline and retries once as plain text when
Telegram rejects malformed entities.  These tests pin the same per-frame
contract onto the edit path:

* mid-stream frames are sent formatted (``format_message`` + MarkdownV2);
* a BadRequest (typically "can't parse entities" from a half-open entity)
  falls back to the plain frame for that frame only;
* "message is not modified" is a success no-op;
* flood control and transient network errors keep their original handling.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    """Install mock telegram modules so TelegramAdapter can be imported."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram import adapter as telegram_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _run(coro):
    """Run a coroutine in a fresh event loop for sync-style tests."""
    return asyncio.run(coro)


class _BadRequest(Exception):
    """Class name is what _is_bad_request_error keys on."""


class _FloodError(Exception):
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    a = TelegramAdapter(config)
    a._bot = MagicMock()
    return a


class TestMidStreamFramesRenderMarkdownV2:
    def test_mid_stream_edit_is_formatted(self, adapter):
        """finalize=False frames go out via format_message + MarkdownV2."""
        adapter._bot.edit_message_text = AsyncMock()

        content = "## Heading\n\n**bold** progress"
        result = _run(
            adapter.edit_message("12345", "7", content, finalize=False)
        )

        assert result.success
        adapter._bot.edit_message_text.assert_awaited_once()
        kwargs = adapter._bot.edit_message_text.await_args.kwargs
        assert kwargs["text"] == adapter.format_message(content)
        assert kwargs["parse_mode"] is telegram_adapter.ParseMode.MARKDOWN_V2

    def test_bad_entities_fall_back_to_plain_frame(self, adapter):
        """A rejected formatted frame degrades to plain text for that frame."""
        calls = []

        async def _edit(**kwargs):
            calls.append(kwargs)
            if "parse_mode" in kwargs:
                raise _BadRequest("Bad Request: can't parse entities")

        adapter._bot.edit_message_text = AsyncMock(side_effect=_edit)

        content = "**unterminated bold"
        result = _run(
            adapter.edit_message("12345", "7", content, finalize=False)
        )

        assert result.success
        assert len(calls) == 2
        assert "parse_mode" in calls[0]
        assert "parse_mode" not in calls[1]
        assert calls[1]["text"] == content

    def test_not_modified_on_formatted_frame_is_success_noop(self, adapter):
        """'message is not modified' must not trigger the plain fallback."""
        adapter._bot.edit_message_text = AsyncMock(
            side_effect=_BadRequest("Bad Request: message is not modified")
        )

        result = _run(
            adapter.edit_message("12345", "7", "same text", finalize=False)
        )

        assert result.success
        adapter._bot.edit_message_text.assert_awaited_once()

    def test_flood_control_keeps_original_handling(self, adapter):
        """Non-BadRequest errors re-raise into the original flood handler."""
        adapter._bot.edit_message_text = AsyncMock(
            side_effect=_FloodError("Flood control exceeded. Retry in 30", 30.0)
        )

        result = _run(
            adapter.edit_message("12345", "7", "some progress", finalize=False)
        )

        # Long flood waits surface as the standard retryable flood failure,
        # exactly as before this change.
        assert not result.success
        assert "flood_control" in (result.error or "")
        assert result.retry_after == 30.0

    def test_finalize_path_unchanged(self, adapter):
        """finalize=True still sends the formatted final edit."""
        adapter._bot.edit_message_text = AsyncMock()

        content = "plain final text"
        result = _run(
            adapter.edit_message("12345", "7", content, finalize=True)
        )

        assert result.success
        kwargs = adapter._bot.edit_message_text.await_args.kwargs
        assert kwargs["text"] == adapter.format_message(content)
