"""Regression test for #48648 — Telegram infinite reply chain on mid-stream overflow.

When streamed content exceeds Telegram's 4096 UTF-16 code-unit limit,
edit_message must NOT call _edit_overflow_split during streaming
(finalize=False) because that spawns continuation messages, updates the
active message_id, and the next token edit carries the full accumulated
text (still > limit) → infinite reply chain.  Instead the preview must be
truncated to keep updating the same message until finalize=True does the
real split.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_adapter():
    """Create a minimal TelegramAdapter with a mocked bot."""
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._bot = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter._metadata_thread_id = MagicMock(return_value=None)
    adapter._thread_kwargs_for_send = MagicMock(return_value={})
    adapter._rich_eligible = MagicMock(return_value=False)
    # name is a read-only property on the base class; mock it at class level
    type(adapter).name = property(lambda self: "telegram")
    return adapter


class TestMidStreamOverflowTruncation:
    """#48648: mid-stream overflow must truncate, not split."""

    @pytest.mark.asyncio
    async def test_finalize_false_truncates_instead_of_splitting(self):
        """When finalize=False and content > 4096, edit_message must truncate
        the content and edit the same message — NOT call _edit_overflow_split."""
        adapter = _make_adapter()
        long_content = "x" * 5000  # exceeds 4096

        # Mock edit_message_text to succeed
        adapter._bot.edit_message_text = AsyncMock()

        # Mock _edit_overflow_split — should NOT be called
        adapter._edit_overflow_split = AsyncMock()

        from gateway.platforms.base import SendResult

        result = await adapter.edit_message(
            chat_id="12345",
            message_id="999",
            content=long_content,
            finalize=False,
        )

        # Must NOT have called _edit_overflow_split
        adapter._edit_overflow_split.assert_not_called()

        # Must have edited with truncated content (≤ 4000 chars)
        call_args = adapter._bot.edit_message_text.call_args
        edited_text = call_args.kwargs.get("text") or call_args[1].get("text")
        assert len(edited_text) <= 4000, (
            f"expected truncated content ≤ 4000, got {len(edited_text)}"
        )
        assert result.success is True
        assert result.message_id == "999"  # same message, not a continuation

    @pytest.mark.asyncio
    async def test_finalize_true_splits_normally(self):
        """When finalize=True and content > 4096, edit_message must still call
        _edit_overflow_split to deliver all segments."""
        adapter = _make_adapter()
        long_content = "x" * 5000

        # Mock _edit_overflow_split to return a success result
        from gateway.platforms.base import SendResult

        adapter._edit_overflow_split = AsyncMock(
            return_value=SendResult(success=True, message_id="1001")
        )

        result = await adapter.edit_message(
            chat_id="12345",
            message_id="999",
            content=long_content,
            finalize=True,
        )

        # Must have called _edit_overflow_split
        adapter._edit_overflow_split.assert_called_once()
        assert result.message_id == "1001"

    @pytest.mark.asyncio
    async def test_under_limit_unchanged(self):
        """Content under the limit should not be truncated."""
        adapter = _make_adapter()
        short_content = "hello"  # well under 4096

        adapter._bot.edit_message_text = AsyncMock()
        adapter._edit_overflow_split = AsyncMock()

        result = await adapter.edit_message(
            chat_id="12345",
            message_id="999",
            content=short_content,
            finalize=False,
        )

        adapter._edit_overflow_split.assert_not_called()
        call_args = adapter._bot.edit_message_text.call_args
        edited_text = call_args.kwargs.get("text") or call_args[1].get("text")
        assert edited_text == "hello"
        assert result.success is True
