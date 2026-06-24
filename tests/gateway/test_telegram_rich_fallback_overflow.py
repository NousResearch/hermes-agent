"""Regression tests for Telegram streaming overflow + rich fallback.  (#51961)"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter


def _msg(message_id: int | str) -> SimpleNamespace:
    return SimpleNamespace(message_id=message_id)


@pytest.fixture
def adapter() -> TelegramAdapter:
    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    a._bot = MagicMock()
    a._bot.do_api_request = AsyncMock()  # makes _bot_supports_rich() True
    return a


@pytest.mark.asyncio
async def test_finalize_rich_failure_falls_through_to_overflow_split(adapter):
    """When _try_edit_rich fails for oversized content, edit_message must
    fall through to _edit_overflow_split instead of returning the failure.

    Before the fix, the rich failure was returned directly, leaving the user
    with a truncated 4096-char preview and no continuation messages.  (#51961)
    """
    oversized = "x" * 6000
    # _rich_eligible needs _needs_rich_rendering → True.  Patch it.
    with patch.object(adapter, "_rich_eligible", return_value=True):
        # _try_edit_rich returns a transient failure (non-None, success=False)
        with patch.object(
            adapter,
            "_try_edit_rich",
            return_value=SendResult(success=False, error="timed out", retryable=True),
        ):
            # _edit_overflow_split should be called — mock it to track the call.
            with patch.object(
                adapter,
                "_edit_overflow_split",
                new_callable=AsyncMock,
                return_value=SendResult(
                    success=True,
                    message_id="456",
                    continuation_message_ids=("457",),
                ),
            ) as mock_split:
                result = await adapter.edit_message(
                    "123", "456", oversized, finalize=True,
                )

    # The overflow split must have been called with the full oversized content.
    mock_split.assert_awaited_once()
    call_kwargs = mock_split.call_args
    assert call_kwargs[0][2] == oversized  # content arg
    assert call_kwargs[1]["finalize"] is True

    # The overall result must be success (from the split, not the rich failure).
    assert result.success is True
    assert result.message_id == "456"


@pytest.mark.asyncio
async def test_finalize_rich_failure_returns_directly_when_content_within_limit(adapter):
    """When _try_edit_rich fails for content within the legacy limit,
    the failure must be returned directly (no overflow split needed)."""
    short_content = "hello world"
    with patch.object(adapter, "_rich_eligible", return_value=True):
        with patch.object(
            adapter,
            "_try_edit_rich",
            return_value=SendResult(success=False, error="timed out", retryable=True),
        ):
            with patch.object(
                adapter, "_edit_overflow_split", new_callable=AsyncMock,
            ) as mock_split:
                result = await adapter.edit_message(
                    "123", "456", short_content, finalize=True,
                )

    # No overflow split — content fits in one message.
    mock_split.assert_not_awaited()
    assert result.success is False
    assert result.retryable is True


@pytest.mark.asyncio
async def test_finalize_rich_success_returns_immediately(adapter):
    """When _try_edit_rich succeeds, the result must be returned directly
    without reaching the overflow check (existing behavior, regression guard)."""
    content = "x" * 6000
    with patch.object(adapter, "_rich_eligible", return_value=True):
        with patch.object(
            adapter,
            "_try_edit_rich",
            return_value=SendResult(success=True, message_id="456"),
        ):
            with patch.object(
                adapter, "_edit_overflow_split", new_callable=AsyncMock,
            ) as mock_split:
                result = await adapter.edit_message(
                    "123", "456", content, finalize=True,
                )

    # Rich succeeded — no overflow split needed.
    mock_split.assert_not_awaited()
    assert result.success is True
    assert result.message_id == "456"
