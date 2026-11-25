"""Regression coverage for partial Telegram overflow delivery."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.stream_consumer import GatewayStreamConsumer


def _message(message_id: int | str) -> SimpleNamespace:
    return SimpleNamespace(message_id=message_id)


@pytest.fixture
def telegram_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    object.__setattr__(adapter, "MAX_MESSAGE_LENGTH", 160)
    return adapter


@pytest.mark.asyncio
async def test_edit_overflow_split_reports_later_partial_failure_after_some_continuations_land(telegram_adapter):
    """Partial metadata tracks the last delivered continuation before failure."""
    content = "word " * 120
    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)
    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[
            _message(202),
            RuntimeError("telegram send failed"),
            RuntimeError("telegram send failed"),
        ]
    )

    result = await telegram_adapter._edit_overflow_split(
        "12345", "201", content, finalize=False, metadata={"thread_id": "77"}
    )

    assert result.success is False
    assert result.message_id == "202"
    assert result.raw_response["partial_overflow"] is True
    assert result.raw_response["delivered_chunks"] == 2
    assert result.raw_response["last_message_id"] == "202"
    assert result.continuation_message_ids == ("202",)


class BadRequest(Exception):
    """Name matches ``_is_bad_request_error`` when python-telegram-bot is absent."""


@pytest.mark.asyncio
async def test_finalize_overflow_timeout_is_not_resent_as_plain_text(telegram_adapter):
    """A MarkdownV2 continuation whose reply timed out may already be in the chat.

    Falling through to plain text posts a second copy and reports success.
    """
    calls: list[object] = []

    async def send_message(**kwargs):
        calls.append(kwargs.get("parse_mode"))
        if kwargs.get("parse_mode") is not None:
            raise TimeoutError("Timed out")
        return _message(602)

    telegram_adapter._bot.send_message = send_message
    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)

    result = await telegram_adapter._edit_overflow_split(
        "12345", "201", "word " * 120, finalize=True, metadata={"thread_id": "77"}
    )

    assert len(calls) == 1 and calls[0] is not None
    assert result.success is False
    assert result.retryable is False
    assert result.message_id == "201"
    assert result.raw_response["partial_overflow"] is True
    assert result.raw_response["delivery_uncertain"] is True
    assert result.raw_response["delivered_chunks"] == 1
    assert len(result.raw_response["delivered_prefix"]) > telegram_adapter.MAX_MESSAGE_LENGTH
    assert "undelivered_chunks" not in result.raw_response


@pytest.mark.asyncio
async def test_finalize_overflow_entity_refusal_still_falls_back_to_plain(telegram_adapter):
    """A real entity refusal posted nothing, so the plain-text continuation still lands."""
    calls: list[object] = []

    async def send_message(**kwargs):
        calls.append(kwargs.get("parse_mode"))
        if kwargs.get("parse_mode") is not None:
            raise BadRequest("Bad Request: can't parse entities")
        return _message(700 + len(calls))

    telegram_adapter._bot.send_message = send_message
    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)

    result = await telegram_adapter._edit_overflow_split(
        "12345", "201", "word " * 120, finalize=True, metadata={"thread_id": "77"}
    )

    assert calls and calls[0] is not None and None in calls
    assert result.success is True


