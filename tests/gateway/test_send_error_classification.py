"""Tests for structured send-error classification (SendResult.error_kind).

Covers the platform-neutral ``classify_send_error`` vocabulary in
``gateway/platforms/base.py`` and its wiring into the Telegram adapter's
``send()`` failure path, so consumers can branch on a typed category instead
of substring-matching the raw provider message.
"""

import pytest

from gateway.platforms.base import (
    SEND_ERROR_KINDS,
    SendResult,
    classify_send_error,
)


class _FakeBadRequest(Exception):
    """Stand-in for a provider BadRequest carrying a message string."""


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Message_too_long", "too_long"),
        ("Bad Request: message is too long", "too_long"),
        ("Bad Request: can't parse entities: unsupported start tag", "bad_format"),
        ("Bad Request: can't find end of the entity", "bad_format"),
        ("Forbidden: bot was blocked by the user", "forbidden"),
        ("Forbidden: user is deactivated", "forbidden"),
        ("Bad Request: not enough rights to send text messages", "forbidden"),
        ("Bad Request: chat not found", "not_found"),
        ("Bad Request: message to edit not found", "not_found"),
        ("Too Many Requests: retry after 12", "rate_limited"),
        ("Flood control exceeded", "rate_limited"),
        ("ConnectError: connection refused", "transient"),
        ("ConnectTimeout", "transient"),
        ("some entirely novel provider message", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_send_error_text(text, expected):
    assert classify_send_error(None, text) == expected


def test_every_classification_is_in_the_vocabulary():
    samples = [
        "message_too_long",
        "can't parse entities",
        "forbidden",
        "chat not found",
        "flood",
        "connecterror",
        "mystery",
        "",
    ]
    for s in samples:
        assert classify_send_error(None, s) in SEND_ERROR_KINDS


def test_telegram_send_failure_populates_error_kind():
    """Telegram send() failures carry a typed error_kind alongside error."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    cfg = PlatformConfig(enabled=True, token="fake-token", extra={})
    adapter = TelegramAdapter(cfg)

    # Minimal bot whose send_message raises a parse/entity rejection.
    bot = MagicMock()
    bot.send_message = AsyncMock(
        side_effect=Exception("Bad Request: can't parse entities: bad tag")
    )
    bot.send_chat_action = AsyncMock()
    # Force the legacy (non-rich) path and a connected bot.
    adapter._bot = bot
    adapter._rich_messages_enabled = False

    result = asyncio.run(adapter.send("123", "<b>broken"))
    assert result.success is False
    # Telegram has a plain-text fallback for parse errors inside the send loop,
    # so a raw parse failure that still escapes is classified for consumers.
    assert result.error_kind in SEND_ERROR_KINDS
    assert result.error_kind != "unknown" or result.error


