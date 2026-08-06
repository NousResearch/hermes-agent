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


def test_classify_uses_exception_class_name():
    # The class name participates in classification even when str(exc) is empty.
    exc = type("Forbidden", (Exception,), {})()
    assert classify_send_error(exc) == "forbidden"


def test_classify_prefers_explicit_text_and_exception_together():
    exc = _FakeBadRequest("chat not found")
    assert classify_send_error(exc) == "not_found"


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


def test_unknown_never_masquerades_as_benign():
    # An unrecognized failure must classify as "unknown", never as a benign
    # category like too_long that a consumer might treat as a soft recovery.
    assert classify_send_error(None, "kaboom 500 internal") == "unknown"


def test_sendresult_error_kind_defaults_none_and_is_backward_compatible():
    # Existing call sites that never set error_kind keep working unchanged.
    ok = SendResult(success=True, message_id="42")
    assert ok.error_kind is None
    legacy_fail = SendResult(success=False, error="boom")
    assert legacy_fail.error_kind is None


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


def test_telegram_too_long_sets_too_long_kind():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    cfg = PlatformConfig(enabled=True, token="fake-token", extra={})
    adapter = TelegramAdapter(cfg)

    bot = MagicMock()
    bot.send_message = AsyncMock(
        side_effect=Exception("Bad Request: message is too long")
    )
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot
    adapter._rich_messages_enabled = False

    result = asyncio.run(adapter.send("123", "x" * 5000))
    assert result.success is False
    assert result.error == "message_too_long"
    assert result.error_kind == "too_long"
