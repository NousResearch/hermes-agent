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
        # Slack permanently-dead-target signatures (#82791-adjacent scope: classify_send_error was
        # Telegram-only; these are the well-documented, unambiguous Slack Web API error codes for a
        # channel/DM that no longer exists or can never receive messages again).
        ("The server responded with: {'ok': False, 'error': 'channel_not_found'}", "not_found"),
        ("The server responded with: {'ok': False, 'error': 'is_archived'}", "forbidden"),
        # Slack rate-limit errors must NOT be swept into a dead-target classification (neither
        # "forbidden" nor "not_found" -- both are in gateway/dead_targets.py's _DEAD_ERROR_KINDS).
        ("SlackApiError: 429 Too Many Requests, Retry-After: 30", "rate_limited"),
        # account_inactive means the BOT'S OWN token/account was deactivated -- every future send
        # (to any chat) would fail the same way, so it is deliberately NOT treated as evidence that
        # this one target is dead (see gateway/platforms/base.py's _SEND_ERROR_CLASSIFIERS comment).
        ("The server responded with: {'ok': False, 'error': 'account_inactive'}", "unknown"),
    ],
)
def test_classify_send_error_text(text, expected):
    assert classify_send_error(None, text) == expected


class TestSlackDeadTargetClassification:
    """Scoped follow-up to the Telegram-only classify_send_error table: Discord's 403/50007
    ("Cannot send messages to this user") already matched the pre-existing generic "forbidden"
    substring and needed no change. Discord's 10003 ("Unknown Channel") is deliberately NOT
    added: Discord threads are their own channel ids, so the identical error text also fires when
    only a thread (not the parent channel) was deleted, and the dead-target short-circuit in
    gateway/delivery.py would then permanently blacklist a live channel with no way to self-heal.
    """

    def test_channel_not_found_is_chat_level_not_found(self):
        from gateway.platforms.base import is_chat_level_not_found

        # Unlike Discord, a Slack thread is a message field (thread_ts) inside its parent
        # channel, not a separate addressable object -- channel_not_found can only mean the
        # channel/DM itself is gone, never "just this thread".
        assert is_chat_level_not_found(
            error_text="The server responded with: {'ok': False, 'error': 'channel_not_found'}"
        ) is True

    def test_discord_forbidden_blocked_dm_already_classified_dead(self):
        """Documents existing behavior (no code change needed): Discord's Forbidden/50007 text
        already contains the generic "forbidden" substring."""
        assert classify_send_error(
            None, "403 Forbidden (error code: 50007): Cannot send messages to this user"
        ) == "forbidden"

    def test_discord_unknown_channel_deliberately_not_classified_dead(self):
        """Documents the deliberate exclusion: ambiguous between a dead channel and a dead
        thread inside a live channel, so it must stay 'unknown' (never dead-target-marked)."""
        assert classify_send_error(
            None, "404 Not Found (error code: 10003): Unknown Channel"
        ) == "unknown"

    @pytest.mark.parametrize(
        "text",
        [
            "SlackApiError: 429 Too Many Requests, Retry-After: 30",
            "The server responded with: {'ok': False, 'error': 'ratelimited'}",
            "httpx.ReadTimeout: connection timed out",
        ],
    )
    def test_slack_transient_errors_are_never_dead_target_kinds(self, text):
        """A clearly-transient Slack failure (rate limit, timeout) must never fall into
        gateway/dead_targets.py's DEAD_ERROR_KINDS ('forbidden'/'not_found'), or a temporary
        429/timeout would wrongly and permanently blacklist a perfectly-alive target."""
        from gateway.dead_targets import _DEAD_ERROR_KINDS

        assert classify_send_error(None, text) not in _DEAD_ERROR_KINDS


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


