"""Reproduce: TimedOut on send_message causes duplicate message delivery.

When Telegram's server receives and delivers a message but the HTTP
response times out, the adapter retries the same send_message call.
The user receives the same message multiple times.

This test simulates the exact scenario:
1. send_message delivers the message to the user (side effect)
2. send_message raises TimedOut (HTTP response didn't come back in time)
3. The retry loop calls send_message AGAIN with the same content
4. The user now has 2 copies of the same message

The fix: TimedOut should NOT trigger a retry for non-idempotent
send_message calls, because the request may have already succeeded.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig, Platform
from gateway.platforms.base import SendResult


# ── Fake telegram.error hierarchy ──────────────────────────────────────

class FakeNetworkError(Exception):
    pass

class FakeBadRequest(FakeNetworkError):
    pass

class FakeTimedOut(FakeNetworkError):
    """Simulates telegram.error.TimedOut.

    In python-telegram-bot, TimedOut is a subclass of NetworkError.
    This means the retry loop's `except NetworkError` catches it,
    treating it as a transient error worth retrying — but for
    send_message, the request may have already been delivered.
    """
    pass


# Build fake telegram module tree
_fake_telegram = types.ModuleType("telegram")
_fake_telegram_error = types.ModuleType("telegram.error")
_fake_telegram_error.NetworkError = FakeNetworkError
_fake_telegram_error.BadRequest = FakeBadRequest
_fake_telegram_error.TimedOut = FakeTimedOut
_fake_telegram.error = _fake_telegram_error
_fake_telegram_constants = types.ModuleType("telegram.constants")
_fake_telegram_constants.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
_fake_telegram.constants = _fake_telegram_constants


@pytest.fixture(autouse=True)
def _inject_fake_telegram(monkeypatch):
    monkeypatch.setitem(sys.modules, "telegram", _fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", _fake_telegram_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", _fake_telegram_constants)


def _make_adapter():
    from gateway.platforms.telegram import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._dm_topics = {}
    adapter._dm_topics_config = []
    adapter._reply_to_mode = "first"
    adapter._fallback_ips = []
    adapter._polling_conflict_count = 0
    adapter._polling_network_error_count = 0
    adapter._polling_error_callback_ref = None
    adapter.platform = Platform.TELEGRAM
    return adapter


# ── The reproduction ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_after_delivery_must_not_duplicate():
    """Reproduce: message delivered but TimedOut → retry → duplicate.

    Scenario:
        - User sends "hello" to the bot
        - Bot generates response "Here is your answer"
        - send_message() delivers it to the user's Telegram chat
        - BUT the HTTP response times out (FakeTimedOut)
        - The retry loop catches NetworkError (parent of TimedOut)
        - It calls send_message() AGAIN with the same text
        - User sees "Here is your answer" TWICE

    This test counts how many times send_message is called.
    If the fix works, it should be called exactly once (no retry).
    """
    adapter = _make_adapter()

    delivered_messages = []  # simulates what the user actually receives

    async def mock_send_message(**kwargs):
        text = kwargs.get("text", "")
        # Message IS delivered to the user (side effect happens)
        delivered_messages.append(text)
        # But the HTTP response times out
        raise FakeTimedOut("Timed out")

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123456",
        content="Here is your answer",
    )

    # The send failed (TimedOut), that's expected
    assert result.success is False
    assert "Timed out" in result.error

    # CRITICAL ASSERTION: the message must be delivered AT MOST once.
    # Before the fix, the retry loop would call send_message up to 3
    # times (inner retry) × 1 (outer retry) = 3 duplicate deliveries.
    assert len(delivered_messages) == 1, (
        f"DUPLICATE DELIVERY: send_message was called {len(delivered_messages)} times. "
        f"User would receive {len(delivered_messages)} copies of the same message. "
        f"Messages delivered: {delivered_messages}"
    )


@pytest.mark.asyncio
async def test_connection_error_still_retries_safely():
    """ConnectionError (request never reached server) should still retry.

    Unlike TimedOut, a ConnectionError means the TCP connection failed
    before the request was sent — so retrying is safe and won't cause
    duplicates.
    """
    adapter = _make_adapter()

    delivered_messages = []
    attempt = [0]

    async def mock_send_message(**kwargs):
        attempt[0] += 1
        text = kwargs.get("text", "")
        if attempt[0] < 3:
            # Connection failed — message was NOT delivered
            raise FakeNetworkError("Connection reset by peer")
        # Third attempt succeeds
        delivered_messages.append(text)
        return SimpleNamespace(message_id=42)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123456",
        content="Here is your answer",
    )

    assert result.success is True
    # Message delivered exactly once (after 2 failed connection attempts)
    assert len(delivered_messages) == 1
    assert attempt[0] == 3  # 2 failures + 1 success
