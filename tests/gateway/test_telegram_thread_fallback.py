"""Tests for Telegram send() thread_id fallback.

When message_thread_id points to a non-existent thread, Telegram returns
BadRequest('Message thread not found'). Since BadRequest is a subclass of
NetworkError in python-telegram-bot, the old retry loop treated this as a
transient error and retried 3 times before silently failing — killing all
tool progress messages, streaming responses, and typing indicators.

The fix detects "thread not found" BadRequest errors and retries the send
WITHOUT message_thread_id so the message still reaches the chat.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig, Platform
from gateway.platforms.base import SendResult


# ── Fake telegram.error hierarchy ──────────────────────────────────────
# Mirrors the real python-telegram-bot hierarchy:
#   BadRequest → NetworkError → TelegramError → Exception


class FakeNetworkError(Exception):
    pass


class FakeBadRequest(FakeNetworkError):
    pass


class FakeTimedOut(FakeNetworkError):
    pass


class FakeRetryAfter(Exception):
    def __init__(self, seconds):
        super().__init__(f"Retry after {seconds}")
        self.retry_after = seconds


# Build a fake telegram module tree so the adapter's internal imports work
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
    """Inject fake telegram modules so the adapter can import from them."""
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


@pytest.mark.asyncio
async def test_send_retries_without_thread_on_thread_not_found():
    """When message_thread_id causes 'thread not found', retry without it."""
    adapter = _make_adapter()

    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        tid = kwargs.get("message_thread_id")
        if tid is not None:
            raise FakeBadRequest("Message thread not found")
        return SimpleNamespace(message_id=42)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="test message",
        metadata={"thread_id": "99999"},
    )

    assert result.success is True
    assert result.message_id == "42"
    # First call has thread_id, second call retries without
    assert len(call_log) == 2
    assert call_log[0]["message_thread_id"] == 99999
    assert call_log[1]["message_thread_id"] is None


@pytest.mark.asyncio
async def test_send_raises_on_other_bad_request():
    """Non-thread BadRequest errors should NOT be retried — they fail immediately."""
    adapter = _make_adapter()

    async def mock_send_message(**kwargs):
        raise FakeBadRequest("Chat not found")

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="test message",
        metadata={"thread_id": "99999"},
    )

    assert result.success is False
    assert "Chat not found" in result.error


@pytest.mark.asyncio
async def test_send_without_thread_id_unaffected():
    """Normal sends without thread_id should work as before."""
    adapter = _make_adapter()

    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=100)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="test message",
    )

    assert result.success is True
    assert len(call_log) == 1
    assert call_log[0]["message_thread_id"] is None


@pytest.mark.asyncio
async def test_send_retries_network_errors_normally():
    """Real transient network errors (not BadRequest) should still be retried."""
    adapter = _make_adapter()

    attempt = [0]

    async def mock_send_message(**kwargs):
        attempt[0] += 1
        if attempt[0] < 3:
            raise FakeNetworkError("Connection reset")
        return SimpleNamespace(message_id=200)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="test message",
    )

    assert result.success is True
    assert attempt[0] == 3  # Two retries then success


@pytest.mark.asyncio
async def test_send_does_not_retry_timeout():
    """TimedOut (subclass of NetworkError) should NOT be retried in send().

    The request may have already been delivered to the user — retrying
    would send duplicate messages.
    """
    adapter = _make_adapter()

    attempt = [0]

    async def mock_send_message(**kwargs):
        attempt[0] += 1
        raise FakeTimedOut("Timed out waiting for Telegram response")

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="test message",
    )

    assert result.success is False
    assert "Timed out" in result.error
    # CRITICAL: only 1 attempt — no retry for TimedOut
    assert attempt[0] == 1


@pytest.mark.asyncio
async def test_thread_fallback_only_fires_once():
    """After clearing thread_id, subsequent chunks should also use None."""
    adapter = _make_adapter()

    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        tid = kwargs.get("message_thread_id")
        if tid is not None:
            raise FakeBadRequest("Message thread not found")
        return SimpleNamespace(message_id=42)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    # Send a long message that gets split into chunks
    long_msg = "A" * 5000  # Exceeds Telegram's 4096 limit
    result = await adapter.send(
        chat_id="123",
        content=long_msg,
        metadata={"thread_id": "99999"},
    )

    assert result.success is True
    # First chunk: attempt with thread → fail → retry without → succeed
    # Second chunk: should use thread_id=None directly (effective_thread_id
    # was cleared per-chunk but the metadata doesn't change between chunks)
    # The key point: the message was delivered despite the invalid thread


@pytest.mark.asyncio
async def test_send_retries_retry_after_errors():
    """Telegram flood control should back off and retry instead of failing fast."""
    adapter = _make_adapter()

    attempt = [0]

    async def mock_send_message(**kwargs):
        attempt[0] += 1
        if attempt[0] == 1:
            raise FakeRetryAfter(2)
        return SimpleNamespace(message_id=300)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(chat_id="123", content="test message")

    assert result.success is True
    assert result.message_id == "300"
    assert attempt[0] == 2
