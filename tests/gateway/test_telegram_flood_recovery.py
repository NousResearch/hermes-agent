"""Tests for Telegram flood-control recovery.

Issue: gateway.log 2026-06-21 23:39-23:47 showed:

    23:39:43 send OK (3027 chars)
    23:46:52 send FAILED after 2 retries: "Flood control exceeded. Retry in 26 seconds"
    23:47:00 send FAILED after 2 retries: "Retry in 14 seconds"

Telegram's per-chat send rate is ~1 msg/sec; rapid back-and-forth produces
14-26s flood windows.  Pre-fix the adapter aborted on any wait > 5s —
silently losing every flood-control message longer than that.  These tests
cover the recovery path:

  1. ``edit_message`` honors inline flood waits up to 60s.
  2. ``edit_message`` surfaces waits > 60s as ``retryable=True`` so the
     outer ``_send_with_retry`` loop can keep backing off.
  3. The outer loop already honors ``retryable=True`` (verified by
     importing and reading the base class — no change needed there).
  4. Inter-chunk spacing in the multi-message ``send()`` loop sleeps
     ``min(1.0s, flood_window_remaining)`` between chunks.
  5. The per-chat flood-window map records ``retry_after`` so chunk N+1
     stays below Telegram's ~1 msg/s rate ceiling after chunk N hit the
     limit.
  6. ``send()`` raises ``_RetryableFloodError`` after 3 inline attempts so
     the outer error catch returns ``retryable=True``.
  7. Backwards compatibility: ``wait <= 5s`` behavior unchanged.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest

# All tests use the existing _make_adapter pattern from
# test_telegram_thread_fallback.py — bypasses __init__ to avoid PTB /
# network setup.
# ---------------------------------------------------------------------------


class _RetryAfterError(Exception):
    """Mimics ``telegram.error.RetryAfter`` — exception with .retry_after."""

    def __init__(self, seconds):
        super().__init__(f"Retry after {seconds}")
        self.retry_after = seconds


# Build a minimal fake telegram module tree (mirrors test_telegram_thread_fallback.py)
_fake_telegram = types.ModuleType("telegram")
_fake_telegram.Update = object
_fake_telegram.Bot = object
_fake_telegram.Message = object
_fake_telegram.InlineKeyboardButton = object
_fake_telegram.InlineKeyboardMarkup = object
_fake_telegram.InputMediaPhoto = object
_fake_telegram_error = types.ModuleType("telegram.error")
_fake_telegram_error.NetworkError = type("NetworkError", (Exception,), {})
_fake_telegram_error.BadRequest = type("BadRequest", (Exception,), {})
_fake_telegram_error.TimedOut = type("TimedOut", (Exception,), {})
_fake_telegram_error.RetryAfter = _RetryAfterError
_fake_telegram.error = _fake_telegram_error
_fake_telegram_constants = types.ModuleType("telegram.constants")
_fake_telegram_constants.ParseMode = SimpleNamespace(
    MARKDOWN_V2="MarkdownV2", MARKDOWN="Markdown", HTML="HTML"
)
_fake_telegram_constants.ChatType = SimpleNamespace(
    GROUP="group", SUPERGROUP="supergroup", CHANNEL="channel", PRIVATE="private"
)
_fake_telegram.constants = _fake_telegram_constants
_fake_telegram_ext = types.ModuleType("telegram.ext")
_fake_telegram_ext.Application = object
_fake_telegram_ext.CommandHandler = object
_fake_telegram_ext.CallbackQueryHandler = object
_fake_telegram_ext.MessageHandler = object
_fake_telegram_ext.TypeHandler = object
_fake_telegram_ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
_fake_telegram_ext.filters = object
_fake_telegram_request = types.ModuleType("telegram.request")
_fake_telegram_request.HTTPXRequest = object


@pytest.fixture(autouse=True)
def _inject_fake_telegram(monkeypatch):
    """Install fake telegram modules before the adapter is imported."""
    monkeypatch.setitem(sys.modules, "telegram", _fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", _fake_telegram_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", _fake_telegram_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", _fake_telegram_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", _fake_telegram_request)


def _make_adapter():
    """Construct a TelegramAdapter bypassing __init__ — same pattern as
    tests/gateway/test_telegram_thread_fallback.py."""
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from gateway.config import Platform, PlatformConfig

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
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
    # ``name`` is a property in the base class — derive from config like
    # other init-bypass tests do, instead of forcing an attribute.
    return adapter


# Stub asyncio.sleep globally for the test — we'll record the calls instead.
_sleep_log: list[float] = []
_real_sleep = asyncio.sleep


async def _fake_sleep(seconds):
    """Record the sleep duration but don't actually wait (tests run fast)."""
    _sleep_log.append(float(seconds))
    # Yield once so any concurrent tasks can run, but don't actually block.
    await _real_sleep(0)


@pytest.fixture(autouse=True)
def _patch_sleep(monkeypatch):
    """Stub asyncio.sleep so flood waits don't slow the test suite."""
    _sleep_log.clear()
    monkeypatch.setattr("asyncio.sleep", _fake_sleep)
    yield
    _sleep_log.clear()


# ===========================================================================
# 1. Constants & state init
# ===========================================================================


def test_flood_constants_match_spec():
    """The recovery ceiling must be 60s, not the historical 5s."""
    from plugins.platforms.telegram.adapter import TelegramAdapter
    assert TelegramAdapter._FLOOD_WAIT_CEILING_SECONDS == 60.0
    assert TelegramAdapter._FLOOD_WINDOW_TTL_SECONDS == 60.0
    assert TelegramAdapter._FLOOD_CHUNK_SPACING_SECONDS == 1.0


def test_flood_windows_map_lazy_init_for_init_bypass_adapters():
    """``_make_adapter`` (used across the suite) bypasses __init__ — the
    flood-window map must be created on first access, not crash with
    ``AttributeError``."""
    adapter = _make_adapter()
    # Even though we pre-set ``adapter._flood_windows = {}`` above, the
    # production code path uses ``_flood_windows_map()`` which is robust
    # against ``None`` (some other tests pass through adapters without it).
    adapter._flood_windows = None  # simulate __init__ bypass without map
    store = adapter._flood_windows_map()
    assert store == {}
    # Subsequent access returns the same dict.
    store["key"] = 123.0
    assert adapter._flood_windows_map()["key"] == 123.0


# ===========================================================================
# 2. Backwards compatibility — wait <= 5s
# ===========================================================================


@pytest.mark.asyncio
async def test_edit_message_inline_succeeds_for_short_flood_wait():
    """A flood with retry_after=2 must still be honored inline (existing
    behavior).  Confirms we didn't regress the historical happy path."""
    adapter = _make_adapter()
    call_count = [0]
    sleeps_during_test = []

    async def mock_edit(chat_id, message_id, text):
        call_count[0] += 1
        if call_count[0] == 1:
            raise _RetryAfterError(2)
        return SimpleNamespace(message_id=message_id)

    # Patch asyncio.sleep *inside* the adapter module so the production
    # sleep gets recorded too.
    real_sleep = asyncio.sleep

    async def record_sleep(seconds):
        sleeps_during_test.append(float(seconds))
        await real_sleep(0)

    adapter._bot = SimpleNamespace(edit_message_text=mock_edit)

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", record_sleep):
        result = await adapter.edit_message(
            chat_id="123", message_id="999", content="hello world"
        )

    assert result.success is True
    assert result.message_id == "999"
    assert call_count[0] == 2  # one initial + one retry
    # The inline sleep was honored.
    assert 2.0 in sleeps_during_test, (
        f"Expected an inline 2s sleep; observed sleeps={sleeps_during_test}"
    )


# ===========================================================================
# 3. Edit_message — wait in the 5-60s band (inline retry succeeds)
# ===========================================================================


@pytest.mark.asyncio
async def test_edit_message_inline_succeeds_for_mid_flood_wait():
    """A flood with retry_after=6s — historically aborted.  Now sleeps
    inline and retries."""
    adapter = _make_adapter()
    call_count = [0]

    async def mock_edit(chat_id, message_id, text):
        call_count[0] += 1
        if call_count[0] == 1:
            raise _RetryAfterError(6)
        return SimpleNamespace(message_id=message_id)

    adapter._bot = SimpleNamespace(edit_message_text=mock_edit)

    result = await adapter.edit_message(
        chat_id="123", message_id="999", content="hello world"
    )

    assert result.success is True
    assert result.message_id == "999"
    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_edit_message_inline_succeeds_for_typical_30s_flood():
    """A flood with retry_after=30s (gateway.log 23:46:52 evidence) — was
    aborting with ``flood_control:30.0``.  Now sleeps inline and succeeds."""
    adapter = _make_adapter()
    call_count = [0]

    async def mock_edit(chat_id, message_id, text):
        call_count[0] += 1
        if call_count[0] == 1:
            raise _RetryAfterError(30)
        return SimpleNamespace(message_id=message_id)

    adapter._bot = SimpleNamespace(edit_message_text=mock_edit)

    result = await adapter.edit_message(
        chat_id="123", message_id="999", content="hello world"
    )

    assert result.success is True
    assert result.message_id == "999"
    assert call_count[0] == 2
    # The flood window should be recorded so subsequent chunks space
    # themselves (see test_inter_chunk_spacing_* below).
    remaining = adapter._flood_window_remaining("123", None)
    assert remaining > 0.0, "Flood window not recorded after 30s retry_after"


# ===========================================================================
# 4. Edit_message — wait > 60s surfaces retryable=True
# ===========================================================================


@pytest.mark.asyncio
async def test_edit_message_surfaces_retryable_for_wait_above_ceiling():
    """A flood with retry_after=90s exceeds the 60s inline ceiling.
    The edit_message call must return ``success=False, retryable=True``
    so the base class ``_send_with_retry`` outer loop keeps backing off."""
    adapter = _make_adapter()

    async def mock_edit(chat_id, message_id, text):
        raise _RetryAfterError(90)

    adapter._bot = SimpleNamespace(edit_message_text=mock_edit)

    result = await adapter.edit_message(
        chat_id="123", message_id="999", content="hello world"
    )

    assert result.success is False
    assert result.retryable is True
    assert "flood_control" in (result.error or "")
    assert "90" in (result.error or "")


# ===========================================================================
# 5. Outer retry honors retryable=True — base class behavior we depend on
# ===========================================================================


@pytest.mark.asyncio
async def test_outer_send_with_retry_retries_retryable_results():
    """The base class ``_send_with_retry`` already retries on
    ``result.retryable=True`` with exponential backoff.  This test pins
    that behavior so we know the edit_message flood path's
    ``retryable=True`` will actually be picked up — the contract
    Change 2 depends on."""
    from gateway.platforms.base import SendResult

    adapter = _make_adapter()
    # First two attempts return retryable; third succeeds.
    attempts = [0]

    async def mock_send(**kwargs):
        attempts[0] += 1
        if attempts[0] < 3:
            return SendResult(
                success=False,
                error=f"flood_control:{26}",
                retryable=True,
            )
        return SendResult(success=True, message_id="42")

    adapter.send = mock_send
    # Patch send_with_retry's sleep path so we don't wait 2+4 seconds
    real_sleep = asyncio.sleep
    async def fast_sleep(seconds):
        await real_sleep(0)
    with patch("asyncio.sleep", fast_sleep):
        result = await adapter._send_with_retry(
            chat_id="123", content="hello"
        )

    assert result.success is True
    assert result.message_id == "42"
    assert attempts[0] == 3


# ===========================================================================
# 6. Inter-chunk spacing — send() paces chunks after a flood
# ===========================================================================


@pytest.mark.asyncio
async def test_send_chunked_reply_spacing_after_flood():
    """A 3-chunk reply sent after a flood should observe inter-chunk
    pacing.  First chunk triggers RetryAfter(8); second and third chunks
    must be delayed by ~min(1.0, window_remaining)."""
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from gateway.platforms.base import SendResult

    adapter = _make_adapter()
    # Use a content that splits into exactly 3 chunks.
    long_content = ("A" * (TelegramAdapter.MAX_MESSAGE_LENGTH + 100)) * 3
    # Sanity: confirm we get >1 chunk.
    chunks = adapter.truncate_message(
        adapter.format_message(long_content), TelegramAdapter.MAX_MESSAGE_LENGTH, len_fn=lambda s: len(s),
    )
    assert len(chunks) >= 2, "test setup must produce multiple chunks"

    # Capture sleeps triggered between chunks (filter out the flood sleeps).
    chunk_loop_sleeps: list[float] = []

    real_sleep = asyncio.sleep

    async def record_sleep(seconds):
        chunk_loop_sleeps.append(float(seconds))
        await real_sleep(0)

    # Send_attempts needed: chunks >= 3 — first chunk hits flood once,
    # then succeeds; subsequent chunks go through cleanly.
    send_attempts = [0]

    async def mock_send_message(**kwargs):
        send_attempts[0] += 1
        # First chunk: hit flood once with retry_after=8, then succeed.
        # Subsequent chunks: succeed immediately.
        if send_attempts[0] == 1:
            raise _RetryAfterError(8)
        return SimpleNamespace(message_id=send_attempts[0])

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", record_sleep):
        result = await adapter.send(chat_id="123", content=long_content)

    assert result.success is True, f"send failed: {result.error}"
    # We should see at least one inter-chunk spacing sleep of <1.0s.
    spacing_sleeps = [s for s in chunk_loop_sleeps if 0.0 < s <= 1.0]
    assert len(spacing_sleeps) >= 1, (
        f"Expected inter-chunk spacing sleeps <=1.0s; observed={chunk_loop_sleeps}"
    )


@pytest.mark.asyncio
async def test_send_no_spacing_when_no_active_flood_window():
    """If no flood window is active, the chunk loop should NOT sleep
    between chunks — pre-fix behavior was already no-spacing."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = _make_adapter()
    long_content = "B" * (TelegramAdapter.MAX_MESSAGE_LENGTH + 100)
    chunks = adapter.truncate_message(
        adapter.format_message(long_content),
        TelegramAdapter.MAX_MESSAGE_LENGTH,
        len_fn=lambda s: len(s),
    )
    assert len(chunks) >= 2

    async def mock_send_message(**kwargs):
        return SimpleNamespace(message_id=1)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    real_sleep = asyncio.sleep
    sleeps_observed: list[float] = []

    async def record_sleep(seconds):
        sleeps_observed.append(float(seconds))
        await real_sleep(0)

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", record_sleep):
        result = await adapter.send(chat_id="123", content=long_content)

    assert result.success is True
    # No chunk-level pacing sleep expected when flood window is empty.
    assert all(s == 0.0 for s in sleeps_observed), (
        f"Unexpected non-zero sleeps without flood window: {sleeps_observed}"
    )


# ===========================================================================
# 7. Flood window state — recording & expiry
# ===========================================================================


def test_record_flood_window_caps_at_ttl():
    """A retry_after > _FLOOD_WINDOW_TTL_SECONDS must be capped, so the
    map doesn't outlive its usefulness across long bot uptimes."""
    adapter = _make_adapter()
    adapter._record_flood_window("chat", None, 9999.0)
    remaining = adapter._flood_window_remaining("chat", None)
    # Capped at TTL — 60s.
    assert remaining <= adapter._FLOOD_WINDOW_TTL_SECONDS + 0.1
    assert remaining > adapter._FLOOD_WINDOW_TTL_SECONDS - 1.0


def test_record_flood_window_ignores_zero_and_negative():
    """retry_after <= 0 should not pollute the map."""
    adapter = _make_adapter()
    adapter._record_flood_window("chat", None, 0)
    adapter._record_flood_window("chat", None, -5)
    assert adapter._flood_window_remaining("chat", None) == 0.0
    assert adapter._flood_windows == {}


def test_flood_window_remaining_returns_zero_when_expired():
    """A recorded window that has expired returns 0.0 and prunes itself."""
    adapter = _make_adapter()
    # Record with retry_after = 0.05s — way shorter than our test will run.
    adapter._record_flood_window("chat", None, 0.05)
    time.sleep(0.1)
    assert adapter._flood_window_remaining("chat", None) == 0.0
    assert "chat" not in {k[0] for k in adapter._flood_windows.keys()}


def test_flood_windows_isolated_per_chat_and_thread():
    """Two chats (or threads) must have independent flood windows."""
    adapter = _make_adapter()
    adapter._record_flood_window("chat-A", None, 10)
    adapter._record_flood_window("chat-B", None, 10)
    adapter._record_flood_window("chat-A", "thread-X", 10)
    assert adapter._flood_window_remaining("chat-A", None) > 0
    assert adapter._flood_window_remaining("chat-B", None) > 0
    assert adapter._flood_window_remaining("chat-A", "thread-X") > 0
    # Unrecorded key returns 0.0.
    assert adapter._flood_window_remaining("chat-C", None) == 0.0
    assert adapter._flood_window_remaining("chat-A", "thread-Y") == 0.0


# ===========================================================================
# 8. send() — inline retry exhaustion surfaces retryable via _RetryableFloodError
# ===========================================================================


@pytest.mark.asyncio
async def test_send_surfaces_retryable_when_inline_retries_exhausted_on_flood():
    """``send()`` makes 3 inline attempts per chunk.  When all 3 hit
    flood control, it should raise ``_RetryableFloodError`` so the outer
    except clause returns ``SendResult(retryable=True)``."""
    from plugins.platforms.telegram.adapter import _RetryableFloodError

    adapter = _make_adapter()

    async def mock_send_message(**kwargs):
        raise _RetryAfterError(15)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(chat_id="123", content="hello")

    assert result.success is False
    assert result.retryable is True
    assert "flood_control" in (result.error or "")