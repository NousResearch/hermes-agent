"""TelegramAdapter.send_draft flood-control handling.

sendMessageDraft can return Telegram's RetryAfter error just like any other
Bot API call.  We distinguish two cases:

  * Short wait (≤5 s): sleep inline and immediately retry the same frame so
    this transient hiccup is invisible to the caller.
  * Long wait (>5 s): return immediately with retryable=True so the
    GatewayStreamConsumer does NOT count the frame against _draft_failures and
    does NOT disable draft streaming for the rest of the response.

These tests cover those paths without spinning up a real PTB bot.
"""
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

import plugins.platforms.telegram.adapter as tg_mod  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _make_retry_after_error(seconds: float):
    """Return an exception with a retry_after attribute, like PTB's RetryAfter."""
    exc = Exception(f"flood control — retry after {seconds}")
    exc.retry_after = seconds
    return exc


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_message_draft = AsyncMock(return_value=True)
    return adapter


@pytest.mark.asyncio
async def test_short_flood_control_sleeps_and_retries_successfully():
    """A RetryAfter ≤5 s must sleep that duration, retry the frame, and
    return success — the caller sees no failure at all."""
    adapter = _make_adapter()
    adapter.format_message = lambda c: c

    calls = []

    async def _draft(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _make_retry_after_error(2.0)
        return True  # retry succeeds

    adapter._bot.send_message_draft = AsyncMock(side_effect=_draft)

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await adapter.send_draft("123", 5, "hello")

    assert result.success is True
    assert not getattr(result, "retryable", False)
    # One initial attempt + one retry after the sleep.
    assert len(calls) == 2
    mock_sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_short_flood_control_retry_failure_returns_non_retryable():
    """If the post-sleep retry also fails, return a non-retryable failure
    so the consumer counts it against _draft_failures."""
    adapter = _make_adapter()
    adapter.format_message = lambda c: c

    async def _always_fails(**kwargs):
        raise _make_retry_after_error(3.0)

    adapter._bot.send_message_draft = AsyncMock(side_effect=_always_fails)

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await adapter.send_draft("123", 5, "hello")

    assert result.success is False
    assert not getattr(result, "retryable", False)
    mock_sleep.assert_awaited_once_with(3.0)


@pytest.mark.asyncio
async def test_long_flood_control_returns_retryable_without_sleeping():
    """A RetryAfter >5 s must return immediately with retryable=True and
    must NOT sleep (sleeping 280 s inline would block the event loop)."""
    adapter = _make_adapter()
    adapter.format_message = lambda c: c

    async def _long_flood(**kwargs):
        raise _make_retry_after_error(280.0)

    adapter._bot.send_message_draft = AsyncMock(side_effect=_long_flood)

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await adapter.send_draft("123", 5, "hello")

    assert result.success is False
    assert getattr(result, "retryable", False) is True
    assert "flood_control" in (getattr(result, "error", "") or "")
    mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_boundary_five_seconds_treated_as_short():
    """retry_after == 5.0 is on the short side of the threshold (≤5 s)
    and must trigger the sleep+retry path, not the retryable path."""
    adapter = _make_adapter()
    adapter.format_message = lambda c: c

    calls = []

    async def _draft(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _make_retry_after_error(5.0)
        return True

    adapter._bot.send_message_draft = AsyncMock(side_effect=_draft)

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await adapter.send_draft("123", 5, "boundary")

    assert result.success is True
    assert not getattr(result, "retryable", False)
    mock_sleep.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_boundary_just_above_five_seconds_treated_as_long():
    """retry_after just above 5.0 (e.g. 5.01 s) must take the long-wait
    path — retryable=True, no sleep — confirming the threshold is exclusive."""
    adapter = _make_adapter()
    adapter.format_message = lambda c: c

    async def _just_over(**kwargs):
        raise _make_retry_after_error(5.01)

    adapter._bot.send_message_draft = AsyncMock(side_effect=_just_over)

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await adapter.send_draft("123", 5, "just-over")

    assert result.success is False
    assert getattr(result, "retryable", False) is True
    assert "flood_control" in (getattr(result, "error", "") or "")
    mock_sleep.assert_not_awaited()
