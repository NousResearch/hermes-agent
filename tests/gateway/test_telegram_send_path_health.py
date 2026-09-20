"""TelegramAdapter send-path health gating after reconnect storms.

After sustained Bad Gateway / TimedOut reconnect cycles, the PTB httpx client
can enter a wedged state where ``bot.send_message()`` returns a valid Message
but nothing reaches the recipient.  ``_send_path_degraded`` short-circuits
``send()`` so cron's live-adapter branch falls through to standalone HTTP.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    return adapter


@pytest.mark.asyncio
async def test_send_short_circuits_when_path_degraded():
    """Degraded adapter returns failure WITHOUT calling send_message,
    so cron's live-adapter branch falls through to standalone HTTP."""
    adapter = _make_adapter()
    adapter._send_path_degraded = True

    result = await adapter.send("123", "hello")

    assert result.success is False
    assert result.error == "send_path_degraded"
    assert result.retryable is True
    adapter._bot.send_message.assert_not_awaited()


class _FloodError(Exception):
    def __init__(self, seconds: float):
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


@pytest.mark.asyncio
async def test_send_long_flood_fails_closed_without_inline_sleep(monkeypatch):
    """A 97-minute RetryAfter must not pin send() for the full penalty."""
    adapter = _make_adapter()
    adapter._rich_send_disabled = True
    adapter._bot.send_message = AsyncMock(side_effect=_FloodError(5827.0))
    sleep = AsyncMock()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", sleep)

    result = await adapter.send("123", "hello")

    assert result.success is False
    assert result.error == "flood_control:5827.0"
    assert result.retry_after == 5827.0
    assert result.retryable is False
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_short_flood_still_retries_inline(monkeypatch):
    """Waits of a few seconds keep the existing inline retry."""
    adapter = _make_adapter()
    adapter._rich_send_disabled = True
    ok = MagicMock(message_id=7)
    adapter._bot.send_message = AsyncMock(side_effect=[_FloodError(2.0), ok])
    sleep = AsyncMock()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", sleep)

    result = await adapter.send("123", "hello")

    assert result.success is True
    assert result.message_id == "7"
    sleep.assert_awaited_once_with(2.0)


def test_mark_connected_publishes_connected_when_healthy():
    """A normal connect (never degraded) still publishes platform_state=connected."""
    adapter = _make_adapter()
    adapter._send_path_degraded = False

    with patch.object(adapter, "_write_runtime_status_safe") as write_status:
        adapter._mark_connected()

    write_status.assert_called_once()
    _, kwargs = write_status.call_args
    assert kwargs["platform_state"] == "connected"


def test_mark_connected_publishes_retrying_when_send_path_degraded():
    """connect() can return True while polling never proved a first getUpdates
    round-trip (the degraded branch, or a reconnect where require_progress is
    skipped). _mark_connected() must not publish "connected" for that case --
    it is indistinguishable from a healthy adapter to anything reading
    gateway_state.json (#101391)."""
    adapter = _make_adapter()
    adapter._send_path_degraded = True

    with patch.object(adapter, "_write_runtime_status_safe") as write_status:
        adapter._mark_connected()

    write_status.assert_called_once()
    _, kwargs = write_status.call_args
    assert kwargs["platform_state"] == "retrying"


def test_record_polling_progress_republishes_connected_after_degraded_connect():
    """Once getUpdates actually proves a round-trip after a degraded connect,
    the previously-published "retrying" status must be corrected back to
    "connected" -- otherwise it stays wedged until the next disconnect."""
    adapter = _make_adapter()
    generation, _event = adapter._begin_polling_generation()
    # Simulate connect() having already run and published the degraded state.
    adapter._running = True

    with patch.object(adapter, "_write_runtime_status_safe") as write_status:
        adapter._record_polling_progress(generation)

    write_status.assert_called_once()
    _, kwargs = write_status.call_args
    assert kwargs["platform_state"] == "connected"
    assert adapter._send_path_degraded is False


def test_mid_session_polling_death_publishes_retrying_while_running():
    """#101391's measured incident: a HEALTHY connect published "connected",
    then getUpdates silently died mid-session and nothing republished for 11h.
    The recovery ladder's entry point must flip the file to "retrying"."""
    adapter = _make_adapter()
    adapter._running = True
    adapter._send_path_degraded = False
    adapter._polling_error_task = None

    class _Loop:
        def create_task(self, coro):
            coro.close()
            return MagicMock(done=lambda: False)

    with patch.object(adapter, "_write_runtime_status_safe") as write_status, \
            patch("asyncio.get_running_loop", return_value=_Loop()):
        adapter._schedule_polling_recovery(RuntimeError("boom"), reason="heartbeat probe")

    assert adapter._send_path_degraded is True
    write_status.assert_called_once()
    _, kwargs = write_status.call_args
    assert kwargs["platform_state"] == "retrying"
    assert kwargs["error_message"] == TelegramAdapter.DEGRADED_STATUS_MESSAGE


def test_polling_death_before_connect_does_not_publish():
    """Not yet running (cold connect still in progress): connect()'s own
    _mark_connected publishes; the recovery path must not write early."""
    adapter = _make_adapter()
    adapter._running = False
    adapter._polling_error_task = None

    class _Loop:
        def create_task(self, coro):
            coro.close()
            return MagicMock(done=lambda: False)

    with patch.object(adapter, "_write_runtime_status_safe") as write_status, \
            patch("asyncio.get_running_loop", return_value=_Loop()):
        adapter._schedule_polling_recovery(RuntimeError("boom"), reason="polling bootstrap")

    write_status.assert_not_called()


@pytest.mark.parametrize("running, fatal", [(False, False), (True, True)])
def test_record_polling_progress_does_not_flip_when_not_running_or_fatal(running, fatal):
    """Cold connect: progress arrives while _running is still False -- the
    connect path publishes, not the flip. Fatal: never overwrite "fatal"."""
    adapter = _make_adapter()
    generation, _event = adapter._begin_polling_generation()
    adapter._running = running
    if fatal:
        adapter._fatal_error_message = "dead"

    with patch.object(adapter, "_write_runtime_status_safe") as write_status:
        adapter._record_polling_progress(generation)

    write_status.assert_not_called()
    assert adapter._send_path_degraded is False
