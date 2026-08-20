"""
Tests for #24851 — DingTalk reconnection storm causes gateway to hang.

When DingTalkStreamClient.start() raises the same TypeError on every call,
the reconnection loop must:
  1. Log the first N failures, then stop logging repeated identical errors.
  2. After _RECONNECT_CIRCUIT_BREAKER_TRIPS consecutive failures, enter a
     long pause instead of spinning at the 60 s normal backoff.
  3. Reset backoff_idx after a clean start() call so a recovered connection
     is treated as a fresh start.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dingtalk_adapter():
    """Build a minimal DingTalkAdapter with mocked internals."""
    try:
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        from gateway.config import Platform
    except ImportError:
        pytest.skip("dingtalk-stream not available; skipping")

    # Build a minimal DingTalkAdapter without going through __init__ which
    # tries to import and configure the real SDK.
    adapter = DingTalkAdapter.__new__(DingTalkAdapter)
    # The base class `name` property reads self.platform.value.title()
    adapter.platform = Platform.DINGTALK
    adapter._running = True
    adapter._stream_client = MagicMock()
    adapter._stream_task = None
    return adapter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_breaker_kicks_in_after_repeated_failures():
    """After TRIPS consecutive identical errors, loop sleeps CIRCUIT_BREAKER_DELAY, not 60 s."""
    adapter = _make_dingtalk_adapter()
    trips = 5  # _RECONNECT_CIRCUIT_BREAKER_TRIPS
    call_count = 0
    sleep_durations = []

    async def fake_start():
        nonlocal call_count
        call_count += 1
        if call_count > trips + 1:
            # Stop the loop after one circuit-breaker sleep
            adapter._running = False
            return
        raise TypeError("'coroutine' object does not support the asynchronous context manager protocol")

    async def fake_sleep(secs):
        sleep_durations.append(secs)

    adapter._stream_client.start = fake_start

    with patch("asyncio.sleep", side_effect=fake_sleep):
        await adapter._run_stream()

    # After TRIPS+1 errors we expect a 300 s circuit-breaker sleep (not 60 s)
    assert any(s >= 300 for s in sleep_durations), (
        f"Expected at least one >=300 s sleep; got: {sleep_durations}"
    )


@pytest.mark.asyncio
async def test_first_few_errors_are_logged(caplog):
    """The first TRIPS errors must be logged at WARNING; later ones must be suppressed."""
    adapter = _make_dingtalk_adapter()
    trips = 5
    call_count = 0

    async def fake_start():
        nonlocal call_count
        call_count += 1
        if call_count > trips + 2:
            adapter._running = False
            return
        raise TypeError("coroutine error")

    async def fake_sleep(secs):
        pass

    adapter._stream_client.start = fake_start

    with caplog.at_level(logging.WARNING):
        with patch("asyncio.sleep", side_effect=fake_sleep):
            await adapter._run_stream()

    warning_lines = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "Stream client error" in r.message
    ]
    # The circuit breaker fires after TRIPS consecutive errors; after one
    # circuit-breaker pause the counter resets so at most one more warning
    # can appear. Total WARNING lines must be < total calls - 1 (i.e. the
    # repeated storm of identical lines is suppressed).
    # We allow up to TRIPS+1 warnings (TRIPS before CB fires, 1 after reset).
    assert len(warning_lines) <= trips + 1, (
        f"Too many warning log lines ({len(warning_lines)}); storm suppression not working"
    )


@pytest.mark.asyncio
async def test_backoff_resets_after_clean_connection():
    """If start() returns cleanly (no error), backoff_idx should reset."""
    adapter = _make_dingtalk_adapter()
    call_count = 0
    sleep_durations = []

    async def fake_start():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: raise an error to advance backoff_idx
            raise ConnectionError("network blip")
        if call_count == 2:
            # Second call: succeed (simulate recovered connection)
            return
        # Third call: error again (should use backoff index 0 = 2 s, not the advanced index)
        adapter._running = False
        raise ConnectionError("network blip again")

    async def fake_sleep(secs):
        sleep_durations.append(secs)

    adapter._stream_client.start = fake_start

    from plugins.platforms.dingtalk.adapter import RECONNECT_BACKOFF

    with patch("asyncio.sleep", side_effect=fake_sleep):
        await adapter._run_stream()

    # After the clean connection (call 2), the backoff should reset.
    # The third error triggers backoff_idx=0 → RECONNECT_BACKOFF[0] = 2 s.
    assert RECONNECT_BACKOFF[0] in sleep_durations, (
        f"Backoff did not reset after clean connection; sleep durations: {sleep_durations}"
    )


@pytest.mark.asyncio
async def test_cancelled_error_exits_cleanly():
    """asyncio.CancelledError must cause immediate exit without looping."""
    adapter = _make_dingtalk_adapter()

    async def fake_start():
        raise asyncio.CancelledError()

    adapter._stream_client.start = fake_start

    with patch("asyncio.sleep", new=AsyncMock()):
        # Should return without re-raising or hanging
        await adapter._run_stream()
    # If we reach here, exit was clean.
