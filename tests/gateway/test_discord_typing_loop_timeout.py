"""Tests for Discord adapter _typing_loop timeout behavior (#64874).

When the Discord HTTP request to the typing endpoint hangs (network blip,
API stall, WS reconnect), the _typing_loop task must not get stuck
permanently.  The fix wraps the request with asyncio.wait_for(timeout=10)
and bounds stop_typing's await with asyncio.wait_for(shield(task), 5).
"""

import asyncio
from unittest.mock import patch

import pytest

# Production timeout used in _typing_loop's wait_for call.
_TYPING_REQUEST_TIMEOUT = 10.0
# Production timeout used in stop_typing's wait_for call.
_STOP_TYPING_TIMEOUT = 5.0


def _make_adapter(monkeypatch, request_fn):
    """Build a minimal DiscordAdapter with a fake HTTP client."""
    discord_adapter = pytest.importorskip("plugins.platforms.discord.adapter")

    adapter = object.__new__(discord_adapter.DiscordAdapter)
    adapter._typing_tasks = {}

    class FakeHTTP:
        request = staticmethod(request_fn)

    class FakeClient:
        http = FakeHTTP()

    adapter._client = FakeClient()

    class FakeRoute:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("discord.http.Route", FakeRoute)
    return adapter


@pytest.mark.asyncio
async def test_typing_loop_retries_after_timeout(monkeypatch):
    """A hanging HTTP request should time out and be retried on the next cycle."""
    call_count = 0
    hang_calls = 2  # first N calls hang, then succeed

    async def _fake_request(route):
        nonlocal call_count
        call_count += 1
        if call_count <= hang_calls:
            await asyncio.sleep(999)
        return None

    adapter = _make_adapter(monkeypatch, _fake_request)

    # Patch asyncio.wait_for to use a tiny timeout for the typing request,
    # so the test doesn't need to wait the full 10 seconds.
    real_wait_for = asyncio.wait_for

    async def _fast_wait_for(fut, *, timeout=None):
        if timeout == _TYPING_REQUEST_TIMEOUT:
            timeout = 0.05
        return await real_wait_for(fut, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait_for", _fast_wait_for)

    await adapter.send_typing("test-channel")
    assert "test-channel" in adapter._typing_tasks

    # Wait for 2 fast timeouts (0.05s each) + the successful 3rd call + margin
    await asyncio.sleep(0.4)

    # Verify retries: 2 timed-out calls + at least 1 success
    assert call_count >= 3, f"Expected >=3 calls (2 timeout + 1 success), got {call_count}"

    # Clean up
    task = adapter._typing_tasks.get("test-channel")
    if task:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    adapter._typing_tasks.pop("test-channel", None)


@pytest.mark.asyncio
async def test_stop_typing_bounded_when_task_ignores_cancel(monkeypatch):
    """stop_typing must return within its timeout even if the task ignores cancellation."""

    async def _stubborn_request(route):
        """Simulate a coroutine that catches CancelledError and keeps blocking."""
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            # Ignore cancellation — keep blocking
            await asyncio.sleep(999)

    adapter = _make_adapter(monkeypatch, _stubborn_request)

    # Patch the 5s stop_typing timeout to 0.2s for speed.
    real_wait_for = asyncio.wait_for

    async def _fast_wait_for(fut, *, timeout=None):
        if timeout == _STOP_TYPING_TIMEOUT:
            timeout = 0.2
        return await real_wait_for(fut, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait_for", _fast_wait_for)

    await adapter.send_typing("chan-stuck")
    assert "chan-stuck" in adapter._typing_tasks

    # Let the loop start its hanging request
    await asyncio.sleep(0.05)

    # stop_typing must complete within the bounded timeout, not hang forever
    await asyncio.wait_for(adapter.stop_typing("chan-stuck"), timeout=1.0)
    assert "chan-stuck" not in adapter._typing_tasks


@pytest.mark.asyncio
async def test_stop_typing_clean_cancel(monkeypatch):
    """stop_typing completes quickly when the task responds to cancellation promptly."""
    call_count = 0

    async def _normal_request(route):
        nonlocal call_count
        call_count += 1
        return None

    adapter = _make_adapter(monkeypatch, _normal_request)

    await adapter.send_typing("chan-clean")

    # Let the loop issue at least one successful request + enter its sleep(12)
    await asyncio.sleep(0.05)

    # Should complete nearly instantly since the task handles CancelledError
    await asyncio.wait_for(adapter.stop_typing("chan-clean"), timeout=1.0)
    assert "chan-clean" not in adapter._typing_tasks
    assert call_count >= 1
