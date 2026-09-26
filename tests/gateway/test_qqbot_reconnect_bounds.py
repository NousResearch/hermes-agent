"""Invariant tests for the QQBot reconnect path (silent-channel-death class).

A messaging adapter must never park its listener forever and must never retire it on a transient
outage: either failure mode leaves the channel silently deaf until the next process restart, with
no error and no further log line. These assert the CONTRACTS (bounded wait, listener survives),
not any particular values.

Observed failure that motivated them (production, 2026-09-14): the QQ WS got op7/4009
(session timed out), the adapter logged "Reconnecting in 2s (attempt 1)..." and then produced no
further qqbot log line for 30h — neither "Reconnected" nor "Reconnect failed" — while weixin and
wecom recovered on their own.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.qqbot import adapter as qq

# Generous outer guard: the adapter must return well inside this, so a regression shows up as a
# clean test failure instead of a hung suite.
_OUTER_GUARD_SECONDS = 5.0


def _adapter() -> qq.QQAdapter:
    adapter = qq.QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "12345", "client_secret": "x" * 8}))
    adapter._app_id = "12345"
    return adapter


class _SilentWS:
    """A socket that stays open but never delivers a frame (half-open peer)."""

    closed = False

    async def receive(self):
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_open_and_read_waits_are_bounded(monkeypatch):
    """Neither the WS open nor the read loop may wait forever.

    Unbounded, the first parks the listener task on a stalled open (no reconnect attempt is ever
    logged again) and the second parks it on a half-open connection the kernel never reports.
    """
    monkeypatch.setattr(qq, "OPEN_WS_TIMEOUT_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(qq, "RECONNECT_BACKOFF", [0])
    adapter = _adapter()

    async def stalled_open(*, log_url: bool = False):
        await asyncio.sleep(3600)

    adapter._open_gateway_ws = stalled_open
    started = time.monotonic()
    try:
        returned = await asyncio.wait_for(adapter._reconnect(0), timeout=_OUTER_GUARD_SECONDS)
    except asyncio.TimeoutError:
        pytest.fail("reconnect parked the listener: the gateway-WS open has no timeout")
    assert returned is False, "a stalled open must fall back to the backoff loop"
    assert time.monotonic() - started < _OUTER_GUARD_SECONDS

    # The failed attempt must not leave a session/WS behind for the next attempt to inherit.
    assert adapter._ws is None and adapter._session is None

    monkeypatch.setattr(qq, "READ_TIMEOUT_SECONDS", 0.2, raising=False)
    adapter._running = True
    adapter._ws = _SilentWS()
    adapter._heartbeat_interval = 0.05  # the guard is max(READ_TIMEOUT_SECONDS, 3 * interval)
    try:
        await asyncio.wait_for(adapter._read_events(), timeout=_OUTER_GUARD_SECONDS)
    except asyncio.TimeoutError:
        pytest.fail("_read_events parked on a silent socket instead of reporting it")
    except RuntimeError:
        pass
    else:
        pytest.fail("a silent socket must raise so _listen_loop reconnects")


@pytest.mark.asyncio
async def test_listener_keeps_retrying_past_max_attempts(monkeypatch):
    """Exhausting MAX_RECONNECT_ATTEMPTS resets the backoff instead of ending the listener.

    ``_mark_disconnected()`` clears ``_running``, so a "give up and return" path exits the listener
    loop and leaves the channel dead with no fatal error recorded — a silent death.
    """
    adapter = _adapter()
    adapter._running = True
    attempts = {"n": 0}

    async def failing_read():
        raise RuntimeError("WebSocket closed")

    async def failing_reconnect(backoff_idx):
        attempts["n"] += 1
        return False

    adapter._read_events = failing_read
    adapter._reconnect = failing_reconnect
    adapter._mark_transport_disconnected = lambda: None
    adapter._fail_pending = lambda reason: None
    monkeypatch.setattr(qq, "MAX_RECONNECT_ATTEMPTS", 1)
    monkeypatch.setattr(qq, "RATE_LIMIT_DELAY", 0.05)

    task = asyncio.create_task(adapter._listen_loop())
    await asyncio.sleep(1.0)
    alive = not task.done()
    adapter._running = False
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert alive, "listener returned — the channel would be silently dead until a restart"
    assert attempts["n"] >= 2, "backoff was not reset after MAX_RECONNECT_ATTEMPTS"
