"""Regression tests for #59614: start_polling() must be time-bounded.

When both the primary Telegram API server and all fallback IPs are unreachable,
``await app.updater.start_polling(...)`` can block forever inside an exhausted
httpx connection pool — it neither returns nor raises. Unbounded, that wedges:

1. the network-error reconnect ladder (stuck inside attempt 1, never advances),
2. the heartbeat loop (sees the recovery task as alive-but-wedged and skips),
3. the fatal-error escalation (never reached).

The fix wraps every ``start_polling()`` await in the wall-deadline helper with
``_UPDATER_START_TIMEOUT`` so a cancellation-shielded hung call still raises and
feeds the existing retry ladder. These tests patch the timeout down to keep the
suite fast.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


async def _hang_forever(**kwargs):
    await asyncio.sleep(0.2)


def _bare_adapter():
    a = TelegramAdapter.__new__(TelegramAdapter)
    # `name` / `has_fatal_error` are read-only base-class properties; set the
    # backing fields they derive from instead.
    from gateway.config import Platform

    a.platform = Platform.TELEGRAM
    a._fatal_error_code = None
    a._fatal_error_message = None
    a._fatal_error_retryable = True
    a._polling_network_error_count = 0
    a._polling_conflict_count = 0
    a._polling_conflict_recovery_generation = None
    a._polling_error_callback_ref = None
    a._background_tasks = set()
    a._send_path_degraded = False
    return a


@pytest.mark.asyncio
async def test_network_ladder_start_polling_hang_does_not_wedge(monkeypatch):
    """A hung start_polling() in _handle_polling_network_error must time out
    and advance the ladder instead of blocking forever (#59614 core repro)."""
    monkeypatch.setattr(tg_adapter, "_UPDATER_START_TIMEOUT", 0.2)
    a = _bare_adapter()
    a._polling_network_error_count = 0  # attempt 1 → 5s backoff before start_polling

    app = MagicMock()
    app.updater = AsyncMock()
    app.updater.start_polling = _hang_forever
    app.updater.running = False
    a._app = app

    with patch.object(a, "_drain_polling_connections", new=AsyncMock()), \
         patch.object(
             tg_adapter.asyncio, "ensure_future",
             side_effect=lambda coro: (coro.close(), asyncio.get_event_loop().create_future())[1],
         ):
        # Unbounded, this await hangs past the 30s wait_for and fails the
        # test; bounded, the handler waits its 5s backoff, times out the hung
        # start_polling() in 0.2s, schedules the chained retry (captured by
        # the ensure_future patch), and returns.
        await asyncio.wait_for(
            a._handle_polling_network_error(Exception("net down")), timeout=30
        )


@pytest.mark.asyncio
async def test_initial_connect_succeeds_on_current_generation_progress(monkeypatch):
    """Strict cold start returns True once THIS generation records progress."""
    monkeypatch.setattr(tg_adapter, "_INITIAL_POLLING_PROGRESS_TIMEOUT", 5.0)
    a = _bare_adapter()
    app = MagicMock()
    app.updater = AsyncMock()

    async def start_polling_with_progress(**_kwargs):
        # PTB's instrumented getUpdates request records progress for the
        # generation started by this call.
        a._record_polling_progress(a._polling_generation)

    app.updater.start_polling = AsyncMock(side_effect=start_polling_with_progress)
    a._app = app

    ok = await asyncio.wait_for(
        a._start_polling_resilient(
            drop_pending_updates=True,
            error_callback=None,
            require_progress=True,
        ),
        timeout=10,
    )
    assert ok is True
    assert a._send_path_degraded is False
    # Strict mode owns readiness itself; the background verifier must not be
    # racing it on the cold-start application.
    assert getattr(a, "_polling_progress_verifier_task", None) is None


@pytest.mark.asyncio
async def test_initial_connect_polling_error_fails_fast_not_background(monkeypatch):
    """#67498 idle-threads shape: a polling error during strict cold start
    must surface immediately as a connect failure — NOT be swallowed into a
    background recovery task that restarts polling on the partial app while
    the readiness gate waits out its full deadline (the G1/G2 race from the
    #69240 review)."""
    monkeypatch.setattr(tg_adapter, "_INITIAL_POLLING_PROGRESS_TIMEOUT", 30.0)
    a = _bare_adapter()
    app = MagicMock()
    app.updater = AsyncMock()

    captured_callbacks = []

    async def start_polling_capture(**kwargs):
        captured_callbacks.append(kwargs.get("error_callback"))

    app.updater.start_polling = AsyncMock(side_effect=start_polling_capture)
    a._app = app

    recovery_scheduled = []
    a._schedule_polling_recovery = lambda err, reason: recovery_scheduled.append(err)

    async def run_connect():
        return await a._start_polling_resilient(
            drop_pending_updates=True,
            error_callback=lambda e: recovery_scheduled.append(e),
            require_progress=True,
        )

    task = asyncio.ensure_future(run_connect())
    # Let start_polling run and the strict gate begin waiting.
    for _ in range(10):
        await asyncio.sleep(0)
        if captured_callbacks:
            break
    assert captured_callbacks and captured_callbacks[0] is not None

    # PTB reports a network error on the first getUpdates (G1 error).
    captured_callbacks[0](OSError("getUpdates failed"))

    # The cold attempt must fail promptly (well before the 30s readiness
    # deadline) with a loud OSError, and must NOT have scheduled background
    # recovery (which would start G2 on the same partial application).
    with pytest.raises(OSError, match="errored before first getUpdates"):
        await asyncio.wait_for(task, timeout=5)
    assert recovery_scheduled == []


