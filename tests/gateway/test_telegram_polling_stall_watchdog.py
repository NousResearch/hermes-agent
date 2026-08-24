"""TelegramAdapter polling-stall watchdog (#92991).

A wedged getUpdates long-poll can be invisible to every other probe: the
TCP connection dies mid-read (CLOSE-WAIT behind a TUN/proxy route flip),
``updater.running`` stays True, ``get_me()`` on the general request path
stays healthy, and — while no messages are queued server-side —
``pending_update_count`` stays 0. The gateway then goes silently deaf and
only a full restart recovers it.

``_check_polling_stall`` closes that hole: Telegram answers a long-poll
within ~50s, so a poller with no successful getUpdates round-trip for
``_POLLING_STALL_TIMEOUT`` seconds is unambiguously wedged, and the check
escalates loudly through the existing reconnect ladder
(``_handle_polling_network_error``). ``_polling_heartbeat_loop`` runs the
check every probe, so steady-state wedges are caught without any Bot API
call.
"""
import asyncio
import time as _time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(*, stalled_seconds: float) -> TelegramAdapter:
    """Build a polling-mode adapter whose long-poll last succeeded
    ``stalled_seconds`` ago, with a healthy general request path and an
    EMPTY server-side queue (so the pending-count probe stays blind)."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._webhook_mode = False
    adapter._app = MagicMock()
    adapter._app.updater.running = True
    bot = MagicMock()
    bot.get_me = AsyncMock()
    bot.get_webhook_info = AsyncMock(
        return_value=MagicMock(pending_update_count=0)
    )
    adapter._app.bot = bot
    adapter._bot = bot
    adapter._polling_generation = 2
    now = _time.monotonic()
    adapter._polling_generation_started_monotonic = now - 500
    adapter._polling_last_progress_monotonic = now - stalled_seconds
    return adapter


@pytest.mark.asyncio
async def test_recent_progress_does_not_escalate():
    """A healthy poller (fresh round-trip) must never trip the watchdog."""
    adapter = _make_adapter(stalled_seconds=1)
    with patch.object(adapter, "_handle_polling_network_error", new=AsyncMock()) as rec:
        await adapter._check_polling_stall()
    assert adapter._polling_error_task is None
    rec.assert_not_called()


@pytest.mark.asyncio
async def test_stalled_long_poll_escalates_to_reconnect_ladder():
    """#92991: with an empty queue and healthy get_me(), only the stall
    timestamp can detect the wedged consumer — and it must."""
    adapter = _make_adapter(stalled_seconds=400)
    recovery = AsyncMock()
    with patch.object(adapter, "_handle_polling_network_error", new=recovery):
        await adapter._check_polling_stall()
    task = adapter._polling_error_task
    assert task is not None
    await task
    recovery.assert_awaited_once()


@pytest.mark.asyncio
async def test_generation_with_no_progress_ever_uses_generation_age():
    """A generation that never completes one round-trip still trips the
    watchdog once its age passes the stall threshold (verifier fallback)."""
    adapter = _make_adapter(stalled_seconds=0)
    adapter._polling_last_progress_monotonic = None
    recovery = AsyncMock()
    with patch.object(adapter, "_handle_polling_network_error", new=recovery):
        await adapter._check_polling_stall()
    task = adapter._polling_error_task
    assert task is not None
    await task
    recovery.assert_awaited_once()


@pytest.mark.asyncio
async def test_stall_ignored_while_recovery_in_flight():
    """An in-flight reconnect owns recovery; the watchdog must not pile on."""
    adapter = _make_adapter(stalled_seconds=400)
    inflight = MagicMock()
    inflight.done.return_value = False
    adapter._polling_error_task = inflight
    with patch.object(adapter, "_handle_polling_network_error", new=AsyncMock()) as rec:
        await adapter._check_polling_stall()
    rec.assert_not_called()
    assert adapter._polling_error_task is inflight


@pytest.mark.asyncio
async def test_stall_check_skipped_in_webhook_mode():
    """Webhook mode has no long-poll socket to wedge."""
    adapter = _make_adapter(stalled_seconds=400)
    adapter._webhook_mode = True
    with patch.object(adapter, "_handle_polling_network_error", new=AsyncMock()) as rec:
        await adapter._check_polling_stall()
    rec.assert_not_called()
    assert adapter._polling_error_task is None


@pytest.mark.asyncio
async def test_heartbeat_detects_wedged_long_poll_with_empty_queue():
    """End-to-end (#92991): drive the lifetime heartbeat loop against a
    wedged poller with a healthy general path and an empty queue. Before the
    stall watchdog, this setup produces total silence — no probe fires and
    no recovery is ever scheduled. After it, the first stall observation
    escalates through the reconnect ladder."""
    adapter = _make_adapter(stalled_seconds=400)
    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *args, **kwargs):
        await real_sleep(0)

    with patch("asyncio.sleep", new=fast_sleep):
        with patch.object(adapter, "_handle_polling_network_error", new=AsyncMock()) as rec:
            loop_task = asyncio.ensure_future(adapter._polling_heartbeat_loop())
            try:
                # Run probe cycles until the stall watchdog escalates (the
                # fix) — bounded so the pre-fix state fails cleanly instead
                # of hanging.
                for _ in range(50):
                    await real_sleep(0)
                    if adapter._polling_error_task is not None:
                        break
                # Let the loop observe the teardown flag and exit on its own.
                # Never cancel(): CPython 3.11's wait_for can swallow task
                # cancellation while its inner future is already done, which
                # leaves the busy loop un-cancellable and the test spinning
                # forever.
                adapter._polling_teardown_started = True
                await asyncio.wait_for(loop_task, 10)
            finally:
                if not loop_task.done():
                    loop_task.cancel()
    task = adapter._polling_error_task
    assert task is not None, (
        "heartbeat probes saw a wedged long-poll (no getUpdates progress for "
        "400s) but scheduled no recovery"
    )
    await task
    rec.assert_awaited_once()
