"""TelegramAdapter wedged-getUpdates detection via pending_update_count.

PTB can report ``updater.running == True`` while its long-poll consumer is
silently stuck (observed on WSL2), so DMs queue in the Bot API and never reach
handlers (#42909). ``get_me()`` stays healthy (general request path), so the
CLOSE-WAIT heartbeat is blind to it. ``_probe_pending_updates`` watches
``get_webhook_info().pending_update_count`` and escalates to the existing
network-error recovery ladder after two stuck probes — with the confirming
probe arriving after a short interval rather than a full heartbeat, so
power-napping hosts can complete the pair inside one awake window (#139281).

The same probe also covers the harsher case where the updater has stopped
entirely (``running=False``) with no reconnect in flight — the long-poll task
is gone, so the gateway silently stops receiving messages while the process
stays alive (#55769) — and feeds it into the same recovery ladder.
"""

import asyncio
import time as _time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(*, pending: int) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._webhook_mode = False
    adapter._app = MagicMock()
    adapter._app.updater.running = True
    bot = MagicMock()
    bot.get_webhook_info = AsyncMock(
        return_value=MagicMock(pending_update_count=pending)
    )
    adapter._app.bot = bot
    adapter._bot = bot
    return adapter


@pytest.mark.asyncio
async def test_single_stopped_updater_probe_does_not_escalate():
    """One probe finding a stopped updater only increments the counter (#55769)."""
    adapter = _make_adapter(pending=9)
    adapter._app.updater.running = False
    adapter._polling_pending_stuck_count = 1
    with patch.object(adapter, "_handle_polling_network_error", new=AsyncMock()) as rec:
        await adapter._probe_pending_updates(adapter._app.bot, 5)
    # Stopped updater means no live consumer to query for a queue.
    adapter._app.bot.get_webhook_info.assert_not_called()
    assert adapter._polling_pending_stuck_count == 0
    assert adapter._polling_not_running_count == 1
    rec.assert_not_called()


@pytest.mark.asyncio
async def test_two_stopped_updater_probes_trigger_recovery():
    """A stopped updater that stays stopped routes into recovery (#55769)."""
    adapter = _make_adapter(pending=9)
    adapter._app.updater.running = False
    recovery = AsyncMock()
    with patch.object(adapter, "_handle_polling_network_error", new=recovery):
        await adapter._probe_pending_updates(adapter._app.bot, 5)
        assert adapter._polling_not_running_count == 1
        await adapter._probe_pending_updates(adapter._app.bot, 5)
        task = adapter._polling_error_task
        assert task is not None
        await task
    recovery.assert_awaited_once()
    assert adapter._polling_not_running_count == 0


@pytest.mark.asyncio
async def test_reconnect_in_flight_skips_stopped_updater_escalation():
    """A stopped updater during an in-flight reconnect must not re-escalate."""
    adapter = _make_adapter(pending=9)
    adapter._app.updater.running = False
    adapter._polling_not_running_count = 1
    inflight = MagicMock()
    inflight.done.return_value = False
    adapter._polling_error_task = inflight
    with patch.object(adapter, "_handle_polling_network_error", new=AsyncMock()) as rec:
        await adapter._probe_pending_updates(adapter._app.bot, 5)
    # The in-flight reconnect owns recovery; the stopped-updater counter resets
    # so the transient stop()->start_polling() window never trips a re-trigger.
    assert adapter._polling_not_running_count == 0
    rec.assert_not_called()


@pytest.mark.asyncio
async def test_first_stuck_pending_probe_requests_prompt_recheck():
    """A first stuck-queue observation returns True so the heartbeat loop re-probes
    after the short confirm interval instead of a full heartbeat (#139281)."""
    adapter = _make_adapter(pending=9)
    with patch.object(adapter, "_handle_polling_network_error", new=AsyncMock()) as rec:
        prompt = await adapter._probe_pending_updates(adapter._app.bot, 5)
    assert prompt is True
    assert adapter._polling_pending_stuck_count == 1
    rec.assert_not_called()


@pytest.mark.asyncio
async def test_second_stuck_pending_probe_escalates_and_ends_recheck():
    """The confirming probe escalates through the recovery ladder and returns
    False so the loop goes back to the full heartbeat interval."""
    adapter = _make_adapter(pending=9)
    adapter._polling_pending_stuck_count = 1
    recovery = AsyncMock()
    with patch.object(adapter, "_handle_polling_network_error", new=recovery):
        prompt = await adapter._probe_pending_updates(adapter._app.bot, 5)
        task = adapter._polling_error_task
        assert task is not None
        await task
    assert prompt is False
    recovery.assert_awaited_once()
    assert adapter._polling_pending_stuck_count == 0


@pytest.mark.asyncio
async def test_drained_queue_ends_recheck():
    """A queue that drained by the time of the confirming probe clears the
    re-check — the short interval must not turn a transient into an escalation."""
    adapter = _make_adapter(pending=9)
    adapter._polling_pending_stuck_count = 1
    adapter._app.bot.get_webhook_info = AsyncMock(
        return_value=MagicMock(pending_update_count=0)
    )
    with patch.object(adapter, "_handle_polling_network_error", new=AsyncMock()) as rec:
        prompt = await adapter._probe_pending_updates(adapter._app.bot, 5)
    assert prompt is False
    assert adapter._polling_pending_stuck_count == 0
    rec.assert_not_called()


@pytest.mark.asyncio
async def test_first_stopped_updater_probe_requests_prompt_recheck():
    """A first stopped-updater observation also asks for the short re-check."""
    adapter = _make_adapter(pending=9)
    adapter._app.updater.running = False
    with patch.object(adapter, "_handle_polling_network_error", new=AsyncMock()) as rec:
        prompt = await adapter._probe_pending_updates(adapter._app.bot, 5)
    assert prompt is True
    assert adapter._polling_not_running_count == 1
    rec.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_reprobes_stuck_queue_after_short_interval():
    """End-to-end (#139281): after a first stuck-queue observation the loop's next
    sleep is the 30s confirm interval, not the 90s heartbeat, so the confirming
    probe can land in the same awake window on a power-napping host (macOS
    Sleep↔DarkWake cycling with ~45s awake windows)."""
    adapter = _make_adapter(pending=9)
    adapter._app.bot.get_me = AsyncMock()
    # Keep the pure local-state watchdogs quiet so the stuck-queue path is the
    # one that escalates.
    adapter._polling_generation = 2
    now = _time.monotonic()
    adapter._polling_generation_started_monotonic = now
    adapter._polling_last_progress_monotonic = now
    real_sleep = asyncio.sleep
    delays = []

    async def tracking_sleep(delay, *args, **kwargs):
        delays.append(delay)
        await real_sleep(0)

    recovery = AsyncMock()
    with patch("asyncio.sleep", new=tracking_sleep):
        with patch.object(adapter, "_handle_polling_network_error", new=recovery):
            loop_task = asyncio.ensure_future(adapter._polling_heartbeat_loop())
            try:
                for _ in range(50):
                    await real_sleep(0)
                    if adapter._polling_error_task is not None:
                        break
                # Let the loop observe the teardown flag and exit on its own.
                # Never cancel(): CPython 3.11's wait_for can swallow task
                # cancellation while its inner future is already done.
                adapter._polling_teardown_started = True
                await asyncio.wait_for(loop_task, 10)
            finally:
                if not loop_task.done():
                    loop_task.cancel()
    task = adapter._polling_error_task
    assert task is not None, "two stuck-queue probes never escalated"
    await task
    recovery.assert_awaited_once()
    # Full heartbeat cadence first, then the short confirm probe (#139281).
    assert len(delays) >= 2
    assert delays[0] == 90
    assert delays[1] == 30
