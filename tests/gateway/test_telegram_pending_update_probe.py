"""TelegramAdapter wedged-getUpdates detection via pending_update_count.

PTB can report ``updater.running == True`` while its long-poll consumer is
silently stuck (observed on WSL2), so DMs queue in the Bot API and never reach
handlers (#42909). ``get_me()`` stays healthy (general request path), so the
CLOSE-WAIT heartbeat is blind to it. ``_probe_pending_updates`` watches
``get_webhook_info().pending_update_count`` and escalates to the existing
network-error recovery ladder after two consecutive stuck probes.

The same probe also covers the harsher case where the updater has stopped
entirely (``running=False``) with no reconnect in flight — the long-poll task
is gone, so the gateway silently stops receiving messages while the process
stays alive (#55769) — and feeds it into the same recovery ladder.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


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
