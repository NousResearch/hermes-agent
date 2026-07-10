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

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402

# Real PTB derives NetworkError from TelegramError, never OSError — this fake
# classifies via the adapter's name-based branch, like production (#62047).
_FakeNetworkError = type("NetworkError", (Exception,), {})


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
async def test_single_stuck_probe_does_not_escalate():
    """One probe with a queued update only increments the counter."""
    adapter = _make_adapter(pending=3)
    with patch.object(adapter, "_handle_polling_network_error", new=AsyncMock()) as rec:
        await adapter._probe_pending_updates(adapter._app.bot, 5)
    assert adapter._polling_pending_stuck_count == 1
    rec.assert_not_called()


@pytest.mark.asyncio
async def test_two_consecutive_stuck_probes_trigger_recovery():
    """Second consecutive stuck probe routes into the recovery ladder."""
    adapter = _make_adapter(pending=2)
    recovery = AsyncMock()
    with patch.object(adapter, "_handle_polling_network_error", new=recovery):
        await adapter._probe_pending_updates(adapter._app.bot, 5)
        assert adapter._polling_pending_stuck_count == 1
        await adapter._probe_pending_updates(adapter._app.bot, 5)
        # Let the scheduled recovery task run.
        task = adapter._polling_error_task
        assert task is not None
        await task
    recovery.assert_awaited_once()
    # Counter resets after escalation so a fresh wedge starts from zero.
    assert adapter._polling_pending_stuck_count == 0


@pytest.mark.asyncio
async def test_zero_pending_resets_counter():
    """A drained queue clears any prior stuck count without escalating."""
    adapter = _make_adapter(pending=0)
    adapter._polling_pending_stuck_count = 1
    with patch.object(adapter, "_handle_polling_network_error", new=AsyncMock()) as rec:
        await adapter._probe_pending_updates(adapter._app.bot, 5)
    assert adapter._polling_pending_stuck_count == 0
    rec.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_info_network_error_defers_to_heartbeat():
    """Connectivity failures of the webhook-info probe are deliberately
    deferred — the get_me() heartbeat owns reconnect scheduling, and a
    transient get_webhook_info() failure must not tear down a healthy poller.
    PTB errors get the same deferral the TimeoutError/OSError path always had
    (#62047)."""
    adapter = _make_adapter(pending=0)
    probe_error = _FakeNetworkError("connection lost")
    adapter._app.bot.get_webhook_info = AsyncMock(side_effect=probe_error)
    recovery = AsyncMock()

    with patch.object(adapter, "_handle_polling_network_error", new=recovery):
        await adapter._probe_pending_updates(adapter._app.bot, 5)

    assert adapter._polling_error_task is None
    recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_webhook_info_non_network_error_propagates():
    """Non-connectivity probe errors propagate to the heartbeat's classifier."""
    adapter = _make_adapter(pending=0)
    adapter._app.bot.get_webhook_info = AsyncMock(
        side_effect=RuntimeError("api contract change")
    )

    with patch.object(adapter, "_handle_polling_network_error", new=AsyncMock()) as rec:
        with pytest.raises(RuntimeError):
            await adapter._probe_pending_updates(adapter._app.bot, 5)

    assert adapter._polling_error_task is None
    rec.assert_not_awaited()


@pytest.mark.asyncio
async def test_webhook_mode_is_noop():
    """Webhook mode holds no server-side queue — probe never runs."""
    adapter = _make_adapter(pending=9)
    adapter._webhook_mode = True
    await adapter._probe_pending_updates(adapter._app.bot, 5)
    adapter._app.bot.get_webhook_info.assert_not_called()
    assert adapter._polling_pending_stuck_count == 0


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
async def test_running_updater_resets_stopped_counter():
    """A recovered (running) updater clears any prior stopped-probe count."""
    adapter = _make_adapter(pending=0)
    adapter._polling_not_running_count = 1
    await adapter._probe_pending_updates(adapter._app.bot, 5)
    assert adapter._polling_not_running_count == 0


@pytest.mark.asyncio
async def test_reconnect_in_flight_skips_probe():
    """An active recovery task owns the connection — don't double-trigger."""
    adapter = _make_adapter(pending=9)
    inflight = MagicMock()
    inflight.done.return_value = False
    adapter._polling_error_task = inflight
    await adapter._probe_pending_updates(adapter._app.bot, 5)
    adapter._app.bot.get_webhook_info.assert_not_called()


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
