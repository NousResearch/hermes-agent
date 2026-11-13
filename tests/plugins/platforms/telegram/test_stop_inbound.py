"""Drain-time inbound stop: getUpdates stops while the handler is installed, and stays stopped."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig


def _adapter(*, webhook_mode=False):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    stops = []

    async def _stop():
        stops.append("stop")

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra={})
    adapter._webhook_mode = webhook_mode
    adapter._polling_teardown_started = False
    adapter._polling_progress_accepting = True
    adapter._polling_generation = 3
    adapter._app = SimpleNamespace(updater=SimpleNamespace(running=True, stop=_stop))
    return adapter, stops


@pytest.mark.asyncio
async def test_stop_inbound_stops_polling_and_fences_the_restart_ladder():
    from plugins.platforms.telegram.adapter import _PollingLifecycleAbort

    adapter, stops = _adapter()
    await adapter.stop_inbound()
    assert stops == ["stop"] and adapter._polling_generation == 4
    # The reconnect/heartbeat ladder must read the stop as teardown, not a wedged fetcher.
    with pytest.raises(_PollingLifecycleAbort):
        await adapter._start_polling_once(adapter._app, drop_pending_updates=False, error_callback=None)
    await adapter.stop_inbound()  # already fenced: no second stop
    assert stops == ["stop"]

    # Shutdown during conflict recovery (updater stopped, restart pending): still fenced.
    recovering, stops = _adapter()
    recovering._app.updater.running = False
    await recovering.stop_inbound()
    assert stops == [] and recovering._teardown_started
    with pytest.raises(_PollingLifecycleAbort):
        await recovering._start_polling_once(recovering._app, drop_pending_updates=False, error_callback=None)

    webhook, stops = _adapter(webhook_mode=True)  # server-pushed: nothing to stop
    await webhook.stop_inbound()
    assert stops == [] and webhook._polling_teardown_started is False
