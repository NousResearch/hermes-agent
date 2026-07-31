"""D59 regression judge: a retryable telegram fatal must be *honored*, not just declared.

Incident (2026-07-26 12:26 / 2026-07-29 07:30, profile-b live gateway): after the
polling network-error ladder exhausted its retries the adapter logged
"Restarting gateway" and escalated a retryable fatal, but the gateway process
stayed alive with zero platforms, an empty reconnect queue, and no shutdown —
a zombie that needed a manual restart hours later.

The judge drives the *real* escalation path end to end: the fatal notification
runs on the adapter's own tracked polling-recovery task (exactly how the
chained-retry call site arranges it), and the gateway handler tears the adapter
down mid-notification. The contract asserted here is behavioral and matches
what the fatal declaration promises:

    after a retryable fatal completes, the platform is queued for background
    reconnection OR the gateway is shutting down. Alive + no adapters + empty
    queue is the zombie state and must be impossible.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_runner(adapter: TelegramAdapter) -> GatewayRunner:
    """Minimal real GatewayRunner (object.__new__ idiom, like test_platform_reconnect)."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="123:TEST")}
    )
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_with_failure = False
    runner._exit_cleanly = False
    runner._failed_platforms = {}
    runner._background_tasks = set()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.delivery_router = MagicMock()
    runner._update_platform_runtime_status = MagicMock()
    runner.session_store = MagicMock()
    return runner


@pytest.mark.asyncio
async def test_network_retry_exhaustion_fatal_is_queued_or_exits():
    """The 07-26/07-29 incident shape, on the real adapter + real gateway handler."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:TEST"))
    runner = _make_runner(adapter)
    adapter.set_fatal_error_handler(runner._handle_adapter_fatal_error)

    async def drive():
        # Ladder already exhausted: the next call is attempt MAX+1 -> fatal path.
        adapter._polling_network_error_count = 10
        await adapter._handle_polling_network_error(
            OSError("httpx.ConnectError: All connection attempts failed")
        )

    # The chained-retry call site (adapter.py) registers the recovery coroutine
    # as _polling_error_task and tracks it in _background_tasks; reproduce that
    # exact task topology so adapter.disconnect() sees the notifier as a
    # cancellable tracked task.
    task = asyncio.create_task(drive())
    adapter._polling_error_task = task
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)

    await asyncio.wait({task}, timeout=30)
    assert task.done(), "fatal escalation wedged (never completed nor died)"
    # Give any detached/shielded handler continuation a chance to finish.
    for _ in range(50):
        if Platform.TELEGRAM in runner._failed_platforms or runner._shutdown_event.is_set():
            break
        await asyncio.sleep(0.1)

    try:
        queued = Platform.TELEGRAM in runner._failed_platforms
        exiting = runner._shutdown_event.is_set() or runner._exit_with_failure or runner._exit_cleanly
        zombie = (not runner.adapters) and (not queued) and (not exiting)
        assert not zombie, (
            "D59 zombie gateway: retryable fatal was declared but the platform is "
            "neither queued for reconnection nor is the gateway exiting "
            f"(adapters={runner.adapters!r}, failed={runner._failed_platforms!r})"
        )
        assert queued or exiting, (
            "fatal declaration was not honored: expected reconnect queueing or shutdown"
        )
    finally:
        # Stop any watcher the fixed code may have spawned so pytest exits clean.
        runner._running = False
        watcher = getattr(runner, "_reconnect_watcher_task", None)
        if watcher is not None and not watcher.done():
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        for t in list(runner._background_tasks) + list(adapter._background_tasks):
            if isinstance(t, asyncio.Task) and not t.done():
                t.cancel()
                await asyncio.gather(t, return_exceptions=True)
