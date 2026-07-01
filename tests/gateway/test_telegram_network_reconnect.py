"""
Tests for Telegram polling network error recovery.

Specifically tests the fix for #3173 — when start_polling() fails after a
network error, the adapter must self-reschedule the next reconnect attempt
rather than silently leaving polling dead.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    """Disable DoH auto-discovery so connect() uses the plain builder chain."""
    async def _noop():
        return []
    monkeypatch.setattr("plugins.platforms.telegram.adapter.discover_fallback_ips", _noop)


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


@pytest.mark.asyncio
async def test_reconnect_self_schedules_on_start_polling_failure():
    """
    When start_polling() raises during a network error retry, the adapter must
    schedule a new _handle_polling_network_error task — otherwise polling stays
    dead with no further error callbacks to trigger recovery.

    Regression test for #3173: gateway becomes unresponsive after Telegram 502.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock(side_effect=Exception("Timed out"))

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    # A retry task must have been added to _background_tasks
    pending = [t for t in adapter._background_tasks if not t.done()]
    assert len(pending) >= 1, (
        "Expected at least one self-rescheduled retry task in _background_tasks "
        f"after start_polling failure, got {len(pending)}"
    )

    # Clean up — cancel the pending retry so it doesn't run after the test
    for t in pending:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_reconnect_does_not_self_schedule_when_fatal_error_set():
    """
    When a fatal error is already set, the failed reconnect should NOT create
    another retry task — the gateway is already shutting down this adapter.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1
    adapter._set_fatal_error("telegram_network_error", "already fatal", retryable=True)

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock(side_effect=Exception("Timed out"))

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    initial_count = len(adapter._background_tasks)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Timed out"))

    assert len(adapter._background_tasks) == initial_count, (
        "Should not schedule a retry when a fatal error is already set"
    )


@pytest.mark.asyncio
async def test_reconnect_chained_retry_updates_polling_error_task():
    """
    When start_polling() fails and the handler self-schedules a retry, that
    retry task must become the new `_polling_error_task` — otherwise the
    reentrancy guard used by the heartbeat loop, the pending-updates probe,
    and the PTB error callback goes stale while a recovery is still in
    flight, letting a second concurrent recovery start for the same outage.

    Regression test for the race behind the "half-destroyed adapter" bug
    (gateway reports connected but silently stops processing messages).
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock(side_effect=Exception("Timed out"))

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    assert adapter._polling_error_task is not None
    assert not adapter._polling_error_task.done()

    adapter._polling_error_task.cancel()
    try:
        await adapter._polling_error_task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_reconnect_success_resets_error_count():
    """
    When start_polling() succeeds, _polling_network_error_count should reset to 0.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 3

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()  # succeeds

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock(return_value=MagicMock())  # heartbeat probe path
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    assert adapter._polling_network_error_count == 0

    # Clean up the heartbeat-probe task scheduled after a successful reconnect.
    pending = [t for t in adapter._background_tasks if not t.done()]
    for t in pending:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_reconnect_triggers_fatal_after_max_retries():
    """
    After MAX_NETWORK_RETRIES attempts, the adapter should set a fatal error
    rather than retrying forever.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 10  # MAX_NETWORK_RETRIES

    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    mock_app = MagicMock()
    adapter._app = mock_app

    await adapter._handle_polling_network_error(Exception("still failing"))

    assert adapter.has_fatal_error
    assert adapter.fatal_error_code == "telegram_network_error"
    fatal_handler.assert_called_once()


# ---------------------------------------------------------------------------
# Connection pool drain tests (PR #16466 salvage)
# ---------------------------------------------------------------------------

def _make_mock_app():
    """Build a mock Application with an explicit polling request object."""
    mock_polling_req = AsyncMock()
    mock_polling_req.shutdown = AsyncMock()
    mock_polling_req.initialize = AsyncMock()

    mock_bot = MagicMock()
    mock_bot._request = (mock_polling_req, MagicMock())  # (getUpdates, general)

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot = mock_bot
    return mock_app, mock_polling_req


@pytest.mark.asyncio
async def test_reconnect_drains_polling_request_only():
    """During reconnect, only the polling request (_request[0]) must be cycled.

    The general request (_request[1]) must NOT be touched — doing so would
    break concurrent send_message / edit_message calls.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_app, mock_polling_req = _make_mock_app()
    adapter._app = mock_app

    general_req = mock_app.bot._request[1]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    # Polling request must be shut down and re-initialized
    mock_polling_req.shutdown.assert_called_once()
    mock_polling_req.initialize.assert_called_once()

    # General request must NOT be touched
    general_req.shutdown.assert_not_called()
    general_req.initialize.assert_not_called()

    # Reconnect must still succeed
    mock_app.updater.start_polling.assert_called_once()
    assert adapter._polling_network_error_count == 0


@pytest.mark.asyncio
async def test_reconnect_continues_if_drain_fails():
    """If the polling request drain raises, start_polling must still proceed."""
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_app, mock_polling_req = _make_mock_app()
    # Both shutdown and initialize fail
    mock_polling_req.shutdown = AsyncMock(side_effect=Exception("shutdown boom"))
    mock_polling_req.initialize = AsyncMock(side_effect=Exception("init boom"))
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    # start_polling must still be called despite drain failure
    mock_app.updater.start_polling.assert_called_once()
    assert adapter._polling_network_error_count == 0


@pytest.mark.asyncio
async def test_initialize_still_runs_when_shutdown_fails():
    """If shutdown() raises, initialize() must still be attempted.

    This prevents a failed shutdown from leaving the request pool in a
    permanently closed state.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_app, mock_polling_req = _make_mock_app()
    mock_polling_req.shutdown = AsyncMock(side_effect=Exception("shutdown boom"))
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    # initialize MUST be called even though shutdown raised
    mock_polling_req.initialize.assert_called_once()
    mock_app.updater.start_polling.assert_called_once()


@pytest.mark.asyncio
async def test_conflict_retry_also_drains_polling_connections():
    """_handle_polling_conflict must also drain the polling pool on retry."""
    adapter = _make_adapter()
    adapter._polling_conflict_count = 0

    mock_app, mock_polling_req = _make_mock_app()
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_conflict(Exception("Conflict: terminated by other getUpdates"))

    # Polling request must be drained during conflict retry too
    mock_polling_req.shutdown.assert_called_once()
    mock_polling_req.initialize.assert_called_once()
    mock_app.updater.start_polling.assert_called_once()


@pytest.mark.asyncio
async def test_drain_helper_noop_without_app():
    """_drain_polling_connections must be a no-op when _app is None."""
    adapter = _make_adapter()
    adapter._app = None
    # Should not raise
    await adapter._drain_polling_connections()


# ── Heartbeat probe ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_heartbeat_probe_no_op_when_polling_healthy():
    """
    Probe scheduled after a successful reconnect: Updater.running=True and
    bot.get_me() returns quickly → recovery confirmed, no further action.
    """
    adapter = _make_adapter()

    mock_updater = MagicMock()
    mock_updater.running = True

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock(return_value=MagicMock())
    adapter._app = mock_app

    adapter._handle_polling_network_error = AsyncMock()

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._verify_polling_after_reconnect()

    mock_app.bot.get_me.assert_awaited_once()
    adapter._handle_polling_network_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_heartbeat_probe_reenters_ladder_when_updater_not_running():
    """
    If Updater.running has flipped to False by the heartbeat delay, treat
    as wedged: re-enter the reconnect ladder.
    """
    adapter = _make_adapter()

    mock_updater = MagicMock()
    mock_updater.running = False

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock()
    adapter._app = mock_app

    adapter._handle_polling_network_error = AsyncMock()

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._verify_polling_after_reconnect()

    mock_app.bot.get_me.assert_not_called()
    adapter._handle_polling_network_error.assert_awaited_once()
    err = adapter._handle_polling_network_error.await_args.args[0]
    assert isinstance(err, RuntimeError)
    assert "not running" in str(err).lower()


@pytest.mark.asyncio
async def test_heartbeat_probe_reenters_ladder_when_get_me_times_out():
    """
    If bot.get_me() hangs longer than PROBE_TIMEOUT, treat as wedged.
    Simulates the connection-pool wedge that motivated this fix.
    """
    adapter = _make_adapter()

    mock_updater = MagicMock()
    mock_updater.running = True

    async def hang_forever(*args, **kwargs):
        await asyncio.sleep(3600)

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock(side_effect=hang_forever)
    adapter._app = mock_app

    adapter._handle_polling_network_error = AsyncMock()

    async def fast_wait_for(coro, timeout):
        if asyncio.iscoroutine(coro):
            coro.close()
        raise asyncio.TimeoutError()

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with patch("plugins.platforms.telegram.adapter.asyncio.wait_for", new=fast_wait_for):
            await adapter._verify_polling_after_reconnect()

    adapter._handle_polling_network_error.assert_awaited_once()
@pytest.mark.asyncio
async def test_heartbeat_probe_reenters_ladder_on_get_me_network_error():
    """
    Any exception raised by bot.get_me() (NetworkError, ConnectionError, etc.)
    should re-enter the reconnect ladder with the original exception.
    """
    adapter = _make_adapter()

    mock_updater = MagicMock()
    mock_updater.running = True

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock(side_effect=ConnectionError("pool wedged"))
    adapter._app = mock_app

    adapter._handle_polling_network_error = AsyncMock()

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._verify_polling_after_reconnect()

    adapter._handle_polling_network_error.assert_awaited_once()
    assert isinstance(
        adapter._handle_polling_network_error.await_args.args[0], ConnectionError
    )


@pytest.mark.asyncio
async def test_heartbeat_probe_skips_when_already_fatal():
    """
    If the adapter is already in fatal-error state by the time the probe
    delay elapses, the probe should bail without further action.
    """
    adapter = _make_adapter()
    adapter._set_fatal_error("telegram_polling_conflict", "already fatal", retryable=False)

    mock_app = MagicMock()
    mock_app.bot.get_me = AsyncMock()
    adapter._app = mock_app

    adapter._handle_polling_network_error = AsyncMock()

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._verify_polling_after_reconnect()

    mock_app.bot.get_me.assert_not_called()
    adapter._handle_polling_network_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconnect_schedules_heartbeat_probe_on_success():
    """
    After a successful start_polling() in the reconnect path, a probe task
    must be added to _background_tasks. Without it, a wedged Updater would
    sit silent indefinitely with no further error_callback to advance the
    reconnect ladder.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()  # succeeds

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock(return_value=MagicMock())
    adapter._app = mock_app

    initial_count = len(adapter._background_tasks)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    assert len(adapter._background_tasks) > initial_count, (
        "Expected a heartbeat probe task to be scheduled after a successful "
        "reconnect's start_polling()"
    )

    # Clean up.
    pending = [t for t in adapter._background_tasks if not t.done()]
    for t in pending:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


# ── Persistent heartbeat loop (_polling_heartbeat_loop) ──────────────────────
#
# These tests cover the continuous CLOSE-WAIT detection loop that fixes the bug
# (#48495) where a dead Telegram TCP socket caused the gateway to stop receiving
# messages silently. The _verify_polling_after_reconnect tests above cover the
# one-shot post-reconnect probe; these cover the background loop that runs for
# the gateway's full lifetime in polling mode.
#
# Loop structure: while True: sleep(INTERVAL) → fatal/app checks → get_me().
# So with cancel raised on the Nth patched sleep, get_me() fires (N-1) times.


@pytest.mark.asyncio
async def test_heartbeat_loop_exits_cleanly_on_cancel():
    """The heartbeat loop must exit without raising when cancelled (normal shutdown)."""
    adapter = _make_adapter()

    mock_app = MagicMock()
    mock_app.bot.get_me = AsyncMock(return_value=MagicMock())
    adapter._app = mock_app

    sleep_count = 0

    async def fast_sleep(seconds):
        nonlocal sleep_count
        sleep_count += 1
        # sleep #1 → get_me, sleep #2 → get_me, sleep #3 → cancel.
        if sleep_count >= 3:
            raise asyncio.CancelledError()

    with patch("asyncio.sleep", side_effect=fast_sleep):
        # Should not raise — CancelledError is swallowed internally.
        await adapter._polling_heartbeat_loop()

    assert mock_app.bot.get_me.await_count == 2


@pytest.mark.asyncio
async def test_heartbeat_loop_triggers_reconnect_on_timeout():
    """A TimeoutError from get_me() must schedule a reconnect via _handle_polling_network_error."""
    adapter = _make_adapter()
    adapter._handle_polling_network_error = AsyncMock()

    mock_app = MagicMock()
    adapter._app = mock_app

    sleep_call = 0

    async def fast_sleep(seconds):
        nonlocal sleep_call
        sleep_call += 1
        if sleep_call >= 3:
            raise asyncio.CancelledError()

    async def fast_wait_for(coro, timeout):
        if asyncio.iscoroutine(coro):
            coro.close()
        raise asyncio.TimeoutError()

    with patch("asyncio.sleep", side_effect=fast_sleep):
        with patch("plugins.platforms.telegram.adapter.asyncio.wait_for", side_effect=fast_wait_for):
            await adapter._polling_heartbeat_loop()

    # A reconnect task must have been created.
    assert adapter._polling_error_task is not None


@pytest.mark.asyncio
async def test_heartbeat_loop_triggers_reconnect_on_os_error():
    """An OSError (e.g. connection reset) from get_me() must trigger a reconnect."""
    adapter = _make_adapter()
    adapter._handle_polling_network_error = AsyncMock()

    mock_app = MagicMock()
    adapter._app = mock_app

    sleep_call = 0

    async def fast_sleep(seconds):
        nonlocal sleep_call
        sleep_call += 1
        if sleep_call >= 3:
            raise asyncio.CancelledError()

    async def os_error_wait_for(coro, timeout):
        if asyncio.iscoroutine(coro):
            coro.close()
        raise OSError("Connection reset by peer")

    with patch("asyncio.sleep", side_effect=fast_sleep):
        with patch("plugins.platforms.telegram.adapter.asyncio.wait_for", side_effect=os_error_wait_for):
            await adapter._polling_heartbeat_loop()

    assert adapter._polling_error_task is not None


@pytest.mark.asyncio
async def test_heartbeat_loop_skips_reconnect_if_already_in_progress():
    """If a reconnect task is already running, the heartbeat must not spawn another."""
    adapter = _make_adapter()

    # Simulate an already-running reconnect task.
    existing_task = asyncio.get_event_loop().create_task(asyncio.sleep(3600))
    adapter._polling_error_task = existing_task
    adapter._handle_polling_network_error = AsyncMock()

    mock_app = MagicMock()
    adapter._app = mock_app

    sleep_call = 0

    async def fast_sleep(seconds):
        nonlocal sleep_call
        sleep_call += 1
        if sleep_call >= 3:
            raise asyncio.CancelledError()

    async def timeout_wait_for(coro, timeout):
        if asyncio.iscoroutine(coro):
            coro.close()
        raise asyncio.TimeoutError()

    with patch("asyncio.sleep", side_effect=fast_sleep):
        with patch("plugins.platforms.telegram.adapter.asyncio.wait_for", side_effect=timeout_wait_for):
            await adapter._polling_heartbeat_loop()

    # _handle_polling_network_error must NOT have been called — existing task still running.
    adapter._handle_polling_network_error.assert_not_awaited()

    existing_task.cancel()
    try:
        await existing_task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_heartbeat_loop_ignores_non_connectivity_errors():
    """Errors that are not connectivity failures (e.g. TelegramError) must be swallowed."""
    adapter = _make_adapter()
    adapter._handle_polling_network_error = AsyncMock()

    mock_app = MagicMock()
    adapter._app = mock_app

    sleep_call = 0

    async def fast_sleep(seconds):
        nonlocal sleep_call
        sleep_call += 1
        if sleep_call >= 3:
            raise asyncio.CancelledError()

    async def telegram_error_wait_for(coro, timeout):
        if asyncio.iscoroutine(coro):
            coro.close()
        raise RuntimeError("TelegramError: Unauthorized")  # non-OSError, non-TimeoutError

    with patch("asyncio.sleep", side_effect=fast_sleep):
        with patch("plugins.platforms.telegram.adapter.asyncio.wait_for", side_effect=telegram_error_wait_for):
            await adapter._polling_heartbeat_loop()

    # No reconnect should have been triggered for a non-connectivity error.
    adapter._handle_polling_network_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_heartbeat_loop_exits_on_fatal_error():
    """A fatal error short-circuits the loop before probing get_me()."""
    adapter = _make_adapter()
    adapter._set_fatal_error("telegram_network_error", "boom", retryable=True)

    mock_app = MagicMock()
    mock_app.bot.get_me = AsyncMock(return_value=MagicMock())
    adapter._app = mock_app

    async def fast_sleep(seconds):
        return None

    with patch("asyncio.sleep", side_effect=fast_sleep):
        await adapter._polling_heartbeat_loop()

    # Fatal error returns before the get_me() probe.
    mock_app.bot.get_me.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_cancels_heartbeat_task():
    """disconnect() must cancel the heartbeat task before shutting down the app."""
    adapter = _make_adapter()

    # Simulate a running heartbeat.
    heartbeat_task = asyncio.get_event_loop().create_task(asyncio.sleep(3600))
    adapter._polling_heartbeat_task = heartbeat_task

    mock_app = MagicMock()
    mock_app.updater = MagicMock()
    mock_app.updater.running = False
    mock_app.running = False
    mock_app.shutdown = AsyncMock()
    adapter._app = mock_app

    await adapter.disconnect()

    assert heartbeat_task.cancelled(), "Heartbeat task must be cancelled by disconnect()"
    assert adapter._polling_heartbeat_task is None


# ── Bootstrap degradation: keep polling alive during outages (#47508) ────


@pytest.mark.asyncio
async def test_delete_webhook_network_error_is_recoverable():
    """deleteWebhook timeouts must not fail gateway startup.

    A transient Bot API outage during bootstrap should be treated as
    recoverable and continue toward polling, so it never becomes a systemd
    service failure.
    """
    adapter = _make_adapter()
    mock_bot = MagicMock()
    mock_bot.delete_webhook = AsyncMock(side_effect=ConnectionError("api.telegram.org timeout"))
    adapter._bot = mock_bot

    result = await adapter._delete_webhook_best_effort()

    assert result is False
    assert adapter._send_path_degraded is True
    mock_bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=False)
    assert not adapter.has_fatal_error


@pytest.mark.asyncio
async def test_polling_bootstrap_network_error_schedules_background_recovery():
    """Initial start_polling() network failure should degrade, not raise."""
    adapter = _make_adapter()
    mock_updater = MagicMock()
    mock_updater.start_polling = AsyncMock(side_effect=ConnectionError("bootstrap timeout"))
    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app
    adapter._schedule_polling_recovery = MagicMock()

    result = await adapter._start_polling_resilient(
        drop_pending_updates=True,
        error_callback=lambda error: None,
    )

    assert result is False
    adapter._schedule_polling_recovery.assert_called_once()
    err = adapter._schedule_polling_recovery.call_args.args[0]
    assert isinstance(err, ConnectionError)
    assert adapter._schedule_polling_recovery.call_args.kwargs["reason"] == "polling bootstrap"
    assert not adapter.has_fatal_error


@pytest.mark.asyncio
async def test_polling_bootstrap_conflict_schedules_conflict_recovery_task():
    """Initial 409 polling conflict should also be recovered in background."""
    adapter = _make_adapter()
    mock_updater = MagicMock()
    mock_updater.start_polling = AsyncMock(
        side_effect=Exception("Conflict: terminated by other getUpdates request")
    )
    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app
    adapter._handle_polling_conflict = AsyncMock()

    result = await adapter._start_polling_resilient(
        drop_pending_updates=True,
        error_callback=lambda error: None,
    )

    assert result is False
    pending = [t for t in adapter._background_tasks if not t.done()]
    assert pending, "expected background conflict recovery task"
    for task in pending:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    assert not adapter.has_fatal_error


@pytest.mark.asyncio
async def test_schedule_polling_recovery_tracks_background_task():
    """Background recovery task is registered so it isn't GC'd mid-flight."""
    adapter = _make_adapter()
    adapter._handle_polling_network_error = AsyncMock()

    adapter._schedule_polling_recovery(ConnectionError("boom"), reason="unit test")

    assert adapter._send_path_degraded is True
    assert adapter._polling_error_task is not None
    assert adapter._polling_error_task in adapter._background_tasks
    await adapter._polling_error_task
    adapter._handle_polling_network_error.assert_awaited_once()
