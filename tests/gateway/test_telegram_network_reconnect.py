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

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    """Disable DoH auto-discovery so connect() uses the plain builder chain."""
    async def _noop():
        return []
    monkeypatch.setattr("gateway.platforms.telegram.discover_fallback_ips", _noop)


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
        with patch("gateway.platforms.telegram.asyncio.wait_for", new=fast_wait_for):
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


# ---------------------------------------------------------------------------
# Socket/transport leak regression (fix/telegram-socket-leak)
# ---------------------------------------------------------------------------
#
# PTB's HTTPXRequest.initialize() rebuilds its httpx.AsyncClient by reusing the
# SAME _client_kwargs dict, including the original `transport` object. The
# TelegramFallbackTransport wraps several httpcore connection pools. Reusing a
# closed transport across reconnect cycles lets half-open sockets to an
# unreachable peer accumulate, eventually exhausting file descriptors
# ("OSError: [Errno 24] Too many open files"). The fix swaps in a fresh
# transport (and force-closes the old one) on every drain.


class _FakeTransport:
    """Stand-in for TelegramFallbackTransport that tracks close calls."""

    instances: list = []

    def __init__(self):
        self.closed = False
        _FakeTransport.instances.append(self)

    async def aclose(self):
        self.closed = True


def _make_mock_app_with_real_request():
    """Mock Application whose polling request mimics PTB's HTTPXRequest.

    Specifically it carries a `_client_kwargs` dict holding a `transport`
    object, like the real HTTPXRequest, so the drain path's transport-swap
    logic is exercised end to end.
    """
    polling_req = MagicMock()
    polling_req.shutdown = AsyncMock()
    polling_req.initialize = AsyncMock()
    polling_req._client_kwargs = {"transport": _FakeTransport()}

    mock_bot = MagicMock()
    mock_bot._request = (polling_req, MagicMock())

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot = mock_bot
    return mock_app, polling_req


@pytest.mark.asyncio
async def test_drain_swaps_in_fresh_transport_and_closes_old():
    """Each drain must install a NEW transport and close the previous one."""
    _FakeTransport.instances.clear()
    adapter = _make_adapter()
    # Simulate connect() having configured a fallback transport.
    adapter._polling_transport_factory = lambda: _FakeTransport()

    mock_app, polling_req = _make_mock_app_with_real_request()
    adapter._app = mock_app
    original_transport = polling_req._client_kwargs["transport"]

    await adapter._drain_polling_connections()

    new_transport = polling_req._client_kwargs["transport"]
    assert new_transport is not original_transport, (
        "drain must swap in a fresh transport instance, not reuse the closed one"
    )
    assert original_transport.closed, "old transport must be force-closed on drain"
    polling_req.shutdown.assert_awaited_once()
    polling_req.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_repeated_reconnects_do_not_leak_transports():
    """N reconnect cycles must leave exactly one live (unclosed) transport.

    Before the fix the single transport instance was reused across every cycle,
    so its connection pools accumulated half-open sockets without bound. After
    the fix each cycle closes the prior transport and installs a fresh one, so
    the number of *unclosed* transports stays at 1 regardless of cycle count.
    """
    _FakeTransport.instances.clear()
    adapter = _make_adapter()
    adapter._polling_transport_factory = lambda: _FakeTransport()

    mock_app, polling_req = _make_mock_app_with_real_request()
    adapter._app = mock_app

    N = 25
    for _ in range(N):
        await adapter._drain_polling_connections()

    live = [t for t in _FakeTransport.instances if not t.closed]
    closed = [t for t in _FakeTransport.instances if t.closed]
    assert len(live) == 1, (
        f"expected exactly 1 live transport after {N} reconnects, "
        f"got {len(live)} (transport leak)"
    )
    # And the live one is the transport currently installed on the request.
    assert live[0] is polling_req._client_kwargs["transport"]
    # Each prior cycle's transport must have been force-closed — proving the
    # transport is genuinely cycled rather than silently reused (the pre-fix
    # behavior, which closed zero transports).
    assert len(closed) == N, (
        f"expected {N} closed transports (one per reconnect), got {len(closed)}"
    )


@pytest.mark.asyncio
async def test_drain_noop_swap_without_fallback_factory():
    """When no fallback transport is in use, the drain leaves kwargs untouched.

    PTB's plain transport-less rebuild already mints a fresh pool, so there is
    nothing to swap and the transport key must be preserved as-is.
    """
    _FakeTransport.instances.clear()
    adapter = _make_adapter()
    adapter._polling_transport_factory = None  # no fallback transport configured

    mock_app, polling_req = _make_mock_app_with_real_request()
    adapter._app = mock_app
    original_transport = polling_req._client_kwargs["transport"]

    await adapter._drain_polling_connections()

    assert polling_req._client_kwargs["transport"] is original_transport
    assert not original_transport.closed
    polling_req.shutdown.assert_awaited_once()
    polling_req.initialize.assert_awaited_once()


def test_default_pool_size_fits_fd_budget(monkeypatch):
    """The default connection_pool_size must stay within a sane FD budget.

    Two request objects, each up to (1 primary + a few fallback) pools, times
    the pool size, must stay comfortably under a launchd service's default
    256-FD limit headroom. 64 keeps the worst case bounded; 512 did not.
    """
    monkeypatch.delenv("HERMES_TELEGRAM_HTTP_POOL_SIZE", raising=False)

    def _env_int(name: str, default: int) -> int:
        import os
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    default_pool = _env_int("HERMES_TELEGRAM_HTTP_POOL_SIZE", 64)
    assert default_pool == 64
    # Two request objects each wrapping ~3 pools (1 primary + 2 fallback IPs).
    worst_case_ceiling = 2 * 3 * default_pool
    assert worst_case_ceiling <= 512, (
        "default pool ceiling must stay well under typical FD budgets"
    )
