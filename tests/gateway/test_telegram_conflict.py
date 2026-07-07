import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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

    # Provide real exception classes so ``except (NetworkError, ...)`` in
    # connect() doesn't blow up with "catching classes that do not inherit
    # from BaseException" when another xdist worker pollutes sys.modules.
    telegram_mod.error.NetworkError = type("NetworkError", (OSError,), {})
    telegram_mod.error.TimedOut = type("TimedOut", (OSError,), {})
    telegram_mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)
    sys.modules.setdefault("telegram.error", telegram_mod.error)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    """Disable DoH auto-discovery so connect() uses the plain builder chain."""
    async def _noop():
        return []
    monkeypatch.setattr("plugins.platforms.telegram.adapter.discover_fallback_ips", _noop)
    # Mock HTTPXRequest so the builder chain doesn't fail
    monkeypatch.setattr("plugins.platforms.telegram.adapter.HTTPXRequest", lambda **kwargs: MagicMock())


async def _cancel_heartbeat(adapter):
    """Cancel the lifetime heartbeat task connect() starts in polling mode.

    These tests call the real connect() but never disconnect(), so the
    _polling_heartbeat_loop task would otherwise outlive the test. With
    asyncio.sleep monkeypatched to instant, leaving it running busy-spins the
    event loop and starves the test (CI per-file timeout). disconnect() does
    this in production; tests that only connect() must do it themselves.
    """
    task = getattr(adapter, "_polling_heartbeat_task", None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    adapter._polling_heartbeat_task = None


@pytest.mark.asyncio
async def test_connect_rejects_same_host_token_lock(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secret-token"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (False, {"pid": 4242}),
    )

    ok = await adapter.connect()

    assert ok is False
    assert adapter.fatal_error_code == "telegram-bot-token_lock"
    assert adapter.has_fatal_error is True
    assert "already in use" in adapter.fatal_error_message


@pytest.mark.asyncio
async def test_polling_conflict_retries_before_fatal(monkeypatch):
    """A single 409 should trigger a retry, not an immediate fatal error."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    captured = {}

    async def fake_start_polling(**kwargs):
        captured["error_callback"] = kwargs["error_callback"]

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=fake_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock(), delete_webhook=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr("plugins.platforms.telegram.adapter.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

    # Speed up retries for testing
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    ok = await adapter.connect()

    assert ok is True
    bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=False)
    assert callable(captured["error_callback"])

    conflict = type("Conflict", (Exception,), {})

    # First conflict: should retry, NOT be fatal
    captured["error_callback"](conflict("Conflict: terminated by other getUpdates request"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    # Give the scheduled task a chance to run
    for _ in range(10):
        await asyncio.sleep(0)

    assert adapter.has_fatal_error is False, "First conflict should not be fatal"
    assert adapter._polling_conflict_count == 0, "Count should reset after successful retry"

    # connect() now starts a lifetime _polling_heartbeat_loop task. With
    # asyncio.sleep mocked to instant above, it must not be left running or it
    # busy-spins on the event loop and starves the test. Cancel it explicitly.
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_polling_conflict_becomes_fatal_after_retries(monkeypatch):
    """After exhausting retries, the conflict should become fatal."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    captured = {}

    async def fake_start_polling(**kwargs):
        captured["error_callback"] = kwargs["error_callback"]

    # Make start_polling fail on retries to exhaust retries
    call_count = {"n": 0}

    async def failing_start_polling(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call (initial connect) succeeds
            captured["error_callback"] = kwargs["error_callback"]
        else:
            # Retry calls fail
            raise Exception("Connection refused")

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=failing_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock(), delete_webhook=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr("plugins.platforms.telegram.adapter.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

    # Speed up retries for testing
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    ok = await adapter.connect()
    assert ok is True

    conflict = type("Conflict", (Exception,), {})

    # Directly call _handle_polling_conflict to avoid event-loop scheduling
    # complexity.  Each call simulates one 409 from Telegram.
    for i in range(6):
        await adapter._handle_polling_conflict(
            conflict("Conflict: terminated by other getUpdates request")
        )

    # Retries 1-4 each schedule a background recovery task via
    # loop.create_task(self._handle_polling_conflict(...)) that this test
    # never awaits.  Cancel the last one so a leaked task can't get a
    # scheduler turn under load and re-drive the counter into the fatal
    # branch a second time — which would fire _notify_fatal_error twice and
    # break assert_awaited_once() non-deterministically.
    leaked = adapter._polling_error_task
    if leaked is not None and not leaked.done():
        leaked.cancel()
        try:
            await leaked
        except (asyncio.CancelledError, Exception):
            pass

    # After 5 failed retries (count 1-5 each enter the retry branch but
    # start_polling raises), the 6th conflict pushes count to 6 which
    # exceeds MAX_CONFLICT_RETRIES (5), entering the fatal branch.
    assert adapter.fatal_error_code == "telegram_polling_conflict", (
        f"Expected fatal after 6 conflicts, got code={adapter.fatal_error_code}, "
        f"count={adapter._polling_conflict_count}"
    )
    assert adapter.has_fatal_error is True
    fatal_handler.assert_awaited_once()
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_connect_marks_retryable_fatal_error_for_startup_network_failure(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    app = SimpleNamespace(
        bot=SimpleNamespace(delete_webhook=AsyncMock(), set_my_commands=AsyncMock()),
        updater=SimpleNamespace(),
        add_handler=MagicMock(),
        initialize=AsyncMock(side_effect=RuntimeError("Temporary failure in name resolution")),
        start=AsyncMock(),
    )
    builder.build.return_value = app
    monkeypatch.setattr("plugins.platforms.telegram.adapter.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

    ok = await adapter.connect()

    assert ok is False
    assert adapter.fatal_error_code == "telegram_connect_error"
    assert adapter.fatal_error_retryable is True
    assert "Temporary failure in name resolution" in adapter.fatal_error_message


@pytest.mark.asyncio
async def test_connect_clears_webhook_before_polling(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    updater = SimpleNamespace(
        start_polling=AsyncMock(),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(
        delete_webhook=AsyncMock(),
        set_my_commands=AsyncMock(),
    )
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )

    ok = await adapter.connect()

    assert ok is True
    bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=False)
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_connect_does_not_block_on_post_connect_housekeeping(monkeypatch):
    """Regression for #46298.

    Command-menu registration and DM-topic setup make Bot API calls that can
    stall for certain tokens. If they run inside connect() (which the gateway
    wraps in a connect timeout), one slow call blows the whole connect and the
    adapter never comes up. connect() must return as soon as polling/webhook is
    live and defer that housekeeping to a cancellable background task.
    """
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    async def _hang_forever(*args, **kwargs):
        await asyncio.Future()

    # Make the entire housekeeping coroutine hang. connect() must still return
    # promptly and expose the still-running task; disconnect() must cancel it.
    monkeypatch.setattr(adapter, "_run_post_connect_housekeeping", _hang_forever)

    updater = SimpleNamespace(
        start_polling=AsyncMock(),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(
        delete_webhook=AsyncMock(),
        set_my_commands=AsyncMock(),
    )
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
        running=True,
        stop=AsyncMock(),
        shutdown=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )

    # A tight timeout: if connect() awaited the hanging set_my_commands this
    # would raise TimeoutError instead of returning.
    ok = await asyncio.wait_for(adapter.connect(), timeout=0.5)

    assert ok is True
    assert adapter._post_connect_task is not None
    assert not adapter._post_connect_task.done()

    # disconnect() must cancel the still-hanging housekeeping task cleanly.
    await adapter.disconnect()
    assert adapter._post_connect_task is None
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_disconnect_skips_inactive_updater_and_app(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    updater = SimpleNamespace(running=False, stop=AsyncMock())
    app = SimpleNamespace(
        updater=updater,
        running=False,
        stop=AsyncMock(),
        shutdown=AsyncMock(),
    )
    adapter._app = app

    warning = MagicMock()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.logger.warning", warning)

    await adapter.disconnect()

    updater.stop.assert_not_awaited()
    app.stop.assert_not_awaited()
    app.shutdown.assert_awaited_once()
    warning.assert_not_called()


@pytest.mark.asyncio
async def test_polling_conflict_reschedule_uses_running_loop(monkeypatch):
    """Regression for #19471.

    When a conflict-retry's start_polling raises and we are still below the
    retry ceiling, the handler reschedules itself via loop.create_task. The
    old code used the deprecated asyncio.get_event_loop(), which raises
    "RuntimeError: There is no current event loop in thread 'MainThread'" on
    Python 3.11+ when no loop is attached to the thread (as happens when PTB
    dispatches this error callback). That left the gateway alive but silent
    and drove the --replace crash loop. The fix uses get_running_loop(), which
    is always valid inside a coroutine. Force get_event_loop() to raise so a
    regression would surface as the original RuntimeError, not pass silently.
    """
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.set_fatal_error_handler(AsyncMock())

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    captured = {}
    call_count = {"n": 0}

    async def failing_start_polling(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            captured["error_callback"] = kwargs["error_callback"]
        else:
            # Retry attempt fails so the handler enters the reschedule branch.
            raise Exception("Connection refused")

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=failing_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock(), delete_webhook=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    ok = await adapter.connect()
    assert ok is True

    # If the fix regresses to get_event_loop(), this makes it raise — the same
    # RuntimeError users hit in #19471. The running-loop path ignores it.
    def _boom():
        raise RuntimeError("There is no current event loop in thread 'MainThread'.")

    monkeypatch.setattr("asyncio.get_event_loop", _boom)

    conflict = type("Conflict", (Exception,), {})

    # One conflict: count goes to 1 (< MAX), retry's start_polling raises,
    # handler reschedules via loop.create_task — the previously-broken line.
    await adapter._handle_polling_conflict(
        conflict("Conflict: terminated by other getUpdates request")
    )

    assert adapter.has_fatal_error is False
    assert adapter._polling_error_task is not None
    # The rescheduled task must be schedulable on the running loop.
    adapter._polling_error_task.cancel()
    try:
        await adapter._polling_error_task
    except (asyncio.CancelledError, Exception):
        pass
    await _cancel_heartbeat(adapter)


def _build_polling_app(monkeypatch):
    """Wire a mock PTB Application whose start_polling captures kwargs."""
    captured = {}

    async def fake_start_polling(**kwargs):
        captured.update(kwargs)

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=fake_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock(), delete_webhook=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    return captured


@pytest.mark.asyncio
async def test_cold_connect_drops_pending_updates(monkeypatch):
    """A cold first boot (is_reconnect=False) drops the stale Bot API queue."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    captured = _build_polling_app(monkeypatch)

    ok = await adapter.connect()  # default is_reconnect=False

    assert ok is True
    assert captured["drop_pending_updates"] is True
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_reconnect_preserves_pending_updates(monkeypatch):
    """A watcher reconnect (is_reconnect=True) preserves the queue Telegram
    accumulated during the outage — the core of #46621."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    captured = _build_polling_app(monkeypatch)

    ok = await adapter.connect(is_reconnect=True)

    assert ok is True
    assert captured["drop_pending_updates"] is False
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_polling_conflict_retry_routes_network_errors_to_network_handler(monkeypatch):
    """A conflict retry that fails with Bad Gateway must not recurse as conflict."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._handle_polling_network_error = AsyncMock()
    adapter._polling_conflict_count = 0

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    monkeypatch.setattr(adapter, "_drain_polling_connections", AsyncMock())

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=Exception("Bad Gateway")),
        stop=AsyncMock(),
        running=True,
    )
    adapter._app = SimpleNamespace(updater=updater)

    conflict = type("Conflict", (Exception,), {})
    await adapter._handle_polling_conflict(
        conflict("Conflict: terminated by other getUpdates request")
    )
    if adapter._polling_error_task is not None:
        await adapter._polling_error_task

    adapter._handle_polling_network_error.assert_awaited_once()
    err = adapter._handle_polling_network_error.await_args.args[0]
    assert "Bad Gateway" in str(err)
    assert adapter.has_fatal_error is False


@pytest.mark.asyncio
async def test_polling_conflict_entry_routes_plain_network_errors(monkeypatch):
    """Plain API outages must not consume conflict retry budget."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._handle_polling_network_error = AsyncMock()

    try:
        from telegram.error import NetworkError
        error = NetworkError("Bad Gateway")
    except ImportError:
        error = Exception("Bad Gateway")

    await adapter._handle_polling_conflict(error)

    adapter._handle_polling_network_error.assert_awaited_once_with(error)
    assert adapter._polling_conflict_count == 0


@pytest.mark.asyncio
async def test_disarm_sets_ptb_stop_event():
    """_disarm_ptb_retry_loop sets PTB's name-mangled polling stop_event.

    This is the root-cause fix for the 409 conflict loop (#30122): the
    error_callback must synchronously signal PTB's internal network_retry_loop
    to stop BEFORE our async recovery task restarts polling, otherwise the two
    polling sessions overlap and produce a fresh 409.
    """
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    stop_event = asyncio.Event()
    # PTB stores it name-mangled as _Updater__polling_task_stop_event.
    updater = SimpleNamespace(running=True)
    setattr(updater, "_Updater__polling_task_stop_event", stop_event)
    adapter._app = SimpleNamespace(updater=updater)

    assert not stop_event.is_set()
    adapter._disarm_ptb_retry_loop()
    assert stop_event.is_set(), "disarm must set PTB's polling stop_event"
    # Must not flip _running — the recovery handler's stop() guards on running
    # and stop() raises if running is already False.
    assert updater.running is True


@pytest.mark.asyncio
async def test_disarm_noop_when_stop_event_absent():
    """When PTB exposes no stop_event, disarm is a safe no-op (no regression).

    It must NOT flip _running (which would make the handler skip stop() and
    leave the loop wedged) — it just falls back to the prior async stop() race.
    """
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    updater = SimpleNamespace(running=True, _running=True)
    adapter._app = SimpleNamespace(updater=updater)

    adapter._disarm_ptb_retry_loop()  # no stop_event attribute present

    assert updater.running is True
    assert updater._running is True, "disarm must not flip _running as a fallback"


@pytest.mark.asyncio
async def test_conflict_callback_disarms_before_scheduling(monkeypatch):
    """The polling error_callback disarms PTB synchronously, then schedules
    recovery — proving the fix is wired into the live callback, not just the
    helper (#30122)."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    captured = {}

    async def fake_start_polling(**kwargs):
        captured["error_callback"] = kwargs["error_callback"]

    stop_event = asyncio.Event()
    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=fake_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    setattr(updater, "_Updater__polling_task_stop_event", stop_event)
    bot = SimpleNamespace(set_my_commands=AsyncMock(), delete_webhook=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )

    ok = await adapter.connect()
    assert ok is True

    conflict = type("Conflict", (Exception,), {})
    # Fire a 409 through the live callback. The disarm must happen
    # synchronously (before any await), so the stop_event is set immediately
    # on return — before the scheduled recovery task gets a chance to run.
    assert not stop_event.is_set()
    captured["error_callback"](conflict("Conflict: terminated by other getUpdates"))
    assert stop_event.is_set(), "callback must disarm PTB synchronously"
    assert adapter._polling_error_task is not None, "recovery task must be scheduled"

    # Drain the scheduled recovery task so it doesn't outlive the test.
    for _ in range(10):
        await asyncio.sleep(0)
    await _cancel_heartbeat(adapter)
