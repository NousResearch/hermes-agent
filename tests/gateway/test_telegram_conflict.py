import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
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
        # Cold connect requires real getUpdates readiness (#67498) — simulate
        # the first successful poll for the generation this call started, but
        # only on the initial connect: the conflict-retry generation must NOT
        # make progress here, or it would legitimately reset the conflict
        # count this test asserts on.
        if not captured.get("initial_done"):
            captured["initial_done"] = True
            adapter._record_polling_progress(adapter._polling_generation)

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
    await adapter._polling_error_task

    assert adapter.has_fatal_error is False, "First conflict should not be fatal"
    assert adapter._polling_conflict_count == 1, (
        "Count must remain until the retried generation makes getUpdates progress"
    )
    assert adapter._send_path_degraded is True

    # connect() now starts a lifetime _polling_heartbeat_loop task. With
    # asyncio.sleep mocked to instant above, it must not be left running or it
    # busy-spins on the event loop and starves the test. Cancel it explicitly.
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_conflict_retry_drops_pending_updates(monkeypatch):
    """Conflict recovery must use drop_pending_updates=True (#75017).

    Without this, each retry starts a new getUpdates session that
    immediately gets 409'd by the previous still-expiring session,
    creating the very conflict we are trying to recover from.
    """
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.set_fatal_error_handler(AsyncMock())
    adapter._drain_polling_connections = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    captured = {}

    async def fake_start_polling(**kwargs):
        captured["drop_pending_updates"] = kwargs.get("drop_pending_updates")

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=fake_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    adapter._app = SimpleNamespace(updater=updater)

    conflict = type("Conflict", (Exception,), {})
    await adapter._handle_polling_conflict(
        conflict("Conflict: terminated by other getUpdates request")
    )

    assert captured.get("drop_pending_updates") is True, (
        "Conflict retry must use drop_pending_updates=True to terminate "
        "stale getUpdates sessions on Telegram's servers (#75017)"
    )


@pytest.mark.asyncio
async def test_conflict_retry_progress_does_not_reset_retry_ladder(monkeypatch):
    """First getUpdates progress after a conflict retry is not durable recovery.

    Telegram can accept the first long-poll after a retry and then return a 409
    from the still-expiring previous session. That transient success must not
    reset the retry counter back to 0, or every new 409 looks like attempt 1/5
    and the backoff never reaches the server-side expiry window.
    """
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.set_fatal_error_handler(AsyncMock())
    adapter._drain_polling_connections = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    calls = {"n": 0}

    async def fake_start_polling(**_kwargs):
        calls["n"] += 1
        adapter._record_polling_progress(adapter._polling_generation)

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=fake_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    adapter._app = SimpleNamespace(updater=updater)

    conflict = type("Conflict", (Exception,), {})
    await adapter._handle_polling_conflict(
        conflict("Conflict: terminated by other getUpdates request")
    )

    assert calls["n"] == 1
    assert adapter._polling_conflict_count == 1
    assert adapter._polling_conflict_recovery_generation is None
    assert adapter._send_path_degraded is False


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
            # Cold connect requires getUpdates readiness (#67498).
            adapter._record_polling_progress(adapter._polling_generation)
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

    async def _start_polling_with_progress(**_kwargs):
        # Cold connect requires getUpdates readiness (#67498).
        adapter._record_polling_progress(adapter._polling_generation)

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=_start_polling_with_progress),
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

    async def _start_polling_with_progress(**_kwargs):
        # Cold connect requires getUpdates readiness (#67498).
        adapter._record_polling_progress(adapter._polling_generation)

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=_start_polling_with_progress),
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
            # Cold connect requires getUpdates readiness (#67498).
            adapter._record_polling_progress(adapter._polling_generation)
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


def _build_polling_app(monkeypatch, adapter):
    """Wire a mock PTB Application whose start_polling captures kwargs."""
    captured = {}

    async def fake_start_polling(**kwargs):
        captured.update(kwargs)
        # Cold connect requires getUpdates readiness (#67498).
        adapter._record_polling_progress(adapter._polling_generation)

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
async def test_cold_connect_preserves_pending_updates_when_enabled(monkeypatch):
    """An opted-in cold first boot preserves the Bot API queue."""
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={"preserve_backlog": True},
        )
    )
    captured = _build_polling_app(monkeypatch, adapter)

    ok = await adapter.connect()

    assert ok is True
    assert captured["drop_pending_updates"] is False
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_reconnect_preserves_pending_updates(monkeypatch):
    """A watcher reconnect (is_reconnect=True) preserves the queue Telegram
    accumulated during the outage — the core of #46621."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    captured = _build_polling_app(monkeypatch, adapter)

    ok = await adapter.connect(is_reconnect=True)

    assert ok is True
    assert captured["drop_pending_updates"] is False
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_reconnect_preserves_pending_updates_when_backlog_preservation_enabled(
    monkeypatch,
):
    """An opted-in watcher reconnect keeps its existing queue behavior."""
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={"preserve_backlog": True},
        )
    )
    captured = _build_polling_app(monkeypatch, adapter)

    ok = await adapter.connect(is_reconnect=True)

    assert ok is True
    assert captured["drop_pending_updates"] is False
    await _cancel_heartbeat(adapter)


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
        # Cold connect requires getUpdates readiness (#67498).
        adapter._record_polling_progress(adapter._polling_generation)

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


# ---------------------------------------------------------------------------
# Startup conflict under backlog preservation.
#
# Conflict recovery restarts polling with drop_pending_updates=True to evict
# the competing getUpdates session, which also discards everything Telegram
# has queued. That is the right trade by default. It is the wrong trade when
# extra.preserve_backlog is on, because the queue is the whole point of the
# setting, so a 409 seen during the cold-start readiness gate is terminal
# instead.
#
# A first successful getUpdates is readiness, not proof of exclusive
# ownership: Telegram can answer one getUpdates and 409 the next (#75017).
# ---------------------------------------------------------------------------

_CONFLICT_TEXT = "Conflict: terminated by other getUpdates request"


def _build_conflicting_polling_app(
    monkeypatch, adapter, *, also_make_progress: bool
):
    """Polling app whose first generation reports a 409 to the error callback.

    ``also_make_progress`` additionally records getUpdates progress for that
    generation, which is the readiness/conflict race: both signals land before
    the gate resolves.
    """
    captured = {}
    conflict_cls = type("Conflict", (Exception,), {})

    async def fake_start_polling(**kwargs):
        captured.update(kwargs)
        captured.setdefault("start_polling_calls", []).append(
            kwargs.get("drop_pending_updates")
        )
        if also_make_progress:
            adapter._record_polling_progress(adapter._polling_generation)
        cb = kwargs.get("error_callback")
        if cb is not None:
            cb(conflict_cls(_CONFLICT_TEXT))

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
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock", lambda scope, identity: None
    )
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    captured["app"] = app
    return captured


def _preserving_adapter():
    return TelegramAdapter(
        PlatformConfig(
            enabled=True, token="***", extra={"preserve_backlog": True}
        )
    )


@pytest.mark.asyncio
async def test_pre_progress_conflict_is_terminal_and_not_retryable(monkeypatch):
    """1. A 409 before readiness fails startup non-retryably."""
    adapter = _preserving_adapter()
    adapter.set_fatal_error_handler(AsyncMock())
    _build_conflicting_polling_app(monkeypatch, adapter, also_make_progress=False)

    ok = await adapter.connect()

    assert ok is False
    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "telegram_polling_conflict"
    assert adapter._fatal_error_retryable is False, (
        "a startup conflict must not be retried; the competing instance has "
        "to be stopped first"
    )
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_pre_progress_conflict_skips_recovery_and_destructive_restart(
    monkeypatch,
):
    """2. It neither runs conflict recovery nor restarts destructively."""
    adapter = _preserving_adapter()
    adapter.set_fatal_error_handler(AsyncMock())
    recovery = AsyncMock()
    monkeypatch.setattr(adapter, "_handle_polling_conflict", recovery)
    captured = _build_conflicting_polling_app(
        monkeypatch, adapter, also_make_progress=False
    )

    await adapter.connect()

    recovery.assert_not_awaited()
    assert adapter._polling_conflict_count == 0, (
        "the conflict-retry ladder must not have been entered"
    )
    assert True not in captured["start_polling_calls"], (
        "no start_polling may run with drop_pending_updates=True, which is "
        f"what discards the backlog; saw {captured['start_polling_calls']}"
    )
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_stored_conflict_wins_a_simultaneous_progress_race(monkeypatch):
    """3. Readiness does not out-race a conflict captured in the same gate.

    Progress proves the transport answered once, not that we own the token
    (#75017). Before the guard this case connected successfully and the 409
    was discarded.
    """
    adapter = _preserving_adapter()
    adapter.set_fatal_error_handler(AsyncMock())
    _build_conflicting_polling_app(monkeypatch, adapter, also_make_progress=True)

    ok = await adapter.connect()

    assert ok is False, "a captured conflict must decide the outcome"
    assert adapter.fatal_error_code == "telegram_polling_conflict"
    assert adapter._fatal_error_retryable is False
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_conflict_after_readiness_still_uses_the_recovery_ladder(
    monkeypatch,
):
    """4. Post-readiness conflicts keep recovering, even with the flag on.

    The gate has closed and the backlog has already been collected, so the
    established retry ladder still applies.
    """
    adapter = _preserving_adapter()
    adapter.set_fatal_error_handler(AsyncMock())
    captured = _build_polling_app(monkeypatch, adapter)

    ok = await adapter.connect()
    assert ok is True
    assert adapter.has_fatal_error is False

    conflict = type("Conflict", (Exception,), {})
    captured["error_callback"](conflict(_CONFLICT_TEXT))
    await adapter._polling_error_task

    assert adapter._polling_conflict_count == 1, (
        "a conflict after readiness must still enter the retry ladder"
    )
    assert adapter.has_fatal_error is False
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_default_config_keeps_the_recovery_ladder_for_startup_conflicts(
    monkeypatch,
):
    """5a. Without the flag, a startup 409 behaves exactly as before."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.set_fatal_error_handler(AsyncMock())
    _build_conflicting_polling_app(monkeypatch, adapter, also_make_progress=False)

    ok = await adapter.connect()

    assert ok is False
    assert adapter.fatal_error_code != "telegram_polling_conflict", (
        "the terminal startup conflict is opt-in via preserve_backlog; the "
        "default path must keep its previous generic classification"
    )
    assert adapter._fatal_error_retryable is True
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_reconnect_is_unaffected_by_the_startup_conflict_guard(monkeypatch):
    """5b. Reconnects have no readiness gate, so the guard cannot fire."""
    adapter = _preserving_adapter()
    adapter.set_fatal_error_handler(AsyncMock())
    captured = _build_conflicting_polling_app(
        monkeypatch, adapter, also_make_progress=False
    )

    ok = await adapter.connect(is_reconnect=True)

    assert ok is True, "a reconnect must stay alive and recover in background"
    assert captured["drop_pending_updates"] is False
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_startup_conflict_message_does_not_leak_the_token(monkeypatch):
    """5c. Redaction still applies to the terminal conflict message.

    The token uses a realistic shape (35+ chars after the colon) because that
    is what the shared redactor matches. A short placeholder passes straight
    through and would prove nothing.
    """
    secret = "123456789:AAHrealistic_len_token_35chars_abcdef"
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True, token=secret, extra={"preserve_backlog": True}
        )
    )
    adapter.set_fatal_error_handler(AsyncMock())

    captured = {}
    conflict_cls = type("Conflict", (Exception,), {})

    async def fake_start_polling(**kwargs):
        captured.update(kwargs)
        cb = kwargs.get("error_callback")
        if cb is not None:
            cb(conflict_cls(f"Conflict on https://api.telegram.org/bot{secret}/getUpdates"))

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=fake_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock(), delete_webhook=AsyncMock())
    app = SimpleNamespace(
        bot=bot, updater=updater, add_handler=MagicMock(),
        initialize=AsyncMock(), start=AsyncMock(),
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
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock", lambda scope, identity: None
    )
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    await adapter.connect()

    assert secret not in (adapter._fatal_error_message or "")
    await _cancel_heartbeat(adapter)
