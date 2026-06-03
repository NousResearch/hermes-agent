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

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    """Disable DoH auto-discovery so connect() uses the plain builder chain."""
    async def _noop():
        return []
    monkeypatch.setattr("gateway.platforms.telegram.discover_fallback_ips", _noop)
    # Mock HTTPXRequest so the builder chain doesn't fail
    monkeypatch.setattr("gateway.platforms.telegram.HTTPXRequest", lambda **kwargs: MagicMock())


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
    monkeypatch.setattr("gateway.platforms.telegram.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

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
    monkeypatch.setattr("gateway.platforms.telegram.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

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

    # After 5 failed retries (count 1-5 each enter the retry branch but
    # start_polling raises), the 6th conflict pushes count to 6 which
    # exceeds MAX_CONFLICT_RETRIES (5), entering the fatal branch.
    assert adapter.fatal_error_code == "telegram_polling_conflict", (
        f"Expected fatal after 6 conflicts, got code={adapter.fatal_error_code}, "
        f"count={adapter._polling_conflict_count}"
    )
    assert adapter.has_fatal_error is True
    fatal_handler.assert_awaited_once()


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
    monkeypatch.setattr("gateway.platforms.telegram.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

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
        "gateway.platforms.telegram.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )

    ok = await adapter.connect()

    assert ok is True
    bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=False)


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
    monkeypatch.setattr("gateway.platforms.telegram.logger.warning", warning)

    await adapter.disconnect()

    updater.stop.assert_not_awaited()
    app.stop.assert_not_awaited()
    app.shutdown.assert_awaited_once()
    warning.assert_not_called()


@pytest.mark.asyncio
async def test_polling_heartbeat_detects_wedged_poll(monkeypatch):
    """A wedged long-poll (get_me() hangs/fails) must re-enter the reconnect ladder."""
    monkeypatch.setenv("HERMES_TELEGRAM_HEARTBEAT_INTERVAL", "0.01")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._polling_error_task = None

    calls = []

    async def _handler(err):
        calls.append(err)
        # Trip fatal so the heartbeat loop exits after this one iteration —
        # robust against tests that globally mock asyncio.sleep.
        adapter._set_fatal_error("test_heartbeat_stop", "stop", retryable=False)

    adapter._handle_polling_network_error = _handler
    adapter._app = SimpleNamespace(
        updater=SimpleNamespace(running=True),
        bot=SimpleNamespace(get_me=AsyncMock(side_effect=RuntimeError("wedged"))),
    )

    await asyncio.wait_for(adapter._polling_heartbeat_loop(), timeout=5)

    assert len(calls) >= 1
    assert isinstance(calls[0], Exception)


@pytest.mark.asyncio
async def test_polling_heartbeat_detects_stopped_updater(monkeypatch):
    """An updater that has stopped running must re-enter the reconnect ladder."""
    monkeypatch.setenv("HERMES_TELEGRAM_HEARTBEAT_INTERVAL", "0.01")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._polling_error_task = None

    calls = []

    async def _handler(err):
        calls.append(err)
        adapter._set_fatal_error("test_heartbeat_stop", "stop", retryable=False)

    adapter._handle_polling_network_error = _handler
    adapter._app = SimpleNamespace(
        updater=SimpleNamespace(running=False),
        bot=SimpleNamespace(get_me=AsyncMock(return_value=SimpleNamespace(username="bot"))),
    )

    await asyncio.wait_for(adapter._polling_heartbeat_loop(), timeout=5)

    assert len(calls) >= 1


@pytest.mark.asyncio
async def test_polling_heartbeat_quiet_when_healthy(monkeypatch):
    """A healthy poll (updater running, get_me() succeeds) must not trip recovery."""
    monkeypatch.setenv("HERMES_TELEGRAM_HEARTBEAT_INTERVAL", "0.01")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._polling_error_task = None

    handler = AsyncMock()
    adapter._handle_polling_network_error = handler

    async def _get_me():
        # Trip fatal so the loop exits after one healthy probe.
        adapter._set_fatal_error("test_heartbeat_stop", "stop", retryable=False)
        return SimpleNamespace(username="bot")

    adapter._app = SimpleNamespace(
        updater=SimpleNamespace(running=True),
        bot=SimpleNamespace(get_me=_get_me),
    )

    await asyncio.wait_for(adapter._polling_heartbeat_loop(), timeout=5)

    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_polling_heartbeat_survives_handler_raise(monkeypatch):
    """If the recovery handler itself raises, the heartbeat must NOT propagate
    the exception out (a raising handler can't be allowed to kill the loop)."""
    monkeypatch.setenv("HERMES_TELEGRAM_HEARTBEAT_INTERVAL", "0.01")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._polling_error_task = None

    # Recovery handler always blows up — the loop must swallow it.
    adapter._handle_polling_network_error = AsyncMock(
        side_effect=RuntimeError("recovery exploded")
    )

    # updater.running is False -> enters the `not updater.running` recovery
    # branch every iteration, where the handler raises. A counter on the
    # `running` property trips fatal after a couple of iterations so the loop
    # self-terminates (get_me() is never reached in this branch).
    term = {"n": 0}

    class _FlakyUpdater:
        @property
        def running(self):
            term["n"] += 1
            if term["n"] >= 2:
                adapter._set_fatal_error(
                    "test_heartbeat_stop", "stop", retryable=False
                )
            return False

    adapter._app = SimpleNamespace(
        updater=_FlakyUpdater(),
        bot=SimpleNamespace(get_me=AsyncMock()),
    )

    # Must NOT raise the RuntimeError out of the loop.
    await asyncio.wait_for(adapter._polling_heartbeat_loop(), timeout=5)

    # Handler was invoked (and raised) at least once but was swallowed.
    assert adapter._handle_polling_network_error.await_count >= 1


@pytest.mark.asyncio
async def test_polling_heartbeat_continues_when_app_none(monkeypatch):
    """When self._app is transiently None, the loop must `continue`, not return
    permanently — otherwise a reconnect (which rebuilds the app) permanently
    kills the heartbeat."""
    monkeypatch.setenv("HERMES_TELEGRAM_HEARTBEAT_INTERVAL", "0.01")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._polling_error_task = None

    handler = AsyncMock()
    adapter._handle_polling_network_error = handler

    # First iteration: app is None (transient reconnect window). The loop must
    # NOT return here. Second iteration: a healthy app whose get_me() trips
    # fatal so the loop self-terminates — proving it survived the None.
    healthy_app = SimpleNamespace(
        updater=SimpleNamespace(running=True),
        bot=None,  # filled in below
    )

    async def _get_me():
        adapter._set_fatal_error("test_heartbeat_stop", "stop", retryable=False)
        return SimpleNamespace(username="bot")

    healthy_app.bot = SimpleNamespace(get_me=_get_me)

    # Start with _app None (transient reconnect window), then flip it to a
    # healthy app after the first loop tick by intercepting asyncio.sleep. On
    # the FIRST iteration _app is still None, exercising the `continue` path;
    # on the SECOND, get_me() trips fatal and the loop self-terminates.
    adapter._app = None
    ticks = {"n": 0}
    real_sleep = asyncio.sleep

    async def _sleep(delay, *a, **k):
        ticks["n"] += 1
        if ticks["n"] >= 2:
            adapter._app = healthy_app
        return await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _sleep)

    await asyncio.wait_for(adapter._polling_heartbeat_loop(), timeout=5)

    # The loop reached the healthy app and tripped fatal — proving the None
    # iteration did NOT return-and-die. Recovery handler was never needed.
    handler.assert_not_awaited()
    assert adapter.has_fatal_error is True
