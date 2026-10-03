import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_await_with_thread_deadline_abandons_and_runs_cleanup_on_timeout():
    """A wedged awaitable must raise TimeoutError promptly AND trigger the
    best-effort on_abandon cleanup (the httpx-pool-leak guard).

    This exercises the REAL _await_with_thread_deadline (not a monkeypatched
    stub), covering the abandonment + cleanup mechanism directly.
    """
    import asyncio as _asyncio
    import time as _time

    cleanup_ran = _asyncio.Event()

    async def _wedged():
        # Swallows cancellation for a bounded window — long enough that the
        # helper must return control BEFORE this finishes (proving it doesn't
        # await cancellation, the #58236 shielded-scope behavior), but bounded
        # so the abandoned task can't outlive the test and wedge teardown.
        for _ in range(20):
            try:
                await _asyncio.sleep(0.05)
            except _asyncio.CancelledError:
                # Keep going despite cancellation, like the shielded scope.
                pass

    async def _cleanup():
        cleanup_ran.set()

    started = _time.monotonic()
    with pytest.raises(_asyncio.TimeoutError):
        await tg_adapter._await_with_thread_deadline(
            _wedged(), timeout=0.2, on_abandon=_cleanup
        )
    elapsed = _time.monotonic() - started

    # Returned control promptly — well before the wedged coroutine's ~1s span.
    assert elapsed < 0.8
    # The detached cleanup was scheduled; give the loop a tick to run it.
    await _asyncio.wait_for(cleanup_ran.wait(), timeout=2.0)
    assert cleanup_ran.is_set()


@pytest.mark.asyncio
async def test_await_with_thread_deadline_cleanup_error_is_swallowed():
    """A cleanup that raises must not surface as an unhandled task error."""
    import asyncio as _asyncio

    async def _wedged():
        for _ in range(20):
            try:
                await _asyncio.sleep(0.05)
            except _asyncio.CancelledError:
                pass

    def _boom():
        raise RuntimeError("cleanup blew up")

    # Must still raise TimeoutError (not the cleanup error) and not crash.
    with pytest.raises(_asyncio.TimeoutError):
        await tg_adapter._await_with_thread_deadline(
            _wedged(), timeout=0.2, on_abandon=_boom
        )
    # Let the detached cleanup task run and be observed (no unraised error).
    await _asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_blocked_loop_after_expiry_dumps_diagnostics(monkeypatch):
    """#63309: when the loop thread is stuck in a synchronous call, the expiry
    callback never runs and every asyncio timeout goes silent. The off-loop
    watchdog must detect that state and emit diagnostics from its own thread."""
    import asyncio as _asyncio
    import time as _time

    from agent import deadline as _deadline

    dumps = []
    monkeypatch.setattr(
        _deadline,
        "_dump_blocked_loop_diagnostics",
        lambda label, timeout_s: dumps.append((label, timeout_s)),
    )
    monkeypatch.setattr(_deadline, "_LOOP_BLOCKED_DUMP_GRACE_S", 0.15)

    hung = _asyncio.get_running_loop().create_future()  # never completes
    task = _asyncio.ensure_future(
        tg_adapter._await_with_thread_deadline(hung, timeout=0.05)
    )
    # Let the helper start its deadline + watchdog timers…
    await _asyncio.sleep(0)
    # …then block the event loop straight through deadline (0.05s) AND the
    # watchdog grace (0.15s): call_soon_threadsafe stays queued, exactly like
    # a sync call pinning the loop during Application.initialize().
    # Margin matters: the watchdog thread only dumps if the loop is STILL
    # blocked when it wakes, and thread wakeup lags under parallel-suite load.
    # 0.2s (= deadline+grace exactly) flaked in a 40-worker full-suite run.
    _time.sleep(1.0)
    with pytest.raises(_asyncio.TimeoutError):
        await task

    assert dumps == [("telegram", 0.05)]
    hung.cancel()




class _Request:
    """PTB request double: ``initialize`` reopens a closed client, ``shutdown`` closes it."""

    def __init__(self, events=None, **kwargs):
        self.closed, self._events = False, events if events is not None else []

    async def initialize(self):
        self.closed = False

    async def shutdown(self):
        self._events.append("request closed")
        self.closed = True


def _app(initialize, requests):
    from unittest.mock import AsyncMock, MagicMock

    app = MagicMock()
    app.initialize = initialize
    app.shutdown = AsyncMock()  # PTB: no-op for an app that never finished initialize()
    app.start = AsyncMock()
    app.running = False
    app.updater.running = False
    app.updater.start_polling = AsyncMock()
    app.bot._request = requests
    return app


async def _built_requests(monkeypatch, adapter):
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _Request)
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "1")
    return await adapter._build_ptb_requests()


@pytest.mark.asyncio
async def test_cancelled_connect_closes_the_abandoned_init_transports_after_it_unwinds():
    """A cancelled connect closes abandoned transports during disconnect and after any late reopen."""
    import asyncio as _asyncio
    from unittest.mock import MagicMock

    events = []
    requests = (_Request(events), _Request(events))
    entered, release = _asyncio.Event(), _asyncio.Event()

    async def _shielded_initialize():
        for request in requests:
            await request.initialize()
        entered.set()
        while not release.is_set():  # an anyio-shielded httpcore scope swallows the cancel
            try:
                await release.wait()
            except _asyncio.CancelledError:
                events.append("init cancelled")
        events.append("init exited")

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    app = _app(_shielded_initialize, requests)
    adapter._app, adapter._bot = app, app.bot
    ladder = _asyncio.ensure_future(adapter._initialize_app_with_retries(MagicMock()))
    await _asyncio.wait_for(entered.wait(), timeout=5)

    ladder.cancel()  # the runner's connect deadline cancels connect() mid-initialize
    with pytest.raises(_asyncio.CancelledError):
        await ladder
    owners = tuple(adapter._abandoned_initialize_owners)
    _asyncio.get_running_loop().call_later(0.2, release.set)
    await _asyncio.wait_for(adapter.disconnect(), timeout=10)
    await _asyncio.gather(*owners)

    assert all(request.closed for request in requests)
    assert events.index("init exited") < len(events) - 1
    # The abandoned app never started, so it cannot run a getUpdates loop beside a successor.
    app.start.assert_not_awaited()
    app.updater.start_polling.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_closes_transports_after_the_init_ladder_is_exhausted(monkeypatch):
    """``app.shutdown()`` no-ops for the last attempt's never-initialized app, so disconnect() must
    close its request transports itself."""
    import asyncio as _asyncio
    from unittest.mock import AsyncMock, MagicMock

    requests = (_Request([]), _Request([]))

    async def _unreachable():
        for request in adapter._app.bot._request:
            await request.initialize()
        raise OSError("api.telegram.org unreachable")

    builder = MagicMock()
    builder.general, builder.updates = requests

    def set_request(request):
        builder.general = request
        return builder

    def set_updates_request(request):
        builder.updates = request
        return builder

    builder.request.side_effect = set_request
    builder.get_updates_request.side_effect = set_updates_request
    builder.build.side_effect = lambda: _app(_unreachable, (builder.general, builder.updates))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    async def fresh_requests():
        return _Request([]), _Request([])

    monkeypatch.setattr(adapter, "_build_ptb_requests", fresh_requests)
    adapter._app = builder.build()
    adapter._bot = adapter._app.bot

    with pytest.raises(OSError):
        await adapter._initialize_app_with_retries(builder)
    last_requests = adapter._app.bot._request
    assert all(request is not old for request, old in zip(last_requests, requests))
    assert not any(request.closed for request in last_requests)  # the last attempt's pools are still open

    await _asyncio.wait_for(adapter.disconnect(), timeout=10)

    assert all(request.closed for request in last_requests)


@pytest.mark.asyncio
async def test_cancelled_abandoned_init_owner_still_closes_transports():
    import asyncio
    from unittest.mock import MagicMock

    requests = (_Request([]), _Request([]))
    entered, release = asyncio.Event(), asyncio.Event()

    async def initialize():
        for request in requests:
            await request.initialize()
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        for request in requests:
            await request.initialize()  # The in-flight init rebuilds a client after an early close.

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    app = _app(initialize, requests)
    adapter._app, adapter._bot = app, app.bot
    ladder = asyncio.create_task(adapter._initialize_app_with_retries(MagicMock()))
    await asyncio.wait_for(entered.wait(), timeout=1)
    ladder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ladder
    owner = next(iter(adapter._abandoned_initialize_owners))

    owner.cancel()  # asyncio.run also cancels remaining tasks during shutdown.
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(owner, return_exceptions=True)

    assert all(request.closed for request in requests)


@pytest.mark.asyncio
async def test_disconnect_closes_abandoned_transports_before_shielded_init_unwinds(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock

    from plugins.platforms.telegram import telegram_abandoned_init

    # Scale the 2 s disconnect / 15 s owner clocks down; init exits at simulated 5 s.
    monkeypatch.setattr(tg_adapter, "_DISCONNECT_STEP_TIMEOUT", 0.05)
    monkeypatch.setattr(telegram_abandoned_init, "_ABANDONED_INIT_UNWIND_TIMEOUT", 0.375)
    requests = (_Request([]), _Request([]))
    entered, release = asyncio.Event(), asyncio.Event()

    async def initialize():
        for request in requests:
            await request.initialize()
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        for request in requests:
            await request.initialize()  # A shielded init can reopen after disconnect's close.

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    app = _app(initialize, requests)
    adapter._app, adapter._bot = app, app.bot
    ladder = asyncio.create_task(adapter._initialize_app_with_retries(MagicMock()))
    await asyncio.wait_for(entered.wait(), timeout=1)
    ladder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ladder
    owners = tuple(adapter._abandoned_initialize_owners)

    try:
        await asyncio.wait_for(adapter.disconnect(), timeout=2)
        assert all(request.closed for request in requests)
        await asyncio.sleep(0.125)  # Simulated 5 s unwind on the scaled clock.
    finally:
        release.set()
        await asyncio.gather(*owners, return_exceptions=True)
    assert all(request.closed for request in requests)


@pytest.mark.asyncio
async def test_cancelled_ladder_retains_app_detached_during_await(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock

    requests = (_Request([]), _Request([]))
    release = asyncio.Event()

    async def initialize():
        for request in requests:
            await request.initialize()
        await release.wait()

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    app = _app(initialize, requests)
    adapter._app, adapter._bot = app, app.bot

    async def detached_while_waiting(*args, **kwargs):
        adapter._app = None  # A concurrent teardown cleared the app during the await.
        raise asyncio.CancelledError

    monkeypatch.setattr(tg_adapter, "_await_with_thread_deadline", detached_while_waiting)
    with pytest.raises(asyncio.CancelledError):
        await adapter._initialize_app_with_retries(MagicMock())
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(*adapter._abandoned_initialize_owners, return_exceptions=True)

    assert all(request.closed for request in requests)


@pytest.mark.asyncio
async def test_cancelled_initialize_cannot_reopen_requests_after_disconnect(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    requests = await _built_requests(monkeypatch, adapter)
    entered, reopen, reopened, finish = (asyncio.Event() for _ in range(4))
    init_tasks = []

    async def initialize():
        init_tasks.append(asyncio.current_task())
        for request in requests:
            await request.initialize()
        entered.set()
        while not reopen.is_set():
            try:
                await reopen.wait()
            except asyncio.CancelledError:
                pass
        for request in requests:
            await request.initialize()
        reopened.set()
        await finish.wait()

    app = _app(initialize, requests)
    adapter._app, adapter._bot = app, app.bot
    ladder = asyncio.create_task(adapter._initialize_app_with_retries(MagicMock()))
    await asyncio.wait_for(entered.wait(), timeout=1)
    ladder.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ladder
    owners = tuple(adapter._abandoned_initialize_owners)

    try:
        await asyncio.wait_for(adapter.disconnect(), timeout=5)
        reopen.set()
        await asyncio.wait_for(reopened.wait(), timeout=1)
        assert not init_tasks[0].done()
        assert all(request.closed for request in requests)
    finally:
        finish.set()
        await asyncio.gather(*owners, return_exceptions=True)


@pytest.mark.asyncio
async def test_timed_out_initialize_cannot_reopen_discarded_requests_after_disconnect(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock

    monkeypatch.setenv("HERMES_TELEGRAM_INIT_TIMEOUT", "0.05")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    requests = await _built_requests(monkeypatch, adapter)
    entered, reopen, reopened, finish = (asyncio.Event() for _ in range(4))
    init_tasks = []

    async def initialize():
        init_tasks.append(asyncio.current_task())
        for request in requests:
            await request.initialize()
        entered.set()
        while not reopen.is_set():
            try:
                await reopen.wait()
            except asyncio.CancelledError:
                pass
        for request in requests:
            await request.initialize()
        reopened.set()
        await finish.wait()

    async def failed_retry():
        raise ValueError("stop after the first retry")

    first_app = _app(initialize, requests)
    successor = _app(failed_retry, await _built_requests(monkeypatch, adapter))
    builder = MagicMock()
    builder.build.return_value = successor
    adapter._app, adapter._bot = first_app, first_app.bot
    ladder = asyncio.create_task(adapter._initialize_app_with_retries(builder))
    await asyncio.wait_for(entered.wait(), timeout=1)

    try:
        with pytest.raises(ValueError, match="stop after the first retry"):
            await asyncio.wait_for(ladder, timeout=3)
        assert adapter._app is successor
        await asyncio.wait_for(adapter.disconnect(), timeout=5)
        reopen.set()
        await asyncio.wait_for(reopened.wait(), timeout=1)
        assert not init_tasks[0].done()
        assert all(request.closed for request in requests)
    finally:
        finish.set()
        await asyncio.gather(*init_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_timeout_retry_initializes_with_its_own_open_requests(monkeypatch):
    import asyncio

    monkeypatch.setenv("HERMES_TELEGRAM_INIT_TIMEOUT", "0.05")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    first_requests = await _built_requests(monkeypatch, adapter)
    first_started, release = asyncio.Event(), asyncio.Event()
    initialized_pairs = []
    first_task = None

    class ReusingBuilder:
        # PTB 22.8 stores each setter's object and build() reads those same objects:
        # https://github.com/python-telegram-bot/python-telegram-bot/blob/v22.8/src/telegram/ext/_applicationbuilder.py#L486-L500
        # https://github.com/python-telegram-bot/python-telegram-bot/blob/v22.8/src/telegram/ext/_applicationbuilder.py#L693-L708
        # https://github.com/python-telegram-bot/python-telegram-bot/blob/v22.8/src/telegram/ext/_applicationbuilder.py#L226-L229
        # https://github.com/python-telegram-bot/python-telegram-bot/blob/v22.8/src/telegram/ext/_applicationbuilder.py#L267-L280
        def __init__(self, requests):
            self.general, self.updates = requests
            self.builds = 0

        def request(self, request):
            self.general = request
            return self

        def get_updates_request(self, request):
            self.updates = request
            return self

        def build(self):
            requests = (self.general, self.updates)
            is_first = self.builds == 0
            self.builds += 1

            async def bot_initialize():
                for request in requests:
                    await request.initialize()
                if any(request.closed for request in requests):
                    raise RuntimeError("bot initialized with closed requests")
                initialized_pairs.append(requests)

            async def app_initialize():
                nonlocal first_task
                await app.bot.initialize()
                if is_first:
                    first_task = asyncio.current_task()
                    first_started.set()
                    while not release.is_set():
                        try:
                            await release.wait()
                        except asyncio.CancelledError:
                            pass

            app = _app(app_initialize, requests)
            app.bot.initialize = bot_initialize
            return app

    builder = ReusingBuilder(first_requests)
    adapter._app = builder.build()
    adapter._bot = adapter._app.bot
    monkeypatch.setattr(adapter, "_wire_plugin_handlers", lambda app: None)
    monkeypatch.setattr(adapter, "_register_handlers", lambda app: None)

    try:
        ladder = asyncio.create_task(adapter._initialize_app_with_retries(builder))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        await asyncio.wait_for(ladder, timeout=5)

        retry_requests = adapter._app.bot._request
        assert builder.builds == 2
        assert all(new is not old for old, new in zip(first_requests, retry_requests))
        assert initialized_pairs == [first_requests, retry_requests]
        assert all(not request.closed for request in retry_requests)
        assert first_task is not None and not first_task.done()
        assert all(request.closed for request in first_requests)
    finally:
        release.set()
        if first_task is not None:
            await asyncio.wait_for(first_task, timeout=2)
