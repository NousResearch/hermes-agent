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

    def __init__(self, events):
        self.closed, self._events = False, events

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


@pytest.mark.asyncio
async def test_cancelled_connect_closes_the_abandoned_init_transports_after_it_unwinds():
    """The runner cancelling connect() mid-initialize abandons the child with no on_abandon; its httpx
    transports must still close, and only after the child stops using them."""
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
    _asyncio.get_running_loop().call_later(0.2, release.set)
    await _asyncio.wait_for(adapter.disconnect(), timeout=10)

    assert all(request.closed for request in requests)
    assert events.index("init exited") < events.index("request closed")
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
        for request in requests:  # Bot.initialize() reopens the shared requests every attempt
            await request.initialize()
        raise OSError("api.telegram.org unreachable")

    builder = MagicMock()
    builder.build.side_effect = lambda: _app(_unreachable, requests)
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._app = builder.build()
    adapter._bot = adapter._app.bot

    with pytest.raises(OSError):
        await adapter._initialize_app_with_retries(builder)
    assert not any(request.closed for request in requests)  # the last attempt's pools are still open

    await _asyncio.wait_for(adapter.disconnect(), timeout=10)

    assert all(request.closed for request in requests)
