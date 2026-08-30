"""
Tests for Telegram polling network error recovery.

Specifically tests the fix for #3173 — when start_polling() fails after a
network error, the adapter must self-reschedule the next reconnect attempt
rather than silently leaving polling dead.
"""

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402
from gateway.run import GatewayRunner  # noqa: E402


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    """Disable DoH auto-discovery so connect() uses the plain builder chain."""
    async def _noop():
        return []
    monkeypatch.setattr("plugins.platforms.telegram.adapter.discover_fallback_ips", _noop)


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


async def _complete_current_polling_generation(adapter: TelegramAdapter) -> None:
    verifier = adapter._polling_progress_verifier_task
    adapter._record_polling_progress(adapter._polling_generation)
    if verifier is not None:
        await verifier


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
async def test_retry_exhaustion_queues_reconnect_before_child_disconnect(tmp_path):
    """Fatal teardown must not cancel the gateway's reconnect handoff.

    The gateway runs ``disconnect()`` in a bounded child task.  If the current
    polling-recovery owner remains in ``_polling_error_task``, Telegram teardown
    cancels that parent while it is still awaiting the fatal handler, so the
    handler never gets to queue background reconnection.
    """
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _make_adapter()
    adapter._polling_network_error_count = 10  # MAX_NETWORK_RETRIES
    adapter.set_fatal_error_handler(runner._handle_adapter_fatal_error)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.delivery_router.adapters = runner.adapters

    recovery_task = asyncio.create_task(
        adapter._handle_polling_network_error(Exception("still failing"))
    )
    adapter._polling_error_task = recovery_task
    result = await asyncio.gather(recovery_task, return_exceptions=True)

    assert result == [None]
    assert runner.adapters == {}
    assert Platform.TELEGRAM in runner._failed_platforms
    assert runner._failed_platforms[Platform.TELEGRAM]["attempts"] == 0


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
    general_req = mock_app.bot._request[1]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    # initialize MUST be called even though shutdown raised
    mock_polling_req.initialize.assert_called_once()
    # Generic polling errors must leave concurrent Bot API sends untouched.
    general_req.shutdown.assert_not_called()
    general_req.initialize.assert_not_called()
    mock_app.updater.start_polling.assert_called_once()


@pytest.mark.asyncio
async def test_reconnect_drains_general_pool_after_pool_timeout():
    """A confirmed bootstrap pool timeout must rebuild both request pools."""
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_app, mock_polling_req = _make_mock_app()
    general_req = AsyncMock()
    general_req.shutdown = AsyncMock()
    general_req.initialize = AsyncMock()
    mock_app.bot._request = (mock_polling_req, general_req)
    adapter._app = mock_app

    error = Exception(
        "Pool timeout: All connections in the connection pool are occupied. "
        "Request was not sent to Telegram."
    )
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(error)

    general_req.shutdown.assert_awaited_once()
    general_req.initialize.assert_awaited_once()
    mock_polling_req.shutdown.assert_awaited_once()
    mock_polling_req.initialize.assert_awaited_once()
    mock_app.updater.start_polling.assert_awaited_once()
    await _complete_current_polling_generation(adapter)


@pytest.mark.asyncio
async def test_general_pool_drain_is_bounded_when_close_hangs(monkeypatch):
    """A wedged general-pool close must not freeze the reconnect ladder."""
    adapter = _make_adapter()
    mock_app, mock_polling_req = _make_mock_app()

    async def _hang(*args, **kwargs):
        await asyncio.Event().wait()

    general_req = AsyncMock()
    general_req.shutdown = AsyncMock(side_effect=_hang)
    general_req.initialize = AsyncMock(side_effect=_hang)
    mock_app.bot._request = (mock_polling_req, general_req)
    adapter._app = mock_app
    monkeypatch.setattr(tg_adapter, "_DRAIN_TIMEOUT", 0.05, raising=False)

    await asyncio.wait_for(
        adapter._drain_general_connections_after_pool_timeout(), timeout=1
    )

    general_req.shutdown.assert_awaited_once()
    general_req.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconnect_continues_if_drain_hangs(monkeypatch):
    """If the polling request drain HANGS (wedged httpx pool close on a
    CLOSE-WAIT socket), the reconnect ladder must still advance rather than
    freezing the tracked _polling_error_task forever.

    Regression test for #66377: an unbounded ``shutdown()`` /
    ``initialize()`` in ``_drain_polling_connections`` leaves the handler
    task pending, which gates every escalation path and silently kills the
    gateway. The drain awaits are bounded by ``_DRAIN_TIMEOUT``, so the
    handler must complete and reach ``start_polling`` within a hard bound.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_app, mock_polling_req = _make_mock_app()

    async def _hang(*args, **kwargs):
        await asyncio.Event().wait()  # never returns

    # Both drain awaits wedge indefinitely.
    mock_polling_req.shutdown = AsyncMock(side_effect=_hang)
    mock_polling_req.initialize = AsyncMock(side_effect=_hang)
    adapter._app = mock_app

    # Keep the drain timeout tiny so the test stays fast; the real default
    # is generous enough not to truncate healthy closes.
    monkeypatch.setattr(tg_adapter, "_DRAIN_TIMEOUT", 0.01, raising=False)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        # Hard outer bound: on unfixed code the drain hangs forever and this
        # trips; with the fix the inner wait_for releases well before it.
        await asyncio.wait_for(
            adapter._handle_polling_network_error(Exception("Timed out")),
            timeout=5,
        )

    # Ladder advanced past the wedged drain despite it never returning.
    mock_app.updater.start_polling.assert_called_once()
    assert adapter._polling_network_error_count == 2
    # The tracked task must not be stuck pending — otherwise every
    # escalation path stays gated behind an in-flight guard.
    assert (
        adapter._polling_error_task is None
        or adapter._polling_error_task.done()
    )


@pytest.mark.asyncio
async def test_reconnect_stop_deadline_does_not_wait_for_cancel_cleanup(monkeypatch):
    """A cancellation-resistant PTB stop must not freeze the retry ladder.

    ``asyncio.wait_for`` waits for the cancelled coroutine to finish.  AnyIO's
    cancellation-shielded httpcore cleanup can therefore leave ``stop()``
    pending forever after the timeout fires: the gateway process stays alive,
    but no later Telegram retry runs.  The wall-clock deadline must abandon
    that task and escalate to a fresh adapter without reusing the Updater.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    release_stop = asyncio.Event()
    stop_cancelled = asyncio.Event()
    lifecycle_lock = asyncio.Lock()

    async def _cancellation_resistant_stop():
        async with lifecycle_lock:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stop_cancelled.set()
                await release_stop.wait()

    async def _start_polling_with_same_lock(*args, **kwargs):
        async with lifecycle_lock:
            return None

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock(side_effect=_cancellation_resistant_stop)
    mock_updater.start_polling = AsyncMock(side_effect=_start_polling_with_same_lock)

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot = MagicMock()
    mock_app.bot._request = ()
    adapter._app = mock_app
    adapter._notify_fatal_error = AsyncMock()

    monkeypatch.setattr(tg_adapter, "_UPDATER_STOP_TIMEOUT", 0.01)
    with patch("asyncio.sleep", new_callable=AsyncMock):
        recovery = asyncio.create_task(
            adapter._handle_polling_network_error(Exception("Timed out"))
        )
        done, _ = await asyncio.wait({recovery}, timeout=0.2)

    try:
        assert recovery in done, (
            "reconnect remained blocked waiting for cancellation-shielded "
            "updater.stop() cleanup"
        )
        assert stop_cancelled.is_set()
        assert adapter.has_fatal_error
        adapter._notify_fatal_error.assert_awaited_once()
        mock_updater.start_polling.assert_not_awaited()
    finally:
        release_stop.set()
        if not recovery.done():
            recovery.cancel()
        await asyncio.gather(recovery, return_exceptions=True)


@pytest.mark.asyncio
async def test_heartbeat_force_escalates_wedged_recovery_task(monkeypatch):
    """#66377: the heartbeat is an independent, cause-agnostic watchdog.

    Every recovery path (ladder re-entry, pending-update probe, PTB error
    callback) gates new recovery on ``_polling_error_task.done()``. If that task
    wedges on ANY hung await — not just the drain closed by #66492 — the gateway
    stays alive but deaf with nothing retrying. The heartbeat must detect a
    recovery task that stays in-flight past ``_POLLING_ERROR_TASK_STUCK_TIMEOUT``
    and force a retryable-fatal so the background reconnector rebuilds the
    adapter.
    """
    adapter = _make_adapter()

    async def _wedged():
        await asyncio.Event().wait()  # never completes — simulates the hang

    wedged_task = asyncio.ensure_future(_wedged())
    adapter._polling_error_task = wedged_task

    mock_bot = MagicMock()
    mock_bot.get_me = AsyncMock()
    mock_app = MagicMock()
    mock_app.bot = mock_bot
    adapter._app = mock_app
    adapter._probe_pending_updates = AsyncMock()
    adapter._notify_fatal_error = AsyncMock()

    # Controllable monotonic clock advanced by each (mocked) heartbeat sleep so
    # the same wedged task is observed across the stuck threshold deterministically.
    clock = [1000.0]

    async def _fake_sleep(*_a, **_k):
        clock[0] += 200.0

    monkeypatch.setattr(tg_adapter.time, "monotonic", lambda: clock[0])

    with patch("asyncio.sleep", new=AsyncMock(side_effect=_fake_sleep)):
        await asyncio.wait_for(adapter._polling_heartbeat_loop(), timeout=5)

    assert adapter.has_fatal_error, "wedged recovery task must force a fatal escalation"
    adapter._notify_fatal_error.assert_awaited()

    wedged_task.cancel()
    try:
        await wedged_task
    except asyncio.CancelledError:
        pass


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
async def test_heartbeat_probe_reenters_ladder_when_updater_not_running(monkeypatch):
    """
    If Updater.running is False at the progress deadline, re-enter recovery.
    """
    adapter = _make_adapter()

    mock_updater = MagicMock()
    mock_updater.running = False

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock()
    adapter._app = mock_app

    adapter._handle_polling_network_error = AsyncMock()
    generation, progress = adapter._begin_polling_generation()
    monkeypatch.setattr(tg_adapter, "_POLLING_PROGRESS_TIMEOUT", 0)

    await adapter._verify_polling_after_reconnect(generation, progress)

    mock_app.bot.get_me.assert_not_called()
    # Recovery is scheduled through _schedule_polling_recovery (#63243), so
    # the ladder runs as the tracked _polling_error_task.
    task = adapter._polling_error_task
    assert task is not None
    await task
    adapter._handle_polling_network_error.assert_awaited_once()
    err = adapter._handle_polling_network_error.await_args.args[0]
    assert isinstance(err, RuntimeError)
    assert "not running" in str(err).lower()


@pytest.mark.asyncio
async def test_heartbeat_probe_ignores_auth_errors(monkeypatch):
    """
    Auth/validation failures from the post-reconnect probe must not enter the
    network-reconnect ladder (#63243): a revoked token would otherwise churn
    through stop/drain/start_polling cycles that mask the real failure.
    """
    adapter = _make_adapter()

    mock_updater = MagicMock()
    mock_updater.running = True

    # Name-shaped like PTB's InvalidToken; _looks_like_network_error excludes
    # it by class name, matching real PTB semantics.
    invalid_token = type("InvalidToken", (Exception,), {})("token revoked")

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock(side_effect=invalid_token)
    adapter._app = mock_app

    adapter._handle_polling_network_error = AsyncMock()
    generation, progress = adapter._begin_polling_generation()
    monkeypatch.setattr(tg_adapter, "_POLLING_PROGRESS_TIMEOUT", 0)

    await adapter._verify_polling_after_reconnect(generation, progress)

    assert adapter._polling_error_task is None
    adapter._handle_polling_network_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_heartbeat_probe_defers_to_inflight_recovery(monkeypatch):
    """
    A probe failure while another recovery is mid-flight must not start a
    second concurrent stop/drain/start_polling sequence (#63243) — overlapping
    recoveries produce dueling getUpdates sessions (self-inflicted 409s).
    """
    adapter = _make_adapter()

    mock_updater = MagicMock()
    mock_updater.running = True

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock(side_effect=ConnectionError("pool wedged"))
    adapter._app = mock_app

    inflight = MagicMock()
    inflight.done.return_value = False
    adapter._polling_error_task = inflight

    adapter._handle_polling_network_error = AsyncMock()
    generation, progress = adapter._begin_polling_generation()
    monkeypatch.setattr(tg_adapter, "_POLLING_PROGRESS_TIMEOUT", 0)

    await adapter._verify_polling_after_reconnect(generation, progress)

    assert adapter._polling_error_task is inflight
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
async def test_heartbeat_loop_skips_reconnect_if_already_in_progress():
    """If a reconnect task is already running, the heartbeat must not spawn another."""
    adapter = _make_adapter()

    # Simulate an already-running reconnect task.
    existing_task = asyncio.get_event_loop().create_task(asyncio.sleep(0.2))
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


async def _heartbeat_exception_case(exc, *, pending_probe=False):
    adapter = _make_adapter()
    reconnect_handler = AsyncMock()
    adapter._handle_polling_network_error = reconnect_handler  # type: ignore[method-assign]
    mock_app = MagicMock()
    mock_app.updater.running = True
    if pending_probe:
        mock_app.bot.get_me = AsyncMock(return_value=MagicMock())
        mock_app.bot.get_webhook_info = AsyncMock(side_effect=exc)
    else:
        mock_app.bot.get_me = AsyncMock(side_effect=exc)
    adapter._app = mock_app

    sleep_calls = 0

    async def fast_sleep(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError()

    with patch("asyncio.sleep", side_effect=fast_sleep):
        await adapter._polling_heartbeat_loop()
    await asyncio.sleep(0)
    return adapter


def _calls_shared_network_classifier(node):
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "_looks_like_network_error"
        for child in ast.walk(node)
    )


# ── Bootstrap degradation: keep polling alive during outages (#47508) ────


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
async def test_handle_polling_network_error_updater_stop_timeout():
    """updater.stop() hanging (CLOSE-WAIT) must not block the reconnect ladder.

    When the underlying TCP connection is in CLOSE-WAIT, PTB's polling task is
    blocked on epoll on the dead socket.  updater.stop() awaits that task and
    therefore hangs indefinitely.  The wall-clock deadline abandons the stop
    task and escalates to fresh-adapter recovery instead of calling
    start_polling() while PTB's shared lifecycle lock may still be held.

    This test simulates the hang by making stop() outlive the deadline and
    verifies that the current Updater is not drained or restarted afterward.
    Refs: NousResearch/hermes-agent#58270
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 0

    # Build a fake app whose updater.stop() hangs forever.
    app = MagicMock()
    app.updater = MagicMock()
    app.updater.running = True

    async def _hanging_stop():
        await asyncio.sleep(0.2)  # simulate CLOSE-WAIT block

    app.updater.stop = _hanging_stop
    app.updater.start_polling = AsyncMock()
    adapter._app = app
    adapter._notify_fatal_error = AsyncMock()

    drain_called = []

    async def _fake_drain():
        drain_called.append(True)

    adapter._drain_polling_connections = _fake_drain

    start_polling_called = []

    async def _fake_start_polling(**kwargs):
        start_polling_called.append(True)

    app.updater.start_polling = AsyncMock(side_effect=_fake_start_polling)

    # Shrink the stop() watchdog bound so the test completes fast instead of
    # waiting the full _UPDATER_STOP_TIMEOUT. Patching the named constant is
    # cleaner than monkeypatching asyncio.wait_for process-wide.
    import plugins.platforms.telegram.adapter as _mod

    with patch.object(_mod, "_UPDATER_STOP_TIMEOUT", 0.05):
        await adapter._handle_polling_network_error(OSError("CLOSE-WAIT test"))

    # A timed-out stop may still hold PTB's lifecycle lock. Reusing this
    # Updater would wedge start_polling() behind it, so recovery must hand the
    # runner a retryable fatal and rebuild the adapter instead.
    assert adapter.has_fatal_error
    adapter._notify_fatal_error.assert_awaited_once()
    assert not drain_called
    assert not start_polling_called


@pytest.mark.asyncio
async def test_disconnect_releases_token_lock_before_wedged_app_shutdown(monkeypatch):
    """#80598: token lock must drop even when app.shutdown() never returns.

    The reconnect watcher creates a fresh adapter that re-acquires the bot-token
    lock. If disconnect only releases the lock after a wedged PTB shutdown, the
    watcher fails forever with a lock conflict while the process stays alive.
    """
    adapter = _make_adapter()
    released = []
    monkeypatch.setattr(
        adapter, "_release_platform_lock", lambda: released.append(True)
    )
    monkeypatch.setattr(adapter, "_set_status_indicator", AsyncMock())
    monkeypatch.setattr(adapter, "_cancel_pending_delivery_tasks", AsyncMock())

    app = MagicMock()
    app.updater = MagicMock()
    app.updater.running = False
    app.running = True
    app.stop = AsyncMock()

    async def _hanging_shutdown():
        await asyncio.Event().wait()

    app.shutdown = _hanging_shutdown
    adapter._app = app
    adapter._bot = MagicMock()

    monkeypatch.setattr(tg_adapter, "_DISCONNECT_STEP_TIMEOUT", 0.01)

    await asyncio.wait_for(adapter.disconnect(), timeout=1.0)

    assert released, "token lock must be released before wedged shutdown"
    assert adapter._app is None


@pytest.mark.asyncio
async def test_disconnect_advances_past_cancellation_swallowing_lifecycle(monkeypatch):
    """#80598: lifecycle tasks that swallow CancelledError must not wedge disconnect."""
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_release_platform_lock", MagicMock())
    monkeypatch.setattr(adapter, "_set_status_indicator", AsyncMock())
    monkeypatch.setattr(adapter, "_cancel_pending_delivery_tasks", AsyncMock())

    release = asyncio.Event()

    async def swallow_cancel():
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue

    wedged = asyncio.create_task(swallow_cancel())
    adapter._polling_error_task = wedged
    adapter._app = None
    adapter._bot = None

    monkeypatch.setattr(tg_adapter, "_DISCONNECT_STEP_TIMEOUT", 0.01)

    await asyncio.wait_for(adapter.disconnect(), timeout=1.0)
    assert adapter._polling_error_task is None

    release.set()
    await asyncio.wait({wedged}, timeout=0.2)
