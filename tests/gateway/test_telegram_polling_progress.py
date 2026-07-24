"""Behavior contract for generation-safe Telegram polling progress."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


class _ControlledRequest:
    """Minimal PTB request double with controllable completion."""

    instances = []

    @staticmethod
    def parse_json_payload(payload):
        """Match PTB's response authority used by the progress observer."""
        return json.loads(payload.decode("utf-8", "replace"))

    def __init__(self, *args, result=None, error=None, entered=None, release=None, **kwargs):
        self.result = result
        self.error = error
        self.entered = entered
        self.release = release
        self.args = args
        self.kwargs = kwargs
        type(self).instances.append(self)

    async def do_request(self, *args, **kwargs):
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.result


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


def _mock_polling_app(*, get_me=None):
    app = MagicMock()
    app.updater = MagicMock()
    app.updater.running = True
    app.updater.stop = AsyncMock()
    app.updater.start_polling = AsyncMock()
    app.bot = MagicMock()
    app.bot.get_me = get_me or AsyncMock(return_value=MagicMock())
    app.running = False
    app.shutdown = AsyncMock()
    return app


class _LifecycleBuilder:
    def __init__(self, app):
        self.app = app
        self.polling_request = None

    def token(self, _token):
        return self

    def request(self, _request):
        return self

    def get_updates_request(self, request):
        self.polling_request = request
        return self

    def build(self):
        return self.app


def _lifecycle_app():
    app = MagicMock()
    app.updater = MagicMock()
    app.updater.running = True
    app.updater.start_polling = AsyncMock()
    app.updater.start_webhook = AsyncMock()
    app.updater.stop = AsyncMock()
    app.bot = MagicMock()
    app.bot.delete_webhook = AsyncMock()
    app.initialize = AsyncMock()
    app.start = AsyncMock()
    app.stop = AsyncMock()
    app.shutdown = AsyncMock()
    app.running = True
    return app


def _configure_lifecycle_connect(monkeypatch, adapter, apps):
    builders = [_LifecycleBuilder(app) for app in apps]
    remaining = iter(builders)

    class _Application:
        @staticmethod
        def builder():
            return next(remaining)

    async def _no_fallback_ips():
        return []

    monkeypatch.setattr(tg_adapter, "Application", _Application)
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _ControlledRequest)
    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _no_fallback_ips)
    monkeypatch.setattr(tg_adapter, "resolve_proxy_url", lambda *args, **kwargs: None)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args, **kwargs: True)
    monkeypatch.setattr(adapter, "_release_platform_lock", MagicMock())
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: [])
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", MagicMock())
    return builders


async def _cancel_task(task):
    if task is None or task.done():
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _request_for_generation(generation, request, *args):
    """Run a direct request double under the production polling context."""
    generation_context = tg_adapter._POLLING_GENERATION_CONTEXT
    token = generation_context.set(generation)
    try:
        return await request.do_request(*args)
    finally:
        generation_context.reset(token)


@pytest.mark.asyncio
async def test_polling_disconnect_webhook_reconnect_heals_webhook_send_path(monkeypatch):
    adapter = _make_adapter()
    polling_app = _lifecycle_app()
    webhook_app = _lifecycle_app()
    _configure_lifecycle_connect(monkeypatch, adapter, [polling_app, webhook_app])
    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)

    assert await adapter.connect() is True
    assert adapter._webhook_mode is False
    assert adapter._send_path_degraded is True
    await adapter.disconnect()

    monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "https://example.test/telegram")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-secret")
    try:
        assert await adapter.connect(is_reconnect=True) is True
        webhook_app.updater.start_webhook.assert_awaited_once()
        assert adapter._webhook_mode is True
        assert adapter._polling_progress_accepting is False
        assert adapter._send_path_degraded is False
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_webhook_disconnect_polling_reconnect_resets_mode_and_waits_for_progress(
    monkeypatch,
):
    adapter = _make_adapter()
    webhook_app = _lifecycle_app()
    polling_app = _lifecycle_app()
    builders = _configure_lifecycle_connect(
        monkeypatch, adapter, [webhook_app, polling_app]
    )
    heartbeat_started = asyncio.Event()
    heartbeat_modes = []

    async def heartbeat():
        heartbeat_modes.append(adapter._webhook_mode)
        heartbeat_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(adapter, "_polling_heartbeat_loop", heartbeat)
    monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "https://example.test/telegram")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-secret")

    assert await adapter.connect() is True
    assert adapter._webhook_mode is True
    assert adapter._polling_heartbeat_task is None
    await adapter.disconnect()

    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL")
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET")
    try:
        assert await adapter.connect(is_reconnect=True) is True
        assert adapter._webhook_mode is False
        assert adapter._polling_heartbeat_task is not None
        assert not adapter._polling_heartbeat_task.done()
        await asyncio.wait_for(heartbeat_started.wait(), timeout=1)
        assert heartbeat_modes == [False]
        assert adapter._send_path_degraded is True

        generation = adapter._polling_generation
        polling_request = builders[1].polling_request
        polling_request.result = (200, b'{"ok":true,"result":[]}')
        await _request_for_generation(generation, polling_request, "getUpdates")
        await asyncio.wait_for(adapter._polling_progress_verifier_task, timeout=1)
        assert adapter._send_path_degraded is False
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_current_polling_generation_success_records_progress():
    adapter = _make_adapter()
    generation, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 3
    request = _ControlledRequest(result=(200, b'{"ok":true,"result":[]}'))

    instrumented = adapter._instrument_polling_request(request)
    result = await _request_for_generation(
        generation, instrumented, "https://api.telegram.org/getUpdates"
    )

    assert instrumented is request
    assert result == (200, b'{"ok":true,"result":[]}')
    assert progress.is_set()
    assert adapter._polling_network_error_count == 0
    assert adapter._send_path_degraded is False
    assert generation > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_unsuccessful_polling_request_does_not_record_progress(error_type):
    adapter = _make_adapter()
    generation, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 3
    request = adapter._instrument_polling_request(
        _ControlledRequest(error=error_type("request did not complete"))
    )

    with pytest.raises(error_type):
        await _request_for_generation(
            generation, request, "https://api.telegram.org/getUpdates"
        )

    assert not progress.is_set()
    assert adapter._polling_network_error_count == 3
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
async def test_http_error_response_does_not_record_polling_progress():
    adapter = _make_adapter()
    generation, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 3
    request = adapter._instrument_polling_request(
        _ControlledRequest(result=(500, b"bad"))
    )

    result = await _request_for_generation(
        generation, request, "https://api.telegram.org/getUpdates"
    )

    assert result == (500, b"bad")
    assert not progress.is_set()
    assert adapter._polling_network_error_count == 3
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
async def test_general_request_success_cannot_record_polling_progress(monkeypatch):
    class _StopConnect(Exception):
        pass

    class _Builder:
        def __init__(self):
            self.general_request = None
            self.polling_request = None

        def token(self, _token):
            return self

        def request(self, request):
            self.general_request = request
            return self

        def get_updates_request(self, request):
            self.polling_request = request
            return self

        def build(self):
            raise _StopConnect

    builder = _Builder()

    class _Application:
        @staticmethod
        def builder():
            return builder

    _ControlledRequest.instances = []

    async def _no_fallback_ips():
        return []

    monkeypatch.setattr(tg_adapter, "Application", _Application)
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _ControlledRequest)
    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _no_fallback_ips)
    monkeypatch.setattr(tg_adapter, "resolve_proxy_url", lambda *args, **kwargs: None)

    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args, **kwargs: True)
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: [])
    _, progress = adapter._begin_polling_generation()

    assert await adapter.connect() is False
    assert builder.general_request is _ControlledRequest.instances[0]
    assert builder.polling_request is _ControlledRequest.instances[1]

    builder.general_request.result = (200, b'{"ok":true}')
    result = await builder.general_request.do_request("https://api.telegram.org/sendMessage")

    assert result == (200, b'{"ok":true}')
    assert not progress.is_set()
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
async def test_late_previous_generation_completion_cannot_heal_current_generation():
    adapter = _make_adapter()
    generation_1, _ = adapter._begin_polling_generation()
    entered = asyncio.Event()
    release = asyncio.Event()
    request = adapter._instrument_polling_request(
        _ControlledRequest(
            result=(200, b'{"ok":true,"result":[]}'),
            entered=entered,
            release=release,
        )
    )

    completion = asyncio.create_task(
        _request_for_generation(generation_1, request, "getUpdates")
    )
    await entered.wait()
    generation_2, progress_2 = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 4
    release.set()

    assert await completion == (200, b'{"ok":true,"result":[]}')
    assert generation_2 == generation_1 + 1
    assert not progress_2.is_set()
    assert adapter._polling_network_error_count == 4
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
async def test_old_polling_child_keeps_generation_when_request_entry_is_delayed():
    adapter = _make_adapter()
    adapter._app = _mock_polling_app()
    release_old_request = asyncio.Event()
    start_count = 0
    old_child = None
    request = adapter._instrument_polling_request(
        _ControlledRequest(result=(200, b'{"ok":true,"result":[]}'))
    )

    async def start_polling(**_kwargs):
        nonlocal start_count, old_child
        start_count += 1
        if start_count == 1:

            async def delayed_old_request():
                await release_old_request.wait()
                return await request.do_request("getUpdates")

            old_child = asyncio.create_task(delayed_old_request())

    adapter._app.updater.start_polling = start_polling

    await adapter._start_polling_once(
        adapter._app,
        drop_pending_updates=False,
        error_callback=MagicMock(),
    )
    generation_1 = adapter._polling_generation
    await adapter._start_polling_once(
        adapter._app,
        drop_pending_updates=False,
        error_callback=MagicMock(),
    )
    generation_2 = adapter._polling_generation
    progress_2 = adapter._polling_progress_event
    verifier_2 = adapter._polling_progress_verifier_task

    try:
        release_old_request.set()
        assert await old_child == (200, b'{"ok":true,"result":[]}')

        assert generation_2 == generation_1 + 1
        assert not progress_2.is_set()
        assert adapter._send_path_degraded is True
    finally:
        await _cancel_task(verifier_2)


@pytest.mark.asyncio
async def test_error_callback_is_bound_to_its_polling_generation():
    adapter = _make_adapter()
    adapter._app = _mock_polling_app()
    callbacks = []
    delegated = MagicMock()

    async def capture_start(**kwargs):
        callbacks.append(kwargs["error_callback"])

    adapter._app.updater.start_polling = capture_start
    await adapter._start_polling_once(
        adapter._app,
        drop_pending_updates=False,
        error_callback=delegated,
    )
    await adapter._start_polling_once(
        adapter._app,
        drop_pending_updates=False,
        error_callback=delegated,
    )
    verifier = adapter._polling_progress_verifier_task
    stale_error = ConnectionError("stale generation")
    current_error = ConnectionError("current generation")

    try:
        callbacks[0](stale_error)
        delegated.assert_not_called()

        callbacks[1](current_error)
        delegated.assert_called_once_with(current_error)
    finally:
        await _cancel_task(verifier)


@pytest.mark.asyncio
async def test_cold_start_waits_for_get_updates_progress_before_healing():
    adapter = _make_adapter()
    adapter._app = _mock_polling_app()

    started = await adapter._start_polling_resilient(
        drop_pending_updates=True,
        error_callback=MagicMock(),
    )

    verifier = adapter._polling_progress_verifier_task
    assert started is True
    assert adapter._app.updater.running is True
    assert adapter._send_path_degraded is True
    assert verifier is not None and not verifier.done()
    assert verifier in adapter._background_tasks
    assert [task for task in adapter._background_tasks if not task.done()] == [verifier]
    await _cancel_task(verifier)


@pytest.mark.asyncio
async def test_matching_get_updates_progress_heals_and_stops_verifier(monkeypatch):
    adapter = _make_adapter()
    adapter._app = _mock_polling_app()
    recovery = MagicMock()
    monkeypatch.setattr(adapter, "_schedule_polling_recovery", recovery)

    await adapter._start_polling_resilient(
        drop_pending_updates=False,
        error_callback=MagicMock(),
    )
    verifier = adapter._polling_progress_verifier_task
    request = adapter._instrument_polling_request(
        _ControlledRequest(result=(200, b'{"ok":true,"result":[]}'))
    )
    await _request_for_generation(
        adapter._polling_generation, request, "getUpdates"
    )
    await asyncio.wait_for(verifier, timeout=1)

    assert adapter._send_path_degraded is False
    assert verifier.done()
    recovery.assert_not_called()


@pytest.mark.asyncio
async def test_general_path_success_without_get_updates_progress_recovers_once(monkeypatch):
    adapter = _make_adapter()
    adapter._app = _mock_polling_app()
    recovery = MagicMock()
    monkeypatch.setattr(tg_adapter, "_POLLING_PROGRESS_TIMEOUT", 0.01, raising=False)
    monkeypatch.setattr(adapter, "_schedule_polling_recovery", recovery)

    await adapter._start_polling_resilient(
        drop_pending_updates=False,
        error_callback=MagicMock(),
    )
    verifier = adapter._polling_progress_verifier_task
    await asyncio.wait_for(verifier, timeout=1)

    assert adapter._app.bot.get_me.await_count == 1
    recovery.assert_called_once()
    error = recovery.call_args.args[0]
    assert isinstance(error, RuntimeError)
    assert str(error) == "getUpdates made no progress before verifier deadline"
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("probe_error", "should_recover"),
    [
        (ConnectionError("pool wedged"), True),
        (type("InvalidToken", (Exception,), {})("token revoked"), False),
    ],
)
async def test_general_path_error_only_recovers_connectivity_failures(
    monkeypatch, probe_error, should_recover
):
    adapter = _make_adapter()
    adapter._app = _mock_polling_app(get_me=AsyncMock(side_effect=probe_error))
    recovery = MagicMock()
    monkeypatch.setattr(tg_adapter, "_POLLING_PROGRESS_TIMEOUT", 0.01, raising=False)
    monkeypatch.setattr(adapter, "_schedule_polling_recovery", recovery)

    await adapter._start_polling_resilient(
        drop_pending_updates=False,
        error_callback=MagicMock(),
    )
    await asyncio.wait_for(adapter._polling_progress_verifier_task, timeout=1)

    assert recovery.called is should_recover
    if should_recover:
        assert recovery.call_args.args[0] is probe_error
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_kind", ["network", "conflict"])
async def test_retry_start_requires_matching_progress_to_heal(monkeypatch, retry_kind):
    adapter = _make_adapter()
    adapter._app = _mock_polling_app()
    adapter._polling_error_callback_ref = MagicMock()
    adapter._polling_network_error_count = 3
    monkeypatch.setattr(tg_adapter.asyncio, "sleep", AsyncMock())

    if retry_kind == "network":
        await adapter._handle_polling_network_error(ConnectionError("offline"))
        assert adapter._polling_network_error_count == 4
    else:
        await adapter._handle_polling_conflict(
            RuntimeError("Conflict: terminated by other getUpdates request")
        )
        assert adapter._polling_network_error_count == 3
        assert adapter._polling_conflict_count == 1

    generation = adapter._polling_generation
    verifier = adapter._polling_progress_verifier_task
    assert generation > 0
    assert verifier is not None and not verifier.done()
    assert adapter._send_path_degraded is True

    request = adapter._instrument_polling_request(
        _ControlledRequest(result=(200, b'{"ok":true,"result":[]}'))
    )
    await _request_for_generation(generation, request, "getUpdates")
    await asyncio.wait_for(verifier, timeout=1)
    assert adapter._polling_network_error_count == 0
    assert adapter._polling_conflict_count == 0
    assert adapter._send_path_degraded is False


@pytest.mark.asyncio
async def test_repeated_starts_replace_verifier_and_stale_verifier_cannot_heal():
    adapter = _make_adapter()
    adapter._app = _mock_polling_app()

    await adapter._start_polling_resilient(
        drop_pending_updates=False, error_callback=MagicMock()
    )
    generation_1 = adapter._polling_generation
    progress_1 = adapter._polling_progress_event
    verifier_1 = adapter._polling_progress_verifier_task

    await adapter._start_polling_resilient(
        drop_pending_updates=False, error_callback=MagicMock()
    )
    verifier_2 = adapter._polling_progress_verifier_task
    await asyncio.sleep(0)

    assert adapter._polling_generation == generation_1 + 1
    assert verifier_1.cancelled()
    assert verifier_2 is not verifier_1 and not verifier_2.done()
    assert [task for task in adapter._background_tasks if not task.done()] == [verifier_2]

    progress_1.set()
    await asyncio.sleep(0)
    assert adapter._send_path_degraded is True
    await _cancel_task(verifier_2)


@pytest.mark.asyncio
async def test_disconnect_fences_verifier_and_late_progress_completion():
    adapter = _make_adapter()
    adapter._app = _mock_polling_app()
    await adapter._start_polling_resilient(
        drop_pending_updates=False, error_callback=MagicMock()
    )
    generation = adapter._polling_generation
    verifier = adapter._polling_progress_verifier_task

    entered = asyncio.Event()
    release = asyncio.Event()
    request = adapter._instrument_polling_request(
        _ControlledRequest(
            result=(200, b'{"ok":true,"result":[]}'),
            entered=entered,
            release=release,
        )
    )
    completion = asyncio.create_task(
        _request_for_generation(generation, request, "getUpdates")
    )
    await entered.wait()

    await adapter.disconnect()

    assert verifier.done()
    assert adapter._polling_progress_verifier_task is None
    assert adapter._polling_progress_accepting is False
    assert adapter._polling_generation > generation
    assert adapter._send_path_degraded is True

    release.set()
    assert await completion == (200, b'{"ok":true,"result":[]}')
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
async def test_disconnect_during_polling_start_returns_false_without_recovery():
    adapter = _make_adapter()
    adapter._app = _mock_polling_app()
    entered = asyncio.Event()
    release = asyncio.Event()
    recovery = MagicMock()
    adapter._schedule_polling_recovery = recovery

    async def blocked_start(**_kwargs):
        entered.set()
        await release.wait()

    adapter._app.updater.start_polling = blocked_start
    start = asyncio.create_task(
        adapter._start_polling_resilient(
            drop_pending_updates=False,
            error_callback=MagicMock(),
        )
    )
    await entered.wait()

    await adapter.disconnect()
    release.set()
    result = await start

    assert result is False
    recovery.assert_not_called()
    assert adapter._polling_error_task is None
    assert not adapter.has_fatal_error
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
async def test_caller_cancellation_during_polling_start_still_propagates():
    adapter = _make_adapter()
    adapter._app = _mock_polling_app()
    entered = asyncio.Event()
    release = asyncio.Event()
    recovery = MagicMock()
    adapter._schedule_polling_recovery = recovery

    async def blocked_start(**_kwargs):
        entered.set()
        await release.wait()

    adapter._app.updater.start_polling = blocked_start
    start = asyncio.create_task(
        adapter._start_polling_resilient(
            drop_pending_updates=False,
            error_callback=MagicMock(),
        )
    )
    await entered.wait()

    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start

    assert start.cancelled()
    recovery.assert_not_called()
    assert not adapter.has_fatal_error
    assert adapter._send_path_degraded is True
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_disconnect_cancels_recovery_before_it_can_rearm_progress(monkeypatch):
    adapter = _make_adapter()
    adapter._app = _mock_polling_app()
    adapter._app.updater.running = False
    adapter._polling_error_callback_ref = MagicMock()

    drain_entered = asyncio.Event()
    release_drain = asyncio.Event()
    start_entered = asyncio.Event()
    release_start = asyncio.Event()
    teardown_paused = asyncio.Event()
    release_teardown = asyncio.Event()

    async def immediate_backoff(_delay):
        return None

    async def blocked_drain():
        drain_entered.set()
        await release_drain.wait()

    async def blocked_start_polling(**_kwargs):
        start_entered.set()
        await release_start.wait()

    async def blocked_status_indicator(*, online):
        assert online is False
        teardown_paused.set()
        await release_teardown.wait()

    monkeypatch.setattr(tg_adapter.asyncio, "sleep", immediate_backoff)
    monkeypatch.setattr(adapter, "_drain_polling_connections", blocked_drain)
    monkeypatch.setattr(
        adapter._app.updater, "start_polling", blocked_start_polling
    )
    monkeypatch.setattr(adapter, "_set_status_indicator", blocked_status_indicator)

    recovery = asyncio.create_task(
        adapter._handle_polling_network_error(ConnectionError("offline"))
    )
    adapter._polling_error_task = recovery
    await drain_entered.wait()

    disconnect = asyncio.create_task(adapter.disconnect())
    await teardown_paused.wait()

    try:
        # Before the fix, disconnect pauses here before cancelling recovery.
        # Releasing the recovery lets it begin a fresh generation after the
        # teardown fence, and matching progress can then heal the adapter.
        if not recovery.done():
            release_drain.set()
            await start_entered.wait()

        rearmed_after_fence = adapter._polling_progress_accepting
        adapter._record_polling_progress(adapter._polling_generation)

        assert rearmed_after_fence is False
        assert getattr(adapter, "_polling_teardown_started", False) is True
        assert adapter._polling_progress_accepting is False
        assert adapter._send_path_degraded is True
        assert recovery.done()
    finally:
        release_drain.set()
        release_start.set()
        release_teardown.set()
        for task in (recovery, disconnect):
            if not task.done():
                task.cancel()
        await asyncio.gather(recovery, disconnect, return_exceptions=True)
