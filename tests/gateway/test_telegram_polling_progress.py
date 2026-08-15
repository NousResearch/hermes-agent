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

    async def start_polling_with_progress(**_kwargs):
        adapter._record_polling_progress(adapter._polling_generation)

    polling_app.updater.start_polling = AsyncMock(
        side_effect=start_polling_with_progress
    )
    _configure_lifecycle_connect(monkeypatch, adapter, [polling_app, webhook_app])
    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)

    assert await adapter.connect() is True
    assert adapter._webhook_mode is False
    assert adapter._send_path_degraded is False
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
async def test_fallback_disabled_skips_doh_discovery_on_connect(monkeypatch):
    """The fallback kill switch must bypass DoH discovery, not just transport use."""
    adapter = _make_adapter()
    polling_app = _lifecycle_app()

    async def start_polling_with_progress(**_kwargs):
        adapter._record_polling_progress(adapter._polling_generation)

    polling_app.updater.start_polling = AsyncMock(
        side_effect=start_polling_with_progress
    )
    builders = _configure_lifecycle_connect(monkeypatch, adapter, [polling_app])
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "true")

    async def fail_if_discovered():
        raise AssertionError("fallback discovery should be skipped when disabled")

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", fail_if_discovered)

    assert await adapter.connect() is True
    assert builders[0].polling_request is _ControlledRequest.instances[-1]
    assert "transport" not in (
        builders[0].polling_request.kwargs.get("httpx_kwargs") or {}
    )
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_fallback_discovery_timeout_falls_back_to_plain_connect(monkeypatch):
    """A stuck DoH fallback lookup must not block Telegram cold connect."""
    adapter = _make_adapter()
    polling_app = _lifecycle_app()

    async def start_polling_with_progress(**_kwargs):
        adapter._record_polling_progress(adapter._polling_generation)

    polling_app.updater.start_polling = AsyncMock(
        side_effect=start_polling_with_progress
    )
    builders = _configure_lifecycle_connect(monkeypatch, adapter, [polling_app])
    monkeypatch.setenv("HERMES_TELEGRAM_FALLBACK_DISCOVERY_TIMEOUT", "0.05")

    async def stuck_discovery():
        await asyncio.Event().wait()

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", stuck_discovery)

    assert await adapter.connect() is True
    assert "transport" not in (
        builders[0].polling_request.kwargs.get("httpx_kwargs") or {}
    )
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_non_finite_fallback_discovery_timeout_uses_finite_default(monkeypatch):
    """NaN/Inf timeout values must not defeat the cold-connect deadline."""
    adapter = _make_adapter()
    polling_app = _lifecycle_app()

    async def start_polling_with_progress(**_kwargs):
        adapter._record_polling_progress(adapter._polling_generation)

    polling_app.updater.start_polling = AsyncMock(
        side_effect=start_polling_with_progress
    )
    builders = _configure_lifecycle_connect(monkeypatch, adapter, [polling_app])
    monkeypatch.setenv("HERMES_TELEGRAM_FALLBACK_DISCOVERY_TIMEOUT", "nan")

    async def stuck_discovery():
        await asyncio.Event().wait()

    original_deadline = tg_adapter._await_with_thread_deadline

    async def deadline(awaitable, timeout, **_kwargs):
        if getattr(getattr(awaitable, "cr_code", None), "co_name", "") == "stuck_discovery":
            assert timeout == 5.0
            awaitable.close()
            raise asyncio.TimeoutError()
        return await original_deadline(awaitable, timeout, **_kwargs)

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", stuck_discovery)
    monkeypatch.setattr(tg_adapter, "_await_with_thread_deadline", deadline)

    assert await adapter.connect() is True
    assert "transport" not in (
        builders[0].polling_request.kwargs.get("httpx_kwargs") or {}
    )
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_fallback_disabled_excludes_configured_ips_from_proxy_targets(monkeypatch):
    """Disabled fallback IPs must not affect proxy bypass decisions."""
    adapter = _make_adapter()
    polling_app = _lifecycle_app()

    async def start_polling_with_progress(**_kwargs):
        adapter._record_polling_progress(adapter._polling_generation)

    polling_app.updater.start_polling = AsyncMock(
        side_effect=start_polling_with_progress
    )
    builders = _configure_lifecycle_connect(monkeypatch, adapter, [polling_app])
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "true")
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: ["149.154.167.220"])

    proxy_targets = []

    def resolve_proxy(_env_name, *, target_hosts):
        proxy_targets.append(list(target_hosts))
        return "http://127.0.0.1:8080"

    monkeypatch.setattr(tg_adapter, "resolve_proxy_url", resolve_proxy)

    assert await adapter.connect() is True
    assert proxy_targets == [["api.telegram.org"]]
    assert builders[0].polling_request.kwargs.get("proxy") == "http://127.0.0.1:8080"
    assert "transport" not in (
        builders[0].polling_request.kwargs.get("httpx_kwargs") or {}
    )
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
