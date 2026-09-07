"""Behavior contract for generation-safe Telegram polling progress."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import TimedOut

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
async def test_webhook_mode_disables_unprovable_background_location_continuity(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"background_locations": True},
        )
    )
    app = _lifecycle_app()
    _configure_lifecycle_connect(monkeypatch, adapter, [app])
    adapter._prepare_background_locations_for_connect = AsyncMock(return_value={})
    monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "https://example.test/telegram")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-secret")

    try:
        assert await adapter.connect() is True
        adapter._prepare_background_locations_for_connect.assert_awaited_once()
        assert adapter._webhook_mode is True
        assert adapter._background_locations_configured is True
        assert adapter._background_locations_enabled is False
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
async def test_fallback_discovery_timeout_uses_seed_ipv4(monkeypatch):
    """A stuck DoH lookup must not block connect; seed IPv4 IPs are used instead."""
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
    httpx_kwargs = builders[0].polling_request.kwargs.get("httpx_kwargs") or {}
    transport = httpx_kwargs.get("transport")
    assert isinstance(transport, tg_adapter.TelegramFallbackTransport)
    assert transport._fallback_ips == list(tg_adapter.SEED_FALLBACK_IPS)
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
    httpx_kwargs = builders[0].polling_request.kwargs.get("httpx_kwargs") or {}
    transport = httpx_kwargs.get("transport")
    assert isinstance(transport, tg_adapter.TelegramFallbackTransport)
    assert transport._fallback_ips == list(tg_adapter.SEED_FALLBACK_IPS)
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
@pytest.mark.parametrize(
    "payload",
    [
        b'{"ok":true,"result":{}}',
        b'{"ok":true,"result":[{}]}',
        b'{"ok":true,"result":[{"update_id":true}]}',
        b'{"ok":true,"result":[{"update_id":"1"}]}',
    ],
)
async def test_malformed_success_cannot_establish_polling_continuity(payload):
    adapter = _make_adapter()
    generation, progress = adapter._begin_polling_generation()
    observer_error = MagicMock()
    adapter._polling_generation_error_callback = observer_error
    request = _ControlledRequest(result=(200, payload))

    await _request_for_generation(
        generation,
        adapter._instrument_polling_request(request),
        "https://api.telegram.org/getUpdates",
    )

    assert not progress.is_set()
    assert adapter._send_path_degraded is True
    assert adapter._polling_progress_accepting is False
    observer_error.assert_called_once()


@pytest.mark.asyncio
async def test_nested_malformed_update_cannot_establish_polling_continuity():
    """A valid update_id cannot hide a location object PTB cannot deserialize."""
    adapter = _make_adapter()
    generation, progress = adapter._begin_polling_generation()
    observer_error = MagicMock()
    adapter._polling_generation_error_callback = observer_error
    request = _ControlledRequest(
        result=(
            200,
            b'{"ok":true,"result":[{"update_id":1,'
            b'"edited_message":{"location":{}}}]}',
        )
    )

    await _request_for_generation(
        generation,
        adapter._instrument_polling_request(request),
        "https://api.telegram.org/getUpdates",
    )

    assert not progress.is_set()
    assert adapter._send_path_degraded is True
    assert adapter._polling_progress_accepting is False
    observer_error.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (401, b'{"ok":false,"description":"Unauthorized"}'),
        (200, b'{"ok":false,"description":"Bad Request"}'),
        (200, b'not-json'),
    ],
)
async def test_failed_poll_response_revokes_previously_healthy_generation(
    status, payload
):
    adapter = _make_adapter()
    generation, progress = adapter._begin_polling_generation()
    adapter._record_polling_progress(generation, None, backlog_drained=True)
    assert progress.is_set()
    assert adapter._send_path_degraded is False
    observer_error = MagicMock()
    adapter._polling_generation_error_callback = observer_error

    request = _ControlledRequest(result=(status, payload))
    await _request_for_generation(
        generation,
        adapter._instrument_polling_request(request),
        "https://api.telegram.org/getUpdates",
    )

    assert adapter._send_path_degraded is True
    assert adapter._polling_progress_accepting is False
    observer_error.assert_called_once()


@pytest.mark.asyncio
async def test_live_observer_auth_then_typed_callback_stays_terminal(monkeypatch):
    adapter = _make_adapter()
    captured = {}

    async def capture_polling_start(**kwargs):
        captured["error_callback"] = kwargs["error_callback"]
        return True

    monkeypatch.setattr(
        adapter, "_delete_webhook_best_effort", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(adapter, "_start_polling_resilient", capture_polling_start)
    handoff = AsyncMock()
    monkeypatch.setattr(adapter, "_handoff_polling_fatal_error", handoff)
    await adapter._start_polling_mode(is_reconnect=True)
    generation, _progress = adapter._begin_polling_generation()
    adapter._polling_generation_error_callback = captured["error_callback"]

    class InvalidToken(Exception):
        pass

    request = _ControlledRequest(
        result=(
            401,
            b'{"ok":false,"error_code":401,"description":"Unauthorized"}',
        )
    )
    await _request_for_generation(
        generation,
        adapter._instrument_polling_request(request),
        "https://api.telegram.org/getUpdates",
    )
    # PTB parses the same response only after the raw observer returns.
    captured["error_callback"](InvalidToken("invalid token"))

    await asyncio.wait_for(adapter._polling_error_task, timeout=1)
    assert adapter.fatal_error_code == "telegram_auth_error"
    assert adapter._fatal_error_retryable is False
    handoff.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_observer_conflict_then_typed_callback_uses_conflict_ladder(
    monkeypatch,
):
    adapter = _make_adapter()
    captured = {}

    async def capture_polling_start(**kwargs):
        captured["error_callback"] = kwargs["error_callback"]
        return True

    monkeypatch.setattr(
        adapter, "_delete_webhook_best_effort", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(adapter, "_start_polling_resilient", capture_polling_start)
    conflict_recovery = AsyncMock()
    disarm = MagicMock()
    monkeypatch.setattr(adapter, "_handle_polling_conflict", conflict_recovery)
    monkeypatch.setattr(adapter, "_disarm_ptb_retry_loop", disarm)
    await adapter._start_polling_mode(is_reconnect=True)
    generation, _progress = adapter._begin_polling_generation()
    adapter._polling_generation_error_callback = captured["error_callback"]

    class Conflict(Exception):
        pass

    request = _ControlledRequest(
        result=(
            409,
            b'{"ok":false,"error_code":409,"description":"Conflict"}',
        )
    )
    await _request_for_generation(
        generation,
        adapter._instrument_polling_request(request),
        "https://api.telegram.org/getUpdates",
    )
    captured["error_callback"](Conflict("other getUpdates request"))

    await asyncio.wait_for(adapter._polling_error_task, timeout=1)
    conflict_recovery.assert_awaited_once()
    disarm.assert_called_once()


@pytest.mark.asyncio
async def test_typed_auth_supersedes_inflight_generic_observer_recovery(
    monkeypatch,
):
    adapter = _make_adapter()
    captured = {}

    async def capture_polling_start(**kwargs):
        captured["error_callback"] = kwargs["error_callback"]
        return True

    generic_started = asyncio.Event()
    release_generic = asyncio.Event()

    async def blocked_generic_recovery(_error):
        generic_started.set()
        await release_generic.wait()

    monkeypatch.setattr(
        adapter, "_delete_webhook_best_effort", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(adapter, "_start_polling_resilient", capture_polling_start)
    monkeypatch.setattr(
        adapter, "_handle_polling_network_error", blocked_generic_recovery
    )
    monkeypatch.setattr(adapter, "_handoff_polling_fatal_error", AsyncMock())
    await adapter._start_polling_mode(is_reconnect=True)

    captured["error_callback"](RuntimeError("unclassified response"))
    generic_task = adapter._polling_error_task
    await asyncio.wait_for(generic_started.wait(), timeout=1)

    class InvalidToken(Exception):
        pass

    captured["error_callback"](InvalidToken("invalid token"))
    await asyncio.gather(generic_task, return_exceptions=True)
    await asyncio.wait_for(adapter._polling_error_task, timeout=1)

    assert generic_task.cancelled()
    assert adapter.fatal_error_code == "telegram_auth_error"
    release_generic.set()


@pytest.mark.asyncio
async def test_cold_observer_auth_then_typed_callback_classifies_wrapped_error(
    monkeypatch,
):
    adapter = _make_adapter()
    app = _lifecycle_app()

    class InvalidToken(Exception):
        pass

    async def polling_fails_auth(**kwargs):
        request = SimpleNamespace(
            parse_json_payload=lambda raw: json.loads(raw.decode("utf-8"))
        )
        adapter._observe_polling_request_result(
            request,
            adapter._polling_generation,
            (
                401,
                b'{"ok":false,"error_code":401,"description":"Unauthorized"}',
            ),
        )
        # The strict gate intentionally keeps only the first error. It must
        # still classify the observer's status-preserving synthetic exception.
        kwargs["error_callback"](InvalidToken("invalid token"))

    app.updater.start_polling = AsyncMock(side_effect=polling_fails_auth)
    _configure_lifecycle_connect(monkeypatch, adapter, [app])
    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL", raising=False)

    assert await adapter.connect() is False
    assert adapter.fatal_error_code == "telegram_auth_error"
    assert adapter._fatal_error_retryable is False


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, TimedOut])
async def test_physical_polling_failure_fences_before_ptb_retry(error_type):
    adapter = _make_adapter()
    generation, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 3
    adapter._send_path_degraded = False
    observed = MagicMock()
    adapter._polling_generation_error_callback = observed
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
    assert adapter._polling_progress_accepting is False
    observed.assert_called_once()
    assert isinstance(observed.call_args.args[0], error_type)


@pytest.mark.asyncio
async def test_polling_request_cancellation_does_not_schedule_recovery():
    adapter = _make_adapter()
    generation, progress = adapter._begin_polling_generation()
    adapter._send_path_degraded = False
    observed = MagicMock()
    adapter._polling_generation_error_callback = observed
    request = adapter._instrument_polling_request(
        _ControlledRequest(error=asyncio.CancelledError("ordinary teardown"))
    )

    with pytest.raises(asyncio.CancelledError):
        await _request_for_generation(
            generation, request, "https://api.telegram.org/getUpdates"
        )

    assert not progress.is_set()
    assert adapter._send_path_degraded is False
    assert adapter._polling_progress_accepting is True
    observed.assert_not_called()


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
