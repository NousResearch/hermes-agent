"""Behavior contract for generation-safe Telegram polling progress."""

import asyncio

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


class _ControlledRequest:
    """Minimal PTB request double with controllable completion."""

    instances = []

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


@pytest.mark.asyncio
async def test_current_polling_generation_success_records_progress():
    adapter = _make_adapter()
    generation, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 3
    request = _ControlledRequest(result=(200, b'{"ok":true}'))

    instrumented = adapter._instrument_polling_request(request)
    result = await instrumented.do_request("https://api.telegram.org/getUpdates")

    assert instrumented is request
    assert result == (200, b'{"ok":true}')
    assert progress.is_set()
    assert adapter._polling_network_error_count == 0
    assert adapter._send_path_degraded is False
    assert generation > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_unsuccessful_polling_request_does_not_record_progress(error_type):
    adapter = _make_adapter()
    _, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 3
    request = adapter._instrument_polling_request(
        _ControlledRequest(error=error_type("request did not complete"))
    )

    with pytest.raises(error_type):
        await request.do_request("https://api.telegram.org/getUpdates")

    assert not progress.is_set()
    assert adapter._polling_network_error_count == 3
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
async def test_http_error_response_does_not_record_polling_progress():
    adapter = _make_adapter()
    _, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 3
    request = adapter._instrument_polling_request(
        _ControlledRequest(result=(500, b"bad"))
    )

    result = await request.do_request("https://api.telegram.org/getUpdates")

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
        _ControlledRequest(result=(200, b'{"ok":true}'), entered=entered, release=release)
    )

    completion = asyncio.create_task(request.do_request("getUpdates"))
    await entered.wait()
    generation_2, progress_2 = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 4
    release.set()

    assert await completion == (200, b'{"ok":true}')
    assert generation_2 == generation_1 + 1
    assert not progress_2.is_set()
    assert adapter._polling_network_error_count == 4
    assert adapter._send_path_degraded is True
