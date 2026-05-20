import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import telegram as telegram_mod
from gateway.platforms.telegram import TelegramAdapter


class FakeHTTPXRequest:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)


class FakeBuilder:
    def __init__(self):
        self.request_obj = None
        self.get_updates_request_obj = None

    def token(self, _token):
        return self

    def request(self, request):
        self.request_obj = request
        return self

    def get_updates_request(self, request):
        self.get_updates_request_obj = request
        return self

    def build(self):
        return SimpleNamespace(
            bot=SimpleNamespace(),
            add_handler=lambda *_args, **_kwargs: None,
            add_error_handler=lambda *_args, **_kwargs: None,
        )


@pytest.mark.asyncio
async def test_connect_disables_keepalive_only_for_get_updates(monkeypatch):
    FakeHTTPXRequest.calls = []
    monkeypatch.setattr(telegram_mod, "HTTPXRequest", FakeHTTPXRequest)
    monkeypatch.setattr(
        telegram_mod.Application,
        "builder",
        staticmethod(lambda: FakeBuilder()),
    )
    monkeypatch.setattr(telegram_mod, "discover_fallback_ips", lambda: asyncio.sleep(0, result=[]))
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "1")

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    await adapter.connect()

    assert len(FakeHTTPXRequest.calls) == 2
    general_request, polling_request = FakeHTTPXRequest.calls
    assert "httpx_kwargs" not in general_request
    limits = polling_request["httpx_kwargs"]["limits"]
    assert limits.max_keepalive_connections == 0
    assert limits.max_connections == polling_request["connection_pool_size"]


def test_polling_timeout_and_interval_env_are_opt_in_and_clamped(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))

    assert adapter._polling_kwargs() == {}

    monkeypatch.setenv("HERMES_TELEGRAM_POLL_TIMEOUT", "120")
    monkeypatch.setenv("HERMES_TELEGRAM_POLL_INTERVAL", "-3")
    assert adapter._polling_kwargs() == {"timeout": 60, "poll_interval": 0.0}

    monkeypatch.setenv("HERMES_TELEGRAM_POLL_TIMEOUT", "bad")
    monkeypatch.setenv("HERMES_TELEGRAM_POLL_INTERVAL", "bad")
    assert adapter._polling_kwargs() == {"timeout": 1, "poll_interval": 1.0}
