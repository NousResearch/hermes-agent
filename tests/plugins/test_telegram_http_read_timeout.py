"""Telegram gateway request defaults are configurable without crossing profile scopes."""

import asyncio
import contextlib

import pytest

from agent.secret_scope import set_multiplex_active
from gateway.config import Platform, PlatformConfig, load_gateway_config
from hermes_cli.plugins import discover_plugins
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.http_settings import http_read_timeout


@pytest.fixture(autouse=True)
def isolated_transport(monkeypatch):
    discover_plugins()
    monkeypatch.delenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", raising=False)
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "true")
    yield
    set_multiplex_active(False)


@pytest.fixture
def real_requests():
    return pytest.importorskip("telegram.request").HTTPXRequest


async def check_requests(adapter, expected):
    requests = await adapter._build_ptb_requests()
    try:
        assert [r.read_timeout for r in requests] == [expected, expected]
        for request in requests:
            timeout = request._client.timeout
            assert (timeout.connect, timeout.pool, timeout.write) == (10.0, 8.0, 20.0)
    finally:
        for request in requests:
            await request.shutdown()


def test_default_requests_use_twenty_seconds(real_requests):
    asyncio.run(check_requests(TelegramAdapter(PlatformConfig()), 20.0))


def test_yaml_read_timeout_reaches_requests(tmp_path, monkeypatch, real_requests):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "platforms:\n  telegram:\n    extra:\n      http_read_timeout: 120\n",
        encoding="utf-8",
    )
    adapter = TelegramAdapter(load_gateway_config().platforms[Platform.TELEGRAM])
    asyncio.run(check_requests(adapter, 120.0))


@pytest.mark.parametrize("invalid", ["bad", 0, -1, True, False, [], {}, "nan", "inf", "-inf"])
def test_invalid_yaml_uses_default(invalid, caplog):
    assert http_read_timeout({"http_read_timeout": invalid}) == 20.0
    assert "Invalid Telegram http_read_timeout" in caplog.text


@pytest.mark.parametrize("invalid", ["bad", "0", "-1", "nan", "inf", "-inf"])
def test_invalid_env_uses_default_not_yaml(invalid, monkeypatch):
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", invalid)
    assert http_read_timeout({"http_read_timeout": 120}) == 20.0


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_yaml_uses_default(value):
    assert http_read_timeout({"http_read_timeout": value}) == 20.0
    assert http_read_timeout({}) == 20.0


@pytest.mark.parametrize("value", [120, 0.5, "120.5"])
def test_positive_seconds(value):
    assert http_read_timeout({"http_read_timeout": value}) == float(value)


@pytest.mark.parametrize("env", ["90.5", "", "   "])
def test_legacy_env_precedence(env, monkeypatch, real_requests):
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", env)
    adapter = TelegramAdapter(PlatformConfig(extra={"http_read_timeout": 120}))
    asyncio.run(check_requests(adapter, 90.5 if env.strip() else 120.0))


@pytest.mark.parametrize("secondary_first", [False, True])
@pytest.mark.parametrize("secondary_env", [None, "60"])
@pytest.mark.parametrize("launch_env", [None, "120"])
def test_profile_separation_in_both_orders(
    tmp_path, monkeypatch, real_requests, secondary_first, secondary_env, launch_env
):
    from gateway.run import _profile_runtime_scope

    launch = tmp_path / "launch"
    secondary = launch / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    if launch_env is not None:
        monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", launch_env)
    (launch / "config.yaml").write_text(
        "platforms:\n  telegram:\n    extra:\n      http_read_timeout: 120\n", encoding="utf-8"
    )
    (secondary / "config.yaml").write_text(
        "platforms:\n  telegram:\n    extra:\n      http_read_timeout: 20\n", encoding="utf-8"
    )
    scope = {} if secondary_env is None else {"HERMES_TELEGRAM_HTTP_READ_TIMEOUT": secondary_env}
    set_multiplex_active(True)
    for home in ([secondary, launch] if secondary_first else [launch, secondary]):
        context = (
            _profile_runtime_scope(home, prepared_secret_scope=scope)
            if home == secondary else contextlib.nullcontext()
        )
        with context:
            adapter = TelegramAdapter(load_gateway_config().platforms[Platform.TELEGRAM])
            expected = (60.0 if secondary_env else 20.0) if home == secondary else 120.0
            asyncio.run(check_requests(adapter, expected))
    # YAML must never bridge the setting into the shared process environment.
    import os
    assert os.environ.get("HERMES_TELEGRAM_HTTP_READ_TIMEOUT") == launch_env


@pytest.mark.parametrize("route", ["direct", "proxy", "fallback"])
def test_effective_api_and_download_timeout(monkeypatch, real_requests, route):
    import httpx

    adapter = TelegramAdapter(PlatformConfig(extra={"http_read_timeout": 120}))
    if route == "proxy":
        adapter.config.extra["proxy_url"] = "http://127.0.0.1:18080"
    elif route == "fallback":
        monkeypatch.delenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS")
        monkeypatch.setattr(adapter, "_fallback_ips", lambda: ["149.154.167.220"])
    # Avoid host proxy auto-detection; use only the explicit test proxy.
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.resolve_proxy_url",
        lambda *args, configured=None, **kwargs: configured,
    )

    async def check():
        requests = await adapter._build_ptb_requests()
        seen = []

        async def respond(request):
            seen.append((request.method, request.extensions["timeout"]))
            return httpx.Response(200, content=b"offline response")

        try:
            for request in requests:
                await request._client.aclose()
                request._client = httpx.AsyncClient(
                    transport=httpx.MockTransport(respond), timeout=request._client.timeout
                )
                await request.do_request(url="https://api.telegram.org/offline", method="POST")
            assert await requests[0].retrieve("https://api.telegram.org/file/offline") == b"offline response"
            assert [method for method, _ in seen] == ["POST", "POST", "GET"]
            assert [timeout["read"] for _, timeout in seen] == [120.0] * 3
            assert all(
                (timeout["connect"], timeout["pool"], timeout["write"]) == (10.0, 8.0, 20.0)
                for _, timeout in seen
            )
        finally:
            for request in requests:
                await request.shutdown()

    asyncio.run(check())
