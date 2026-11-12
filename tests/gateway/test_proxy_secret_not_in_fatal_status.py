"""A connect failure through a proxy whose URL carries a secret never puts that secret into the
adapter's fatal message, the runtime status (``hermes gateway status`` / Desktop) or the startup
"queued for retry" warning: those all show ``adapter.fatal_error_message``, built from the error
text that aiohttp/httpx echo the proxy URL into.

Drives the real ``GatewayRunner.start()`` / reconnect watcher with each real adapter whose connect
goes through a proxy; only the network endpoint is unreachable (``127.0.0.1:1``)."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch
from urllib.parse import unquote

import pytest

from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig

PASSWORD = "p{a}ss|w0rd^99"
# Scheme-less (yarl re-quotes it into InvalidURL), '#' (httpx/aiohttp "Invalid port: '<pw>'"),
# and '/' in the password (parses as host ``bob``, the password in the path).
SHAPES = ["bob:{pw}@127.0.0.1:1", "http://bob:{pw}#x@127.0.0.1:1", "http://bob:ab/{pw}@127.0.0.1:1"]
ENV_PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "WSS_PROXY",
                  "https_proxy", "http_proxy", "all_proxy", "wss_proxy")


def _leaks(text: str) -> list[str]:
    pieces = {PASSWORD[i:j] for i in range(len(PASSWORD)) for j in range(i + 4, len(PASSWORD) + 1)}
    return sorted(p for p in pieces if p in text or p in unquote(text))


def _aiohttp_proxy_error(proxy):
    """What aiohttp raises for a request through ``proxy`` (discord.py's HTTP client is aiohttp)."""
    aiohttp = pytest.importorskip("aiohttp")

    async def _go():
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:1/", proxy=proxy):
                pass
    try:
        asyncio.run(_go())
    except Exception as exc:
        return exc
    raise AssertionError("aiohttp accepted the proxy")


def _discord(monkeypatch, proxy):
    from hermes_agent.plugins.platforms.discord import adapter as dc

    if not dc.DISCORD_AVAILABLE:
        pytest.skip("discord.py not installed")
    monkeypatch.setenv("DISCORD_PROXY", proxy)
    monkeypatch.setattr(dc.discord.opus, "is_loaded", lambda: True)
    error = _aiohttp_proxy_error(proxy)

    def _bot(**kwargs):
        assert kwargs.get("proxy") == proxy
        raise error
    monkeypatch.setattr(dc.commands, "Bot", _bot)
    return dc.DiscordAdapter(PlatformConfig(enabled=True, token="MTA.fake.token"))


def _qqbot(monkeypatch, proxy):
    from hermes_agent.plugins.platforms.qqbot import adapter as qq

    for name in ENV_PROXY_VARS:
        monkeypatch.setenv(name, proxy)

    async def _token(self):
        self._access_token = "t"

    async def _gateway_url(self):
        return "wss://127.0.0.1:1/ws"
    monkeypatch.setattr(qq.QQAdapter, "_ensure_token", _token)
    monkeypatch.setattr(qq.QQAdapter, "_get_gateway_url", _gateway_url)
    return qq.QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "1", "client_secret": "s"}))


def _wecom(monkeypatch, proxy):
    from hermes_agent.plugins.platforms.wecom import adapter as wc

    for name in ENV_PROXY_VARS:
        monkeypatch.setenv(name, proxy)
    return wc.WeComAdapter(PlatformConfig(
        enabled=True, extra={"bot_id": "b", "secret": "s", "websocket_url": "wss://127.0.0.1:1/"}))


ADAPTERS = {Platform.DISCORD: _discord, Platform.QQBOT: _qqbot, Platform.WECOM: _wecom}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("platform", list(ADAPTERS), ids=lambda p: p.value)
def test_connect_failure_keeps_proxy_secret_out_of_fatal_status_and_retry(
        monkeypatch, tmp_path, caplog, platform, shape):
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.status import read_runtime_status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    proxy = shape.format(pw=PASSWORD)
    built = []

    def _make_adapter(_platform, _config):
        adapter = ADAPTERS[platform](monkeypatch, proxy)
        monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a, **k: True)
        built.append(adapter)
        return adapter

    runner = GatewayRunner(GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="MTA.fake.token")},
        sessions_dir=tmp_path / "sessions"))
    monkeypatch.setattr(runner, "_create_adapter", _make_adapter)
    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", AsyncMock(return_value=0))

    async def _start_then_reconnect():
        await runner.start()
        assert platform in runner._failed_platforms, "the connect failure was not queued for retry"
        runner._failed_platforms[platform]["next_retry"] = 0
        await runner._reconnect_failed_platform(platform, now=float("inf"))
        await runner.stop()

    with caplog.at_level(logging.DEBUG), patch("hermes_agent.gateway.platforms.base.gateway_trust_env", return_value=True):
        asyncio.run(_start_then_reconnect())

    assert len(built) == 2 and all(a.fatal_error_message for a in built)
    fatal = [a.fatal_error_message for a in built]
    assert all("via" in f for f in fatal), fatal  # the proxy-error path, not an unrelated failure
    status = json.dumps(read_runtime_status() or {})
    retry = "\n".join(r.getMessage() for r in caplog.records if "queued for retry" in r.getMessage())
    assert retry, "no startup 'queued for retry' warning"
    assert platform.value in status
    logs = "\n".join(r.getMessage() for r in caplog.records)
    for where, text in (("fatal", "\n".join(fatal)), ("status", status), ("retry", retry), ("logs", logs)):
        assert not _leaks(text), (where, _leaks(text), text[:600])


def test_qq_connect_redacts_the_https_proxy_httpx_parses_beside_a_clean_wss_proxy(monkeypatch, caplog):
    # The WebSocket dials the clean WSS_PROXY, but the httpx token client is built from HTTPS_PROXY
    # and fails first with ``Invalid port: '<pw>'``: no SDK mocks, the real client construction.
    pytest.importorskip("httpx")
    from hermes_agent.plugins.platforms.qqbot.adapter import QQAdapter

    for name in (*ENV_PROXY_VARS, "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WSS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", f"http://bob:{PASSWORD}#x@127.0.0.1:1")
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "mixed-proxy", "client_secret": "s"}))
    with caplog.at_level(logging.DEBUG):
        assert asyncio.run(adapter.connect()) is False
    logs = "\n".join(logging.Formatter().format(r) for r in caplog.records)
    assert adapter.fatal_error_message and " via " in adapter.fatal_error_message
    for text in (adapter.fatal_error_message, logs):
        assert not _leaks(text), text[-2000:]
