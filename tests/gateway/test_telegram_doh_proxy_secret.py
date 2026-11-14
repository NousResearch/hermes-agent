"""Telegram connect builds httpx clients that read the proxy environment themselves: fallback-IP
discovery (before TELEGRAM_PROXY is resolved) and the plain PTB requests when no proxy is resolved
(NO_PROXY bypass, fallback IPs disabled). A malformed secret-bearing HTTPS_PROXY must not reach the
connect logs or the fatal status through either."""

import asyncio
import logging
from unittest.mock import patch

import httpx
import pytest

from gateway.config import PlatformConfig

SECRET = "AgentTok987"
SHAPES = [f"http://bob:{SECRET}#x@127.0.0.1:1", f"ftp://{SECRET}:pw@127.0.0.1:1"]


class _PTBLikeRequest:
    """python-telegram-bot 22.8 ``HTTPXRequest`` builds ``httpx.AsyncClient(proxy=..., **httpx_kwargs)``
    in ``__init__`` without ``trust_env``, so httpx parses the proxy environment right there."""

    def __init__(self, proxy=None, httpx_kwargs=None, **_kw):
        kwargs = dict(httpx_kwargs or {})
        httpx.AsyncClient(proxy=proxy, **kwargs)
        raise ConnectionError("stop after request building")


def _connect(monkeypatch, caplog, proxy, **env):
    from plugins.platforms.telegram import adapter as tg

    for name in ("TELEGRAM_PROXY", "HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "TELEGRAM_FALLBACK_IPS", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    adapter = tg.TelegramAdapter(PlatformConfig(enabled=True, token="123456:TOKEN"))
    with caplog.at_level(logging.DEBUG), patch.object(tg, "HTTPXRequest", _PTBLikeRequest), \
            patch.object(adapter, "_acquire_platform_lock", return_value=True), \
            patch.object(adapter, "_write_runtime_status_safe"):
        assert asyncio.run(adapter.connect()) is False
    return adapter


@pytest.mark.parametrize("proxy", SHAPES)
def test_fallback_ip_discovery_error_hides_the_env_proxy_secret(monkeypatch, caplog, proxy):
    adapter = _connect(monkeypatch, caplog, proxy)
    assert any("fallback-IP discovery failed" in r.getMessage() for r in caplog.records)
    assert SECRET not in caplog.text
    assert SECRET not in (adapter.fatal_error_message or "")


@pytest.mark.parametrize("proxy", SHAPES)
def test_plain_ptb_client_error_hides_the_env_proxy_secret(monkeypatch, caplog, proxy):
    adapter = _connect(monkeypatch, caplog, proxy, NO_PROXY="api.telegram.org", HERMES_TELEGRAM_DISABLE_FALLBACK_IPS="true")
    assert adapter.fatal_error_message and " via " in adapter.fatal_error_message, adapter.fatal_error_message
    assert SECRET not in caplog.text
    assert SECRET not in adapter.fatal_error_message
