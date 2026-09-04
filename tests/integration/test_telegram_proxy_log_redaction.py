"""Integration coverage for Telegram proxy log redaction."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

telegram = pytest.importorskip("telegram")


def test_standalone_send_redacts_proxy_with_real_telegram_imports(
    monkeypatch, caplog, tmp_path
):
    from gateway import run as gateway_run
    from tools.send_message_tool import _send_telegram

    proxy_url = (
        "http://agent-vault-token:hermes@proxy.example:14322/route"
        "?token=secret#fragment"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_PROXY", proxy_url)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: None)

    async def _send_message(self, **kwargs):
        return SimpleNamespace(message_id=42)

    monkeypatch.setattr(telegram.Bot, "send_message", _send_message)
    caplog.set_level(logging.INFO, logger="tools.send_message_tool")

    result = asyncio.run(_send_telegram("tok", "123", "hello world"))

    assert result["success"] is True
    assert "agent-vault-token" not in caplog.text
    assert "secret" not in caplog.text
    assert "fragment" not in caplog.text
    assert "http://proxy.example:14322/.../route" in caplog.text


def test_adapter_connect_redacts_proxy_with_real_telegram_requests(
    monkeypatch, caplog, tmp_path
):
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram import adapter as telegram_adapter
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from telegram.ext import Application

    proxy_url = "http://agent-vault-token:hermes@proxy.example:14322"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_PROXY", proxy_url)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    async def _no_fallback_ips():
        return []

    async def _stop_before_network(self):
        raise asyncio.CancelledError

    monkeypatch.setattr(telegram_adapter, "discover_fallback_ips", _no_fallback_ips)
    monkeypatch.setattr(Application, "initialize", _stop_before_network)
    caplog.set_level(logging.INFO, logger=telegram_adapter.__name__)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123456:test-token"))
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(adapter.connect())

    requests = adapter._app.bot._request
    assert all(request._client_kwargs["proxy"] == proxy_url for request in requests)
    assert "agent-vault-token" not in caplog.text
    assert "http://proxy.example:14322" in caplog.text
