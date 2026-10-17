"""Proxy URLs may embed credentials (``http://user:pass@host``, see
``resolve_proxy_url``). The adapters that announce their proxy at connect must log
the host, never the userinfo: ``RedactingFormatter`` deliberately passes
``user:pass@`` through for http(s) URLs, so the log line itself is the last line
of defence. Slack already logs through ``safe_url_for_log``; these tests pin the
same contract for Telegram, Discord, Matrix and the standalone Telegram sender."""

import asyncio
import logging
import types
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig

PROXY = "http://alice:S3cretPass@127.0.0.1:3128"
SECRET = "S3cretPass"


def _proxy_lines(caplog):
    return [r.getMessage() for r in caplog.records if "proxy" in r.getMessage().lower()]


def _assert_host_logged_without_secret(caplog):
    lines = _proxy_lines(caplog)
    assert lines, "the adapter must still announce that a proxy is in use"
    assert all(SECRET not in line and "alice" not in line for line in lines), lines
    assert any("127.0.0.1:3128" in line for line in lines), lines


def test_telegram_request_builder_logs_proxy_without_credentials(monkeypatch, caplog):
    from plugins.platforms.telegram import adapter as tg

    monkeypatch.setenv("TELEGRAM_PROXY", PROXY)
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "true")
    adapter = tg.TelegramAdapter(PlatformConfig(enabled=True, token="123456:TOKEN"))
    with caplog.at_level(logging.INFO), \
            patch.object(tg, "HTTPXRequest", lambda **kw: types.SimpleNamespace(**kw)), \
            patch.object(adapter, "_instrument_polling_request", side_effect=lambda r: r):
        request, _updates = asyncio.run(adapter._build_ptb_requests())
    assert request.proxy == PROXY  # the transport still gets the real credentials
    _assert_host_logged_without_secret(caplog)


def test_matrix_adapter_logs_proxy_without_credentials(monkeypatch, caplog):
    from plugins.platforms.matrix import adapter as matrix_adapter

    monkeypatch.setenv("MATRIX_PROXY", PROXY)
    with caplog.at_level(logging.INFO):
        adapter = matrix_adapter.MatrixAdapter(PlatformConfig(
            enabled=True, token="syt_test_token",
            extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}))
    assert adapter._proxy_url == PROXY
    _assert_host_logged_without_secret(caplog)


def test_discord_connect_logs_proxy_without_credentials(monkeypatch, caplog):
    from plugins.platforms.discord import adapter as discord_adapter

    if not discord_adapter.DISCORD_AVAILABLE:
        pytest.skip("discord.py not installed")
    monkeypatch.setenv("DISCORD_PROXY", PROXY)
    adapter = discord_adapter.DiscordAdapter(PlatformConfig(enabled=True, token="bot-token"))
    built: list[dict] = []

    def _stop_at_client(**kw):
        built.append(kw)
        raise RuntimeError("stop before any network I/O")

    with caplog.at_level(logging.INFO), \
            patch.object(discord_adapter.discord.opus, "is_loaded", return_value=True), \
            patch.object(discord_adapter.commands, "Bot", _stop_at_client), \
            patch.object(adapter, "_acquire_platform_lock", return_value=True), \
            patch.object(adapter, "_snapshot_gate_env"), \
            patch.object(adapter, "_get_allowed_users", return_value=set()), \
            patch.object(adapter, "_get_allowed_roles", return_value=set()):
        asyncio.run(adapter.connect())
    assert built and built[0].get("proxy") == PROXY
    _assert_host_logged_without_secret(caplog)


def test_standalone_telegram_sender_logs_proxy_without_credentials(monkeypatch, caplog):
    import sys
    import telegram
    import telegram.request
    from gateway.platforms import base as gw_base
    from tools import send_message_senders

    monkeypatch.setattr(gw_base, "resolve_proxy_url", lambda *a, **k: PROXY)
    seen: list = []
    # ``_telegram_bot`` imports both names inside the function, so patch the modules
    # ``from telegram import Bot`` / ``from telegram.request import HTTPXRequest`` resolve to.
    monkeypatch.setattr(sys.modules["telegram"], "Bot", lambda **kw: seen.append(kw) or MagicMock())
    monkeypatch.setattr(sys.modules["telegram.request"], "HTTPXRequest",
                        lambda **kw: types.SimpleNamespace(**kw))
    with caplog.at_level(logging.INFO):
        send_message_senders._telegram_bot("123456:TOKEN")
    assert seen and seen[0]["request"].proxy == PROXY
    _assert_host_logged_without_secret(caplog)


SOCKS_PROXY = "socks5://alice:S3cretPass@127.0.0.1:1080"


def _assert_socks_warning_without_secret(caplog):
    lines = [r.getMessage() for r in caplog.records if "SOCKS proxy" in r.getMessage()]
    assert lines, "the missing-aiohttp_socks warning must still fire"
    assert all(SECRET not in line and "alice" not in line for line in lines), lines
    assert any("127.0.0.1:1080" in line for line in lines), lines


def test_shared_socks_connector_warning_logs_proxy_without_credentials(monkeypatch, caplog):
    import sys

    from gateway.platforms import base as gw_base

    monkeypatch.setitem(sys.modules, "aiohttp_socks", None)  # import raises ImportError
    with caplog.at_level(logging.WARNING):
        assert gw_base._aiohttp_socks_connector(SOCKS_PROXY) is None
    _assert_socks_warning_without_secret(caplog)


def test_matrix_socks_session_warning_logs_proxy_without_credentials(monkeypatch, caplog):
    import sys

    from plugins.platforms.matrix import adapter as matrix_adapter

    monkeypatch.setitem(sys.modules, "aiohttp_socks", None)
    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(ClientSession=lambda **kw: kw))
    with caplog.at_level(logging.WARNING):
        matrix_adapter._create_matrix_session(SOCKS_PROXY)
    _assert_socks_warning_without_secret(caplog)
