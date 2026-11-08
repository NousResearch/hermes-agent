"""A transport error raised with a proxy configured logs and replies with only its exception type
and the proxy's ``scheme://host:port``: yarl/httpx percent-encode, split or reformat a malformed
proxy URL into the error text, so no substring redaction of that text keeps the password out.

Every case drives the real call site with the real aiohttp/httpx exception for that proxy value."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import unquote

import httpx
import pytest

from gateway.config import Platform, PlatformConfig

# Characters yarl/httpx re-quote, non-ASCII, whitespace, a short one, a scheme word, a repeat.
PASSWORDS = ['p{a}ss|w0rd^9"<>', "pässwörd99", "pa ss wo rd", "abc", "socks5", "qzxqzxqzx"]
# Scheme-less (re-quoted into the error) and an unencoded '#' (``Invalid port: '<pw>'``).
SHAPES = ["bob:{pw}@127.0.0.1:1", "http://bob:{pw}#x@127.0.0.1:1"]
CASES = [pytest.param(shape.format(pw=pw), pw, id=f"{pw}|{shape.split(':')[0]}")
         for pw in PASSWORDS for shape in SHAPES]


def _aiohttp_error(proxy, *, as_url=False):
    aiohttp = pytest.importorskip("aiohttp")
    url, kwargs = (proxy, {}) if as_url else ("http://127.0.0.1:1/", {"proxy": proxy})

    async def _go():
        async with aiohttp.ClientSession() as session:
            async with session.get(url, **kwargs):
                pass
    try:
        asyncio.run(_go())
    except Exception as exc:
        return exc
    raise AssertionError("aiohttp accepted the malformed proxy")


def _httpx_error(proxy):
    try:
        httpx.AsyncClient(proxy=proxy)
    except Exception as exc:
        return exc
    raise AssertionError("httpx accepted the malformed proxy")


def _assert_safe(text, password, exc):
    assert type(exc).__name__ in text, text  # the user still learns what failed
    decoded = unquote(text)
    # Every piece with 3+ non-space chars (``"o r"`` in ``"... to recover"`` is English, not a leak).
    pieces = {password[i:j] for i in range(len(password)) for j in range(i + 3, len(password) + 1)
              if len(password[i:j].replace(" ", "")) >= 3}
    leaked = sorted(p for p in pieces if p in text or p in decoded)
    assert not leaked, (leaked, text)


def _logs(caplog):
    return "\n".join(r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("proxy, password", CASES)
def test_proxy_mode_turn_error(monkeypatch, caplog, proxy, password):
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    monkeypatch.setenv("GATEWAY_PROXY_URL", proxy)
    monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
    runner = object.__new__(GatewayRunner)
    runner.adapters, runner._running_agents, runner._agent_cache = {}, {}, {}
    runner.config = MagicMock()
    source = SessionSource(platform=Platform.MATRIX, chat_id="!r:s", chat_type="group", user_id="@u:s")
    exc = _aiohttp_error(f"{proxy}/v1/chat/completions", as_url=True)  # the upstream itself, not a proxy
    with patch("gateway.run._load_gateway_config", return_value={}), caplog.at_level(logging.DEBUG):
        reply = asyncio.run(runner._run_agent_via_proxy(
            message="hi", context_prompt="", history=[], source=source, session_id="s"))["final_response"]
    _assert_safe(reply, password, exc)
    _assert_safe(_logs(caplog), password, exc)


def _telegram_adapter(monkeypatch, proxy):
    from plugins.platforms.telegram import adapter as tg

    monkeypatch.setenv("TELEGRAM_PROXY", proxy)
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "true")
    return tg, tg.TelegramAdapter(PlatformConfig(enabled=True, token="123456:TOKEN"))


@pytest.mark.parametrize("proxy, password", CASES)
def test_telegram_connect_error(monkeypatch, caplog, proxy, password):
    tg, adapter = _telegram_adapter(monkeypatch, proxy)
    with caplog.at_level(logging.DEBUG), \
            patch.object(tg, "HTTPXRequest", lambda **kw: httpx.AsyncClient(proxy=kw.get("proxy"))), \
            patch.object(adapter, "_acquire_platform_lock", return_value=True), \
            patch.object(adapter, "_write_runtime_status_safe"):
        assert asyncio.run(adapter.connect()) is False
    exc = _httpx_error(proxy)
    _assert_safe(adapter._fatal_error_message or "", password, exc)
    _assert_safe(_logs(caplog), password, exc)


@pytest.mark.parametrize("proxy, password", CASES)
@pytest.mark.parametrize("site", ["network", "conflict"])
def test_telegram_polling_error_after_retries(monkeypatch, caplog, proxy, password, site):
    _, adapter = _telegram_adapter(monkeypatch, proxy)
    adapter._ptb_proxy_url = proxy  # what _build_ptb_requests() bound at connect
    exc = _httpx_error(proxy)
    with caplog.at_level(logging.DEBUG), patch.object(adapter, "_handoff_polling_fatal_error", AsyncMock()):
        if site == "network":
            adapter._polling_network_error_count = 10
            asyncio.run(adapter._handle_polling_network_error(exc))
        else:
            adapter._polling_conflict_count = 5
            asyncio.run(adapter._handle_polling_conflict(exc))
    _assert_safe(_logs(caplog), password, exc)


@pytest.mark.parametrize("proxy, password", CASES)
def test_standalone_telegram_sender_proxy_fallback(monkeypatch, caplog, proxy, password):
    import sys

    import telegram
    import telegram.request
    from tools import send_message_senders

    monkeypatch.setenv("TELEGRAM_PROXY", proxy)
    monkeypatch.setattr(sys.modules["telegram"], "Bot", lambda **kw: MagicMock())
    monkeypatch.setattr(sys.modules["telegram.request"], "HTTPXRequest",
                        lambda **kw: httpx.AsyncClient(proxy=kw["proxy"]))
    with caplog.at_level(logging.DEBUG):
        send_message_senders._telegram_bot("123456:TOKEN")
    _assert_safe(_logs(caplog), password, _httpx_error(proxy))


@pytest.mark.parametrize("proxy, password", CASES)
def test_discord_connect_error(monkeypatch, caplog, proxy, password):
    from plugins.platforms.discord import adapter as discord_adapter

    if not discord_adapter.DISCORD_AVAILABLE:
        pytest.skip("discord.py not installed")
    monkeypatch.setenv("DISCORD_PROXY", proxy)
    adapter = discord_adapter.DiscordAdapter(PlatformConfig(enabled=True, token="bot-token"))
    exc = _aiohttp_error(proxy)
    with caplog.at_level(logging.DEBUG), \
            patch.object(discord_adapter.discord.opus, "is_loaded", return_value=True), \
            patch.object(discord_adapter.commands, "Bot", side_effect=exc), \
            patch.object(adapter, "_acquire_platform_lock", return_value=True), \
            patch.object(adapter, "_snapshot_gate_env"), \
            patch.object(adapter, "_get_allowed_users", return_value=set()), \
            patch.object(adapter, "_get_allowed_roles", return_value=set()):
        assert asyncio.run(adapter.connect()) is False
    _assert_safe(_logs(caplog), password, exc)


@pytest.mark.parametrize("proxy, password", CASES)
@pytest.mark.parametrize("site", ["whoami", "login", "initial_sync", "sync_loop"])
def test_matrix_error(monkeypatch, caplog, proxy, password, site):
    from plugins.platforms.matrix import adapter as matrix_adapter

    monkeypatch.setenv("MATRIX_PROXY", proxy)
    adapter = matrix_adapter.MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test_token" if site == "whoami" else "",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org", "password": "pw"}))
    exc = _aiohttp_error(proxy)
    client = MagicMock()
    client.whoami = client.login = client.sync = AsyncMock(side_effect=exc)
    client.sync_store.get_next_batch = AsyncMock(return_value="b1")
    real_sleep = asyncio.sleep

    async def _sleep_then_close(delay, *a, **kw):
        adapter._closing = True
        await real_sleep(0)

    async def _run():
        if site in ("whoami", "login"):
            api = MagicMock()
            api.session.close = AsyncMock()
            assert await adapter._connect_authenticate(client, api) is False
        elif site == "initial_sync":
            await adapter._connect_initial_sync(client)
        else:
            adapter._client = client
            with patch.object(matrix_adapter.asyncio, "sleep", _sleep_then_close):
                await adapter._sync_loop()

    with caplog.at_level(logging.DEBUG):
        asyncio.run(_run())
    _assert_safe(_logs(caplog), password, exc)


@pytest.mark.parametrize("proxy, password", [c for c in CASES if c.values[0].startswith("http://")])
def test_slack_connect_error(monkeypatch, caplog, proxy, password):
    from plugins.platforms.slack import adapter as slack_adapter

    for name in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-fake")
    monkeypatch.setattr(slack_adapter, "SLACK_AVAILABLE", True)
    adapter = slack_adapter.SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    exc = _aiohttp_error(proxy)
    with caplog.at_level(logging.DEBUG), \
            patch.object(slack_adapter, "AsyncApp", side_effect=exc, create=True), \
            patch.object(slack_adapter, "AsyncWebClient", MagicMock(), create=True), \
            patch.object(adapter, "_acquire_platform_lock", return_value=True):
        assert asyncio.run(adapter.connect()) is False
    _assert_safe(_logs(caplog), password, exc)


@pytest.mark.parametrize("proxy", [
    None, "http://proxy.example.com:3128", "socks5://10.0.0.1:1080",
    # Rendering lowercases scheme/host and drops port zero-padding; nothing is left out.
    "HTTP://proxy:3128", "http://Proxy.Corp:3128", "http://[2001:DB8::1]:3128", "http://proxy:03128/"])
def test_error_through_a_secret_free_proxy_keeps_its_text(monkeypatch, caplog, proxy):
    """No proxy, or one whose URL is all shown: no secret involved, the user keeps the detail."""
    from plugins.platforms.matrix import adapter as matrix_adapter

    adapter = matrix_adapter.MatrixAdapter(PlatformConfig(
        enabled=True, token="", extra={"homeserver": "https://matrix.example.org"}))
    adapter._proxy_url = proxy
    client = MagicMock()
    client.sync = AsyncMock(side_effect=RuntimeError("homeserver said M_UNKNOWN"))
    with caplog.at_level(logging.DEBUG):
        asyncio.run(adapter._connect_initial_sync(client))
    assert "homeserver said M_UNKNOWN" in _logs(caplog)
