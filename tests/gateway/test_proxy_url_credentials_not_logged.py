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

from hermes_agent.gateway.config import Platform, PlatformConfig

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
    from hermes_agent.plugins.platforms.telegram import adapter as tg

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
    from hermes_agent.plugins.platforms.matrix import adapter as matrix_adapter

    monkeypatch.setenv("MATRIX_PROXY", PROXY)
    with caplog.at_level(logging.INFO):
        adapter = matrix_adapter.MatrixAdapter(PlatformConfig(
            enabled=True, token="syt_test_token",
            extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}))
    assert adapter._proxy_url == PROXY
    _assert_host_logged_without_secret(caplog)


def test_discord_connect_logs_proxy_without_credentials(monkeypatch, caplog):
    from hermes_agent.plugins.platforms.discord import adapter as discord_adapter

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
    from hermes_agent.gateway.platforms import base as gw_base
    from hermes_agent.plugins.platforms.telegram import standalone as send_message_senders  # 150996 moved _telegram_bot here

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

    from hermes_agent.gateway.platforms import base as gw_base

    monkeypatch.setitem(sys.modules, "aiohttp_socks", None)  # import raises ImportError
    with caplog.at_level(logging.WARNING):
        assert gw_base._aiohttp_socks_connector(SOCKS_PROXY) is None
    _assert_socks_warning_without_secret(caplog)


def test_matrix_socks_session_warning_logs_proxy_without_credentials(monkeypatch, caplog):
    import sys

    from hermes_agent.plugins.platforms.matrix import adapter as matrix_adapter

    monkeypatch.setitem(sys.modules, "aiohttp_socks", None)
    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(ClientSession=lambda **kw: kw))
    with caplog.at_level(logging.WARNING):
        matrix_adapter._create_matrix_session(SOCKS_PROXY)
    _assert_socks_warning_without_secret(caplog)


@pytest.mark.parametrize("proxy", [
    "http://alice:S3cretPass@[::1",  # unparseable (unclosed IPv6 bracket)
    "alice:S3cretPass@127.0.0.1:3128",  # no scheme
    "http:/alice:S3cretPass@127.0.0.1:3128",  # no authority
])
def test_malformed_proxy_url_never_echoes_credentials(monkeypatch, caplog, proxy):
    """A mistyped proxy URL reaches the connect log before anything validates it."""
    from hermes_agent.plugins.platforms.matrix import adapter as matrix_adapter

    monkeypatch.setenv("MATRIX_PROXY", proxy)
    with caplog.at_level(logging.INFO):
        matrix_adapter.MatrixAdapter(PlatformConfig(
            enabled=True, token="syt_test_token",
            extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}))
    lines = _proxy_lines(caplog)
    assert lines, "the adapter must still announce that a proxy is in use"
    assert all(SECRET not in line and "alice" not in line for line in lines), lines


TOKEN = "Zq9xK7tWv3"
# Every shape a credential can take in a proxy URL; ``{T}`` is where the secret goes.
SECRET_SHAPES = [
    "http://bob:{T}@proxy:8080",
    "http://{T}@proxy:8080",  # token as the username
    "http://bob:{T}ab/cd@proxy:8080",  # unencoded '/' in the password
    "http://bob:ab/{T}@proxy:8080",
    "socks5://bob:SEK?{T}@proxy:1080",  # unencoded '?'
    "socks5://bob:{T}?x@proxy:1080",
    "http://bob:ab#{T}@proxy:8080",  # unencoded '#'
    "http://bob:{T}#x@proxy:8080",
    "http://bob:a@b:{T}@proxy:8080",  # several '@'
    "http://bob:{T}@[::1]:3128",
    "http://bob:{T}@[::1",
    "bob:{T}@proxy:8080",
    "http:/bob:{T}@proxy:8080",
    "http://bob:{T}@proxy:8080/x?y#z",
    "http://bob:{T}",  # userinfo with no host
    "proxy:8080?auth={T}",  # no scheme, secret in the query
    "proxy:8080#{T}",
    "bob:{T}",  # no scheme, no '@': nothing parses as an authority
    "//bob:{T}",
    "http:bob:{T}",
    "bob:{T}/",
    "proxy:8080/{T}",
    "proxy:8080;{T}",
    "http://proxy:8080/x/{T}",  # secret as the last path segment
    "http://proxy:8080/{T}.json",  # secret shaped like a media file name
    "http://bob:" + "p" * 70 + "{T}@" + "h" * 60 + ".example:8080",  # past the 80-char cut
    "http://bob:" + "p" * 70 + "{T}/x@proxy:8080",  # token straddles the cut
]


def _assert_token_absent(text):
    # No 4+ char prefix either, so a token cut in half by truncation cannot leak.
    assert all(TOKEN[:n] not in text for n in range(4, len(TOKEN) + 1)), text


@pytest.mark.parametrize("shape", SECRET_SHAPES)
def test_proxy_secret_never_logged_in_any_shape(monkeypatch, caplog, shape):
    import sys

    import telegram
    import telegram.request
    from hermes_agent.gateway.platforms import base as gw_base
    from hermes_agent.plugins.platforms.matrix import adapter as matrix_adapter
    from hermes_agent.plugins.platforms.telegram import standalone as send_message_senders  # 150996 moved _telegram_bot here

    proxy = shape.format(T=TOKEN)
    monkeypatch.setenv("MATRIX_PROXY", proxy)
    monkeypatch.setenv("TELEGRAM_PROXY", proxy)
    monkeypatch.setitem(sys.modules, "aiohttp_socks", None)
    monkeypatch.setattr(sys.modules["telegram"], "Bot", lambda **kw: MagicMock())
    monkeypatch.setattr(sys.modules["telegram.request"], "HTTPXRequest", lambda **kw: kw)
    with caplog.at_level(logging.INFO):
        matrix_adapter.MatrixAdapter(PlatformConfig(
            enabled=True, token="syt_test_token",
            extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}))
        send_message_senders._telegram_bot("123456:TOKEN")
        gw_base._aiohttp_socks_connector("socks5://" + proxy.split("://", 1)[-1])
    assert _proxy_lines(caplog), "the adapter must still announce that a proxy is in use"
    _assert_token_absent("\n".join(r.getMessage() for r in caplog.records))


@pytest.mark.parametrize("status, chunks", [
    (500, []),  # HTTP error line
    (200, ['data: {"choices":[{"delta":{"content":"hi"}}]}\n\n']),  # no [DONE] + response line
    (None, []),  # connection error line
])
def test_gateway_proxy_mode_logs_upstream_url_without_credentials(monkeypatch, caplog, status, chunks):
    """``GATEWAY_PROXY_URL`` is logged on every proxied turn; its userinfo must not be."""
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.session import SessionSource

    monkeypatch.setenv("GATEWAY_PROXY_URL", f"http://bob:{TOKEN}@127.0.0.1:8642")
    monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
    runner = object.__new__(GatewayRunner)
    runner.adapters, runner._running_agents, runner._agent_cache = {}, {}, {}
    runner.config = MagicMock()

    class _Resp:
        def __init__(self):
            self.status, self.content = status, self

        async def text(self):
            return "boom"

        async def iter_any(self):
            for chunk in chunks:
                yield chunk.encode()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def post(self, url, **kw):
            if status is None:
                raise ConnectionError("refused")
            return _Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    aiohttp_stub = types.SimpleNamespace(ClientSession=lambda **kw: _Session(), ClientTimeout=MagicMock())
    source = SessionSource(platform=Platform.MATRIX, chat_id="!r:s", chat_type="group", user_id="@u:s")
    with patch.dict("sys.modules", {"aiohttp": aiohttp_stub}), \
            patch("hermes_agent.gateway.run._load_gateway_config", return_value={}), \
            caplog.at_level(logging.INFO):
        asyncio.run(runner._run_agent_via_proxy(
            message="hi", context_prompt="", history=[], source=source, session_id="s"))
    lines = [r.getMessage() for r in caplog.records if "127.0.0.1:8642" in r.getMessage()]
    assert lines, "the proxy-mode turn must still log which upstream it used"
    _assert_token_absent("\n".join(r.getMessage() for r in caplog.records))


@pytest.mark.parametrize("proxy_url", [
    f"http://bob:ab/{TOKEN}@127.0.0.1:1",  # unencoded '/' in the password
    f"http://bob:{TOKEN}#x@127.0.0.1:1",  # unencoded '#'
    f"http://bob:{TOKEN}@[::1",  # unclosed IPv6 bracket
    f"bob:{TOKEN}@127.0.0.1:1",  # no scheme
    f"bob:{TOKEN}",
    f"http://bob:{TOKEN}",  # userinfo with no host
])
def test_gateway_proxy_mode_malformed_url_error_never_echoes_secret(monkeypatch, caplog, proxy_url):
    """aiohttp's InvalidUrlClientError/NonHttpUrlClientError text is the whole request URL; the
    turn logs it and sends it to the chat as the error reply."""
    pytest.importorskip("aiohttp")
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.session import SessionSource

    monkeypatch.setenv("GATEWAY_PROXY_URL", proxy_url)
    monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
    runner = object.__new__(GatewayRunner)
    runner.adapters, runner._running_agents, runner._agent_cache = {}, {}, {}
    runner.config = MagicMock()
    source = SessionSource(platform=Platform.MATRIX, chat_id="!r:s", chat_type="group", user_id="@u:s")
    with patch("hermes_agent.gateway.run._load_gateway_config", return_value={}), caplog.at_level(logging.INFO):
        result = asyncio.run(runner._run_agent_via_proxy(
            message="hi", context_prompt="", history=[], source=source, session_id="s"))
    assert any("Proxy connection error" in r.getMessage() for r in caplog.records)
    assert result["final_response"].startswith("⚠️")
    _assert_token_absent(result["final_response"])
    _assert_token_absent("\n".join(r.getMessage() for r in caplog.records))


def test_telegram_connect_error_never_echoes_proxy_password(monkeypatch, caplog):
    """httpx rejects ``http://bob:<pw>#x@host`` with ``Invalid port: '<pw>'``; that text feeds the
    connect-failure log and the fatal-error status."""
    import httpx
    from hermes_agent.plugins.platforms.telegram import adapter as tg

    monkeypatch.setenv("TELEGRAM_PROXY", f"http://bob:{TOKEN}#x@127.0.0.1:3128")
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "true")
    adapter = tg.TelegramAdapter(PlatformConfig(enabled=True, token="123456:TOKEN"))
    with caplog.at_level(logging.INFO), \
            patch.object(tg, "HTTPXRequest", lambda **kw: httpx.AsyncClient(proxy=kw.get("proxy"))), \
            patch.object(adapter, "_acquire_platform_lock", return_value=True), \
            patch.object(adapter, "_write_runtime_status_safe"):
        assert asyncio.run(adapter.connect()) is False
    assert any("Failed to connect to Telegram" in r.getMessage() for r in caplog.records)
    _assert_token_absent(adapter._fatal_error_message or "")
    _assert_token_absent("\n".join(r.getMessage() for r in caplog.records))
