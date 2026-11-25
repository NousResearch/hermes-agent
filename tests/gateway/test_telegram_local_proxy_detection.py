"""Telegram adapter local-proxy discovery: a Clash/mihomo proxy on 127.0.0.1 is used only when
no proxy is configured. Configured values, environment values and NO_PROXY win over discovery.

Scope is the bundled plugin adapter's proxy-resolution path. The gateway's fallback transport stays
connect-error-only: a ReadTimeout is never retried through another IP because the Bot API may already
have received the send (PR #7028 review).
"""

import asyncio
import socket
import threading

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import telegram_network as tnet

PROXY_VARS = ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
              "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")


def _connect_server(status_line: bytes):
    """A loopback HTTP CONNECT listener that answers with a fixed status line."""
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(2.0)

    def serve():
        try:
            conn, _ = server.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(1.0)
            try:
                conn.recv(4096)
                conn.sendall(status_line + b"\r\n\r\n")
            except OSError:
                pass

    threading.Thread(target=serve, daemon=True).start()
    return server, server.getsockname()[1]


def test_detect_local_proxy_url_finds_a_connect_proxy():
    server, port = _connect_server(b"HTTP/1.1 200 Connection established")
    try:
        assert tnet.detect_local_proxy_url(ports=(port,)) == f"http://127.0.0.1:{port}"
    finally:
        server.close()


def test_detect_local_proxy_url_rejects_a_non_proxy_listener():
    server, port = _connect_server(b"HTTP/1.1 405 Method Not Allowed")
    try:
        assert tnet.detect_local_proxy_url(ports=(port,)) is None
    finally:
        server.close()


def test_detect_local_proxy_url_skips_closed_ports():
    sock = socket.create_server(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    assert tnet.detect_local_proxy_url(ports=(port,)) is None


def _adapter(monkeypatch, **extra):
    from plugins.platforms.telegram import adapter as tg

    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "true")
    adapter = tg.TelegramAdapter(PlatformConfig(enabled=True, token="123:abc", extra=extra))
    monkeypatch.setattr(adapter, "_instrument_polling_request", lambda request: request)
    return adapter


def _capture_requests(monkeypatch, captured):
    from plugins.platforms.telegram import adapter as tg

    def capture(_cls, *, fallback_ips, proxy_url, verify):
        captured["proxy_url"] = proxy_url
        return object(), object()

    monkeypatch.setattr(tg, "build_ptb_requests", capture)


def test_adapter_uses_detected_local_proxy_when_nothing_is_configured(monkeypatch):
    from plugins.platforms.telegram import adapter as tg

    adapter = _adapter(monkeypatch)
    calls = []
    monkeypatch.setattr(tg, "detect_local_proxy_url",
                        lambda: calls.append(True) or "http://127.0.0.1:7890")
    captured = {}
    _capture_requests(monkeypatch, captured)

    asyncio.run(adapter._build_ptb_requests())

    assert calls == [True]
    assert captured["proxy_url"] == "http://127.0.0.1:7890"
    assert adapter._ptb_proxy_url == "http://127.0.0.1:7890"


def test_configured_proxy_wins_over_local_discovery(monkeypatch):
    from plugins.platforms.telegram import adapter as tg

    adapter = _adapter(monkeypatch, proxy_url="http://configured.example:3128")
    monkeypatch.setattr(tg, "detect_local_proxy_url",
                        lambda: pytest.fail("local discovery must not run when a proxy is configured"))
    captured = {}
    _capture_requests(monkeypatch, captured)

    asyncio.run(adapter._build_ptb_requests())

    assert captured["proxy_url"] == "http://configured.example:3128"


def test_environment_proxy_wins_over_local_discovery(monkeypatch):
    from plugins.platforms.telegram import adapter as tg

    adapter = _adapter(monkeypatch)
    monkeypatch.setenv("TELEGRAM_PROXY", "socks5://env-proxy:1080")
    monkeypatch.setattr(tg, "detect_local_proxy_url",
                        lambda: pytest.fail("local discovery must not run when TELEGRAM_PROXY is set"))
    captured = {}
    _capture_requests(monkeypatch, captured)

    asyncio.run(adapter._build_ptb_requests())

    assert captured["proxy_url"] == "socks5://env-proxy:1080"


def test_no_proxy_bypass_skips_local_discovery(monkeypatch):
    from plugins.platforms.telegram import adapter as tg

    adapter = _adapter(monkeypatch)
    monkeypatch.setenv("NO_PROXY", "api.telegram.org")
    monkeypatch.setattr(tg, "detect_local_proxy_url",
                        lambda: pytest.fail("local discovery must not override NO_PROXY"))
    captured = {}
    _capture_requests(monkeypatch, captured)

    asyncio.run(adapter._build_ptb_requests())

    assert captured["proxy_url"] is None
