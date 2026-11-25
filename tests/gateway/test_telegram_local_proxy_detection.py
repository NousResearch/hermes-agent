"""Telegram adapter local-proxy discovery: a Clash/mihomo proxy on 127.0.0.1 is used only when
no proxy is configured. Configured values, environment values and NO_PROXY win over discovery.

Scope is the bundled plugin adapter's proxy-resolution path. The gateway's fallback transport stays
connect-error-only: a ReadTimeout is never retried through another IP because the Bot API may already
have received the send (PR #7028 review).
"""

import asyncio
import socket
import threading
import time

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


def _stalled_connect_server(release):
    """A loopback listener that accepts CONNECT but never answers it until released."""
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
                release.wait(2.0)
            except OSError:
                pass

    threading.Thread(target=serve, daemon=True).start()
    return server, server.getsockname()[1]


def _fragmented_connect_server(chunks):
    """A loopback listener that sends a status line in several TCP writes."""
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
                for chunk in chunks:
                    conn.sendall(chunk)
                    time.sleep(0.02)  # health: allow HX013 -- the delay is the behaviour under test: the status line arrives in separate TCP segments
            except OSError:
                pass

    threading.Thread(target=serve, daemon=True).start()
    return server, server.getsockname()[1]


def test_custom_base_url_skips_local_discovery(monkeypatch):
    """A custom Bot API server stays direct unless a proxy is explicitly configured."""
    from plugins.platforms.telegram import adapter as tg

    adapter = _adapter(monkeypatch, base_url="http://127.0.0.1:8081")
    monkeypatch.setattr(tg, "detect_local_proxy_url",
                        lambda: pytest.fail("local discovery must not run for a custom base_url"))
    captured = {}
    _capture_requests(monkeypatch, captured)

    asyncio.run(adapter._build_ptb_requests())

    assert captured["proxy_url"] is None
    assert adapter._ptb_proxy_url is None


def test_explicit_public_base_url_still_allows_local_discovery(monkeypatch):
    """Only custom endpoints skip discovery; the public Bot API keeps the new path."""
    from plugins.platforms.telegram import adapter as tg

    adapter = _adapter(monkeypatch, base_url="https://api.telegram.org")
    calls = []
    monkeypatch.setattr(tg, "detect_local_proxy_url",
                        lambda: calls.append(True) or "http://127.0.0.1:7890")
    captured = {}
    _capture_requests(monkeypatch, captured)

    asyncio.run(adapter._build_ptb_requests())

    assert calls == [True]
    assert captured["proxy_url"] == "http://127.0.0.1:7890"


def test_custom_base_url_with_configured_proxy_keeps_configured_proxy(monkeypatch):
    from plugins.platforms.telegram import adapter as tg

    adapter = _adapter(monkeypatch, base_url="http://127.0.0.1:8081",
                       proxy_url="http://configured.example:3128")
    monkeypatch.setattr(tg, "detect_local_proxy_url",
                        lambda: pytest.fail("local discovery must not run when a proxy is configured"))
    captured = {}
    _capture_requests(monkeypatch, captured)

    asyncio.run(adapter._build_ptb_requests())

    assert captured["proxy_url"] == "http://configured.example:3128"


def test_stalled_local_probe_does_not_block_the_event_loop(monkeypatch):
    """Other coroutines keep running while a local proxy port stalls on CONNECT."""
    from plugins.platforms.telegram import adapter as tg

    release = threading.Event()
    server, port = _stalled_connect_server(release)
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(tg, "detect_local_proxy_url",
                        lambda: tnet.detect_local_proxy_url(ports=(port,)))
    captured = {}
    _capture_requests(monkeypatch, captured)

    async def scenario():
        loop = asyncio.get_running_loop()
        start = loop.time()
        ticks = []
        loop.call_later(0.05, lambda: ticks.append(loop.time() - start))
        await adapter._build_ptb_requests()
        return ticks, loop.time() - start

    try:
        ticks, elapsed = asyncio.run(scenario())
    finally:
        release.set()
        server.close()

    assert ticks, "the 50 ms timer never ran"
    assert ticks[0] < 0.2, f"event loop was blocked for {ticks[0]:.3f}s by the stalled probe"
    assert elapsed < 1.0, f"discovery took {elapsed:.3f}s without a bounded deadline"


def test_local_proxy_discovery_has_a_bounded_overall_deadline(monkeypatch):
    """A probe that never answers cannot hold up request construction past its deadline."""
    from plugins.platforms.telegram import adapter as tg

    adapter = _adapter(monkeypatch)
    monkeypatch.setenv("HERMES_TELEGRAM_LOCAL_PROXY_DISCOVERY_TIMEOUT", "0.05")
    release = threading.Event()

    def stalled_probe():
        release.wait(2.0)

    monkeypatch.setattr(tg, "detect_local_proxy_url", stalled_probe)
    captured = {}
    _capture_requests(monkeypatch, captured)

    async def scenario():
        loop = asyncio.get_running_loop()
        start = loop.time()
        try:
            await adapter._build_ptb_requests()
        finally:
            release.set()
        return loop.time() - start

    elapsed = asyncio.run(scenario())

    assert elapsed < 0.5, f"discovery took {elapsed:.3f}s despite a 0.05s deadline"
    assert captured["proxy_url"] is None


@pytest.mark.parametrize("status_line", [
    b"HTTP/1.1 2not-a-status",
    b"HTTP/1.1 999 OK",
    b"HTTP/9 200 OK",
])
def test_detect_local_proxy_url_rejects_malformed_connect_status(status_line):
    server, port = _connect_server(status_line)
    try:
        assert tnet.detect_local_proxy_url(ports=(port,)) is None
    finally:
        server.close()


def test_detect_local_proxy_url_reads_fragmented_status_line():
    server, port = _fragmented_connect_server([b"HTTP/", b"1.1 20", b"0 OK\r\n"])
    try:
        assert tnet.detect_local_proxy_url(ports=(port,)) == f"http://127.0.0.1:{port}"
    finally:
        server.close()
