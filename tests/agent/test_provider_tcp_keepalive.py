"""Provider sockets carry TCP keepalive so a silently dead connection fails its read.

A VPN/tunnel restart (or a network switch) leaves a streaming socket ESTABLISHED with nothing
ever arriving: the new tunnel has no state for the flow and a reading client never sends. The
request waited for the provider-silence watchdogs (240-300 s). With keepalive the first probe
after the idle time draws a reset (or goes unanswered) and the stream retry reconnects.
"""

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpcore
import pytest

from agent import process_bootstrap
from agent.process_bootstrap import build_keepalive_http_client, enable_tcp_keepalive

_IDLE_OPTION = getattr(socket, "TCP_KEEPIDLE", None)
if _IDLE_OPTION is None:
    _IDLE_OPTION = getattr(socket, "TCP_KEEPALIVE", None)


@pytest.fixture
def no_proxy_env(monkeypatch):
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy",
                 "NO_PROXY", "no_proxy", "HERMES_TCP_KEEPALIVE_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    process_bootstrap.close_shared_transports()
    yield
    process_bootstrap.close_shared_transports()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def connected_socket(local_server):
    sock = socket.create_connection(local_server)
    try:
        yield sock
    finally:
        sock.close()


def _keepalive(sock):
    return sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0


def _tcp_option(sock, option):
    """The option's value, or None where the platform lacks it or cannot read it back."""
    if option is None:
        return None
    try:
        return sock.getsockopt(socket.IPPROTO_TCP, option)
    except OSError:
        return None


def test_enable_tcp_keepalive_sets_probe_timing(no_proxy_env, connected_socket):
    assert enable_tcp_keepalive(connected_socket) is True
    assert _keepalive(connected_socket)
    assert _tcp_option(connected_socket, _IDLE_OPTION) in (None, 30)
    assert _tcp_option(connected_socket, getattr(socket, "TCP_KEEPINTVL", None)) in (None, 10)
    assert _tcp_option(connected_socket, getattr(socket, "TCP_KEEPCNT", None)) in (None, 3)


def test_idle_is_tunable_and_zero_disables(no_proxy_env, monkeypatch, local_server):
    monkeypatch.setenv("HERMES_TCP_KEEPALIVE_SECONDS", "45")
    with socket.create_connection(local_server) as sock:
        assert enable_tcp_keepalive(sock) is True
        assert _tcp_option(sock, _IDLE_OPTION) in (None, 45)
    monkeypatch.setenv("HERMES_TCP_KEEPALIVE_SECONDS", "0")
    with socket.create_connection(local_server) as sock:
        assert enable_tcp_keepalive(sock) is False
        assert not _keepalive(sock)


class _PickySocket:
    """Records options; rejects the ones named in ``reject`` like a platform that lacks them."""

    def __init__(self, reject):
        self.reject, self.options = reject, []

    def setsockopt(self, level, option, value):
        if (level, option) in self.reject:
            raise OSError(22, "Invalid argument")
        self.options.append((level, option, value))


def test_rejected_tcp_timing_options_keep_so_keepalive(no_proxy_env):
    tcp_options = {(socket.IPPROTO_TCP, opt) for opt in (_IDLE_OPTION, getattr(socket, "TCP_KEEPINTVL", None),
                                                          getattr(socket, "TCP_KEEPCNT", None)) if opt is not None}
    sock = _PickySocket(reject=tcp_options)
    assert enable_tcp_keepalive(sock) is True
    assert sock.options == [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]


def test_rejected_so_keepalive_never_raises(no_proxy_env):
    sock = _PickySocket(reject={(socket.SOL_SOCKET, socket.SO_KEEPALIVE)})
    assert enable_tcp_keepalive(sock) is False
    assert sock.options == []


def test_provider_client_connections_carry_keepalive(no_proxy_env, local_server):
    """The shared sync transport every agent's OpenAI client uses (chat-completions providers)."""
    host, port = local_server
    client = build_keepalive_http_client(f"http://{host}:{port}/v1")
    try:
        # Still httpcore's default backend underneath (no connect racing for other providers).
        backends = [t._pool._network_backend for t in (client._transport, *client._mounts.values())
                    if t is not None and hasattr(t, "_pool")]
        assert backends and all(isinstance(b, httpcore.SyncBackend) for b in backends)
        with client.stream("GET", f"http://{host}:{port}/ping") as response:
            assert _keepalive(response.extensions["network_stream"].get_extra_info("socket"))
            assert response.read() == b"ok"
    finally:
        client.close()


def test_codex_racing_backend_sets_keepalive(no_proxy_env, local_server):
    host, port = local_server
    stream = process_bootstrap._HappyEyeballsSyncBackend().connect_tcp(host, port, timeout=5)
    try:
        assert _keepalive(stream.get_extra_info("socket"))
    finally:
        stream.close()


def test_keepalive_failure_never_fails_the_connect(no_proxy_env, monkeypatch, local_server):
    monkeypatch.setattr(process_bootstrap, "enable_tcp_keepalive",
                        lambda _sock: (_ for _ in ()).throw(RuntimeError("boom")))
    host, port = local_server
    stream = process_bootstrap._keepalive_sync_backend().connect_tcp(host, port, timeout=5)
    stream.close()


def _proxied_client(monkeypatch, local_server, *, async_mode=False):
    """A provider client behind HTTPS_PROXY/HTTP_PROXY. The local server plays the forward
    proxy: an absolute-form ``GET http://provider.invalid/...`` is answered like any other."""
    host, port = local_server
    monkeypatch.setenv("HTTP_PROXY", f"http://{host}:{port}")
    monkeypatch.setenv("HTTPS_PROXY", f"http://{host}:{port}")
    client = build_keepalive_http_client("http://provider.invalid/v1", async_mode=async_mode)
    assert client is not None
    assert any(isinstance(t._pool, httpcore.HTTPProxy if not async_mode else httpcore.AsyncHTTPProxy)
               for t in client._mounts.values()), "not a proxied client"
    return client


def test_proxied_client_puts_keepalive_on_the_proxy_socket(no_proxy_env, monkeypatch, local_server):
    """Behind a proxy the client <-> proxy socket is the one a dropped tunnel leaves dead."""
    client = _proxied_client(monkeypatch, local_server)
    try:
        with client.stream("GET", "http://provider.invalid/ping") as response:
            assert response.read() == b"ok"
            assert _keepalive(response.extensions["network_stream"].get_extra_info("socket"))
    finally:
        client.close()


def test_async_provider_client_connections_carry_keepalive(no_proxy_env, local_server):
    """The auxiliary client's async transports (direct, not shared)."""
    import asyncio

    host, port = local_server

    async def _run():
        client = build_keepalive_http_client(f"http://{host}:{port}/v1", async_mode=True)
        try:
            async with client.stream("GET", f"http://{host}:{port}/ping") as response:
                assert await response.aread() == b"ok"
                return _keepalive(response.extensions["network_stream"].get_extra_info("socket"))
        finally:
            await client.aclose()

    assert asyncio.run(_run())


def test_async_proxied_client_puts_keepalive_on_the_proxy_socket(no_proxy_env, monkeypatch, local_server):
    import asyncio

    client = _proxied_client(monkeypatch, local_server, async_mode=True)

    async def _run():
        try:
            async with client.stream("GET", "http://provider.invalid/ping") as response:
                assert await response.aread() == b"ok"
                return _keepalive(response.extensions["network_stream"].get_extra_info("socket"))
        finally:
            await client.aclose()

    assert asyncio.run(_run())
