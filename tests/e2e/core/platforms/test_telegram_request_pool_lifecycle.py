"""Telegram request pools never orphan a socket, and a failed path never kills requests in flight.

Drives the real ``build_ptb_requests`` output (PTB ``HTTPXRequest`` + ``TelegramFallbackTransport``
+ httpcore) against a local TLS server that holds each response briefly and then FINs idle
keepalive connections, the way Telegram's edge does. On the unfixed transport a
``shutdown()``/``initialize()`` drain or a failed connect closed the pool under requests still
connecting on it; their sockets then lived in no pool, and after the peer's FIN sat in CLOSE_WAIT
until the process exited (#142971, #70830, #69314). The same close also failed every request
already in flight on the retired pool (#96261).
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import ssl
import threading
import time
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address

import httpx
import pytest

pytest.importorskip("telegram")
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from telegram import Bot
from telegram.request import HTTPXRequest

import plugins.platforms.telegram.telegram_network as tnet

pytestmark = pytest.mark.platforms("linux")

_EDGE_IDLE_S = 0.4


def _tls_pair(tmp_path):
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "api.telegram.org")])
    now = datetime.now(UTC)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("api.telegram.org"),
                                                        x509.IPAddress(ip_address("127.0.0.1"))]), critical=False)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "edge.pem", tmp_path / "edge.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert_path, key_path)
    client = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client.load_verify_locations(cert_path)
    return server, client


class _Edge:
    """Bot API edge on 127.0.0.1: answers each request after ``delay`` s, FINs a keepalive
    connection idle for ``_EDGE_IDLE_S``, and resets the connection numbered ``reset_conn``."""

    def __init__(self, ctx: ssl.SSLContext, *, delay: float = 0.05, reset_conn: int | None = None) -> None:
        self._ctx, self._delay, self._reset_conn = ctx, delay, reset_conn
        self._sock = socket.create_server(("127.0.0.1", 0))
        self.port = self._sock.getsockname()[1]
        self.connections = 0
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._serve, args=(conn, self.connections), daemon=True).start()

    def _serve(self, conn: socket.socket, number: int) -> None:
        try:
            if number == self._reset_conn:
                time.sleep(0.05)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
                return
            tls = self._ctx.wrap_socket(conn, server_side=True)
            tls.settimeout(_EDGE_IDLE_S)
            while True:
                head = b""
                while b"\r\n\r\n" not in head:
                    chunk = tls.recv(65536)
                    if not chunk:
                        return
                    head += chunk
                headers, _, body = head.partition(b"\r\n\r\n")
                length = next((int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                               if line.lower().startswith(b"content-length:")), 0)
                while len(body) < length:
                    body += tls.recv(65536)
                time.sleep(self._delay)
                method = headers.split(b" ", 2)[1].rsplit(b"/", 1)[-1]
                result = ({"id": 1, "is_bot": True, "first_name": "edge", "username": "edge_bot"}
                          if method == b"getMe" else
                          {"message_id": 1, "date": 0, "chat": {"id": 5, "type": "private"}, "text": "x"})
                payload = json.dumps({"ok": True, "result": result}).encode()
                tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
                            % len(payload) + payload)
        except OSError:
            pass  # idle timeout or reset: closing the socket below is the edge's FIN
        finally:
            conn.close()

    def close(self) -> None:
        self._sock.close()


def _close_wait_fds() -> int:
    """Sockets this process holds whose peer already closed (TCP state CLOSE_WAIT)."""
    inodes = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    count = 0
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        with open(table) as handle:
            for line in handle.readlines()[1:]:
                fields = line.split()
                count += fields[9] in inodes and fields[3] == "08"
    return count


def _point_at_edge(monkeypatch, edge: _Edge) -> None:
    """Every TCP connect the transport makes to api.telegram.org or a fallback literal lands on the
    edge: the IPv4 literal path and the hostname path both reach 127.0.0.1:edge.port."""
    from httpcore._backends import anyio as anyio_backend

    real_connect = anyio_backend.AnyIOBackend.connect_tcp

    async def connect_tcp(self, host, port, *args, **kwargs):
        return await real_connect(self, "127.0.0.1", edge.port, *args, **kwargs)

    monkeypatch.setattr(anyio_backend.AnyIOBackend, "connect_tcp", connect_tcp)
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "TELEGRAM_PROXY"):
        monkeypatch.delenv(key, raising=False)


async def _settle_and_sweep(bot: Bot) -> int:
    """Wait past the edge's idle FIN, then send once more: httpcore closes every peer-closed idle
    socket still IN a pool on the next request, so any CLOSE_WAIT left is one no pool owns."""
    await asyncio.sleep(_EDGE_IDLE_S + 0.6)
    await bot.send_message(5, "sweep")
    await asyncio.sleep(0.1)
    return _close_wait_fds()


@pytest.fixture
def no_gc():
    import gc

    gc.collect()
    gc.disable()  # an orphaned socket is otherwise freed by a lucky collection, masking the leak
    yield
    gc.enable()
    gc.collect()


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback_ips", [["149.154.166.110", "149.154.167.220"], []], ids=["fallback", "direct"])
async def test_request_drain_mid_flight_orphans_no_socket(tmp_path, monkeypatch, no_gc, fallback_ips):
    server_ctx, client_ctx = _tls_pair(tmp_path)
    edge = _Edge(server_ctx)
    _point_at_edge(monkeypatch, edge)
    general, updates = tnet.build_ptb_requests(HTTPXRequest, fallback_ips=fallback_ips, proxy_url=None,
                                               verify=client_ctx)
    bot = Bot("123:abc", request=general, get_updates_request=updates)
    await bot.initialize()
    try:
        for _ in range(6):
            sends = [asyncio.ensure_future(bot.send_message(5, "x")) for _ in range(4)]
            await asyncio.sleep(0.01)  # the sends are mid-connect / mid-response
            # The adapter's pool-timeout / polling drain (_drain_general_connections_after_pool_timeout).
            await general.shutdown()
            await general.initialize()
            results = await asyncio.gather(*sends, return_exceptions=True)
            assert [r for r in results if isinstance(r, Exception)] == []
        assert await _settle_and_sweep(bot) == 0
    finally:
        await bot.shutdown()
        edge.close()


@pytest.mark.asyncio
async def test_one_failed_connect_keeps_requests_in_flight_on_that_path(tmp_path, monkeypatch, no_gc):
    """One reset connect on the sticky IPv4 path retires that pool; the requests already being
    answered on it still complete, and their sockets are not orphaned."""
    server_ctx, client_ctx = _tls_pair(tmp_path)
    edge = _Edge(server_ctx, delay=0.3, reset_conn=4)
    _point_at_edge(monkeypatch, edge)
    general, updates = tnet.build_ptb_requests(HTTPXRequest, fallback_ips=["149.154.166.110", "149.154.167.220"],
                                               proxy_url=None, verify=client_ctx)
    bot = Bot("123:abc", request=general, get_updates_request=updates)
    await bot.initialize()
    try:
        await bot.get_me()  # sticky path settles on the first literal
        results = await asyncio.gather(*[bot.send_message(5, "x") for _ in range(8)], return_exceptions=True)
        assert edge.connections > 4, "the reset connection was never made; the test did not exercise a failed path"
        assert [r for r in results if isinstance(r, Exception)] == []
        assert await _settle_and_sweep(bot) == 0
    finally:
        await bot.shutdown()
        edge.close()


def test_pool_size_override_fits_the_process_descriptor_budget(monkeypatch):
    """HERMES_TELEGRAM_HTTP_POOL_SIZE=512 across 2 clients × (fallback IPs + hostname) pools must not
    promise more sockets than a quarter of RLIMIT_NOFILE (#82678); a small override is kept."""
    import resource

    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    ips = ["149.154.166.110", "149.154.167.220", "149.154.167.221", "149.154.167.222", "149.154.167.223",
           "149.154.167.224"]
    built = []

    class _Capture:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_POOL_SIZE", "512")
    tnet.build_ptb_requests(_Capture, fallback_ips=ips, proxy_url=None, verify=False)
    transports = [b["httpx_kwargs"]["transport"] for b in built]
    paths = len(transports[0]._fallback_ips) + 1
    assert paths <= 5, "the fallback-address count is unbounded"
    per_pool = [t._transport_kwargs["limits"].max_connections for t in transports]
    assert sum(per_pool) * paths <= max(soft // 4, 4 * 2 * paths)

    built.clear()
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_POOL_SIZE", "4")
    tnet.build_ptb_requests(_Capture, fallback_ips=ips, proxy_url=None, verify=False)
    assert all(b["httpx_kwargs"]["transport"]._transport_kwargs["limits"].max_connections == 4 for b in built)


@pytest.mark.asyncio
async def test_aclosed_transport_serves_the_next_request_on_a_fresh_pool(monkeypatch):
    """PTB wraps its rebuilt client around the same transport after ``shutdown()``: the request
    after ``aclose()`` must not reach the retired pool."""
    served = []

    class _Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"{}"

    class _Inner(httpx.AsyncBaseTransport):
        def __init__(self, **_kwargs):
            self.closed = False

        async def handle_async_request(self, request):
            assert not self.closed, "request reached a retired pool"
            served.append(self)
            return httpx.Response(200, request=request, stream=_Body())

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", _Inner)
    from plugins.platforms.telegram import telegram_pool
    monkeypatch.setattr(telegram_pool.httpx, "AsyncHTTPTransport", _Inner)
    transport = tnet.TelegramFallbackTransport(["149.154.166.110"], proxy=None)
    request = httpx.Request("POST", "https://api.telegram.org/bot1/getMe")
    first = await transport.handle_async_request(request)
    await transport.aclose()  # retired while its response is still open: stays usable until closed
    assert not served[0].closed
    await first.aclose()
    await asyncio.sleep(0)
    assert served[0].closed
    await (await transport.handle_async_request(request)).aclose()
    assert len(served) == 2 and served[1] is not served[0]
