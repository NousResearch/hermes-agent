"""Telegram's opt-in classic key exchange survives a network path that resets large ClientHellos.

OpenSSL 3.5 sends a post-quantum hybrid key share by default, which makes the TLS ClientHello
about 1.5 KB; some network paths reset a hello that large, and every Bot API path then fails with
an empty ``ConnectError`` in a loop (#147179). A local TLS server here resets any first record over
1,400 bytes, and the real adapter request builder drives a real PTB ``get_me`` through it.
"""

from __future__ import annotations

import json
import select
import socket
import ssl
import struct
import threading
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address

import pytest

pytest.importorskip("telegram")
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from telegram import Bot

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

_RESET_ABOVE = 1400


def _default_hello_size() -> int:
    """Bytes in this OpenSSL build's default TLS ClientHello (large only with the PQ hybrid share)."""
    from agent.ssl_verify import platform_ssl_context

    incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
    tls = platform_ssl_context().wrap_bio(incoming, outgoing, server_hostname="127.0.0.1")
    with pytest.raises(ssl.SSLWantReadError):
        tls.do_handshake()
    return len(outgoing.read())


def _require_large_default_hello() -> None:
    if _default_hello_size() <= _RESET_ABOVE:
        pytest.skip("this OpenSSL sends no post-quantum key share by default; nothing for the path to reset")


class _HelloSizeEdge:
    """TLS edge on 127.0.0.1 that resets a connection whose ClientHello record exceeds
    ``_RESET_ABOVE`` bytes and answers ``getMe`` otherwise."""

    def __init__(self, tmp_path) -> None:
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
        now = datetime.now(UTC)
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
                .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip_address("127.0.0.1"))]), critical=False)
                .sign(key, hashes.SHA256()))
        self.cert_path = tmp_path / "edge.pem"
        key_path = tmp_path / "edge.key"
        self.cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
        self._ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._ctx.load_cert_chain(self.cert_path, key_path)
        self._sock = socket.create_server(("127.0.0.1", 0))
        self.port = self._sock.getsockname()[1]
        self.hello_sizes: list[int] = []
        self.answered = 0
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(5)
            header = conn.recv(5, socket.MSG_PEEK)
            size = struct.unpack("!H", header[3:5])[0] if len(header) == 5 else 0
            self.hello_sizes.append(size)
            if size > _RESET_ABOVE:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                return
            tls = self._ctx.wrap_socket(conn, server_side=True)
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = tls.recv(65536)
                if not chunk:
                    return
                request += chunk
            body = json.dumps({"ok": True, "result": {"id": 1, "is_bot": True, "first_name": "edge",
                                                      "username": "edge_bot"}}).encode()
            tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n"
                        b"Connection: close\r\n\r\n" % len(body) + body)
            self.answered += 1
        except OSError:
            pass
        finally:
            conn.close()

    def close(self) -> None:
        self._sock.close()


class _HelloSizeProxy(_HelloSizeEdge):
    """HTTPS CONNECT proxy on 127.0.0.1 with the same reset rule for ITS OWN TLS hello, serving
    ``edge``'s certificate (one trusted cert for both legs), then tunnelling to the target."""

    def __init__(self, tmp_path, edge: _HelloSizeEdge) -> None:
        proxy_dir = tmp_path / "proxy"
        proxy_dir.mkdir()
        super().__init__(proxy_dir)
        self._ctx.load_cert_chain(edge.cert_path, edge.cert_path.with_name("edge.key"))

    def _serve(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(5)
            header = conn.recv(5, socket.MSG_PEEK)
            size = struct.unpack("!H", header[3:5])[0] if len(header) == 5 else 0
            self.hello_sizes.append(size)
            if size > _RESET_ABOVE:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                return
            tls = self._ctx.wrap_socket(conn, server_side=True)
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = tls.recv(65536)
                if not chunk:
                    return
                request += chunk
            host, port = request.split(b" ")[1].decode().rsplit(":", 1)
            with socket.create_connection((host, int(port)), timeout=5) as remote:
                tls.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                while True:
                    ready, _, _ = select.select([tls, remote], [], [], 5)
                    if not ready:
                        return
                    for current in ready:
                        data = current.recv(65536)
                        if not data:
                            return
                        (remote if current is tls else tls).sendall(data)
        except OSError:
            pass
        finally:
            conn.close()


async def _get_me_through_adapter(edge: _HelloSizeEdge, monkeypatch, extra: dict):
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "TELEGRAM_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", str(edge.cert_path))  # trusted on top of the platform store
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", "3")
    base_url = f"https://127.0.0.1:{edge.port}/bot"
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:abc", extra={
        "base_url": base_url, "base_file_url": f"https://127.0.0.1:{edge.port}/file/bot",
        "fallback_ips": ["149.154.166.110"], **extra}))
    general, updates = await adapter._build_ptb_requests()
    bot = Bot("123:abc", request=general, get_updates_request=updates, base_url=base_url)
    try:
        return await bot.get_me()
    finally:
        await bot.shutdown()


@pytest.mark.asyncio
async def test_classic_key_exchange_handshakes_where_the_default_hello_is_reset(tmp_path, monkeypatch):
    edge = _HelloSizeEdge(tmp_path)
    try:
        if _default_hello_size() > _RESET_ABOVE:  # the failing leg needs OpenSSL's PQ default
            with pytest.raises(Exception, match="ConnectError"):
                await _get_me_through_adapter(edge, monkeypatch, {})
            assert edge.hello_sizes and min(edge.hello_sizes) > _RESET_ABOVE
            edge.hello_sizes.clear()

        me = await _get_me_through_adapter(edge, monkeypatch, {"tls_classic_key_exchange": True})
        assert me.username == "edge_bot"
        assert edge.answered == 1
        assert max(edge.hello_sizes) <= _RESET_ABOVE
    finally:
        edge.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["sometimes", 1])
async def test_unrecognised_setting_keeps_the_default_handshake(tmp_path, monkeypatch, caplog, value):
    _require_large_default_hello()
    edge = _HelloSizeEdge(tmp_path)
    try:
        with pytest.raises(Exception, match="ConnectError"):
            await _get_me_through_adapter(edge, monkeypatch, {"tls_classic_key_exchange": value})
        assert min(edge.hello_sizes) > _RESET_ABOVE
        assert "tls_classic_key_exchange" in caplog.text and "expected true or false" in caplog.text
    finally:
        edge.close()


@pytest.mark.asyncio
async def test_classic_key_exchange_reaches_the_https_proxy_handshake(tmp_path, monkeypatch):
    """An ``https://`` proxy has its own outer TLS handshake; the opt-in covers it too, so both the
    proxy's and the Bot API's hello stay small and ``get_me`` goes through the tunnel."""
    edge = _HelloSizeEdge(tmp_path)
    proxy = _HelloSizeProxy(tmp_path, edge)
    try:
        me = await _get_me_through_adapter(edge, monkeypatch, {
            "tls_classic_key_exchange": True, "proxy_url": f"https://127.0.0.1:{proxy.port}"})
        assert me.username == "edge_bot"
        assert proxy.hello_sizes and max(proxy.hello_sizes) <= _RESET_ABOVE
        assert edge.answered == 1 and max(edge.hello_sizes) <= _RESET_ABOVE
    finally:
        proxy.close()
        edge.close()
