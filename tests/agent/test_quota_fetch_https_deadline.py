"""Dispatch 170 — real HTTPS aggregate deadline through the production transport.

Uses an actual localhost TLS server with a certificate trusted only by the
test. The real ``_DeadlineHTTPSHandler`` / ``_DeadlineHTTPSConnection`` path is
exercised (no synthetic TLS-timeout mock), so encrypted header parsing and body
reads are proven to obey the monotonic budget.
"""
from __future__ import annotations

import json
import socket
import ssl
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

import agent.quota_fetch as qf

SCOPE = "d" * 32
TOKEN = "synthetic-https-token-not-real"


@pytest.fixture(scope="module")
def tls_material(tmp_path_factory):
    directory = tmp_path_factory.mktemp("quota-tls")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("localhost"), x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1")),
        ]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path = directory / "cert.pem"
    key_path = directory / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(str(cert_path), str(key_path))
    client_ctx = ssl.create_default_context(cafile=str(cert_path))
    return server_ctx, client_ctx


class _TLSHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.requests.append(self.path)
        if self.server.handshake_delay:
            time.sleep(self.server.handshake_delay)
        raw = json.dumps({
            "data": {"free_model_daily_requests": {"limit": 10, "remaining": 6, "used": 4}}
        }).encode()
        head = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(raw)}\r\n".encode() + b"Connection: close\r\n\r\n")
        try:
            if self.server.mode == "fast":
                self.wfile.write(head + raw)
                self.wfile.flush()
            elif self.server.mode == "slow_body":
                self.wfile.write(head)
                self.wfile.flush()
                for offset in range(0, len(raw), 3):
                    self.wfile.write(raw[offset:offset + 3])
                    self.wfile.flush()
                    time.sleep(0.085)
            else:  # slow_headers
                payload = head + raw
                for offset in range(0, len(payload), 3):
                    self.wfile.write(payload[offset:offset + 3])
                    self.wfile.flush()
                    time.sleep(0.085)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.server.done.set()


class _TLSHTTPServer(ThreadingHTTPServer):
    """Wrap accepted sockets (never the listening socket) for TLS."""

    server_context: ssl.SSLContext | None = None

    def get_request(self):
        sock, address = super().get_request()
        try:
            return self.server_context.wrap_socket(sock, server_side=True), address
        except Exception:
            sock.close()
            raise


@pytest.fixture
def tls_server(tls_material, monkeypatch):
    server_ctx, client_ctx = tls_material
    server = _TLSHTTPServer(("127.0.0.1", 0), _TLSHandler)
    server.server_context = server_ctx
    server.requests = []
    server.done = threading.Event()
    server.mode = "slow_headers"
    server.handshake_delay = 0.0
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(qf, "OPENROUTER_KEY_URL", f"https://127.0.0.1:{server.server_port}/v1/key")
    monkeypatch.setattr(qf, "_OPENROUTER_HOSTS", {"localhost", "127.0.0.1"})
    monkeypatch.setattr(qf.ssl, "create_default_context", lambda: client_ctx)
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _fetch(timeout=0.5):
    started = time.monotonic()
    result = qf.fetch_openrouter_key_limits(TOKEN, credential_scope_id=SCOPE, timeout=timeout)
    return result, time.monotonic() - started


def test_https_slow_headers_respect_aggregate_deadline(tls_server):
    result, elapsed = _fetch()
    assert result["status"] == "unavailable"
    assert result["reason"] == "fetch_timeout"
    assert elapsed <= 1.2, f"encrypted header budget exceeded: {elapsed:.2f}s"
    assert len(tls_server.requests) == 1, "no retry after refusal"
    assert "limit" not in result and "remaining" not in result
    assert TOKEN not in json.dumps(result)


def test_https_slow_body_respects_aggregate_deadline(tls_server):
    tls_server.mode = "slow_body"
    result, elapsed = _fetch()
    assert result["reason"] == "fetch_timeout"
    assert elapsed <= 1.2, f"encrypted body budget exceeded: {elapsed:.2f}s"


def test_https_fast_response_returns_counters(tls_server):
    tls_server.mode = "fast"
    result, elapsed = _fetch(timeout=2.0)
    assert result["status"] == "ok"
    assert (result["limit"], result["remaining"], result["used"]) == (10, 6, 4)
    assert elapsed < 2.0


def test_tls_handshake_and_slow_headers_share_one_budget(tls_server):
    tls_server.handshake_delay = 0.2
    result, elapsed = _fetch()
    assert result["reason"] == "fetch_timeout"
    assert elapsed <= 1.2, f"handshake+headers exceeded budget: {elapsed:.2f}s"


def test_dns_delay_then_tls_headers_share_one_budget(tls_server, monkeypatch):
    original = socket.getaddrinfo

    def delayed(*args, **kwargs):
        time.sleep(0.2)
        return original(*args, **kwargs)

    monkeypatch.setattr(qf.socket, "getaddrinfo", delayed)
    result, elapsed = _fetch()
    assert result["reason"] == "fetch_timeout"
    assert elapsed <= 1.2, f"dns+tls exceeded budget: {elapsed:.2f}s"


def test_no_background_provider_work_after_refusal(tls_server):
    before = {t.ident for t in threading.enumerate()}
    result, _ = _fetch()
    assert result["reason"] == "fetch_timeout"
    assert tls_server.done.wait(3)
    for _ in range(30):
        if len(threading.enumerate()) <= len(before):
            break
        time.sleep(0.1)
    assert {t.ident for t in threading.enumerate()} - before == set()
    assert TOKEN not in json.dumps(result)
