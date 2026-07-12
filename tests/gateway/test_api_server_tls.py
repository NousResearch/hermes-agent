"""Security regression tests for the API server's optional TLS listener."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import ipaddress
import os
from pathlib import Path
import ssl

import aiohttp
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
import pytest

from gateway.config import PlatformConfig
from gateway.platforms import api_server
from gateway.platforms.api_server import APIServerAdapter


def _adapter(**extra):
    config = {
        "host": "127.0.0.1",
        "port": 0,
        "key": "this-is-a-long-enough-test-api-key",
        "ssl": True,
        **extra,
    }
    return APIServerAdapter(PlatformConfig(enabled=True, extra=config))


def test_tls_default_paths_are_profile_scoped(tmp_path, monkeypatch):
    hermes_home = tmp_path / "profile-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    adapter = _adapter()

    assert adapter._ssl_certfile == str(
        hermes_home / "certs" / "hermes-tailscale.crt"
    )
    assert adapter._ssl_keyfile == str(
        hermes_home / "certs" / "hermes-tailscale.key"
    )


def test_tls_context_enforces_tls_1_2_minimum(monkeypatch):
    class FakeContext:
        minimum_version = None
        loaded = None

        def load_cert_chain(self, certfile, keyfile):
            self.loaded = (certfile, keyfile)

    fake = FakeContext()
    monkeypatch.setattr(ssl, "create_default_context", lambda purpose: fake)

    context = api_server._create_server_ssl_context(
        Path("server.crt"), Path("server.key")
    )

    assert context is fake
    assert fake.minimum_version is ssl.TLSVersion.TLSv1_2
    assert fake.loaded == ("server.crt", "server.key")


@pytest.mark.asyncio
async def test_tls_listener_serves_health_over_https(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]
    )
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "server.crt"
    key_path = tmp_path / "server.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    os.chmod(key_path, 0o600)

    adapter = _adapter(
        ssl_certfile=str(cert_path),
        ssl_keyfile=str(key_path),
    )
    assert await adapter.connect() is True
    try:
        assert adapter._site is not None
        server = adapter._site._server
        assert server is not None and server.sockets
        port = server.sockets[0].getsockname()[1]
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://127.0.0.1:{port}/health", ssl=False
            ) as response:
                assert response.status == 200
                assert (await response.json())["status"] == "ok"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_tls_refuses_missing_certificate_files(tmp_path):
    adapter = _adapter(
        ssl_certfile=str(tmp_path / "missing.crt"),
        ssl_keyfile=str(tmp_path / "missing.key"),
    )
    assert await adapter.connect() is False
    assert adapter._runner is None


@pytest.mark.asyncio
async def test_tls_refuses_group_or_world_readable_private_key(tmp_path, monkeypatch):
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    cert.write_text("certificate")
    key.write_text("private key")
    os.chmod(key, 0o644)

    adapter = _adapter(
        ssl_certfile=str(cert),
        ssl_keyfile=str(key),
    )

    class FailIfLoaded:
        def load_cert_chain(self, *args, **kwargs):
            pytest.fail("an over-permissive key must be rejected before loading")

    monkeypatch.setattr(ssl, "create_default_context", lambda *a, **k: FailIfLoaded())
    assert await adapter.connect() is False
    assert adapter._runner is None
