"""``network.tls_max_version`` caps the TLS handshake on every keepalive client.

Some CDN edges and middleboxes accept TLS 1.2 handshakes but kill TLS 1.3
ClientHellos, surfacing as ``[SSL: UNEXPECTED_EOF_WHILE_READING]`` ~15s into
every request while ``curl`` (OS TLS stack) works fine — #44365 hit this on
Windows desktop against DeepSeek's edge. ``network.tls_max_version: "1.2"`` in
config.yaml caps the handshake in ``build_keepalive_http_client``, which both
the primary (``AIAgent._build_keepalive_http_client``) and auxiliary clients
go through.

The cap must compose with the resolved ``verify`` value (per-provider
``ssl_ca_cert`` context, platform-store context, ``True``, ``False``) and must
not break process-wide transport sharing, which keys pools by the ``verify``
object's identity.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
import ssl
from threading import Thread

import _ssl

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
import httpx
import pytest
from truststore._ssl_constants import _original_SSLContext

from agent import process_bootstrap, ssl_verify
from agent.process_bootstrap import (
    _apply_tls_max_version,
    _configured_tls_max_version,
    build_keepalive_http_client,
)

_PROXY_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
               "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    # Capping is in place on shared contexts; never leak a capped one into other tests.
    monkeypatch.setattr(ssl_verify, "_CA_CONTEXTS", {})
    monkeypatch.setattr(process_bootstrap, "_CAPPED_DEFAULT_CONTEXTS", {})
    process_bootstrap.close_shared_transports()
    yield
    process_bootstrap.close_shared_transports()


@pytest.fixture
def tls_cap(monkeypatch):
    """Set ``network.tls_max_version`` as config.yaml would deliver it."""
    def _set(value):
        monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                            lambda: {"network": {"force_ipv4": False, "tls_max_version": value}})
    return _set


def _pool_ssl_contexts(client: httpx.Client):
    contexts = []
    for transport in (client._transport, *client._mounts.values()):
        pool = getattr(transport, "_pool", None) if transport is not None else None
        ctx = getattr(pool, "_ssl_context", None)
        if ctx is not None:
            contexts.append(ctx)
    return contexts


# ─── parsing the config value ───────────────────────────────────────────────


@pytest.mark.parametrize("value", ["", None, False])
def test_unset_means_no_cap(tls_cap, value):
    tls_cap(value)
    assert _configured_tls_max_version() is None


def test_missing_network_section_means_no_cap(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", dict)
    assert _configured_tls_max_version() is None


def test_config_read_failure_means_no_cap(monkeypatch):
    def boom():
        raise RuntimeError("config unreadable")
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", boom)
    assert _configured_tls_max_version() is None


@pytest.mark.parametrize("value", ["1.2", "TLSv1.2", "tls1.2", "TLS1.2", "  1.2  ", 1.2])
def test_accepts_1_2_spellings(tls_cap, value):
    # An unquoted ``tls_max_version: 1.2`` parses as a YAML float.
    tls_cap(value)
    assert _configured_tls_max_version() == ssl.TLSVersion.TLSv1_2


def test_accepts_1_3(tls_cap):
    tls_cap("1.3")
    assert _configured_tls_max_version() == ssl.TLSVersion.TLSv1_3


@pytest.mark.parametrize("value", ["1.1", "1.0", "garbage", "ssl3"])
def test_invalid_and_deprecated_values_are_ignored(tls_cap, value):
    # 1.0/1.1 are rejected: the knob dodges broken TLS 1.3 paths, it does not
    # re-enable deprecated protocol versions.
    tls_cap(value)
    assert _configured_tls_max_version() is None


def test_default_config_ships_cap_off():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["network"]["tls_max_version"] == ""


# ─── composing with the resolved verify value ───────────────────────────────


def test_no_cap_leaves_verify_untouched(tls_cap):
    tls_cap("")
    existing = ssl.create_default_context()
    assert _apply_tls_max_version(True) is True
    assert _apply_tls_max_version(False) is False
    assert _apply_tls_max_version(existing) is existing
    assert existing.maximum_version == ssl.TLSVersion.MAXIMUM_SUPPORTED


def test_true_becomes_a_stable_capped_default_context(tls_cap):
    tls_cap("1.2")
    first = _apply_tls_max_version(True)
    assert isinstance(first, ssl.SSLContext)
    assert first.maximum_version == ssl.TLSVersion.TLSv1_2
    assert first.verify_mode == ssl.CERT_REQUIRED and first.check_hostname
    # Same identity on every call, so transports keyed by it stay shared.
    assert _apply_tls_max_version(True) is first


def test_existing_context_is_capped_in_place(tls_cap):
    tls_cap("1.2")
    provided = ssl.create_default_context()
    result = _apply_tls_max_version(provided)
    assert result is provided  # CA material and identity survive; only the ceiling moves
    assert result.maximum_version == ssl.TLSVersion.TLSv1_2


def test_verify_false_is_left_alone(tls_cap):
    tls_cap("1.2")
    assert _apply_tls_max_version(False) is False


def test_provider_ca_context_is_capped_despite_a_broken_ambient_ca(tls_cap, tmp_path, monkeypatch):
    """Sweeper regression: capping a provider's own CA context must not depend on
    any other CA lookup succeeding (a stale ambient bundle used to leave it uncapped)."""
    import certifi

    for variable in ("HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        monkeypatch.setenv(variable, str(tmp_path / "missing-ca.pem"))
    tls_cap("1.2")
    provided = ssl_verify.resolve_httpx_verify(ca_bundle=certifi.where())
    # A provider bundle context is the stdlib class, not truststore's injected one.
    assert isinstance(provided, _original_SSLContext)
    assert _apply_tls_max_version(provided) is provided
    assert provided.maximum_version == ssl.TLSVersion.TLSv1_2


def test_platform_store_context_keeps_truststore_and_is_capped(tls_cap, monkeypatch):
    monkeypatch.setenv("SSL_CERT_DIR", "/nonexistent-ca-dir")
    tls_cap("1.2")
    verify = ssl_verify.resolve_httpx_verify()
    assert isinstance(verify, ssl.SSLContext)
    capped = _apply_tls_max_version(verify)
    assert capped is verify
    assert type(capped).__module__.startswith("truststore")
    assert capped.maximum_version == ssl.TLSVersion.TLSv1_2


# ─── end-to-end through the keepalive client builder ────────────────────────


def test_keepalive_client_pools_get_the_cap(tls_cap):
    from run_agent import AIAgent

    tls_cap("1.2")
    client = AIAgent._build_keepalive_http_client("https://api.deepseek.com")
    assert isinstance(client, httpx.Client)
    try:
        contexts = _pool_ssl_contexts(client)
        assert contexts, "expected the keepalive transport to expose an ssl context"
        assert all(ctx.maximum_version == ssl.TLSVersion.TLSv1_2 for ctx in contexts)
    finally:
        client.close()


def test_async_client_pools_get_the_cap(tls_cap):
    tls_cap("1.2")
    client = build_keepalive_http_client("https://api.deepseek.com", async_mode=True)
    assert isinstance(client, httpx.AsyncClient)
    contexts = [getattr(getattr(m, "_pool", None), "_ssl_context", None) for m in client._mounts.values()]
    contexts = [c for c in contexts if c is not None]
    assert contexts and all(c.maximum_version == ssl.TLSVersion.TLSv1_2 for c in contexts)


def test_proxy_mount_gets_the_cap(tls_cap, monkeypatch):
    tls_cap("1.2")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    client = build_keepalive_http_client("https://api.deepseek.com")
    assert isinstance(client, httpx.Client)
    try:
        pools = [type(m._pool).__name__ for m in client._mounts.values() if m is not None and hasattr(m, "_pool")]
        assert "HTTPProxy" in pools, pools
        contexts = _pool_ssl_contexts(client)
        assert contexts and all(c.maximum_version == ssl.TLSVersion.TLSv1_2 for c in contexts)
    finally:
        client.close()


def test_capped_clients_still_share_one_transport(tls_cap):
    tls_cap("1.2")
    first = build_keepalive_http_client("https://api.deepseek.com")
    second = build_keepalive_http_client("https://api.deepseek.com")
    try:
        assert first._transport._inner is second._transport._inner
    finally:
        first.close()
        second.close()


def test_no_cap_keeps_default_tls(tls_cap):
    tls_cap("")
    client = build_keepalive_http_client("https://api.deepseek.com")
    assert isinstance(client, httpx.Client)
    try:
        for ctx in _pool_ssl_contexts(client):
            assert ctx.maximum_version == ssl.TLSVersion.MAXIMUM_SUPPORTED
    finally:
        client.close()


# ─── the cap reaches the wire ───────────────────────────────────────────────


@pytest.fixture
def tls13_only_server(tmp_path, monkeypatch):
    """A loopback HTTPS server that refuses anything below TLS 1.3."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test loopback")])
    now = datetime.now(UTC)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip_address("127.0.0.1"))]), critical=False)
            .sign(key, hashes.SHA256()))
    bundle = tmp_path / "loopback.pem"
    bundle.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private_key = tmp_path / "loopback.key"
    private_key.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                            serialization.PrivateFormat.PKCS8,
                                            serialization.NoEncryption()))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    context = _original_SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # The Python setter recurses once truststore is injected; use the C descriptor.
    _ssl._SSLContext.minimum_version.__set__(context, ssl.TLSVersion.TLSv1_3)
    context.load_cert_chain(bundle, private_key)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    try:
        yield f"https://127.0.0.1:{server.server_port}", bundle
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cap_limits_the_real_handshake(tls13_only_server, tls_cap):
    url, bundle = tls13_only_server

    tls_cap("")
    verify = ssl_verify.resolve_httpx_verify(ca_bundle=str(bundle))
    with build_keepalive_http_client(url, verify=verify) as client:
        assert client.get(url).text == "ok"

    process_bootstrap.close_shared_transports()
    tls_cap("1.2")
    verify = ssl_verify.resolve_httpx_verify(ca_bundle=str(bundle))
    with build_keepalive_http_client(url, verify=verify) as client:
        with pytest.raises(httpx.ConnectError):
            client.get(url)
