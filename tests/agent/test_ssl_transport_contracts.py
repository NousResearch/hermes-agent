"""TLS assertions reach real client transports, never the host certificate store."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
import ssl
from threading import Thread

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
import httpx
import pytest
from truststore._ssl_constants import _original_SSLContext

from agent import model_metadata, process_bootstrap, ssl_verify


@pytest.fixture
def local_tls(tmp_path, monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test loopback")])
    now = datetime.now(timezone.utc)
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
            body = b'{"data":[{"id":"test-model","context_length":2048}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    context = _original_SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(bundle, private_key)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    process_bootstrap.close_shared_transports()
    try:
        yield f"https://127.0.0.1:{server.server_port}", bundle
    finally:
        process_bootstrap.close_shared_transports()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("probe", ["metadata", "catalog"])
def test_provider_ca_reaches_the_real_probe_transport(local_tls, monkeypatch, probe):
    import certifi
    from hermes_cli import models

    url, bundle = local_tls
    monkeypatch.setattr(model_metadata, "detect_local_server_type", lambda *args, **kwargs: None)
    settings = {"name": "loopback", "base_url": url, "ssl_ca_cert": certifi.where()}
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", lambda config=None: [dict(settings)])
    monkeypatch.setenv("SSL_CERT_FILE", "/missing-ambient.pem")

    def discover():
        if probe == "metadata":
            return list(model_metadata.fetch_endpoint_model_metadata(url, force_refresh=True))
        return models.probe_api_models(None, url)["models"]

    assert discover() == ([] if probe == "metadata" else None)
    settings["ssl_ca_cert"] = str(bundle)
    assert discover() == ["test-model"]

    settings["ssl_ca_cert"] = certifi.where()
    settings["ssl_verify"] = False
    assert discover() == ["test-model"]


@pytest.mark.parametrize("variable", ["HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_ambient_ca_cannot_break_the_shared_client(variable, tmp_path, monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setenv(variable, str(tmp_path / "missing-ca"))
    for _ in range(2):
        assert ssl_verify.install_truststore() is True
    verify = ssl_verify.resolve_httpx_verify()
    with httpx.Client(verify=verify) as client:
        context = client._transport._pool._ssl_context
        assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
        assert type(context).__module__.startswith("truststore")
    client = AIAgent._build_keepalive_http_client("https://example.invalid", verify=verify)
    assert client is not None
    client.close()


@pytest.mark.parametrize("insecure", [False, True])
def test_pinned_clients_share_transport_not_close_state(local_tls, insecure):
    from run_agent import AIAgent
    url, bundle = local_tls
    first = ssl_verify.resolve_httpx_verify(ca_bundle=str(bundle), ssl_verify=not insecure)
    second = ssl_verify.resolve_httpx_verify(ca_bundle=str(bundle), ssl_verify=not insecure)
    if not insecure:
        assert isinstance(first, _original_SSLContext)
        assert not type(first).__module__.startswith("truststore")
        assert first.verify_mode == ssl.CERT_REQUIRED and first.check_hostname
        assert len(first.get_ca_certs()) == 1
    assert first is second
    a = AIAgent._build_keepalive_http_client(url, verify=first)
    b = process_bootstrap.build_keepalive_http_client(url, verify=second)
    assert a is not None and b is not None
    try:
        assert a._transport._inner is b._transport._inner
        assert a.get(url, timeout=5).status_code == 200
        a.close()
        assert b.get(url, timeout=5).status_code == 200
    finally:
        a.close()
        b.close()
