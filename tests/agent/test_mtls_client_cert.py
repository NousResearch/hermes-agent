"""mTLS (``agent.mtls``): every client stack presents the configured certificate.

A real TLS server that requires a client certificate (``CERT_REQUIRED`` against a
throwaway CA) is the oracle, so a pass means the handshake carried the certificate.
"""

from __future__ import annotations

import ipaddress
import json
import ssl
import threading
from datetime import datetime, timedelta, timezone, UTC
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from agent import ssl_verify
from agent.anthropic_adapter import build_anthropic_client


# ---------------------------------------------------------------------------
# A throwaway PKI and a TLS server that insists on a client certificate
# ---------------------------------------------------------------------------


def _pki(tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    def _write(name, data: bytes) -> str:
        path = tmp_path / name
        path.write_bytes(data)
        return str(path)

    def _key():
        return ec.generate_private_key(ec.SECP256R1())

    def _pem_key(key, password: bytes | None = None):
        enc = (
            serialization.BestAvailableEncryption(password)
            if password
            else serialization.NoEncryption()
        )
        return key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc
        )

    def _ca(common_name: str):
        key = _key()
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
        now = datetime.now(UTC)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256())
        )
        return key, cert

    def _leaf(ca_key, ca_cert, common_name, usage, san_ip=False, password=None):
        key = _key()
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
        now = datetime.now(UTC)
        builder = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(ca_cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
        )
        if san_ip:
            builder = builder.add_extension(
                x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                critical=False,
            )
        cert = builder.sign(ca_key, hashes.SHA256())
        return key, cert

    ca_key, ca_cert = _ca("Hermes Test CA")
    ca_pem = _write("ca.pem", ca_cert.public_bytes(serialization.Encoding.PEM))

    server_key, server_cert = _leaf(
        ca_key, ca_cert, "localhost", ExtendedKeyUsageOID.SERVER_AUTH, san_ip=True
    )
    server_pem = _write("server.pem", server_cert.public_bytes(serialization.Encoding.PEM))
    server_key_pem = _write("server-key.pem", _pem_key(server_key))

    client_key, client_cert = _leaf(
        ca_key, ca_cert, "hermes-client", ExtendedKeyUsageOID.CLIENT_AUTH
    )
    client_pem = _write("client.pem", client_cert.public_bytes(serialization.Encoding.PEM))
    client_key_pem = _write("client-key.pem", _pem_key(client_key))
    combined = _write(
        "client-combined.pem",
        client_cert.public_bytes(serialization.Encoding.PEM) + _pem_key(client_key),
    )

    # A second, unrelated CA: signing nothing our server trusts.
    other_key, other_cert = _ca("Unrelated CA")
    other_ca_pem = _write("other-ca.pem", other_cert.public_bytes(serialization.Encoding.PEM))
    del other_key

    from cryptography.hazmat.primitives.serialization import BestAvailableEncryption, pkcs12

    p12 = _write(
        "client.p12",
        pkcs12.serialize_key_and_certificates(
            b"hermes-client",
            client_key,
            client_cert,
            None,
            BestAvailableEncryption(b"s3cret"),
        ),
    )

    return {
        "ca": ca_pem,
        "other_ca": other_ca_pem,
        "server_cert": server_pem,
        "server_key": server_key_pem,
        "client_cert": client_pem,
        "client_key": client_key_pem,
        "client_combined": combined,
        "p12": p12,
        "p12_password": "s3cret",
    }


def _is_tls_context(value) -> bool:
    """A resolved ``verify`` that is a context rather than a bool.

    ``isinstance(value, ssl.SSLContext)`` cannot be used: ``install_truststore()``
    rebinds ``ssl.SSLContext`` process-wide, so a context built from the stdlib class
    (or from truststore's) fails the other class's isinstance check.
    """
    return value is not True and value is not False and hasattr(value, "get_ca_certs")


class _MtlsHandler(BaseHTTPRequestHandler):
    def _reply(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply({"ok": True, "client_cert": self._client_cert_subject()})

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        if length:
            self.rfile.read(length)
        if self.path.rstrip("/").endswith("/v1/messages"):
            self._reply({
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })
            return
        self._reply({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    def _client_cert_subject(self):
        try:
            cert = self.connection.getpeercert()
        except Exception:
            return None
        for rdn in (cert or {}).get("subject", ()):
            for key, value in rdn:
                if key == "commonName":
                    return value
        return None

    def log_message(self, *args):  # keep pytest output clean
        return


@pytest.fixture
def pki(tmp_path):
    return _pki(tmp_path)


@pytest.fixture
def mtls_server(pki):
    """A TLS server on 127.0.0.1 that rejects connections without a client cert."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(pki["server_cert"], pki["server_key"])
    context.load_verify_locations(cafile=pki["ca"])
    context.verify_mode = ssl.CERT_REQUIRED

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _MtlsHandler)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def mtls_config(monkeypatch, pki):
    """Point ``agent.mtls`` at the test client certificate."""
    block = {
        "cert_file": pki["client_cert"],
        "key_file": pki["client_key"],
        "ca_cert": pki["ca"],
        "password_env": "",
    }
    monkeypatch.setattr(ssl_verify, "mtls_settings", lambda: dict(block))
    ssl_verify._MTLS_CONTEXTS.clear()
    ssl_verify._MTLS_WARNED.clear()
    return block


@pytest.fixture(autouse=True)
def _clear_mtls_caches():
    ssl_verify._MTLS_CONTEXTS.clear()
    ssl_verify._MTLS_WARNED.clear()
    yield
    ssl_verify._MTLS_CONTEXTS.clear()
    ssl_verify._MTLS_WARNED.clear()


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_every_client_stack_presents_the_global_certificate(mtls_server, mtls_config, pki, monkeypatch):
    """OpenAI-wire factory, Anthropic SDK, and MCP inheritance all carry the identity;
    without agent.mtls the same server refuses the connection (control)."""
    from types import SimpleNamespace

    from agent.agent_runtime_helpers import create_openai_client
    from agent.process_bootstrap import build_keepalive_http_client
    from tools.mcp_tool_errors import _resolve_client_cert, _resolve_ssl_verify, _resolve_tls

    agent = SimpleNamespace(
        provider="", model="test-model", base_url=mtls_server, verbose_logging=False, _client_kwargs={},
        _build_keepalive_http_client=lambda url, verify=None: build_keepalive_http_client(url, verify=verify),
        _client_log_context=lambda: "",
    )
    client = create_openai_client(agent, {"api_key": "k", "base_url": mtls_server}, reason="mtls-test", shared=False)
    try:
        reply = client.chat.completions.create(model="test-model", messages=[{"role": "user", "content": "hi"}])
        assert reply.choices[0].message.content == "ok"
    finally:
        client.close()
    message = build_anthropic_client("k", base_url=mtls_server, timeout=900)
    assert message._client.timeout.read == 900  # the SDK adopts the injected client's timeout
    assert message.messages.create(model="m", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
                                   ).content[0].text == "ok"
    assert _resolve_client_cert("srv", {"url": "https://mcp.example.com"}) == (pki["client_cert"], pki["client_key"])
    assert _resolve_ssl_verify("srv", {"ssl_verify": False}) is False  # an explicit per-server choice wins
    # A per-server CA path beside a certificate becomes a context (httpx refuses verify=<str> + cert=).
    verify, cert = _resolve_tls("srv", {"url": mtls_server, "ssl_verify": pki["ca"]})
    assert _is_tls_context(verify) and cert == (pki["client_cert"], pki["client_key"])

    ssl_verify._MTLS_CONTEXTS.clear()
    monkeypatch.setattr(ssl_verify, "mtls_settings", dict)  # unconfigured: the same server must refuse us
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{mtls_server}/", verify=ssl_verify.resolve_httpx_verify(base_url=mtls_server), timeout=10)


def test_unusable_global_certificate_warns_once_and_connects_without_it(monkeypatch, pki, caplog, tmp_path):
    """A missing file or a wrong PKCS#12 passphrase never takes providers or MCP servers
    down: one warning naming the setting, then the platform trust decision as before."""
    from tools.mcp_tool_errors import _resolve_client_cert

    monkeypatch.setenv("HERMES_TEST_P12_PASSWORD", "not-the-password")
    for block in ({"cert_file": str(tmp_path / "absent.pem"), "key_file": "", "ca_cert": ""},
                  {"cert_file": pki["p12"], "key_file": "", "ca_cert": "", "password_env": "HERMES_TEST_P12_PASSWORD"}):
        monkeypatch.setattr(ssl_verify, "mtls_settings", lambda block=block: dict(block))
        ssl_verify._MTLS_WARNED.clear()
        with caplog.at_level("WARNING"):
            caplog.clear()
            for _ in range(2):
                verify = ssl_verify.resolve_httpx_verify(base_url="https://api.example.com")
                assert verify is True or _is_tls_context(verify)
            assert _resolve_client_cert("srv", {"url": "https://mcp.example.com"}) is None
        unusable = [r for r in caplog.records if "client certificate unusable" in r.getMessage()]
        assert any("WITHOUT a" in r.getMessage() for r in unusable)
        assert sum("mTLS client certificate unusable" in r.getMessage() for r in unusable) == 1
