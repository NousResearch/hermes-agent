"""mTLS (``agent.mtls``) — client certificates for LLM endpoints and MCP servers.

The wire-level tests run a real TLS server that requires a client certificate
(``CERT_REQUIRED`` against a throwaway CA), so a passing test means the handshake
actually carried the certificate — not that a kwarg was threaded somewhere.

Covered here:

1. ``resolve_httpx_verify`` — the one ``verify`` resolver every LLM client stack
   (OpenAI-wire, Gemini native, auxiliary, model metadata) goes through: no-op when
   unconfigured, certificate presented when configured, provider ``ssl_ca_cert``
   outranking the global ``agent.mtls.ca_cert``.
2. ``build_keepalive_http_client`` with that resolver — the exact httpx client the
   OpenAI/Gemini SDKs are handed — completes a request to the mTLS server.
3. The Anthropic SDK path (``http_client``-only: it has no ``verify=`` argument),
   including the timeout the SDK inherits from that client.
4. PKCS#12 bundles, including a passphrase from ``password_env`` and the failure mode
   for a wrong passphrase (warn once, connect without a certificate, never crash).
5. MCP: a server inherits the global certificate, an explicit per-server
   ``client_cert``/``ssl_verify`` still wins, and the per-server trust statement is
   never replaced by the global CA.
6. ``agent.mtls`` config defaults and ``hermes config check`` diagnostics.
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
# The shared resolver + the keepalive client handed to the OpenAI/Gemini SDKs
# ---------------------------------------------------------------------------


def test_unconfigured_mtls_is_a_no_op(monkeypatch):
    monkeypatch.setattr(ssl_verify, "mtls_settings", dict)
    assert ssl_verify.client_cert_spec() is None
    # The trust decision is unchanged: True (platform store) or a trust context.
    verify = ssl_verify.resolve_httpx_verify(base_url="https://api.example.com")
    assert verify is True or _is_tls_context(verify)


def test_configured_client_certificate_completes_an_mtls_handshake(mtls_server, mtls_config):
    """The certificate reaches the wire: the server demands one and the request succeeds."""
    verify = ssl_verify.resolve_httpx_verify(base_url=mtls_server)
    assert _is_tls_context(verify)

    response = httpx.get(f"{mtls_server}/", verify=verify, timeout=10)
    assert response.status_code == 200
    assert response.json()["client_cert"] == "hermes-client"


def test_keepalive_client_carries_the_certificate(mtls_server, mtls_config):
    """The exact httpx client the OpenAI/Gemini SDKs receive presents the certificate."""
    from agent.process_bootstrap import build_keepalive_http_client

    verify = ssl_verify.resolve_httpx_verify(base_url=mtls_server)
    client = build_keepalive_http_client(mtls_server, verify=verify)
    assert client is not None
    try:
        assert client.get(f"{mtls_server}/").json()["client_cert"] == "hermes-client"
    finally:
        client.close()


def test_no_certificate_is_rejected_by_the_same_server(mtls_server, monkeypatch):
    """Control: the server really does require a certificate — no silent pass."""
    monkeypatch.setattr(ssl_verify, "mtls_settings", dict)
    verify = ssl_verify.resolve_httpx_verify(base_url=mtls_server, ca_bundle=None)
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{mtls_server}/", verify=verify, timeout=10)


def test_provider_ca_bundle_outranks_global_mtls_ca(mtls_server, monkeypatch, pki):
    """Precedence: the endpoint's own ``ssl_ca_cert`` wins over ``agent.mtls.ca_cert``."""
    monkeypatch.setattr(
        ssl_verify,
        "mtls_settings",
        lambda: {
            "cert_file": pki["client_cert"],
            "key_file": pki["client_key"],
            "ca_cert": pki["other_ca"],  # wrong signer on purpose
            "password_env": "",
        },
    )
    verify = ssl_verify.resolve_httpx_verify(ca_bundle=pki["ca"], base_url=mtls_server)
    assert httpx.get(f"{mtls_server}/", verify=verify, timeout=10).status_code == 200


def test_global_mtls_ca_is_used_when_the_endpoint_names_none(mtls_server, mtls_config):
    """The global CA is the fallback trust root, so a private signer needs one config block."""
    verify = ssl_verify.resolve_httpx_verify(base_url=mtls_server)
    assert httpx.get(f"{mtls_server}/", verify=verify, timeout=10).status_code == 200


def test_missing_certificate_warns_once_and_connects_without_one(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(
        ssl_verify,
        "mtls_settings",
        lambda: {"cert_file": str(tmp_path / "absent.pem"), "key_file": "", "ca_cert": ""},
    )
    ssl_verify._MTLS_WARNED.clear()
    with pytest.raises(ssl_verify.MtlsConfigError):
        ssl_verify.client_cert_spec()
    with caplog.at_level("WARNING"):
        first = ssl_verify.resolve_httpx_verify(base_url="https://api.example.com")
        second = ssl_verify.resolve_httpx_verify(base_url="https://api.example.com")
    for verify in (first, second):
        assert verify is True or _is_tls_context(verify)
    warnings = [r for r in caplog.records if "mTLS client certificate unusable" in r.getMessage()]
    assert len(warnings) == 1, "a broken global certificate must warn exactly once per process"


def test_pkcs12_bundle_is_accepted(mtls_server, monkeypatch, pki):
    """``cert_file: client.p12`` with the passphrase in ``password_env``.

    ``ssl.load_cert_chain`` reads PEM only, so the bundle is unwrapped once into a
    0600 PEM pair; this asserts the extraction is usable for a real handshake.
    """
    monkeypatch.setenv("HERMES_TEST_P12_PASSWORD", pki["p12_password"])
    monkeypatch.setattr(
        ssl_verify,
        "mtls_settings",
        lambda: {
            "cert_file": pki["p12"],
            "key_file": "",
            "ca_cert": pki["ca"],
            "password_env": "HERMES_TEST_P12_PASSWORD",
        },
    )
    spec = ssl_verify.client_cert_spec()
    assert spec is not None
    cert_path, key_path, _ = spec
    assert cert_path and key_path
    assert cert_path.endswith("-cert.pem") and key_path.endswith("-key.pem")
    verify = ssl_verify.resolve_httpx_verify(base_url=mtls_server)
    assert httpx.get(f"{mtls_server}/", verify=verify, timeout=10).json()["client_cert"] == "hermes-client"


def test_pkcs12_wrong_passphrase_reports_and_degrades(monkeypatch, pki, caplog):
    monkeypatch.setenv("HERMES_TEST_P12_PASSWORD", "not-the-password")
    monkeypatch.setattr(
        ssl_verify,
        "mtls_settings",
        lambda: {
            "cert_file": pki["p12"],
            "key_file": "",
            "ca_cert": "",
            "password_env": "HERMES_TEST_P12_PASSWORD",
        },
    )
    with pytest.raises(ssl_verify.MtlsConfigError) as excinfo:
        ssl_verify.client_cert_spec()
    assert "password_env" in str(excinfo.value)

    with caplog.at_level("WARNING"):
        verify = ssl_verify.resolve_httpx_verify(base_url="https://api.example.com")
    assert verify is True or _is_tls_context(verify)
    assert any("mTLS client certificate unusable" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Anthropic: the SDK takes an http_client, never a verify= argument
# ---------------------------------------------------------------------------


def test_anthropic_client_presents_the_certificate(mtls_server, mtls_config):
    client = build_anthropic_client("test-key", base_url=mtls_server, timeout=30)
    message = client.messages.create(
        model="test-model", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
    )
    assert message.content[0].text == "ok"


def test_anthropic_http_client_keeps_hermes_read_timeout(mtls_server, mtls_config):
    """A 5s httpx default timeout would cut long streams; the SDK adopts this client's."""
    client = build_anthropic_client("test-key", base_url=mtls_server, timeout=900)
    assert client._client.timeout.read == 900


def test_anthropic_client_is_untouched_without_mtls(monkeypatch):
    monkeypatch.setattr(ssl_verify, "mtls_settings", dict)
    client = build_anthropic_client("test-key", base_url="https://api.anthropic.com", timeout=30)
    # The SDK's own wrapper client, not one Hermes injected for mTLS.
    assert client._client.__class__.__name__ == "SyncHttpxClientWrapper"


# ---------------------------------------------------------------------------
# MCP: inherit the global identity, never override the per-server one
# ---------------------------------------------------------------------------


def test_mcp_server_inherits_the_global_certificate(mtls_config, pki):
    from tools.mcp_tool_errors import _resolve_client_cert

    assert _resolve_client_cert("srv", {"url": "https://mcp.example.com"}) == (
        pki["client_cert"], pki["client_key"]
    )


def test_mcp_per_server_certificate_wins_over_the_global_one(mtls_config, pki, tmp_path):
    from tools.mcp_tool_errors import _resolve_client_cert

    own_cert = tmp_path / "own.pem"
    own_key = tmp_path / "own-key.pem"
    own_cert.write_text("cert")
    own_key.write_text("key")
    assert _resolve_client_cert(
        "srv",
        {"client_cert": str(own_cert), "client_key": str(own_key)},
    ) == (str(own_cert), str(own_key))


def test_mcp_explicit_ssl_verify_still_wins(mtls_config, pki):
    from tools.mcp_tool_errors import _resolve_ssl_verify

    # The server opted out of verification: the global CA must not re-enable it.
    assert _resolve_ssl_verify("srv", {"ssl_verify": False}) is False
    # No per-server statement: the global CA bundle becomes the trust root.
    context = _resolve_ssl_verify("srv", {"url": "https://mcp.example.com"})
    assert _is_tls_context(context)
    assert context is ssl_verify.ca_verify_for("mcp:srv")


def test_mcp_without_global_mtls_keeps_todays_behaviour(monkeypatch):
    from tools.mcp_tool_errors import _resolve_client_cert, _resolve_ssl_verify

    monkeypatch.setattr(ssl_verify, "mtls_settings", dict)
    assert _resolve_client_cert("srv", {"url": "https://mcp.example.com"}) is None
    assert _resolve_ssl_verify("srv", {"url": "https://mcp.example.com"}) is True


def test_mcp_broken_global_certificate_does_not_take_the_server_down(monkeypatch, tmp_path, caplog):
    from tools.mcp_tool_errors import _resolve_client_cert

    monkeypatch.setattr(
        ssl_verify,
        "mtls_settings",
        lambda: {"cert_file": str(tmp_path / "gone.pem"), "key_file": "", "ca_cert": ""},
    )
    ssl_verify._MTLS_WARNED.clear()
    with caplog.at_level("WARNING"):
        assert _resolve_client_cert("srv", {"url": "https://mcp.example.com"}) is None
    assert any("agent.mtls client certificate unusable" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_default_config_documents_the_mtls_block():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["agent"]["mtls"] == {
        "cert_file": "",
        "key_file": "",
        "ca_cert": "",
        "password_env": "",
    }


def test_config_check_reports_missing_certificate(tmp_path):
    from hermes_cli.config import validate_config_structure

    issues = validate_config_structure(
        config={"agent": {"mtls": {"cert_file": str(tmp_path / "nope.pem"), "key_file": ""}}}
    )
    assert any(
        issue.severity == "error" and "agent.mtls.cert_file does not exist" in issue.message
        for issue in issues
    ), issues


def test_config_check_accepts_a_working_certificate(pki):
    from hermes_cli.config import validate_config_structure

    issues = [
        issue
        for issue in validate_config_structure(
            config={
                "agent": {
                    "mtls": {
                        "cert_file": pki["client_cert"],
                        "key_file": pki["client_key"],
                        "ca_cert": pki["ca"],
                        "password_env": "",
                    }
                }
            }
        )
        if "mtls" in issue.message
    ]
    assert issues == []


def test_openai_client_factory_presents_the_certificate(mtls_server, mtls_config):
    """The real chokepoint (``create_openai_client``) resolves the global block itself.

    This is the path every OpenAI-wire provider (and Gemini native, which receives the
    same resolved ``verify``) goes through, so no per-client wiring is needed.
    """
    from types import SimpleNamespace

    from agent.agent_runtime_helpers import create_openai_client
    from agent.process_bootstrap import build_keepalive_http_client

    agent = SimpleNamespace(
        provider="",
        model="test-model",
        base_url=mtls_server,
        verbose_logging=False,
        _client_kwargs={},
        _build_keepalive_http_client=lambda url, verify=None: build_keepalive_http_client(
            url, verify=verify
        ),
        _client_log_context=lambda: "",
    )
    client = create_openai_client(
        agent, {"api_key": "test-key", "base_url": mtls_server}, reason="mtls-test", shared=False
    )
    try:
        completion = client.chat.completions.create(
            model="test-model", messages=[{"role": "user", "content": "hi"}]
        )
        assert completion.choices[0].message.content == "ok"
    finally:
        client.close()


def test_doctor_loads_the_configured_identity(mtls_config, capsys):
    """``hermes doctor`` proves the pair actually loads, not just that paths exist."""
    from hermes_cli import doctor_platform

    issues: list[str] = []
    doctor_platform.check_certificates(should_fix=False, issues=issues)
    assert issues == []
    assert "mTLS client certificate loaded" in capsys.readouterr().out


def test_doctor_warns_when_the_global_identity_is_unusable(monkeypatch, tmp_path, capsys):
    from hermes_cli import doctor_platform

    monkeypatch.setattr(
        ssl_verify,
        "mtls_settings",
        lambda: {"cert_file": str(tmp_path / "gone.pem"), "key_file": "", "ca_cert": ""},
    )
    issues: list[str] = []
    doctor_platform.check_certificates(should_fix=False, issues=issues)
    assert any("agent.mtls" in issue for issue in issues)
