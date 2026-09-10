"""Wire-level tests for credential-safe stdlib urllib redirects."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ssl
from threading import Thread
import urllib.error
import urllib.request

import pytest

from hermes_cli.urllib_security import (
    SafeCredentialRedirectHandler,
    open_credentialed_url,
    url_origin,
)


class _Response:
    def __init__(self, payload: bytes = b"{}") -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._payload


class _RecordingHandler(BaseHTTPRequestHandler):
    redirect_to = ""
    redirect_status = 302
    requests: list[tuple[str, dict[str, str]]] = []

    def _record(self) -> None:
        type(self).requests.append((
            self.command,
            {name.lower(): value for name, value in self.headers.items()},
        ))

    def do_GET(self):
        if self.path.startswith("/redirect"):
            self.send_response(type(self).redirect_status)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
            return
        self._record()
        body = json.dumps({"data": []}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path == "/redirect":
            self.send_response(type(self).redirect_status)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
            return
        self._record()
        self.send_response(200)
        self.end_headers()

    def log_message(self, _format, *_args):
        pass


def _server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def _credential_headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer secret",
        "Cookie": "session=secret",
        "CF-Access-Client-Secret": "cloudflare-secret",
        "X-Custom-Auth": "tenant-secret",
        "Accept": "application/json",
        "User-Agent": "hermes-test",
    }


def test_cross_host_redirect_drops_arbitrary_credentials_on_wire():
    source = _server()
    sink = _server()
    _RecordingHandler.requests = []
    _RecordingHandler.redirect_status = 302
    _RecordingHandler.redirect_to = f"http://localhost:{sink.server_port}/sink"
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{source.server_port}/redirect",
            headers=_credential_headers(),
        )
        with open_credentialed_url(request, timeout=3) as response:
            response.read()
    finally:
        source.shutdown()
        sink.shutdown()

    method, headers = _RecordingHandler.requests[-1]
    assert method == "GET"
    assert headers["accept"] == "application/json"
    assert headers["user-agent"] == "hermes-test"
    for name in (
        "authorization",
        "cookie",
        "cf-access-client-secret",
        "x-custom-auth",
    ):
        assert name not in headers


def test_same_host_different_port_drops_credentials_on_wire():
    source = _server()
    sink = _server()
    _RecordingHandler.requests = []
    _RecordingHandler.redirect_status = 302
    _RecordingHandler.redirect_to = f"http://127.0.0.1:{sink.server_port}/sink"
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{source.server_port}/redirect",
            headers=_credential_headers(),
        )
        with open_credentialed_url(request, timeout=3) as response:
            response.read()
    finally:
        source.shutdown()
        sink.shutdown()

    _, headers = _RecordingHandler.requests[-1]
    assert "authorization" not in headers
    assert "cf-access-client-secret" not in headers


def test_post_307_remains_rejected_by_urllib():
    request = urllib.request.Request(
        "https://models.example.test/load",
        data=b"{}",
        headers=_credential_headers(),
        method="POST",
    )
    handler = SafeCredentialRedirectHandler(request.full_url)
    with pytest.raises(urllib.error.HTTPError):
        handler.redirect_request(
            request,
            None,
            307,
            "Temporary Redirect",
            {},
            "https://other.example.test/load",
        )


def test_explicit_opener_factory_is_instrumentable_without_security_bypass():
    calls = []

    class _Opener:
        def open(self, request, *, timeout):
            calls.append((request.full_url, timeout))
            return _Response()

    def factory(*handlers):
        assert any(isinstance(h, SafeCredentialRedirectHandler) for h in handlers)
        return _Opener()

    request = urllib.request.Request(
        "https://models.example.test/models", headers={"Authorization": "secret"}
    )
    with open_credentialed_url(request, timeout=7, opener_factory=factory):
        pass
    assert calls == [("https://models.example.test/models", 7)]


def test_installed_request_processor_cannot_resurrect_cross_origin_secret(
    monkeypatch,
):
    source = _server()
    sink = _server()
    _RecordingHandler.requests = []
    _RecordingHandler.redirect_status = 302
    _RecordingHandler.redirect_to = f"http://localhost:{sink.server_port}/sink"

    class SecretProcessor(urllib.request.BaseHandler):
        handler_order = float("inf")  # type: ignore[assignment]

        def http_request(self, request):
            request.add_header("X-Installed-Secret", "must-not-cross")
            return request

    installed = urllib.request.build_opener(SecretProcessor())
    installed.addheaders = [("X-Opener-Secret", "also-must-not-cross")]
    monkeypatch.setattr(urllib.request, "_opener", installed)
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{source.server_port}/redirect",
            headers={"Authorization": "Bearer secret"},
        )
        with open_credentialed_url(request, timeout=3) as response:
            response.read()
    finally:
        source.shutdown()
        sink.shutdown()

    _, headers = _RecordingHandler.requests[-1]
    assert "authorization" not in headers
    assert "x-installed-secret" not in headers
    assert "x-opener-secret" not in headers


def test_multihop_redirects_never_resurrect_credentials():
    request = urllib.request.Request(
        "https://a.example.test/models", headers=_credential_headers()
    )
    handler = SafeCredentialRedirectHandler(request.full_url)

    same_origin = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://a.example.test/step-two",
    )
    assert same_origin is not None
    same_headers = {name.lower(): value for name, value in same_origin.header_items()}
    assert "authorization" in same_headers

    cross_origin = handler.redirect_request(
        same_origin,
        None,
        302,
        "Found",
        {},
        "https://b.example.test/step-three",
    )
    assert cross_origin is not None
    cross_headers = {name.lower(): value for name, value in cross_origin.header_items()}
    assert "authorization" not in cross_headers
    assert "cf-access-client-secret" not in cross_headers

    returned = handler.redirect_request(
        cross_origin,
        None,
        302,
        "Found",
        {},
        "https://a.example.test/final",
    )
    assert returned is not None
    returned_headers = {name.lower(): value for name, value in returned.header_items()}
    assert "authorization" not in returned_headers
    assert "cf-access-client-secret" not in returned_headers


def test_probe_api_models_drops_custom_credentials_on_wire():
    from hermes_cli.models import probe_api_models

    source = _server()
    sink = _server()
    _RecordingHandler.requests = []
    _RecordingHandler.redirect_status = 302
    _RecordingHandler.redirect_to = f"http://localhost:{sink.server_port}/sink"
    try:
        result = probe_api_models(
            "provider-key",
            f"http://127.0.0.1:{source.server_port}/redirect/..",
            timeout=3,
            request_headers={
                "CF-Access-Client-Secret": "cloudflare-secret",
                "X-Custom-Auth": "tenant-secret",
            },
        )
    finally:
        source.shutdown()
        sink.shutdown()

    assert result["models"] == []
    _, headers = _RecordingHandler.requests[-1]
    assert "authorization" not in headers
    assert "cf-access-client-secret" not in headers
    assert "x-custom-auth" not in headers


class _LmStudioSourceHandler(BaseHTTPRequestHandler):
    redirect_to = ""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(302)
        self.send_header("Location", type(self).redirect_to)
        self.end_headers()

    def log_message(self, format, *_args):
        pass


def test_anthropic_profile_drops_x_api_key_on_redirect(monkeypatch):
    import importlib

    AnthropicProfile = importlib.import_module(
        "plugins.model-providers.anthropic"
    ).AnthropicProfile

    source = _server()
    sink = _server()
    _RecordingHandler.requests = []
    _RecordingHandler.redirect_status = 302
    _RecordingHandler.redirect_to = f"http://localhost:{sink.server_port}/sink"

    original_request = urllib.request.Request

    def local_anthropic_request(url, *args, **kwargs):
        if url == "https://api.anthropic.com/v1/models":
            url = f"http://127.0.0.1:{source.server_port}/redirect"
        return original_request(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "Request", local_anthropic_request)
    try:
        result = AnthropicProfile(name="anthropic").fetch_models(
            api_key="anthropic-secret", timeout=3
        )
    finally:
        source.shutdown()
        sink.shutdown()

    assert result == []
    _, headers = _RecordingHandler.requests[-1]
    assert "x-api-key" not in headers
    assert headers["accept"] == "application/json"


def test_azure_catalog_probe_drops_api_key_and_bearer_on_redirect():
    from hermes_cli import azure_detect

    source = _server()
    sink = _server()
    _RecordingHandler.requests = []
    _RecordingHandler.redirect_status = 302
    _RecordingHandler.redirect_to = f"http://localhost:{sink.server_port}/sink"
    try:
        status, body = azure_detect._http_get_json(
            f"http://127.0.0.1:{source.server_port}/redirect", "azure-secret", timeout=3
        )
    finally:
        source.shutdown()
        sink.shutdown()

    assert status == 200
    assert body == {"data": []}
    _, headers = _RecordingHandler.requests[-1]
    assert "authorization" not in headers
    assert "api-key" not in headers


def test_azure_anthropic_probe_drops_api_key_and_bearer_on_redirect():
    from hermes_cli import azure_detect

    sink = _server()
    source = ThreadingHTTPServer(("127.0.0.1", 0), _LmStudioSourceHandler)
    Thread(target=source.serve_forever, daemon=True).start()
    _RecordingHandler.requests = []
    _LmStudioSourceHandler.redirect_to = f"http://localhost:{sink.server_port}/sink"
    try:
        azure_detect._probe_anthropic_messages(
            f"http://127.0.0.1:{source.server_port}", "azure-secret"
        )
    finally:
        source.shutdown()
        sink.shutdown()

    _, headers = _RecordingHandler.requests[-1]
    assert "authorization" not in headers
    assert "api-key" not in headers


def _clear_ca_bundle_env(monkeypatch) -> None:
    for name in (
        "HERMES_CA_BUNDLE",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_hermes_owned_opener_uses_resolved_https_context(monkeypatch):
    import hermes_cli.urllib_security as urllib_security

    context = ssl.create_default_context()
    monkeypatch.setattr(urllib.request, "_opener", None)
    monkeypatch.setattr(urllib_security, "_resolved_https_context", lambda: context)

    opener = urllib_security._secure_opener_from_installed_policy(
        "https://models.example.test/catalog"
    )

    https_handlers = [
        handler
        for handler in opener.handlers
        if isinstance(handler, urllib.request.HTTPSHandler)
    ]
    assert len(https_handlers) == 1
    assert getattr(https_handlers[0], "_context", None) is context


def test_resolved_https_context_prefers_configured_ca_bundle(monkeypatch, tmp_path):
    import hermes_cli.urllib_security as urllib_security

    _clear_ca_bundle_env(monkeypatch)
    ca_bundle = tmp_path / "corporate-ca.pem"
    ca_bundle.touch()
    expected_context = ssl.create_default_context()
    seen: list[str | None] = []

    def create_default_context(*, cafile=None):
        seen.append(cafile)
        return expected_context

    monkeypatch.setenv("HERMES_CA_BUNDLE", str(ca_bundle))
    monkeypatch.setattr(ssl, "create_default_context", create_default_context)

    assert urllib_security._resolved_https_context() is expected_context
    assert seen == [str(ca_bundle)]


def test_resolved_https_context_uses_certifi_on_macos(monkeypatch):
    import certifi
    import hermes_cli.urllib_security as urllib_security

    _clear_ca_bundle_env(monkeypatch)
    expected_context = ssl.create_default_context()
    seen: list[str | None] = []

    def create_default_context(*, cafile=None):
        seen.append(cafile)
        return expected_context

    monkeypatch.setattr(urllib_security.sys, "platform", "darwin")
    monkeypatch.setattr(certifi, "where", lambda: "/certifi/cacert.pem")
    monkeypatch.setattr(ssl, "create_default_context", create_default_context)

    assert urllib_security._resolved_https_context() is expected_context
    assert seen == ["/certifi/cacert.pem"]


def test_invalid_ca_bundle_falls_back_to_certifi_on_macos(monkeypatch, tmp_path):
    import certifi
    import hermes_cli.urllib_security as urllib_security

    _clear_ca_bundle_env(monkeypatch)
    missing_bundle = tmp_path / "missing-ca.pem"
    expected_context = ssl.create_default_context()
    seen: list[str | None] = []

    def create_default_context(*, cafile=None):
        seen.append(cafile)
        return expected_context

    monkeypatch.setenv("HERMES_CA_BUNDLE", str(missing_bundle))
    monkeypatch.setattr(urllib_security.sys, "platform", "darwin")
    monkeypatch.setattr(certifi, "where", lambda: "/certifi/cacert.pem")
    monkeypatch.setattr(ssl, "create_default_context", create_default_context)

    assert urllib_security._resolved_https_context() is expected_context
    assert seen == ["/certifi/cacert.pem"]


def test_resolved_https_context_keeps_stdlib_default_off_macos(monkeypatch):
    import hermes_cli.urllib_security as urllib_security

    _clear_ca_bundle_env(monkeypatch)
    monkeypatch.setattr(urllib_security.sys, "platform", "linux")

    assert urllib_security._resolved_https_context() is None


def test_installed_https_context_is_preserved(monkeypatch):
    import hermes_cli.urllib_security as urllib_security

    context = ssl.create_default_context()
    installed = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context)
    )
    monkeypatch.setattr(urllib.request, "_opener", installed)

    def unexpected_context_resolution():
        raise AssertionError("installed TLS policy must remain authoritative")

    monkeypatch.setattr(
        urllib_security,
        "_resolved_https_context",
        unexpected_context_resolution,
    )

    opener = urllib_security._secure_opener_from_installed_policy(
        "https://models.example.test/catalog"
    )

    https_handlers = [
        handler
        for handler in opener.handlers
        if isinstance(handler, urllib.request.HTTPSHandler)
    ]
    assert len(https_handlers) == 1
    assert getattr(https_handlers[0], "_context", None) is context
