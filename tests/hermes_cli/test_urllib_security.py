"""Wire-level tests for credential-safe stdlib urllib redirects."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
        type(self).requests.append(
            (self.command, {name.lower(): value for name, value in self.headers.items()})
        )

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


def test_url_origin_normalizes_default_ports_and_trailing_dot():
    assert url_origin("https://EXAMPLE.test./models") == (
        "https",
        "example.test",
        443,
    )
    assert url_origin("https://example.test:443/other") == (
        "https",
        "example.test",
        443,
    )
    assert url_origin("http://example.test") != url_origin("https://example.test")
    assert url_origin("https://example.test:0") == (
        "https",
        "example.test",
        0,
    )
    with pytest.raises(ValueError):
        url_origin("https://example.test:not-a-port")


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


def test_same_origin_redirect_preserves_headers_on_wire():
    server = _server()
    _RecordingHandler.requests = []
    _RecordingHandler.redirect_status = 302
    _RecordingHandler.redirect_to = f"http://127.0.0.1:{server.server_port}/sink"
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/redirect",
            headers=_credential_headers(),
        )
        with open_credentialed_url(request, timeout=3) as response:
            response.read()
    finally:
        server.shutdown()

    _, headers = _RecordingHandler.requests[-1]
    assert headers["authorization"] == "Bearer secret"
    assert headers["cf-access-client-secret"] == "cloudflare-secret"


def test_scheme_downgrade_is_cross_origin():
    request = urllib.request.Request(
        "https://models.example.test/models", headers=_credential_headers()
    )
    handler = SafeCredentialRedirectHandler(request.full_url)
    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "http://models.example.test/models",
    )
    assert redirected is not None
    headers = {name.lower(): value for name, value in redirected.header_items()}
    assert "authorization" not in headers
    assert "cf-access-client-secret" not in headers


def test_post_302_uses_urllib_semantics_and_drops_credentials():
    request = urllib.request.Request(
        "https://models.example.test/load",
        data=b"{}",
        headers={**_credential_headers(), "Content-Type": "application/json"},
        method="POST",
    )
    handler = SafeCredentialRedirectHandler(request.full_url)
    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://other.example.test/load",
    )
    assert redirected is not None
    assert redirected.get_method() == "GET"
    assert redirected.data is None
    headers = {name.lower(): value for name, value in redirected.header_items()}
    assert "authorization" not in headers
    assert "content-type" not in headers


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


def test_installed_custom_opener_policy_is_preserved(monkeypatch):
    opened = []

    class FooHandler(urllib.request.BaseHandler):
        def foo_open(self, request):
            opened.append(request.full_url)
            return _Response(b"custom")

    installed = urllib.request.build_opener(FooHandler())
    installed.addheaders = [
        ("X-Trace-Policy", "installed"),
        ("User-agent", "enterprise-client"),
    ]
    monkeypatch.setattr(urllib.request, "_opener", installed)

    from hermes_cli.urllib_security import _secure_opener_from_installed_policy

    secured = _secure_opener_from_installed_policy(
        "foo://models.example.test/catalog"
    )
    assert secured.addheaders == installed.addheaders

    request = urllib.request.Request(
        "foo://models.example.test/catalog", headers={"Authorization": "secret"}
    )
    with open_credentialed_url(request, timeout=3) as response:
        assert response.read() == b"custom"
    assert opened == ["foo://models.example.test/catalog"]


def test_installed_proxy_handler_is_preserved(monkeypatch):
    installed = urllib.request.build_opener(
        urllib.request.ProxyHandler({"https": "http://proxy.example.test:8443"})
    )
    monkeypatch.setattr(urllib.request, "_opener", installed)

    from hermes_cli.urllib_security import _secure_opener_from_installed_policy

    secured = _secure_opener_from_installed_policy(
        "https://models.example.test/catalog"
    )
    proxy_handlers = [
        handler
        for handler in getattr(secured, "handlers", ())
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert proxy_handlers
    assert getattr(proxy_handlers[0], "proxies", {}) == {
        "https": "http://proxy.example.test:8443"
    }


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
    same_headers = {
        name.lower(): value for name, value in same_origin.header_items()
    }
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
    cross_headers = {
        name.lower(): value for name, value in cross_origin.header_items()
    }
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
    returned_headers = {
        name.lower(): value for name, value in returned.header_items()
    }
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


def test_lmstudio_load_post_drops_bearer_on_redirect(monkeypatch):
    from hermes_cli import models

    sink = _server()
    source = ThreadingHTTPServer(("127.0.0.1", 0), _LmStudioSourceHandler)
    Thread(target=source.serve_forever, daemon=True).start()
    _RecordingHandler.requests = []
    _LmStudioSourceHandler.redirect_to = f"http://localhost:{sink.server_port}/sink"
    monkeypatch.setattr(
        models,
        "_lmstudio_fetch_raw_models",
        lambda **_kwargs: [
            {"id": "model", "max_context_length": 8192, "loaded_instances": []}
        ],
    )
    try:
        loaded = models.ensure_lmstudio_model_loaded(
            "model",
            f"http://127.0.0.1:{source.server_port}",
            api_key="lm-secret",
            target_context_length=4096,
            timeout=3,
        )
    finally:
        source.shutdown()
        sink.shutdown()

    assert loaded == 4096
    method, headers = _RecordingHandler.requests[-1]
    assert method == "GET"
    assert "authorization" not in headers
    assert "content-type" not in headers
