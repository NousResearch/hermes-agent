"""Exercise probe transport redirects, proxies, and early response closure."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from agent import model_metadata, model_metadata_http


@pytest.fixture
def servers():
    active = []

    def start(handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        active.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}"

    yield start
    for server, thread in reversed(active):
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_redirect_credentials_and_proxy_routing_are_preserved(servers, monkeypatch):
    seen = []
    sink_url = ""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, dict(self.headers)))
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", sink_url + "/models")
                self.end_headers()
                return
            body = b'{"data":[]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    source_url = servers(Handler)
    sink_url = servers(Handler)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    headers = {"Authorization": "Bearer dummy", "X-Custom-Auth": "dummy", "Accept": "application/json"}
    assert model_metadata_http.get(source_url + "/redirect", headers=headers).status_code == 200
    assert seen[0][1]["Authorization"] == "Bearer dummy"
    redirected = {key.lower(): value for key, value in seen[-1][1].items()}
    assert "authorization" not in redirected and "x-custom-auth" not in redirected
    assert redirected["accept"] == "application/json"

    # A real HTTP proxy responds to an absolute URI; the .invalid destination
    # cannot answer without the configured proxy. No external network is used.
    monkeypatch.setenv("HTTP_PROXY", source_url)
    monkeypatch.setenv("NO_PROXY", "")
    assert model_metadata_http.get("http://probe.invalid/models").status_code == 200
    assert seen[-1][0] == "http://probe.invalid/models"


@pytest.mark.parametrize("status", [401, 403, 404])
def test_metadata_auth_failure_closes_without_reading_body(servers, monkeypatch, status):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            requests.append(self.path)
            if status == 404 and self.path == "/models":
                body = b'{"data":[{"id":"test/model","context_length":32768}]}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(status)
                self.send_header("Content-Length", "1000000")
                self.end_headers()
            # Intentionally send no body. The client must close on the headers,
            # not wait for a read timeout or try the alternate /v1 URL.

        def log_message(self, *args):
            pass

    url = servers(Handler)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(model_metadata, "detect_local_server_type", lambda *args, **kwargs: None)
    # Patch the module the transport actually resolves, not a hard-coded
    # `httpx` — this test file used to `import httpx` directly, which made
    # it uncollectable on every Python below 3.14 (httpx is installed only
    # for python_version >= '3.14').
    client_mod = model_metadata_http.client_module()

    responses = []
    original_client = client_mod.Client

    def recorded_client(**kwargs):
        kwargs["event_hooks"]["response"] = [responses.append]
        return original_client(**{**kwargs, "timeout": 2.0})

    monkeypatch.setattr(client_mod, "Client", recorded_client)
    url += "/v1"
    result = model_metadata.fetch_endpoint_model_metadata(url, force_refresh=True)
    if status == 404:
        assert result["test/model"]["context_length"] == 32768
        assert requests == ["/v1/models", "/models"]
        assert responses[1].is_stream_consumed
    else:
        assert result == {}
        assert model_metadata.fetch_endpoint_model_metadata(url) == {}
        assert requests == ["/v1/models"]
    assert all(response.is_closed for response in responses)
    assert not responses[0].is_stream_consumed, "rejection read the absent body"


def test_probe_transport_does_not_require_the_httpx_distribution(monkeypatch):
    """The probe must work on Python versions where `httpx` is absent.

    `httpx` is installed only for python_version >= '3.14' (pyproject.toml),
    while Hermes supports 3.11-3.14. A bare `import httpx` in the transport
    raised ModuleNotFoundError on three of the four supported
    interpreters, which failed every model-metadata probe and logged a
    WARNING per estimate.

    The transport resolves `httpx2` first and falls back to `httpx`, so a
    missing `httpx` must not matter — and when neither is importable the
    error must say so instead of naming a single distribution.
    """
    import importlib

    real_import_module = importlib.import_module

    def blocked(*names):
        def _import_module(name, *args, **kwargs):
            if name in names:
                raise ModuleNotFoundError(f"No module named '{name}'")
            return real_import_module(name, *args, **kwargs)

        return _import_module

    # 1. `httpx` missing is fine — httpx2 (or httpx) still resolves.
    monkeypatch.setattr(importlib, "import_module", blocked("httpx"))
    monkeypatch.setattr(model_metadata_http, "_CLIENT_MODULE", None)
    module = model_metadata_http.client_module()
    assert module is not None
    assert module.__name__ in ("httpx2", "httpx")

    # 2. `httpx2` missing falls back to `httpx`.
    monkeypatch.setattr(importlib, "import_module", blocked("httpx2"))
    monkeypatch.setattr(model_metadata_http, "_CLIENT_MODULE", None)
    module = model_metadata_http.client_module()
    assert module is not None

    # 3. Neither available: the error must name both, not just one.
    monkeypatch.setattr(importlib, "import_module", blocked("httpx2", "httpx"))
    monkeypatch.setattr(model_metadata_http, "_CLIENT_MODULE", None)
    with pytest.raises(ImportError) as excinfo:
        model_metadata_http.client_module()
    message = str(excinfo.value)
    assert "httpx2" in message and "httpx" in message
