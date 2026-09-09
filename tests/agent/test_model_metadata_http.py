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


@pytest.mark.parametrize("status", [401, 403])
def test_metadata_auth_failure_closes_without_reading_body(servers, monkeypatch, status):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            requests.append(self.path)
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
    import httpx

    responses = []
    original_client = httpx.Client

    def recorded_client(**kwargs):
        kwargs["event_hooks"]["response"] = [responses.append]
        return original_client(**{**kwargs, "timeout": 2.0})

    monkeypatch.setattr(httpx, "Client", recorded_client)
    assert model_metadata.fetch_endpoint_model_metadata(url, force_refresh=True) == {}
    assert requests == ["/models"]
    assert len(responses) == 1 and responses[0].is_closed
    assert not responses[0].is_stream_consumed, "auth rejection read the absent body"
