"""Exercise probe transport redirects, proxies, and early response closure."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from agent import model_metadata, model_metadata_http


@pytest.mark.parametrize("section", ["providers", "custom_providers"])
@pytest.mark.parametrize("authorization", [None, "custom-auth"])
def test_metadata_reads_context_from_header_authenticated_provider(servers, monkeypatch, section, authorization):
    """A gateway requiring configured headers must supply the served context, not a family guess."""
    from hermes_constants import get_hermes_home

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            authorized = (self.headers.get("Authorization") == (authorization or "Bearer test-token")
                          and self.headers.get("X-Consumer") == "assistant")
            status = (200 if authorized else 401) if self.path == "/inference/v1/models" else 404
            body = b'{"data":[{"id":"served-model","max_model_len":114688}]}' if status == 200 else b'{}'
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    base_url = servers(Handler) + "/inference/v1"
    provider = {"name": "local", "base_url": base_url, "extra_headers": {"X-Consumer": "assistant"}}
    if authorization:
        provider["extra_headers"]["authorization"] = authorization
    config = {section: {"local": provider} if section == "providers" else [provider]}
    (get_hermes_home() / "config.yaml").write_text(json.dumps(config))

    metadata = model_metadata.fetch_endpoint_model_metadata(base_url, api_key="test-token", force_refresh=True)
    assert metadata["served-model"]["context_length"] == 114688


def test_metadata_provider_headers_are_route_scoped_and_stripped_on_redirect(servers, monkeypatch):
    from hermes_constants import get_hermes_home

    seen = []
    sink_url = ""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, {key.lower(): value for key, value in self.headers.items()}))
            if self.path == "/configured/v1/models":
                self.send_response(302)
                self.send_header("Location", sink_url + "/sink/models")
                self.end_headers()
                return
            status = 200 if self.path in {"/sink/models", "/other/v1/models"} else 404
            body = b'{"data":[{"id":"model","max_model_len":32768}]}' if status == 200 else b'{}'
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    source_url, sink_url = servers(Handler), servers(Handler)
    base_url = source_url + "/configured/v1"
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"providers": {"local": {
        "base_url": base_url, "extra_headers": {"X-Consumer": "assistant", "X-Private-Auth": "test-secret"},
    }}}))
    assert model_metadata.fetch_endpoint_model_metadata(base_url, api_key="test-token", force_refresh=True)["model"]["context_length"] == 32768
    original = next(headers for path, headers in seen if path == "/configured/v1/models")
    assert original["x-consumer"] == "assistant"
    redirected = next(headers for path, headers in seen if path == "/sink/models")
    assert not ({"authorization", "x-consumer", "x-private-auth"} & redirected.keys())
    seen.clear()
    model_metadata.fetch_endpoint_model_metadata(source_url + "/other/v1", force_refresh=True)
    assert all(not ({"x-consumer", "x-private-auth"} & headers.keys()) for _, headers in seen)


def test_metadata_cache_refreshes_after_header_or_profile_change(servers, tmp_path, monkeypatch):
    """One tenant's catalog cannot mask another header-authenticated tenant's context."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            tenant = self.headers.get("X-Tenant")
            seen.append(tenant)
            length = 32768 if tenant == "first-tenant" else 65536
            body = json.dumps({"data": [{"id": "model", "max_model_len": length}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    monkeypatch.setenv("HTTP_PROXY", servers(Handler))
    monkeypatch.setenv("NO_PROXY", "")
    base_url = "http://metadata.invalid/v1"
    first_home, second_home = tmp_path / "first", tmp_path / "second"
    for home, tenant in ((first_home, "first-tenant"), (first_home, "second-tenant"), (second_home, "second-tenant")):
        home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        (home / "config.yaml").write_text(json.dumps({"providers": {"gateway": {
            "base_url": base_url, "extra_headers": {"X-Tenant": tenant},
        }}}))
        metadata = model_metadata.fetch_endpoint_model_metadata(base_url, api_key="test-token")
        assert metadata["model"]["context_length"] == (32768 if tenant == "first-tenant" else 65536)
        model_metadata._endpoint_model_metadata_cache.clear()
        model_metadata._endpoint_model_metadata_cache_time.clear()
        assert model_metadata.fetch_endpoint_model_metadata(base_url, api_key="test-token") == metadata
        cache_text = (home / "cache" / "endpoint_model_metadata.json").read_text()
        assert "first-tenant" not in cache_text and "second-tenant" not in cache_text
    assert seen == ["first-tenant", "second-tenant", "second-tenant"]


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
    import httpx

    responses = []
    original_client = httpx.Client

    def recorded_client(**kwargs):
        kwargs["event_hooks"]["response"] = [responses.append]
        return original_client(**{**kwargs, "timeout": 2.0})

    monkeypatch.setattr(httpx, "Client", recorded_client)
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
