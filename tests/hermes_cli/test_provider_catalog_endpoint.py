"""Provider discovery follows the configured inference host, including cached pickers."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import urlparse

import pytest

from hermes_cli.models import _api_key_credentials, cached_provider_model_ids
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_constants import get_hermes_home


@pytest.mark.parametrize(
    ("provider", "key_env"),
    [("merge-gateway", "MERGE_GATEWAY_API_KEY"), ("gmi", "GMI_API_KEY")],
)
def test_cached_catalog_tracks_configured_inference_endpoint(monkeypatch, provider, key_env):
    monkeypatch.setenv(key_env, "catalog-test-key")
    requests = []

    class CatalogHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            requests.append((path, self.headers.get("Authorization")))
            model = "vendor/" + path.split("/")[1]
            body = json.dumps({
                "data": [{
                    "id": model,
                    "model": model,
                    "availability_status": "available",
                    "vendors": {"test": {
                        "availability_status": "available",
                        "capabilities": {"supports_tool_calling": True},
                    }},
                }],
                "has_more": False,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    from hermes_cli import urllib_security
    real_open = urllib_security.open_credentialed_url

    def local_only(req, **kwargs):
        assert urlparse(req.full_url).hostname == "127.0.0.1", req.full_url
        return real_open(req, **kwargs)

    monkeypatch.setattr(urllib_security, "open_credentialed_url", local_only)
    monkeypatch.setattr("hermes_cli.models.open_credentialed_url", local_only)
    server = HTTPServer(("127.0.0.1", 0), CatalogHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for endpoint in ("first", "second", "first"):
            base = f"http://127.0.0.1:{server.server_port}/{endpoint}/v1"
            if provider == "merge-gateway":
                base += "/openai"
            config = {"model": {"provider": provider, "default": "vendor/test", "base_url": base}}
            (get_hermes_home() / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
            runtime = resolve_runtime_provider(requested=provider)
            assert runtime["base_url"] == base
            models = cached_provider_model_ids(provider)
            assert f"vendor/{endpoint}" in models
            assert requests[-1] == (f"/{endpoint}/v1/models", "Bearer catalog-test-key")
            count = len(requests)
            assert cached_provider_model_ids(provider) == models
            assert len(requests) == count
        assert len(requests) == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_catalog_does_not_borrow_another_providers_endpoint(monkeypatch):
    monkeypatch.setenv("MERGE_GATEWAY_API_KEY", "catalog-test-key")
    original = _api_key_credentials("merge-gateway")
    config = {"model": {"provider": "openai", "base_url": "https://other.example/v1"}}
    (get_hermes_home() / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    assert _api_key_credentials("merge-gateway") == original
