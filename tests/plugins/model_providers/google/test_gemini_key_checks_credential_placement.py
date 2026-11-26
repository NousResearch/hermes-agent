"""Where the Gemini key checks carry the API key.

Model requests send a Google AI Studio key in the ``x-goog-api-key`` header (``GeminiNativeClient``).
The setup tier probe and the dashboard's key check make their own requests to the same host; each
carries the key in that header and nowhere else, so it never rides in a URL.
"""

from __future__ import annotations

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from plugins.model_providers.google.native_adapter import GeminiNativeClient, probe_gemini_tier

KEY = "AQ.placement-test-key"


def _model_request_carriers() -> set[str]:
    """Names of the headers a model request carries the key in."""
    client = GeminiNativeClient(api_key=KEY)
    try:
        return {name.lower() for name, value in client._auth_headers().items() if value == KEY}
    finally:
        client.close()


@pytest.fixture
def generate_content_endpoint(monkeypatch):
    """A loopback Gemini host that records the request line and headers of every POST."""
    seen: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            seen.append({"(request line)": self.path, **{name.lower(): value for name, value in self.headers.items()}})
            body = json.dumps({"candidates": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_model_requests_carry_the_key_as_x_goog_api_key():
    assert _model_request_carriers() == {"x-goog-api-key"}


def test_tier_probe_carries_the_key_where_model_requests_do(generate_content_endpoint):
    base_url, seen = generate_content_endpoint

    assert probe_gemini_tier(KEY, base_url) == "paid"

    assert len(seen) == 1
    carriers = {name for name, value in seen[0].items() if KEY in value}
    assert carriers == _model_request_carriers()
    assert seen[0]["(request line)"].endswith(":generateContent")


def test_dashboard_key_check_carries_the_key_where_model_requests_do(monkeypatch):
    import httpx

    import hermes_cli.web_routers.config_env as mod
    from hermes_cli.web_models import EnvVarUpdate

    seen: list[dict[str, str]] = []

    class _Resp:
        status_code = 200
        is_success = True

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, *, headers=None, params=None, **kwargs):
            sent = httpx.Request("GET", url, headers=headers, params=params)
            seen.append({"(request line)": str(sent.url), **{name.lower(): value for name, value in sent.headers.items()}})
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(mod, "_require_token", lambda request: None)

    body = EnvVarUpdate(key="GEMINI_API_KEY", value=KEY)
    out = asyncio.run(mod.validate_provider_credential(body, request=None))  # type: ignore[arg-type]

    assert out["ok"] is True
    assert len(seen) == 1
    carriers = {name for name, value in seen[0].items() if KEY in value}
    assert carriers == _model_request_carriers()
    assert seen[0]["(request line)"] == "https://generativelanguage.googleapis.com/v1beta/models"
