"""Where the Anthropic context-window probe carries the credential.

``_query_anthropic_context_length`` asks ``/v1/models`` with ``x-api-key``. The Messages client puts
only a Console key there and sends every OAuth-shaped credential as a bearer token, so the probe
either carries a credential in the header that client uses for it, or sends nothing.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch

import pytest

from agent.anthropic_adapter import build_anthropic_client
from agent.model_metadata import _query_anthropic_context_length

MODEL = "claude-probe-model"
WINDOW = 345_678

CONSOLE_KEYS = [
    pytest.param("sk-ant-api03-placement-test", id="console-key"),
    pytest.param("sk-ant-admin01-placement-test", id="admin-key"),
]
OAUTH_SHAPED = [
    pytest.param("sk-ant-oat01-placement-test", id="setup-token"),
    pytest.param("eyJhbGciOiJub25lIn0.eyJzdWIiOiJwbGFjZW1lbnQifQ.c2ln", id="jwt"),
    pytest.param("cc-placement-test", id="cc-prefix"),
    pytest.param("sk-ant-new01-placement-test", id="unlisted-sk-ant-prefix"),
]


@pytest.fixture
def models_endpoint(monkeypatch):
    """A loopback ``/v1/models`` that records the headers of every request it receives."""
    seen: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append({name.lower(): value for name, value in self.headers.items()})
            body = json.dumps({"data": [{"id": MODEL, "max_input_tokens": WINDOW}]}).encode()
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


def _carriers(seen: list[dict[str, str]], credential: str) -> set[str]:
    """Names of the request headers whose value holds *credential*, over every recorded request."""
    return {name for headers in seen for name, value in headers.items() if credential in value}


def _messages_client_carrier(credential: str) -> str:
    """The header the Messages client sends *credential* in on ``api.anthropic.com``."""
    with patch("agent.anthropic_adapter._anthropic_sdk") as sdk:
        build_anthropic_client(credential, "https://api.anthropic.com")
    kwargs = sdk.Anthropic.call_args[1]
    return "authorization" if kwargs.get("auth_token") == credential else "x-api-key"


@pytest.mark.parametrize("credential", CONSOLE_KEYS)
def test_console_key_is_probed_as_x_api_key(models_endpoint, credential):
    base_url, seen = models_endpoint
    assert _query_anthropic_context_length(MODEL, base_url, credential) == WINDOW
    assert _carriers(seen, credential) == {"x-api-key"}


@pytest.mark.parametrize("credential", OAUTH_SHAPED)
def test_oauth_shaped_credential_never_rides_as_x_api_key(models_endpoint, credential):
    base_url, seen = models_endpoint
    window = _query_anthropic_context_length(MODEL, base_url, credential)
    assert "x-api-key" not in _carriers(seen, credential)
    assert seen == []
    assert window is None


@pytest.mark.parametrize("credential", CONSOLE_KEYS + OAUTH_SHAPED)
def test_probe_carries_a_credential_only_where_the_messages_client_does(models_endpoint, credential):
    base_url, seen = models_endpoint
    _query_anthropic_context_length(MODEL, base_url, credential)
    assert _carriers(seen, credential) <= {_messages_client_carrier(credential)}
