"""Pricing/accounting ``/models`` probes must authenticate like the real call.

Regression for #75479: the pricing probe ran without the provider's configured
API key, so auth-gated providers answered 401 and costs came back "unknown".
``resolve_probe_api_key`` now resolves the configured key (config entries matched
by base URL, host-gated env vars) and the three ``estimate_usage_cost`` call
sites (aux accounting, insights, langfuse) pass it through.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent.insights import _estimate_cost

_PROBE_KEY = "probe-key-123"
_MODEL = "acme-probe-model"


class _AuthedModelsHandler(BaseHTTPRequestHandler):
    """401s /models without the configured key, like an auth-gated provider."""

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server naming
        if self.path == "/models":
            if self.headers.get("Authorization") == f"Bearer {_PROBE_KEY}":
                self._send(200, {"data": [{
                    "id": _MODEL,
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                }]})
            else:
                self._send(401, {"error": {"message": "auth required"}})
        else:
            self._send(404, {})

    def log_message(self, *args):  # keep the test output clean
        pass


@pytest.fixture()
def authed_models_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AuthedModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join()


@pytest.fixture()
def configured_provider(authed_models_server):
    """A providers.<name> entry with base_url + api_key in the temp HERMES_HOME."""
    home = os.environ["HERMES_HOME"]
    with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(
            "providers:\n"
            "  acme:\n"
            f"    base_url: {authed_models_server}\n"
            f"    api_key: {_PROBE_KEY}\n"
        )
    return authed_models_server


def test_estimate_cost_sends_configured_key_to_authed_models_probe(
    configured_provider,
):
    """The insights call site authenticates the probe: 401 on base, priced here."""
    session = {
        "model": _MODEL,
        "billing_provider": "acme",
        "billing_base_url": configured_provider,
        "input_tokens": 1_000,
        "output_tokens": 500,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    amount, status = _estimate_cost(session)

    assert status == "estimated"
    assert amount == pytest.approx(0.002)


def test_probe_key_never_leaks_to_unrelated_host(configured_provider):
    """Keys are bound to their endpoint: a name match alone never authenticates
    another host, while the endpoint's own key is returned whatever the caller
    named the provider."""
    from agent.model_metadata import resolve_probe_api_key

    assert resolve_probe_api_key("acme", "https://unrelated.example/v1") == ""
    assert resolve_probe_api_key("no-such-provider", configured_provider) == _PROBE_KEY
    assert resolve_probe_api_key("acme", configured_provider) == _PROBE_KEY
