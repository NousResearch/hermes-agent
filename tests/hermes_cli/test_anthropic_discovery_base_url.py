"""Regression tests: Anthropic model discovery ignores ANTHROPIC_BASE_URL, and a
declined live fetch is not flagged as a placeholder.

Drop into tests/hermes_cli/ (same fixture style as
tests/hermes_cli/test_anthropic_models_pagination.py). Both tests FAIL on main
and pass once discovery honours the env endpoint and returns the placeholder
type.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import pytest

from hermes_cli import models as M

STUB_MODELS = ["claude-opus-1", "claude-opus-2", "claude-future-9-99"]


class _Handler(BaseHTTPRequestHandler):
    hits: list[str] = []

    def log_message(self, *args):  # noqa: D102
        pass

    def do_GET(self):  # noqa: N802
        type(self).hits.append(self.path)
        if not urlparse(self.path).path.endswith("/models"):
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({
            "data": [{"id": m, "type": "model"} for m in STUB_MODELS],
            "has_more": False,
            "first_id": STUB_MODELS[0],
            "last_id": STUB_MODELS[-1],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def anthropic_stub(monkeypatch):
    handler = type("Handler", (_Handler,), {"hits": []})
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    try:
        yield handler
    finally:
        srv.shutdown()


def test_picker_probes_the_endpoint_from_anthropic_base_url(anthropic_stub):
    """A relay/gateway endpoint set in the environment must be probed, not api.anthropic.com."""
    result = M.provider_model_ids("anthropic", force_refresh=True)

    assert anthropic_stub.hits, "discovery never contacted ANTHROPIC_BASE_URL"
    assert "claude-future-9-99" in result, "live-only model from the env endpoint is missing"


def test_declined_live_fetch_is_flagged_as_a_placeholder(monkeypatch):
    """A failed probe must be flagged so the disk cache re-probes instead of caching it as live."""
    monkeypatch.setattr(M, "_fetch_anthropic_models", lambda *a, **k: None)

    result = M.provider_model_ids("anthropic", force_refresh=True)

    assert result == list(M._PROVIDER_MODELS["anthropic"]), "content contract changed"
    assert isinstance(result, M.CuratedFallbackModels), (
        "a declined Anthropic probe returned a plain list — the disk cache then stores it with "
        "fallback=False and serves the static catalog for the full stale window"
    )
