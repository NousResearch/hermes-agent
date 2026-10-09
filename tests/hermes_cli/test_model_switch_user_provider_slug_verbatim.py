"""A model the user selected on their own ``providers.<key>`` endpoint survives ``/model``
verbatim — whether the slug arrives as ``custom:<key>`` or as the bare config key the Desktop
picker rows carry. Its ``/v1/models`` listing is a hint: an id it lacks (a newer release, a dated
snapshot) is soft-accepted, never rejected or swapped for a listed sibling (the Desktop picker
kept "going back to deepseek 0731" because the bare-key spelling fell into the built-in
live-listing branch, whose near-miss auto-correct rewrote ``deepseek-v4.1-flash``).

Loopback ``/v1/models`` server; no mocks on the validation chain.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from hermes_cli.model_switch import switch_model

LISTING = ["deepseek-v4-flash-0731", "deepseek-v4-flash", "deepseek-v4-pro"]


class _Listing(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": m} for m in LISTING]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        pass


@pytest.fixture
def endpoint(monkeypatch):
    """Loopback listing + the ``providers.hyper`` block on disk (credential resolution reads
    config.yaml, exactly as the gateway does)."""
    srv = HTTPServer(("127.0.0.1", 0), _Listing)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_port}/v1"
    monkeypatch.setenv("HYPER_KEY", "test-key-12345")
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
        "model:\n  provider: custom:hyper\n  default: deepseek-v4-flash-0731\n"
        f"providers:\n  hyper:\n    base_url: {base_url}\n    api_key_env: HYPER_KEY\n")
    try:
        yield base_url
    finally:
        srv.shutdown()


@pytest.mark.parametrize("explicit_provider", ["hyper", "custom:hyper"])
def test_unlisted_id_on_user_provider_is_kept_verbatim(endpoint, explicit_provider):
    from hermes_cli.config import get_compatible_custom_providers, load_config

    cfg = load_config()
    result = switch_model(
        raw_input="deepseek-v4.1-flash", explicit_provider=explicit_provider,
        current_provider="custom:hyper", current_model=LISTING[0],
        current_base_url=endpoint, current_api_key="test-key-12345",
        user_providers=cfg["providers"], custom_providers=get_compatible_custom_providers(cfg))
    assert result.success is True, result.error_message
    assert result.new_model == "deepseek-v4.1-flash"
    assert result.base_url == endpoint
    assert "not found" in result.warning_message  # warned, not rewritten
