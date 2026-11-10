"""A ``providers:`` block whose config key carries uppercase (``Axonhub``, ``OpenAI``) is the
provider every picker row offers under that key, so selecting it must route to THAT endpoint on
each switch surface. ``resolve_provider_full`` lowercases the request before the user-config
lookup, and a case-sensitive dict read missed the key: ``OpenAI`` fell through to the legacy
``openai`` -> OpenRouter alias and ``hermes model`` / ``/model`` switched to the wrong host.

Real config on a temp ``HERMES_HOME``; loopback ``/v1/models``; no mocks on the switch chain.
"""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


class _Listing(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": "m1"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _write_config(key: str, provider: str = "openrouter") -> tuple[Path, str]:
    srv = HTTPServer(("127.0.0.1", 0), _Listing)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_port}/v1"
    path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    path.write_text(f"model:\n  provider: {provider}\n  default: x\n"
                    f"providers:\n  {key}:\n    base_url: {base_url}\n    api_key_env: MIXED_KEY\n    models: [m1]\n")
    return path, base_url


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("MIXED_KEY", "mixed-key-123456")


@pytest.mark.parametrize("key", ["Axonhub", "OpenAI"])
def test_gateway_picker_slug_switches_to_the_configured_endpoint(key):
    """The Telegram picker forwards the configured key verbatim (``mp:<key>``) to the gateway switch."""
    from gateway.slash_commands_model import GatewayModelCommandsMixin, _ModelSwitchContext

    config_path, base_url = _write_config(key)
    ctx = _ModelSwitchContext(session_key="s", source=None, config_path=config_path, persist_global=False)
    ctx.read_config()
    result, error = asyncio.run(GatewayModelCommandsMixin._perform_model_switch(SimpleNamespace(), ctx, "m1", key, None))
    assert error is None, error
    assert (result.new_model, result.base_url, result.api_key) == ("m1", base_url, "mixed-key-123456")


@pytest.mark.parametrize("requested", ["OpenAI", "openai"])
def test_cli_model_command_switches_to_the_configured_endpoint(requested):
    """CLI ``/model m1 --provider <key>``: the context the CLI loads, then its switch seam."""
    from hermes_cli.cli_model_switch_mixin import _switch_model_from
    from hermes_cli.inventory import load_picker_context

    _, base_url = _write_config("OpenAI")
    ctx = load_picker_context()
    cli = SimpleNamespace(provider="openrouter", model="x", base_url="", api_key="")
    result = _switch_model_from(cli, "m1", is_global=False, explicit_provider=requested,
                                user_providers=ctx.user_providers, custom_providers=ctx.custom_providers)
    assert result.success, result.error_message
    assert (result.base_url, result.api_key) == (base_url, "mixed-key-123456")


def test_hermes_model_marks_the_configured_mixed_case_provider_active():
    """``hermes model`` opens with the configured ``providers.OpenAI`` row selected, not OpenRouter."""
    from hermes_cli.config import load_config
    from hermes_cli.main import _resolve_active_provider
    from hermes_cli.main_provider_setup import _named_custom_provider_map

    _write_config("OpenAI", provider="OpenAI")
    cfg = load_config()
    custom_map = _named_custom_provider_map(cfg)
    assert _resolve_active_provider(cfg, cfg["model"], "OpenAI", custom_map) in custom_map
