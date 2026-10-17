"""Selection must reject unusable contexts before callers commit the route."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_cli.config import atomic_config_write, get_compatible_custom_providers
from hermes_cli.model_switch import switch_model


@pytest.fixture
def selection_home(tmp_path, monkeypatch):
    requests = []

    class Catalog(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(("GET", self.path))
            body = json.dumps({"data": [{"id": "working-chat"}, {"id": "small-chat"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            requests.append(("POST", self.path))
            self.send_error(405)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Catalog)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Only the external vendor catalog is replaced; route/config/client resolution is real.
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    base = f"http://127.0.0.1:{server.server_port}/v1"
    cfg = {
        "model": {"default": "working-chat", "provider": "selection-test", "base_url": base},
        "providers": {"selection-test": {
            "base_url": base, "api_key": "dummy-test-key", "default_model": "working-chat",
            "models": {"working-chat": {"context_length": 131072},
                       "small-chat": {"context_length": 32768}},
        }},
    }
    try:
        yield home, cfg, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("target,auxiliary,compression_enabled,pin,expected", [
    ("small-chat", None, True, None, False),
    ("small-chat", None, True, 65536, False),  # A different route cannot inherit the default's pin.
    ("working-chat", "small-chat", True, None, False),
    ("working-chat", "working-chat", True, None, True),
    ("working-chat", "small-chat", False, None, True),
    ("working-chat", "working-chat", True, 64000, True),
])
def test_selection_context_is_validated_before_persistence(
    selection_home, target, auxiliary, compression_enabled, pin, expected,
):
    home, cfg, requests = selection_home
    cfg["compression"] = {"enabled": compression_enabled}
    if pin is not None:
        cfg["model"]["context_length"] = pin
    if auxiliary:
        cfg["auxiliary"] = {"compression": {"provider": "selection-test", "model": auxiliary}}
    atomic_config_write(home / "config.yaml", cfg)
    before = (home / "config.yaml").read_bytes()
    result = switch_model(
        target, "selection-test", "working-chat", current_base_url=cfg["model"]["base_url"],
        current_api_key="dummy-test-key", user_providers=cfg["providers"],
        custom_providers=get_compatible_custom_providers(cfg), is_global=True,
    )
    assert result.success is expected
    if not expected:
        assert "32,768" in result.error_message
        assert "64,000" in result.error_message
    assert (home / "config.yaml").read_bytes() == before
    assert requests  # Real catalog validation, without an inference request.
    assert all(method == "GET" for method, _path in requests)
