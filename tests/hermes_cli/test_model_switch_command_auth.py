"""Live model switching must verify a command-authenticated custom catalog."""

import json
import shlex
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from openai import OpenAI

from agent import command_token_source
from hermes_cli.config import get_compatible_custom_providers, load_config, save_config
from hermes_cli.model_switch import switch_model
from hermes_cli.runtime_provider import resolve_runtime_provider

MODEL = "example/second-model"
TOKEN = "fixture-bearer"


@pytest.fixture
def authenticated_endpoint():
    seen = []
    auth = SimpleNamespace(token=TOKEN)

    class Endpoint(BaseHTTPRequestHandler):
        def _respond(self, path, payload):
            authorized = self.headers.get("Authorization") == f"Bearer {auth.token}"
            seen.append((self.command, self.path, authorized))
            status = 404 if self.path != path else 200 if authorized else 401
            body = json.dumps(payload if status == 200 else {"error": "not available"}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._respond("/v1/models", {"data": [{"id": MODEL}]})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._respond("/v1/responses", {"id": "resp_fixture", "model": request["model"], "status": "completed"})

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", seen, auth
        finally:
            server.shutdown()
            worker.join()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("same_provider", [True, False], ids=["same-provider", "cross-provider"])
def test_switch_to_catalog_model_with_key_cmd(tmp_path, monkeypatch, authenticated_endpoint, same_provider):
    url, seen, auth = authenticated_endpoint
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(command_token_source, "time", SimpleNamespace(monotonic=lambda: clock.now))
    token_file = tmp_path / "token.json"
    token_file.write_text(json.dumps({"access_token": TOKEN, "expires_in": 120}), encoding="utf-8")
    helper = tmp_path / "mint.py"
    helper.write_text("from pathlib import Path\nimport sys\nprint(Path(sys.argv[1]).read_text())\n", encoding="utf-8")
    current = "example/first-model"
    save_config({
        "model": {"provider": "custom:fixture-relay", "default": current},
        "providers": {"fixture-relay": {
            "api": url,
            "transport": "codex_responses",
            "key_cmd": shlex.join([sys.executable, str(helper), str(token_file)]),
        }},
    })
    config = load_config()
    if same_provider:
        runtime = resolve_runtime_provider(requested="custom:fixture-relay", target_model=current)
        provider, base_url, key = "custom", runtime["base_url"], runtime["api_key"]
    else:
        provider, base_url, key = "openai", "https://api.openai.com/v1", "previous-route-key"

    result = switch_model(
        raw_input=MODEL,
        current_provider=provider,
        current_model=current,
        current_base_url=base_url,
        current_api_key=key,
        explicit_provider="fixture-relay",  # Desktop sends the bare provider slug.
        user_providers=config["providers"],
        custom_providers=get_compatible_custom_providers(config),
    )
    assert result.success, result.error_message
    assert result.new_model == MODEL
    assert result.api_mode == "codex_responses"
    assert not result.warning_message
    assert callable(result.api_key)
    assert seen == [("GET", "/v1/models", True)]

    with OpenAI(base_url=result.base_url, api_key=result.api_key, max_retries=0) as client:
        reply = client.responses.create(model=result.new_model, input="Test")
        assert reply.status == "completed"
        assert reply.model == MODEL

        # Simulate expiry after switching; the same SDK client must send the refreshed token.
        auth.token = "rotated-fixture-bearer"
        token_file.write_text(json.dumps({"access_token": auth.token, "expires_in": 120}), encoding="utf-8")
        clock.now += 121
        refreshed = client.responses.create(model=result.new_model, input="Test after expiry")
        assert refreshed.status == "completed"
        assert refreshed.model == MODEL

    assert seen[1:] == [("POST", "/v1/responses", True)] * 2
    assert load_config() == config
