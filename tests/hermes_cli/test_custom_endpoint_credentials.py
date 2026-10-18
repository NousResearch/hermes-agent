"""Desktop credential commands must behave like the selected profile's runtime."""

import json
import os
import shlex
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@contextmanager
def endpoint_server(catalog_status=200):
    received = []
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.headers.get("Authorization"))
            self.send_response(catalog_status if received[-1] == "Bearer bot-token" else 401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"deployment"}]}')

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            received.append(self.headers.get("Authorization"))
            status = self.server.inference_status if received[-1] == "Bearer bot-token" else 401
            self.send_response(status if self.path == "/v1/responses" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{}')

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.inference_status = 200
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received, requests, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def client(monkeypatch):
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from starlette.testclient import TestClient

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    with TestClient(app) as client:
        client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        yield client


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(("source", "catalog_status", "legacy"), [
    ("command", 200, False), ("command", 404, False), ("command", 403, False), ("command", 200, True),
])
def test_command_credential_test_save_and_runtime_use_bot_profile(client, tmp_path, monkeypatch, source, catalog_status, legacy):
    from hermes_cli.profiles import get_profile_dir
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.web_server_profiles import _config_profile_scope

    home = get_profile_dir("credential-bot")
    home.mkdir(parents=True)
    (home / "config.yaml").touch()
    (home / ".env").write_text("TEST_BOT_TOKEN=bot-token\n")
    monkeypatch.setenv("TEST_BOT_TOKEN", "launch-token")
    helper = tmp_path / "credential.py"
    helper.write_text('import os, json, time\nprint(json.dumps({"access_token": os.environ["TEST_BOT_TOKEN"], "expires_on": int(time.time()) + 3600}))\n')
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(helper))}"
    route = "/api/providers/custom-endpoints"
    with endpoint_server(catalog_status) as (base_url, received, requests, server):
        payload = {"id": "foundry-local", "name": "Foundry local", "base_url": base_url,
                   "model": "deployment", "api_mode": "codex_responses", "make_default": True,
                   "credential_source": source}
        if source == "command":
            payload["key_cmd"] = command
        if legacy:
            from hermes_cli.config import atomic_config_write, DEFAULT_CONFIG

            atomic_config_write(home / "config.yaml", {
                "custom_providers": [{"name": "Foundry local", "base_url": base_url, "key_cmd": command}],
                "_config_version": DEFAULT_CONFIG["_config_version"],
            })
            payload.pop("key_cmd")
            with _config_profile_scope("credential-bot"):
                legacy_runtime = resolve_runtime_provider(requested="custom:foundry-local", target_model="deployment")
                assert legacy_runtime["api_key"]() == "bot-token"
        launch_payload = {**payload, "key_cmd": command} if legacy else payload
        launch = client.post(route + "/validate", json=launch_payload)
        assert launch.json()["ok"] is False
        received.clear()
        tested = client.post(route + "/validate?profile=credential-bot", json=payload)
        assert tested.status_code == 200
        assert tested.json()["ok"], tested.json()
        saved = client.post(route + "?profile=credential-bot", json=payload)
        assert saved.status_code == 200, saved.text
        row = saved.json()["endpoints"][0]
        assert row["credential_source"] == source
        assert row["is_current"] is True
        assert command not in saved.text  # a helper can embed secrets
        # Blank edit/Test fields preserve the saved helper, just like a saved API key.
        payload.pop("key_cmd", None)
        tested = client.post(route + "/validate?profile=credential-bot", json=payload)
        assert tested.json()["ok"], tested.json()
        with _config_profile_scope("credential-bot"):
            runtime = resolve_runtime_provider(requested="custom:foundry-local", target_model="deployment")
            assert callable(runtime["api_key"])
            assert runtime["api_key"]() == "bot-token"
            assert runtime["api_mode"] == "codex_responses"
            from openai import OpenAI

            with OpenAI(api_key=runtime["api_key"], base_url=runtime["base_url"], max_retries=0) as inference:
                inference.responses.create(
                    model="deployment", input="hi", max_output_tokens=16,
                    reasoning={"effort": "low"},
                    tools=[{"type": "function", "name": "check", "parameters": {"type": "object", "properties": {}}}],
                    tool_choice="none",
                )
            assert requests[-1]["model"] == "deployment"
            assert requests[-1]["reasoning"] == {"effort": "low"}
            assert requests[-1]["tools"][0]["name"] == "check"
        assert received and set(received) == {"Bearer bot-token"}
        server.inference_status = 400
        rejected = client.post(route + "/validate?profile=credential-bot", json=payload)
        assert rejected.json()["ok"] is False
        assert "400" in rejected.json()["message"]
        launch = client.post(route + "/validate", json={**payload, "key_cmd": command})
        assert launch.json()["ok"] is False
        assert received[-1] == "Bearer launch-token"
    assert "bot-token" not in (home / "config.yaml").read_text()


@pytest.mark.platforms("posix")
def test_switching_auth_clears_stale_command_and_failed_helper_is_private(client, tmp_path):
    from hermes_cli.config import read_raw_config

    helper = tmp_path / "failure.py"
    helper.write_text('import sys\nprint("private-token")\nprint("private-stderr", file=sys.stderr)\nsys.exit(1)\n')
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(helper))}"
    payload = {"id": "command-test", "name": "Command Test", "base_url": "http://127.0.0.1:1/v1",
               "model": "deployment", "credential_source": "command", "key_cmd": command}
    route = "/api/providers/custom-endpoints"
    failed = client.post(route + "/validate", json=payload)
    assert failed.json()["ok"] is False
    assert "key_cmd" in failed.json()["message"]
    assert all(secret not in failed.text for secret in (command, "private-token", "private-stderr"))
    assert client.post(route, json=payload).status_code == 200
    assert read_raw_config()["providers"]["command-test"]["key_cmd"] == command
    payload.update(credential_source="api_key", api_key="static-token")
    payload.pop("key_cmd")
    assert client.post(route, json=payload).status_code == 200
    entry = read_raw_config()["providers"]["command-test"]
    assert "key_cmd" not in entry
    assert entry["key_env"]
    assert "static-token" not in str(entry)
    payload.update(credential_source="command", key_cmd=command, make_default=True, model="gpt-image-1")
    payload.pop("api_key")
    rejected = client.post(route, json=payload)
    assert rejected.status_code == 400
    assert "generation model" in rejected.json()["detail"]
    assert read_raw_config()["providers"]["command-test"] == entry
    from hermes_constants import get_hermes_home

    assert "static-token" in (get_hermes_home() / ".env").read_text()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("remaining", [3600, -1])
def test_azure_epoch_expiry_controls_command_cache(remaining):
    import time
    from agent.command_token_source import CommandTokenError, CommandTokenSource

    deadline = int(time.time()) + remaining
    source = CommandTokenSource(f"printf '%s' '{{\"access_token\":\"token\",\"expires_on\":\"{deadline}\"}}'", "azure")
    if remaining < 0:
        with pytest.raises(CommandTokenError, match="expired expires_on"):
            source()
    else:
        assert source() == "token"
        assert 3500 < source._expires_at - time.monotonic() < 3600


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("status", [400, 401, 403])
def test_inference_failure_explains_provider_error_without_token(client, monkeypatch, status):
    import httpx
    from hermes_cli.web_routers import config_env

    def endpoint(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "deployment"}]})
        return httpx.Response(status, json={"error": {"message": "Missing RBAC permission. Credential: private-token"}})

    monkeypatch.setattr(config_env, "_endpoint_probe_client", lambda *_args: httpx.AsyncClient(transport=httpx.MockTransport(endpoint)))
    program = 'print("private-token")'
    payload = {"id": "project", "name": "Project", "base_url": "https://resource.services.ai.azure.com/openai/v1",
               "model": "deployment", "api_mode": "codex_responses", "credential_source": "command",
               "key_cmd": f'{shlex.quote(sys.executable)} -c {shlex.quote(program)}'}
    response = client.post("/api/providers/custom-endpoints/validate", json=payload)
    assert response.json()["ok"] is False
    assert f"HTTP {status}" in response.json()["message"]
    assert "Missing RBAC permission" in response.json()["message"]
    assert "private-token" not in response.text
    if status == 403:
        assert "access denied" in response.json()["message"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(("base_url", "model", "limit_field"), [
    ("https://resource.openai.azure.com/openai/v1", "my-deployment", "max_completion_tokens"),
    ("https://api.openai.com/v1", "my-model", "max_completion_tokens"),
    ("https://gateway.example.com/v1", "gpt-5.2", "max_completion_tokens"),
    ("https://gateway.example.com/v1", "llama-3", "max_tokens"),
])
def test_strict_chat_probe_uses_runtime_completion_limit_field(client, monkeypatch, base_url, model, limit_field):
    import httpx
    from hermes_cli.web_routers import config_env

    requests = []

    def endpoint(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": model}]})
        body = json.loads(request.content)
        requests.append(body)
        valid = limit_field in body and request.url.path.endswith("/chat/completions")
        return httpx.Response(200 if valid else 400, json={})

    monkeypatch.setattr(config_env, "_endpoint_probe_client", lambda *_args: httpx.AsyncClient(transport=httpx.MockTransport(endpoint)))
    program = 'print("test-token")'
    payload = {"id": "chat", "name": "Chat", "base_url": base_url, "model": model,
               "api_mode": "chat_completions", "credential_source": "command",
               "key_cmd": f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"}
    response = client.post("/api/providers/custom-endpoints/validate", json=payload)
    assert response.json()["ok"], response.json()
    assert requests[-1][limit_field] > 0


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("mode", ["codex_responses", "anthropic_messages"])
def test_saved_command_probe_preserves_runtime_headers_and_redacts_them(client, monkeypatch, mode):
    import httpx
    from hermes_constants import get_hermes_home
    from hermes_cli.config import atomic_config_write, DEFAULT_CONFIG
    from hermes_cli.web_routers import config_env

    program = 'print("helper-token")'
    atomic_config_write(get_hermes_home() / "config.yaml", {
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "providers": {"gateway": {
            "base_url": "https://gateway.example.com/v1", "model": "deployment",
            "api_mode": mode,
            "key_cmd": f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}",
            "extra_headers": {"X-Gateway-Key": "gateway-private", "authorization": "Bearer override-private",
                              "X-API-KEY": "anthropic-private", "Anthropic-Version": "2023-01-01"},
        }},
    })
    requested = []
    reject = False

    def endpoint(request):
        requested.append(request)
        authorized = (request.headers.get("X-Gateway-Key") == "gateway-private"
                      and request.headers.get("Authorization") == "Bearer override-private")
        if mode == "anthropic_messages":
            authorized = authorized and request.headers.get("x-api-key") == "anthropic-private"
            authorized = authorized and request.headers.get("anthropic-version") == "2023-01-01"
        if not authorized:
            return httpx.Response(401, json={})
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "deployment"}]})
        if reject:
            return httpx.Response(403, json={"error": {"message": "Denied override-private gateway-private anthropic-private"}})
        return httpx.Response(200, json={})

    monkeypatch.setattr(config_env, "_endpoint_probe_client", lambda *_args: httpx.AsyncClient(transport=httpx.MockTransport(endpoint)))
    payload = {"id": "gateway", "name": "Gateway", "base_url": "https://gateway.example.com/v1",
               "model": "deployment", "api_mode": mode, "credential_source": "command"}
    response = client.post("/api/providers/custom-endpoints/validate", json=payload)
    assert response.json()["ok"], response.json()
    assert {request.method for request in requested} == {"GET", "POST"}
    reject = True
    response = client.post("/api/providers/custom-endpoints/validate", json=payload)
    assert response.json()["ok"] is False
    assert "Denied" in response.json()["message"]
    assert "override-private" not in response.text
    assert "gateway-private" not in response.text
    assert "anthropic-private" not in response.text
