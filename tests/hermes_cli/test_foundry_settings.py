"""Native Foundry setup preserves profile ownership and uses the runtime credential path."""

import json
import os
import shlex
import sys
import threading
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from starlette.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").touch()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with TestClient(app) as client:
        client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        yield client


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("profile", ["default", "foundry-bot"])
def test_cli_test_save_and_inference_use_selected_profile(client, tmp_path, monkeypatch, profile):
    pytest.importorskip("azure.identity")
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from agent import secret_scope
    from agent.azure_identity_adapter import reset_credential_cache
    from hermes_cli.config import read_raw_config
    from hermes_cli.profiles import get_profile_dir
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.web_server_profiles import _config_profile_scope
    from openai import OpenAI

    home = get_profile_dir(profile)
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").touch()
    # A broken CLI must never fall through to the launch profile's service principal.
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("AZURE_CLIENT_ID", "launch-client")
    monkeypatch.setenv("AZURE_TENANT_ID", "launch-tenant")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "launch-secret")
    arguments = tmp_path / "az-arguments.jsonl"
    helper = tmp_path / "az.py"
    helper.write_text(
        'import json, sys, time\n'
        f'with open({str(arguments)!r}, "a") as log: log.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        'print(json.dumps({"accessToken":"cli-token", "expires_on":int(time.time())+3600}))\n'
    )
    az = tmp_path / "az"
    az.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(helper))} "$@"\n')
    az.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    requests = []
    model_requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            model_requests.append((self.path, self.headers.get("Authorization"), self.headers.get("api-key")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"value":[{"name":"my-deployment","modelName":"catalog-name","type":"ModelDeployment"}]}')

        def do_POST(self):
            requests.append((self.path, self.headers.get("Authorization"),
                             json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"id":"resp-test","object":"response","output":[]}')

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    route = f"/api/providers/azure-foundry?profile={profile}"
    payload = {"base_url": f"http://127.0.0.1:{server.server_port}/api/projects/test/openai/v1", "model": "my-deployment",
               "api_mode": "codex_responses", "auth": {"kind": "entra_id", "entra": {
                   "credential_source": "azure_cli", "tenant_id": "tenant-selected", "subscription": "subscription-selected", "scope": ""}}}
    selector = "tenant_id" if profile == "default" else "subscription"
    selected_value = "tenant-selected" if selector == "tenant_id" else "subscription-selected"
    payload["auth"]["entra"].pop("subscription" if selector == "tenant_id" else "tenant_id")
    reset_credential_cache()
    try:
        before = (home / "config.yaml").read_text()
        discovered = client.post(f"/api/providers/azure-foundry/deployments?profile={profile}", json={**payload, "model": ""})
        assert discovered.status_code == 200, discovered.text
        assert discovered.json() == {"deployments": ["my-deployment"]}
        assert model_requests == [("/api/projects/test/deployments?api-version=v1", "Bearer cli-token", None)]
        assert (home / "config.yaml").read_text() == before
        tested = client.post(f"/api/providers/azure-foundry/validate?profile={profile}", json=payload)
        assert tested.status_code == 200 and tested.json()["ok"], tested.text
        saved = client.put(route, json=payload)
        assert saved.status_code == 200, saved.text
        assert saved.json()["auth"]["entra"]["credential_source"] == "azure_cli"
        assert "cli-token" not in saved.text
        assert client.get(route).json()["auth"]["entra"][selector] == selected_value
        with _config_profile_scope(profile):
            model = read_raw_config()["model"]
            assert model["provider"] == "azure-foundry"
            assert model["entra"][selector] == selected_value
            runtime = resolve_runtime_provider(requested="azure-foundry", target_model="my-deployment")
            assert runtime["entra"][selector] == selected_value
            with OpenAI(api_key=runtime["api_key"], base_url=runtime["base_url"], max_retries=0) as inference:
                inference.responses.create(model="my-deployment", input="hi")
        assert len(requests) == 2
        assert all(path == "/api/projects/test/openai/v1/responses" and auth == "Bearer cli-token" and body["model"] == "my-deployment"
                   for path, auth, body in requests)
        for args in map(json.loads, arguments.read_text().splitlines()):
            flag = "--tenant" if selector == "tenant_id" else "--subscription"
            assert args[args.index(flag) + 1] == selected_value
            assert ("--subscription" if selector == "tenant_id" else "--tenant") not in args
        assert "cli-token" not in (home / "config.yaml").read_text()
        # Reset the cached bearer and make the CLI fail. A valid launch SP must not be tried.
        helper.write_text('import sys\nprint("sensitive-cli-output", file=sys.stderr)\nsys.exit(1)\n')
        reset_credential_cache()
        rejected = client.post(f"/api/providers/azure-foundry/validate?profile={profile}", json=payload)
        assert rejected.json()["ok"] is False
        assert "az login" in rejected.json()["message"]
        assert "sensitive-cli-output" not in rejected.text
        assert len(requests) == 2
    finally:
        reset_credential_cache()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(("base", "expected"), [
    ("https://resource.services.ai.azure.com/api/projects/project", "https://resource.services.ai.azure.com/api/projects/project/openai/v1"),
    ("https://resource.services.ai.azure.com", "https://resource.services.ai.azure.com/openai/v1"),
    ("https://resource.openai.azure.com/openai/v1/", "https://resource.openai.azure.com/openai/v1"),
    ("https://resource.services.ai.azure.com.example.com/api/projects/project", "https://resource.services.ai.azure.com.example.com/api/projects/project"),
])
def test_foundry_url_and_provider_errors_match_runtime(client, monkeypatch, base, expected):
    from hermes_cli.web_routers import azure_foundry

    requests = []

    def endpoint(request):
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "Missing responses/write. Credential: private-key"}})

    monkeypatch.setattr(azure_foundry, "_endpoint_probe_client", lambda *_: httpx.AsyncClient(transport=httpx.MockTransport(endpoint)))
    payload = {"base_url": base, "model": "deployment-alias", "api_mode": "codex_responses",
               "auth": {"kind": "api_key", "api_key": "private-key"}}
    saved = client.put("/api/providers/azure-foundry?profile=default", json=payload)
    assert saved.status_code == 200, saved.text
    assert saved.json()["base_url"] == expected
    tested = client.post("/api/providers/azure-foundry/validate?profile=default", json=payload)
    assert tested.json()["ok"] is False
    assert "Missing responses/write" in tested.json()["message"]
    assert "private-key" not in tested.text
    assert str(requests[0].url) == expected + "/responses"
    assert json.loads(requests[0].content)["model"] == "deployment-alias"


@pytest.mark.parametrize("base", ["https://[", "", "https://user:password@example.com", "https://example.com?token=secret"])
def test_bad_endpoint_never_overwrites_settings(client, base):
    from hermes_cli.config import read_raw_config

    before = read_raw_config()
    payload = {"base_url": base, "model": "deployment", "api_mode": "codex_responses",
               "auth": {"kind": "entra_id", "entra": {"credential_source": "azure_cli"}}}
    for method, route in ((client.put, "/api/providers/azure-foundry"), (client.post, "/api/providers/azure-foundry/validate"), (client.post, "/api/providers/azure-foundry/deployments")):
        result = method(route + "?profile=default", json=payload)
        assert result.status_code == 400, result.text
    assert read_raw_config() == before


def test_entra_anthropic_probe_uses_runtime_path_and_bearer_only(client, monkeypatch):
    from hermes_cli.web_routers import azure_foundry
    from agent import azure_identity_adapter

    monkeypatch.setattr(azure_identity_adapter, "build_token_provider", lambda **_: lambda: "entra-token")
    requests = []

    def endpoint(request):
        requests.append(request)
        return httpx.Response(200, json={})

    monkeypatch.setattr(azure_foundry, "_endpoint_probe_client", lambda *_: httpx.AsyncClient(transport=httpx.MockTransport(endpoint)))
    payload = {"base_url": "https://resource.services.ai.azure.com/anthropic/v1", "model": "claude-deployment",
               "api_mode": "anthropic_messages", "auth": {"kind": "entra_id", "entra": {"credential_source": "default"}}}
    result = client.post("/api/providers/azure-foundry/validate?profile=default", json=payload)
    assert result.json()["ok"], result.text
    assert str(requests[0].url) == "https://resource.services.ai.azure.com/anthropic/v1/messages"
    assert requests[0].headers["Authorization"] == "Bearer entra-token"
    assert requests[0].headers["anthropic-version"] == "2025-04-15"
    assert "x-api-key" not in requests[0].headers


@pytest.mark.parametrize("mode", ["codex_responses", "chat_completions"])
def test_unreachable_inference_is_an_actionable_test_failure(client, monkeypatch, mode):
    from hermes_cli.web_routers import azure_foundry

    def unreachable(request):
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setattr(azure_foundry, "_endpoint_probe_client", lambda *_: httpx.AsyncClient(transport=httpx.MockTransport(unreachable)))
    payload = {"base_url": "https://resource.services.ai.azure.com/openai/v1", "model": "deployment",
               "api_mode": mode, "auth": {"kind": "api_key", "api_key": "private-key"}}
    result = client.post("/api/providers/azure-foundry/validate?profile=default", json=payload)
    assert result.status_code == 200
    assert result.json()["ok"] is False
    assert "Could not validate the inference route" in result.json()["message"]
    assert "private-key" not in result.text


def test_mixed_account_selectors_are_rejected_before_test_or_save(client):
    from hermes_cli.config import read_raw_config

    before = read_raw_config()
    payload = {"base_url": "https://resource.services.ai.azure.com/openai/v1", "model": "deployment",
               "api_mode": "codex_responses", "auth": {"kind": "entra_id", "entra": {
                   "credential_source": "azure_cli", "tenant_id": "tenant", "subscription": "subscription"}}}
    for method, route in ((client.put, "/api/providers/azure-foundry"), (client.post, "/api/providers/azure-foundry/validate"), (client.post, "/api/providers/azure-foundry/deployments")):
        result = method(route + "?profile=default", json=payload)
        assert result.status_code == 422, result.text
        assert "not both" in result.text
    assert read_raw_config() == before


def test_discovery_failure_does_not_leak_credentials_or_save(client, monkeypatch):
    from hermes_cli.web_routers import azure_foundry
    from agent import azure_identity_adapter
    from hermes_cli.config import read_raw_config

    def failed(**_):
        raise RuntimeError("secret-cli-output")

    monkeypatch.setattr(azure_identity_adapter, "build_token_provider", failed)
    payload = {"base_url": "https://resource.services.ai.azure.com/openai/v1", "model": "", "api_mode": "codex_responses",
               "auth": {"kind": "entra_id", "entra": {"credential_source": "azure_cli"}}}
    before = read_raw_config()
    result = client.post("/api/providers/azure-foundry/deployments?profile=default", json=payload)
    assert result.status_code == 400
    assert "az login" in result.text
    assert "secret-cli-output" not in result.text
    assert read_raw_config() == before


def test_unavailable_model_listing_keeps_manual_setup_possible(client, monkeypatch):
    from hermes_cli import azure_foundry_deployments
    from hermes_cli.config import read_raw_config

    monkeypatch.setattr(azure_foundry_deployments, "discover_deployments", lambda *_: [])
    payload = {"base_url": "https://resource.services.ai.azure.com/openai/v1", "model": "", "api_mode": "codex_responses",
               "auth": {"kind": "api_key", "api_key": "private-key"}}
    before = read_raw_config()
    listed = client.post("/api/providers/azure-foundry/deployments?profile=default", json=payload)
    assert listed.json() == {"deployments": []}
    assert read_raw_config() == before
    saved = client.put("/api/providers/azure-foundry?profile=default", json={**payload, "model": "manual-deployment"})
    assert saved.status_code == 200
    assert saved.json()["model"] == "manual-deployment"


@pytest.mark.parametrize("key", [None, "", "  "])
def test_blank_api_key_reuses_saved_key_for_test_and_save(client, monkeypatch, key):
    from hermes_cli.web_routers import azure_foundry

    requests = []

    def endpoint(request):
        requests.append(request)
        return httpx.Response(200, json={})

    monkeypatch.setattr(azure_foundry, "_endpoint_probe_client", lambda *_: httpx.AsyncClient(transport=httpx.MockTransport(endpoint)))
    route = "/api/providers/azure-foundry?profile=default"
    payload = {"base_url": "https://resource.services.ai.azure.com/openai/v1", "model": "deployment",
               "api_mode": "codex_responses", "auth": {"kind": "api_key", "api_key": "saved-key"}}
    assert client.put(route, json=payload).status_code == 200
    payload["auth"] = {"kind": "api_key", "api_key": key}
    tested = client.post("/api/providers/azure-foundry/validate?profile=default", json=payload)
    assert tested.json()["ok"], tested.text
    assert requests[0].headers["Authorization"] == "Bearer saved-key"
    saved = client.put(route, json=payload)
    assert saved.status_code == 200, saved.text
    assert saved.json()["api_key_set"] is True
    assert "saved-key" not in client.get(route).text
