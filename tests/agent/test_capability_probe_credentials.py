"""Capability probes materialize credentials without consuming the chat source."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import threading
from unittest.mock import patch

import httpx
import pytest

from agent import auxiliary_client, image_routing, model_metadata
from agent.command_token_source import CommandTokenSource
from hermes_cli import models_local


@pytest.fixture
def anthropic_context_endpoint(monkeypatch):
    requests = []
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(
                {
                    "path": self.path,
                    "x-api-key": self.headers.get("x-api-key"),
                    "anthropic-version": self.headers.get("anthropic-version"),
                }
            )
            body = (
                b'{"data":[{"id":"claude-fixture",'
                b'"max_input_tokens":123456}]}'
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("credential,expected", [(lambda: "minted", "minted"), ("static", "static")])
def test_capability_paths_share_concrete_bearer(credential, expected):
    auxiliary_client.set_runtime_main("custom", "fixture", api_key=credential)
    try:
        assert image_routing._resolve_inference_api_key({}, "custom") == expected
        assert model_metadata._auth_headers(credential) == {"Authorization": f"Bearer {expected}"}
        assert models_local._lmstudio_request_headers(credential)["Authorization"] == f"Bearer {expected}"
        requests = []
        def capture(req):
            requests.append(req)
            return httpx.Response(200, json={"capabilities": ["thinking"]})
        client_type = httpx.Client
        with patch("httpx.Client", lambda **kwargs: client_type(
            **kwargs, transport=httpx.MockTransport(capture)
        )):
            models_local.ollama_model_supports_thinking("fixture", "http://localhost:11434/v1", credential)
        assert requests and requests[0].headers["Authorization"] == f"Bearer {expected}"
        assert auxiliary_client._runtime_main_value("api_key") is credential
    finally:
        auxiliary_client.clear_runtime_main()


def test_failed_callable_never_becomes_a_bearer(monkeypatch):
    def failed():
        raise RuntimeError("secret-bearing command failure")
    for value in (failed, lambda: object(), object(), None):
        assert model_metadata._auth_headers(value) == {}
        assert "Authorization" not in models_local._lmstudio_request_headers(value)

    from hermes_cli import models
    url = "http://localhost:11434/v1"
    configured = {"base_url": url, "api_key": "provider-fallback", "extra_headers": {
        "aUtHoRiZaTiOn": "Bearer configured-fallback", "X-Probe-Fixture": "preserved",
    }}
    monkeypatch.setattr(models, "_get_provider_config_dict", lambda _: configured)
    for value in (failed, lambda: object(), lambda: ""):
        auxiliary_client.set_runtime_main("custom", "fixture", api_key=value)
        try:
            for cfg in ({"model": {"api_key": "model-fallback"}},
                        {"providers": {"custom": {"api_key": "provider-fallback"}}}):
                assert image_routing._resolve_inference_api_key(cfg, "custom") == ""
            assert models_local._get_ollama_native_headers(url, api_key=value) == {
                "X-Probe-Fixture": "preserved",
            }
            assert auxiliary_client._runtime_main_value("api_key") is value
        finally:
            auxiliary_client.clear_runtime_main()
    # An absent explicit credential still permits configured authentication.
    assert models_local._get_ollama_native_headers(url)["aUtHoRiZaTiOn"] == "Bearer configured-fallback"
    assert image_routing._resolve_inference_api_key({"model": {"api_key": "model-fallback"}}, "custom") == "model-fallback"


@pytest.mark.parametrize("credential_kind", ["static", "command"])
def test_anthropic_context_probe_materializes_credential(
    anthropic_context_endpoint, credential_kind
):
    base_url, requests = anthropic_context_endpoint
    credential = "minted" if credential_kind == "static" else CommandTokenSource(
        f'"{sys.executable}" -c "print(\'minted\')"', "fixture"
    )

    assert model_metadata._query_anthropic_context_length(
        "claude-fixture", base_url, credential
    ) == 123456
    assert requests == [
        {
            "path": "/v1/models?limit=1000",
            "x-api-key": "minted",
            "anthropic-version": "2023-06-01",
        }
    ]


def test_anthropic_context_probe_skips_unusable_credentials(
    anthropic_context_endpoint,
):
    base_url, requests = anthropic_context_endpoint

    def failed():
        raise RuntimeError("secret-bearing command failure")

    for credential in (
        failed,
        lambda: object(),
        lambda: "",
        lambda: "sk-ant-oat-fixture",
        None,
    ):
        assert model_metadata._query_anthropic_context_length(
            "claude-fixture", base_url, credential
        ) is None
    assert requests == []
