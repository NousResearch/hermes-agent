"""Configured route headers reach discovery, not only inference."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hermes_cli import models
from hermes_constants import get_hermes_home


@pytest.fixture
def workspace_server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append({key.lower(): value for key, value in self.headers.items()})
            allowed = self.headers.get("anthropic-workspace-id") == "workspace-test"
            page_two = "after_id=" in self.path
            payload = {
                "data": [{"id": "claude-fixture-b" if page_two else "claude-fixture-a"}],
                "has_more": not page_two,
                "last_id": "claude-fixture-a",
            } if allowed else {"error": "workspace header required"}
            raw = json.dumps(payload).encode()
            self.send_response(200 if allowed else 400)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def configure(base, shape="providers"):
    entry = {"name": "fixture", "base_url": base, "api_mode": "anthropic_messages",
             "extra_headers": {"anthropic-workspace-id": "workspace-test"}}
    config = {shape: {"fixture": entry} if shape == "providers" else [entry]}
    # JSON is also YAML; real config loader and endpoint matching are exercised.
    (get_hermes_home() / "config.yaml").write_text(json.dumps(config), encoding="utf-8")


@pytest.mark.parametrize("shape", ["providers", "custom_providers"])
def test_workspace_headers_reach_every_catalog_page(workspace_server, shape):
    base, seen = workspace_server
    configure(base, shape)
    result = models._fetch_anthropic_models(base_url=base, api_key="sk-ant-api-fixture")
    assert result == ["claude-fixture-a", "claude-fixture-b"]
    assert len(seen) == 2
    assert all(row.get("anthropic-workspace-id") == "workspace-test" for row in seen)
    assert all(row.get("x-api-key") == "sk-ant-api-fixture" for row in seen)


@pytest.mark.parametrize("base", [None, "https://api.anthropic.com/v1/"])
def test_native_endpoint_uses_configured_headers(monkeypatch, base):
    configure(base or "https://api.anthropic.com")
    captured = []
    def get(url, **kwargs):
        captured.append(kwargs["headers"].copy())
        return {"data": []}
    monkeypatch.setattr(models, "_get_json", get)
    assert models._fetch_anthropic_models(base_url=base, api_key="sk-ant-api-fixture") == []
    assert captured[0]["anthropic-workspace-id"] == "workspace-test"


def test_pool_endpoint_not_discarded_caller_selects_headers(monkeypatch, workspace_server):
    from agent import anthropic_credentials
    base, seen = workspace_server
    configure(base)
    monkeypatch.setattr(anthropic_credentials, "resolve_anthropic_token", lambda: None)
    monkeypatch.setattr(models, "_resolve_anthropic_pool_catalog_credentials", lambda: ("sk-ant-api-fixture", base))
    assert models._fetch_anthropic_models(base_url="https://unrelated.invalid") == ["claude-fixture-a", "claude-fixture-b"]
    assert len(seen) == 2


def test_unmatched_endpoint_keeps_headers_out(monkeypatch):
    configure("https://other.invalid")
    captured = []
    def get(url, **kwargs):
        captured.append(kwargs["headers"].copy())
        return {"data": []}
    monkeypatch.setattr(models, "_get_json", get)
    assert models._fetch_anthropic_models(api_key="sk-ant-api-fixture") == []
    assert "anthropic-workspace-id" not in captured[0]
    assert captured[0]["x-api-key"] == "sk-ant-api-fixture"


def test_failed_catalog_preserves_fallback(monkeypatch):
    configure("https://api.anthropic.com")
    def fail(*args, **kwargs):
        raise OSError("fixture unavailable")
    monkeypatch.setattr(models, "_get_json", fail)
    assert models._fetch_anthropic_models(api_key="sk-ant-api-fixture") is None
    monkeypatch.setattr(models, "_get_model_config_dict", lambda: {"provider": "anthropic", "api_key": "sk-ant-api-fixture"})
    result = models.provider_model_ids("anthropic", force_refresh=True)
    assert result == list(models._PROVIDER_MODELS["anthropic"])


def test_oauth_beta_retry_keeps_workspace_header(monkeypatch):
    import io
    from urllib.error import HTTPError

    from agent.anthropic_adapter import _CONTEXT_1M_BETA

    configure("https://api.anthropic.com")
    captured = []

    def get(url, **kwargs):
        captured.append(kwargs["headers"].copy())
        if len(captured) == 1:
            raise HTTPError(url, 400, "fixture", {}, io.BytesIO(
                b"long context beta is not yet available for this subscription"))
        return {"data": [{"id": "claude-fixture"}]}

    monkeypatch.setattr(models, "_get_json", get)
    token = "sk-ant-oat01-fixture"
    assert models._fetch_anthropic_models(api_key=token) == ["claude-fixture"]
    assert len(captured) == 2
    assert all(row["anthropic-workspace-id"] == "workspace-test" for row in captured)
    assert all(row["Authorization"] == f"Bearer {token}" for row in captured)
    assert _CONTEXT_1M_BETA not in captured[1]["anthropic-beta"]


def test_configured_header_precedence_matches_inference(monkeypatch):
    from hermes_cli import config
    from agent.anthropic_adapter import _custom_provider_extra_headers

    configure("https://api.anthropic.com")
    path = get_hermes_home() / "config.yaml"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["providers"]["fixture"]["extra_headers"]["anthropic-version"] = "fixture-version"
    path.write_text(json.dumps(data), encoding="utf-8")
    captured = []

    def get(url, **kwargs):
        captured.append(kwargs["headers"].copy())
        return {"data": []}

    monkeypatch.setattr(models, "_get_json", get)
    assert models._fetch_anthropic_models(api_key="sk-ant-api-fixture") == []
    expected = _custom_provider_extra_headers("https://api.anthropic.com")
    assert expected == config.get_custom_provider_extra_headers("https://api.anthropic.com")
    assert expected.items() <= captured[0].items()
