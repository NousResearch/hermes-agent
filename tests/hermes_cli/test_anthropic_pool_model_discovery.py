from __future__ import annotations

import json

from hermes_cli import models


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


def test_anthropic_picker_discovers_models_with_pool_api_key(monkeypatch):
    """A direct API key stored only in auth.json must reach /v1/models."""
    monkeypatch.setattr(models, "_get_model_config_dict", lambda: {"provider": "nous"})
    monkeypatch.setattr(
        "agent.anthropic_adapter.resolve_anthropic_token",
        lambda: None,
    )
    monkeypatch.setattr(
        "hermes_cli.auth.read_credential_pool",
        lambda provider: [
            {
                "auth_type": "api_key",
                "access_token": "sk-ant-api03-pool-key",
                "base_url": "https://api.anthropic.com",
            }
        ] if provider == "anthropic" else [],
    )

    captured = {}

    def _open(request, *, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["timeout"] = timeout
        return _Response({"data": [{"id": "claude-opus-5"}]})

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", _open)

    result = models.provider_model_ids("anthropic")

    assert "claude-opus-5" in result
    assert captured["url"] == "https://api.anthropic.com/v1/models"
    assert captured["headers"]["x-api-key"] == "sk-ant-api03-pool-key"
    assert "authorization" not in captured["headers"]


def test_anthropic_pool_api_key_uses_its_configured_endpoint(monkeypatch):
    """Never send a proxy-scoped pool credential to Anthropic's public host."""
    monkeypatch.setattr(
        "agent.anthropic_adapter.resolve_anthropic_token",
        lambda: None,
    )
    monkeypatch.setattr(
        "hermes_cli.auth.read_credential_pool",
        lambda _provider: [
            {
                "auth_type": "api_key",
                "access_token": "proxy-key",
                "base_url": "https://proxy.example/anthropic/v1",
            }
        ],
    )

    captured = {}

    def _open(request, *, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        return _Response({"data": [{"id": "claude-proxy-model"}]})

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", _open)

    assert models._fetch_anthropic_models() == ["claude-proxy-model"]
    assert captured["url"] == "https://proxy.example/anthropic/v1/models"
    assert captured["headers"]["x-api-key"] == "proxy-key"
