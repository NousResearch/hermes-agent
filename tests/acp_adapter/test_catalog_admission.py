"""Keyless custom endpoints cannot borrow a failed Ollama probe's generic result."""
from __future__ import annotations

from unittest.mock import patch

from acp_adapter.model_catalog import _named_custom_provider_catalogs
from models.catalog_endpoint import EndpointModels


def test_no_key_and_no_declaration_does_not_admit_an_unverified_proxy():
    cfg = {"providers": {
        "proxy": {"name": "Proxy", "base_url": "http://127.0.0.1:11434/v1"},
    }}
    # A 11434 endpoint is a *candidate* for Ollama, not proof of native support.
    # If /api/tags failed but /v1/models worked, the former picker required a
    # credential or declared models before admitting the generic proxy.
    with patch("hermes_cli.config.load_config", return_value=cfg), patch(
        "models.catalog_endpoint.discover_endpoint_models",
        return_value=EndpointModels(("unexpected-proxy-model",)),
    ):
        assert _named_custom_provider_catalogs() == []


def test_raw_legacy_model_allowlist_survives_empty_native_catalogue():
    cfg = {"custom_providers": [{
        "name": "Legacy Local", "base_url": "http://127.0.0.1:11434/v1",
        "model": "pinned:latest", "models": ["pinned:latest"],
    }]}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch(
        "models.catalog_endpoint.discover_endpoint_models",
        return_value=EndpointModels((), native_ollama=True),
    ) as discover:
        rows = _named_custom_provider_catalogs()
    assert rows == [("custom:legacy-local", "Legacy Local", [("pinned:latest", "")])]
    assert discover.call_args.kwargs["preserve_native_models"] is True

def test_custom_endpoint_tls_policy_is_resolved_by_application(monkeypatch):
    cfg = {"providers": {
        "internal": {
            "name": "Internal", "base_url": "https://internal.example/v1",
            "api_key": "fixture-key", "default_model": "model-a",
            "ssl_verify": False,
        },
    }}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch(
        "agent.ssl_verify.resolve_httpx_verify", return_value=False,
    ) as resolve_trust, patch(
        "models.catalog_endpoint.discover_endpoint_models",
        return_value=EndpointModels(("model-a",)),
    ) as discover:
        rows = _named_custom_provider_catalogs()
    assert rows and rows[0][2] == [("model-a", "")]
    assert discover.call_args.kwargs["tls_verify"] is False
    resolve_trust.assert_called_once()
