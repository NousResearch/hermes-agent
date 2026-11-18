"""Codex subscription defaults stay separate from extended context and API limits."""

import base64
import json
from unittest.mock import Mock

import pytest

from agent import model_metadata as mm


@pytest.fixture
def codex_catalog(monkeypatch):
    monkeypatch.setattr(mm, "_codex_oauth_context_cache", {})
    monkeypatch.setattr(mm, "_codex_oauth_max_context_cache", {})
    response = Mock(status_code=200)
    monkeypatch.setattr(mm.model_metadata_http, "get", Mock(return_value=response))
    payload = base64.urlsafe_b64encode(json.dumps({"sub": "sol-context-test"}).encode()).decode().rstrip("=")
    return response, f"e30.{payload}.test-signature"


@pytest.mark.parametrize("model,provider,expected", [
    ("gpt-6.1-sol", "openai-codex", 500_000),
    ("openai/gpt-6.1-sol", "openai-codex", 500_000),
    ("gpt-6.1-sol-2026-09-29", "openai-codex", 500_000),
    ("gpt-6.1-sol-900k", "openai-codex", 900_000),
    ("gpt-6-astra", "openai-codex", 272_000),
    ("gpt-6-sol", "openai-codex", 272_000),
    ("gpt-6-luna", "openai-codex", 272_000),
    ("gpt-5.6-sol", "openai-codex", 272_000),
    ("gpt-5.5", "openai-codex", 272_000),
    ("gpt-6.1-sol", "openai", 1_050_000),
])
def test_sol_fallback_preserves_route_and_variant_limits(codex_catalog, model, provider, expected):
    response, token = codex_catalog
    response.status_code = 503
    assert mm.get_model_context_length(model, provider=provider, api_key=token if provider == "openai-codex" else "") == expected


@pytest.mark.parametrize("model,advertised,catalog_max,expected", [
    ("gpt-6.1-sol", 500_000, 872_000, 500_000),
    ("gpt-6.1-sol-900k", 500_000, None, 900_000),
    ("gpt-6.1-sol-900k", 500_000, 872_000, 872_000),
    ("gpt-6.1-sol-900k", 500_000, 1_050_000, 900_000),
    ("gpt-6.1-sol-900k", 272_000, 872_000, 872_000),
    ("openai/gpt-6.1-sol-900k", 500_000, 872_000, 872_000),
    ("gpt-6.1-sol-900k", 200_000, 872_000, 200_000),
    ("gpt-6.1-sol-900k", 600_000, 872_000, 600_000),
    ("gpt-6.1-sol-900k", 1_050_000, 1_050_000, 1_050_000),
    ("gpt-6-sol-900k", 500_000, 872_000, 500_000),
])
def test_sol_catalogue_preserves_default_and_opt_in_cap(codex_catalog, model, advertised, catalog_max, expected):
    response, token = codex_catalog
    slug = mm.strip_codex_context_variant_suffix(model).rsplit("/", 1)[-1]
    item = {"slug": slug, "context_window": advertised}
    if catalog_max is not None:
        item["max_context_window"] = catalog_max
    response.json.return_value = {"models": [item]}

    for _ in range(2):
        assert mm.get_model_context_length(model, provider="openai-codex", api_key=token) == expected
    assert mm.model_metadata_http.get.call_count == 1


@pytest.mark.parametrize("model,window,threshold", [
    ("gpt-6.1-sol", 500_000, 327_000),
    ("gpt-6.1-sol-900k", 900_000, 418_000),
])
def test_sol_default_reaches_compressor_and_model_display(codex_catalog, model, window, threshold):
    from agent.context_compressor import ContextCompressor
    from hermes_cli.model_switch import resolve_display_context_length

    response, token = codex_catalog
    response.status_code = 503
    compressor = ContextCompressor(
        model=model, provider="openai-codex", api_key=token,
        max_tokens=64_000, threshold_percent=0.50, quiet_mode=True,
    )
    assert compressor.context_length == window
    assert compressor.threshold_tokens == threshold
    assert resolve_display_context_length(model, "openai-codex", api_key=token) == window
