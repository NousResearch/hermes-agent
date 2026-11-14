"""The published Sol subscription window replaces its legacy catalog advertisement."""

from types import SimpleNamespace

import pytest

from agent import model_metadata as mm


@pytest.mark.parametrize("catalog", [{}, {"gpt-6.1-sol": 272_000}])
def test_sol_uses_published_default_without_changing_other_codex_models(monkeypatch, catalog):
    monkeypatch.setattr(mm, "_fetch_codex_oauth_context_lengths_with_source", lambda *a, **k: (catalog, True))
    monkeypatch.setattr(mm, "_codex_oauth_max_context_cache", {})
    route = dict(provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex", api_key="test")
    assert mm.get_model_context_length("gpt-6.1-sol", **route) == 500_000
    assert mm.get_model_context_length("gpt-6-luna", **route) == 272_000
    assert mm.get_model_context_length("gpt-6.1-sol-900k", **route) == 900_000


@pytest.mark.parametrize("advertised", [128_000, 500_000, 600_000])
def test_sol_keeps_current_catalog_values_and_caps_explicit_large_window(monkeypatch, advertised):
    monkeypatch.setattr(mm, "_codex_oauth_context_cache", {})
    monkeypatch.setattr(mm, "_codex_oauth_max_context_cache", {})
    response = SimpleNamespace(status_code=200, json=lambda: {"models": [{
        "slug": "gpt-6.1-sol", "context_window": advertised, "max_context_window": 600_000,
    }]})
    monkeypatch.setattr(mm.model_metadata_http, "get", lambda *a, **k: response)
    token = "test.eyJzdWIiOiJzb2wtY29udGV4dC10ZXN0In0.test"
    route = dict(provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex", api_key=token)
    assert mm.get_model_context_length("gpt-6.1-sol", **route) == advertised
    expected = 600_000 if advertised == 500_000 else advertised
    assert mm.get_model_context_length("gpt-6.1-sol-900k", **route) == expected
