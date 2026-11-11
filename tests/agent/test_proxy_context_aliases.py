"""Proxy aliases fall back to their family windows without overriding live metadata."""
import pytest
from agent import model_metadata as metadata


@pytest.mark.parametrize("model,expected", [
    ("opus-latest", 1_000_000),
    ("sonnet-latest", 1_000_000),
    ("agy-opus", 1_000_000),
    ("haiku-latest", 200_000),
])
def test_proxy_alias_probe_down_fallback(monkeypatch, model, expected):
    monkeypatch.setattr(metadata, "_resolve_endpoint_context_length", lambda *a, **kw: None)
    monkeypatch.setattr(metadata, "_probe_local_context_length", lambda *a, **kw: None)
    monkeypatch.setattr(metadata, "_query_ollama_api_show", lambda *a, **kw: None)
    assert metadata._resolve_custom_endpoint_context_length(
        model, "https://proxy.example/v1", "", "custom:proxy"
    ) == expected


def test_endpoint_metadata_precedes_proxy_alias_fallback(monkeypatch):
    monkeypatch.setattr(metadata, "_resolve_endpoint_context_length", lambda *a, **kw: 64_000)
    assert metadata._resolve_custom_endpoint_context_length(
        "opus-latest", "https://proxy.example/v1", "", "custom:proxy"
    ) == 64_000
