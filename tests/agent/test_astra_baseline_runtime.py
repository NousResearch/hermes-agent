"""Focused real-path coverage for the GPT-6 Astra baseline contract."""

from types import SimpleNamespace

import pytest


def test_explicit_astra_resolves_and_uses_official_responses(monkeypatch, tmp_path):
    """A fresh profile resolves metadata and routes the official endpoint without live I/O."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *args, **kwargs: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *args, **kwargs: [])

    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-6-astra",
        provider="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        platform="cli",
        max_iterations=2,
        quiet_mode=True,
        skip_memory=True,
    )

    assert agent.api_mode == "codex_responses"
    assert agent.context_compressor.context_length == 1_050_000
    kwargs = agent._get_transport().build_kwargs(
        model=agent.model,
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        provider=agent.provider,
        base_url=agent.base_url,
        reasoning_config={"enabled": True, "effort": "none"},
    )
    assert kwargs["reasoning"]["effort"] == "low"  # Astra has no ``none`` wire level


def test_astra_codex_oauth_fallback_uses_backend_context_limit():
    """OAuth keeps the Codex backend's 272K fallback; direct API metadata remains 1.05M."""
    from agent.model_metadata import (
        DEFAULT_CONTEXT_LENGTHS,
        _resolve_codex_oauth_context_length_with_source,
    )

    codex_ctx, source = _resolve_codex_oauth_context_length_with_source("gpt-6-astra")
    assert source == "fallback"
    assert codex_ctx < DEFAULT_CONTEXT_LENGTHS["gpt-6-astra"]  # Codex caps below the direct API window


@pytest.mark.parametrize("advertised,expected", [(272_000, 900_000), (200_000, 200_000), (1_050_000, 1_050_000)])
def test_astra_900k_opt_in_preserves_live_limits_and_wire_contract(monkeypatch, tmp_path, advertised, expected):
    """Only the known stale advertisement is lifted; the alias never reaches the wire."""
    from agent import model_metadata as metadata
    from agent.reasoning_effort import CODEX_ASTRA_EFFORTS, codex_supported_efforts
    from agent.transports.codex import ResponsesApiTransport

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(metadata, "_codex_oauth_context_cache", {})
    monkeypatch.setattr(metadata.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        status_code=200,
        json=lambda: {"models": [{"slug": "gpt-6-astra", "context_window": advertised}]},
    ))
    route = {"base_url": "https://chatgpt.com/backend-api/codex", "provider": "openai-codex"}
    assert metadata.get_model_context_length("gpt-6-astra-900k", api_key="test-token", **route) == expected
    assert metadata.get_model_context_length("gpt-6-astra", api_key="test-token", **route) == advertised
    assert codex_supported_efforts("gpt-6-astra-900k") == CODEX_ASTRA_EFFORTS

    for config in ({"effort": "max"}, {"enabled": False}):
        params = dict(
            messages=[{"role": "user", "content": "Hi"}], tools=[],
            is_codex_backend=True, reasoning_config=config,
            request_overrides={"temperature": 0.5, "logprobs": True}, **route,
        )
        kwargs = ResponsesApiTransport().build_kwargs(model="gpt-6-astra-900k", **params)
        assert kwargs["model"] == "gpt-6-astra"
        assert kwargs == ResponsesApiTransport().build_kwargs(model="gpt-6-astra", **params)


@pytest.mark.parametrize("provider,model", [
    ("openai-codex", "gpt-6-astra"), ("openai-codex", "gpt-6-astra-900k"), ("openai-api", "gpt-6-astra"),
])
def test_picker_revalidates_cached_astra_and_never_injects_saved_entitlement(monkeypatch, tmp_path, provider, model):
    from hermes_cli import models
    from hermes_cli.inventory import ConfigContext, _append_unconfigured_rows
    from hermes_cli.model_switch_providers import _finalize_picker_rows

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(models, "_credential_fingerprint", lambda _: "synthetic-account")
    models.update_provider_cache_entry(provider, ["gpt-5.6-sol", model])
    live = []
    calls = []

    def discover(slug, **kwargs):
        calls.append(slug)
        return list(live)

    monkeypatch.setattr(models, "provider_model_ids", discover)
    # Only live discovery writes Astra into a same-credential entry, so a fresh entry IS the
    # entitlement record: served without a round-trip (a per-open revalidation defeated the cache).
    assert model in models.cached_provider_model_ids(provider)
    assert calls == []
    # After the entry ages past the fresh window, a failed refresh must not resurrect Astra from it.
    monkeypatch.setattr(models, "_PROVIDER_MODELS_STALE_SERVE_MAX", 0)
    assert models.cached_provider_model_ids(provider, ttl_seconds=0) == ["gpt-5.6-sol"]
    assert calls == [provider]
    row = {"slug": provider, "is_current": True, "models": ["gpt-5.6-sol"], "total_models": 1}
    assert model not in _finalize_picker_rows([row], {}, model)[0]["models"]
    ctx = ConfigContext(provider, model, "", {}, [])
    assert _append_unconfigured_rows([], ctx, current_only=True)[0]["models"] == []

    live[:] = ["gpt-5.6-sol", model]
    assert model in models.cached_provider_model_ids(provider)
