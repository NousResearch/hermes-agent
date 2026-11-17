"""GPT-6.2 first-party request and billing contracts, with an empty registry cache."""

from datetime import datetime, UTC
from decimal import Decimal
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("model,codex_window", [("gpt-6.2-astra", 272_000), ("gpt-6.2-sol", 500_000)])
def test_first_party_model_routes_keep_capabilities_and_reasoning(monkeypatch, model, codex_window):
    from agent import models_dev
    from agent.model_metadata import _resolve_codex_oauth_context_length_with_source, has_codex_context_variant
    from agent.reasoning_effort import codex_supported_efforts, is_astra_model
    from hermes_cli.codex_models import get_codex_model_ids
    from hermes_cli.model_normalize import normalize_model_for_provider
    from hermes_cli import models
    from run_agent import AIAgent

    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda *a, **k: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **k: SimpleNamespace())
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])
    monkeypatch.setattr("hermes_cli.codex_models._read_default_model", lambda *a: None)
    monkeypatch.setattr("hermes_cli.codex_models._read_cache_models", lambda *a: [])
    monkeypatch.setattr(models, "fetch_api_models", lambda *a, **k: [model, "text-embedding-3-large"])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert models.provider_model_ids("openai-api", force_refresh=True) == [model]
    assert model in get_codex_model_ids()
    assert not is_astra_model(model)  # GPT-6 Astra's preview entitlement gate must not hide 6.2.
    assert not has_codex_context_variant(model)  # No unverified 900K opt-in.
    assert _resolve_codex_oauth_context_length_with_source(model) == (codex_window, "fallback")
    info = models_dev.get_model_info("openai", model)
    assert info is not None
    assert (info.context_window, info.max_input, info.max_output) == (1_050_000, 922_000, 128_000)
    assert info.supports_vision() and info.tool_call and info.reasoning and not info.temperature
    assert info.input_modalities == ("text", "image") and info.output_modalities == ("text",)

    agent = AIAgent(model=model, provider="openai", api_key="test-key", base_url="https://api.openai.com/v1",
                    quiet_mode=True, skip_memory=True, max_iterations=2)
    assert agent.api_mode == "codex_responses"  # Chat Completions cannot call tools on 6.2.
    assert agent.context_compressor.context_length == info.context_window
    transport = agent._get_transport()
    tool = {"type": "function", "function": {"name": "lookup", "description": "Look up a value",
            "parameters": {"type": "object", "properties": {}}}}
    for provider, url in (("openai", "https://api.openai.com/v1"),
                          ("openai", "https://us.api.openai.com/v1"),
                          ("openai", "https://eu.api.openai.com/v1"),
                          ("openai-codex", "https://chatgpt.com/backend-api/codex")):
        normalized = normalize_model_for_provider(f"openai/{model}", "openai-codex")
        assert normalized == model
        for effort in ("none", "minimal", "max", "ultra"):
            kwargs = transport.build_kwargs(
                normalized, [{"role": "user", "content": "Hi"}], tools=[tool], provider=provider,
                base_url=url, is_codex_backend=provider == "openai-codex", reasoning_config={"effort": effort},
            )
            assert kwargs["reasoning"]["effort"] == ("low" if effort in ("none", "minimal") else "max")
            assert kwargs["tools"][0]["name"] == "lookup"
        assert "none" not in codex_supported_efforts(f"openai/{model}")
        overrides = {"temperature": 0.5, "top_p": 0.5, "top_logprobs": 1, "logprobs": True,
                     "include": ["reasoning.encrypted_content", "message.output_text.logprobs"],
                     "reasoning": {"effort": "none"}, "prompt_cache_retention": "24h"}
        kwargs = transport.build_kwargs(model, [{"role": "user", "content": "Hi"}], provider=provider,
                                       base_url=url, is_codex_backend=provider == "openai-codex",
                                       request_overrides=overrides)
        wire = transport.preflight_kwargs(kwargs, sanitize_harmony_tokens=True)
        assert wire["reasoning"]["effort"] == "low"
        assert not ({"temperature", "top_p", "top_logprobs", "logprobs", "prompt_cache_retention"} & wire.keys())
        assert wire["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize("model,rate", [("gpt-6.2-astra", Decimal(8)), ("gpt-6.2-sol", Decimal(2))])
def test_published_tiers_price_the_served_request_not_the_subscription(monkeypatch, model, rate):
    from agent import usage_pricing as pricing
    from hermes_cli.models import resolve_fast_mode_overrides

    # The launch discount changes at midnight Pacific on March 1, not midnight UTC.
    for now, multiplier in ((datetime(2027, 3, 1, 7, 59, tzinfo=UTC), 3),
                            (datetime(2027, 3, 1, 8, 0, tzinfo=UTC), 6)):
        monkeypatch.setattr(pricing, "_UTC_NOW", lambda: now)
        for prompt in (272_000, 272_001):
            long = prompt > 272_000
            usage = pricing.CanonicalUsage(input_tokens=prompt - 1500, cache_read_tokens=1000,
                                           cache_write_tokens=500, output_tokens=2000)
            expected = (Decimal(usage.input_tokens) * rate * (2 if long else 1)
                        + Decimal(1000) * rate / 20 * (2 if long else 1)
                        + Decimal(500) * rate * Decimal("1.25") * (2 if long else 1)
                        + Decimal(2000) * rate * 5 * (Decimal("1.5") if long else 1)) / 1_000_000
            standard = pricing.estimate_usage_cost(model, usage, provider="openai-api")
            assert standard.status == "estimated" and standard.amount_usd == expected
            fast = pricing.with_served_service_tier(usage, SimpleNamespace(service_tier="priority"))
            assert pricing.estimate_usage_cost(model, fast, provider="openai-api").amount_usd == expected * 2
            served = pricing.with_served_service_tier(usage, SimpleNamespace(service_tier="ultrafast"))
            ultra = pricing.estimate_usage_cost(f"openai/{model}", served, provider="openai-api")
            assert ultra.amount_usd == expected * multiplier
            assert pricing.estimate_usage_cost(model, served, provider="openai-codex").status == "included"
    for host in ("api.openai.com", "us.api.openai.com", "eu.api.openai.com"):
        route = {"provider": "openai-api", "base_url": f"https://{host}/v1"}
        assert resolve_fast_mode_overrides(model, **route) == {"service_tier": "priority"}
        ultra = resolve_fast_mode_overrides(model, tier="ultrafast", **route)
        assert ultra == (None if host.startswith("eu.") else {"service_tier": "ultrafast"})


def test_new_contract_does_not_capture_predecessors_or_invent_variants():
    from agent.reasoning_effort import CODEX_LEGACY_EFFORTS, codex_supported_efforts
    from hermes_cli.models import model_supports_ultrafast, resolve_fast_mode_overrides

    assert "none" in codex_supported_efforts("gpt-6-sol")
    for model in ("gpt-6.2-astra-pro", "gpt-6.2-sol-mini", "gpt-6.2-sol-900k", "gpt-6.20-sol"):
        assert codex_supported_efforts(model) == CODEX_LEGACY_EFFORTS
        assert not model_supports_ultrafast(model)
    for model in ("gpt-6.2-astra", "gpt-6.2-sol"):
        assert resolve_fast_mode_overrides(model, provider="openrouter", base_url="https://openrouter.ai/api/v1",
                                           tier="ultrafast") is None
