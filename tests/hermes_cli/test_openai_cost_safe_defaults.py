"""Implicit first-party model selection must not escalate the previous Sol price budget."""

import pytest

from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
from hermes_cli.models import detect_static_provider_for_model, get_default_model_for_provider


@pytest.mark.parametrize("provider", ["openai-api", "openai-codex"])
def test_static_default_stays_within_sol_price_budget(provider):
    # Use API rates for both catalogs: Codex's included billing would hide an expensive default.
    model = get_default_model_for_provider(provider)
    for usage in (CanonicalUsage(input_tokens=1_000), CanonicalUsage(output_tokens=1_000)):
        baseline = estimate_usage_cost("gpt-6.1-sol", usage, provider="openai-api")
        selected = estimate_usage_cost(model, usage, provider="openai-api")
        assert selected.amount_usd is not None and baseline.amount_usd is not None
        assert selected.amount_usd <= baseline.amount_usd, model


@pytest.mark.parametrize("provider", ["openai-api", "openai-codex"])
def test_bare_provider_switch_uses_the_cost_safe_default(provider):
    selected = detect_static_provider_for_model(provider, "anthropic")
    assert selected is not None
    assert selected == (provider, get_default_model_for_provider(provider))
    usage = CanonicalUsage(input_tokens=1_000, output_tokens=1_000)
    baseline = estimate_usage_cost("gpt-6.1-sol", usage, provider="openai-api")
    cost = estimate_usage_cost(selected[1], usage, provider="openai-api")
    assert cost.amount_usd is not None and baseline.amount_usd is not None
    assert cost.amount_usd <= baseline.amount_usd, selected
