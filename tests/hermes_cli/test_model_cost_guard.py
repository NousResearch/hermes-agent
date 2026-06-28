from decimal import Decimal

import pytest

from agent.models_dev import ModelInfo
from agent.usage_pricing import PricingEntry
from hermes_cli.model_cost_guard import expensive_model_warning


def test_no_warning_when_known_prices_are_at_threshold():
    info = ModelInfo(
        id="edge/model",
        name="edge/model",
        family="",
        provider_id="anthropic",
        cost_input=20.0,
        cost_output=100.0,
    )

    assert expensive_model_warning("edge/model", provider="anthropic", model_info=info) is None


def test_warns_when_models_dev_input_price_exceeds_threshold():
    info = ModelInfo(
        id="expensive/input",
        name="expensive/input",
        family="",
        provider_id="anthropic",
        cost_input=20.01,
        cost_output=1.0,
    )

    warning = expensive_model_warning(
        "expensive/input",
        provider="anthropic",
        model_info=info,
    )

    assert warning is not None
    assert warning.input_cost_per_million == Decimal("20.01")
    assert "EXPENSIVE MODEL WARNING" in warning.message
    assert "$20/M input" in warning.message


@pytest.mark.parametrize("provider", ["custom", "custom:routerai", "routerai"])
def test_skips_foreign_models_dev_pricing_for_custom_or_unknown_providers(provider):
    info = ModelInfo(
        id="openai/gpt-5.5-pro",
        name="openai/gpt-5.5-pro",
        family="",
        provider_id="openrouter",
        cost_input=25.0,
        cost_output=125.0,
    )

    assert (
        expensive_model_warning(
            "openai/gpt-5.5-pro",
            provider=provider,
            model_info=info,
        )
        is None
    )


def test_skips_untrusted_provider_pricing_lookup_for_custom_provider(monkeypatch):
    monkeypatch.setattr("agent.models_dev.get_model_info", lambda *_args, **_kwargs: None)
    pricing_calls = []

    def fake_get_pricing_entry(*_args, **_kwargs):
        pricing_calls.append(_args)
        return PricingEntry(
            input_cost_per_million=Decimal("25"),
            output_cost_per_million=Decimal("125"),
            source="provider_models_api",
        )

    monkeypatch.setattr("agent.usage_pricing.get_pricing_entry", fake_get_pricing_entry)

    warning = expensive_model_warning(
        "openai/gpt-5.5-pro",
        provider="custom:routerai",
        base_url="https://routerai.example/v1",
    )

    assert warning is None
    assert pricing_calls == []


def test_warns_when_pricing_entry_output_price_exceeds_threshold(monkeypatch):
    monkeypatch.setattr("agent.models_dev.get_model_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "agent.usage_pricing.get_pricing_entry",
        lambda *_args, **_kwargs: PricingEntry(
            input_cost_per_million=Decimal("1.00"),
            output_cost_per_million=Decimal("100.01"),
            source="provider_models_api",
        ),
    )

    warning = expensive_model_warning("provider/expensive-output", provider="openrouter")

    assert warning is not None
    assert warning.output_cost_per_million == Decimal("100.01")
    assert "$100.01/M" in warning.message


def test_openai_gpt55_pro_adds_suggestion(monkeypatch):
    monkeypatch.setattr("agent.models_dev.get_model_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "agent.usage_pricing.get_pricing_entry",
        lambda *_args, **_kwargs: PricingEntry(
            input_cost_per_million=Decimal("25"),
            output_cost_per_million=Decimal("125"),
            source="provider_models_api",
        ),
    )

    warning = expensive_model_warning("openai/gpt-5.5-pro", provider="openrouter")

    assert warning is not None
    assert "did you mean to select openai/gpt-5.5?" in warning.message


def test_openai_gpt55_pro_warns_for_nous_portal_pricing(monkeypatch):
    monkeypatch.setattr("agent.models_dev.get_model_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_endpoint_model_metadata",
        lambda base_url, api_key="": {
            "openai/gpt-5.5-pro": {
                "pricing": {
                    "prompt": "0.000025",
                    "completion": "0.000125",
                }
            }
        },
    )

    warning = expensive_model_warning("openai/gpt-5.5-pro", provider="nous")

    assert warning is not None
    assert warning.input_cost_per_million == Decimal("25.000000")
    assert warning.output_cost_per_million == Decimal("125.000000")
    assert "did you mean to select openai/gpt-5.5?" in warning.message
