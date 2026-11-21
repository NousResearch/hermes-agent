"""OpenRouter OpenAI tier aliases: generic rule, base-pin preservation, final wire shape."""

from __future__ import annotations

import pytest

from agent.model_metadata import get_model_context_length
from agent.transports.chat_completions import ChatCompletionsTransport
from hermes_cli import models as models_module
from hermes_cli.models_reasoning_caps import openrouter_model_reasoning_capabilities
from hermes_constants import openrouter_openai_tier_alias
from providers import get_provider_profile

VALID_ALIASES = [
    ("openai/gpt-6-astra-fast", "openai/gpt-6-astra", "fast"),
    ("openai/gpt-6-astra-flex", "openai/gpt-6-astra", "flex"),
    ("openai/gpt-6-astra-pro-fast", "openai/gpt-6-astra-pro", "fast"),
    ("openai/gpt-6-astra-pro-flex", "openai/gpt-6-astra-pro", "flex"),
    ("openai/gpt-6.1-terra-fast", "openai/gpt-6.1-terra", "fast"),
    ("openai/gpt-6.1-terra-flex", "openai/gpt-6.1-terra", "flex"),
]


@pytest.mark.parametrize("alias,base,tier", VALID_ALIASES)
def test_parser_recognises_openai_tier_aliases(alias, base, tier):
    assert openrouter_openai_tier_alias(alias) == (base, tier)


@pytest.mark.parametrize(
    "model",
    [
        None,
        "",
        "anthropic/claude-opus-5-fast",  # real model id, not an alias
        "openai/-flex",  # malformed empty base
        "openai/gpt-6.1-terra-fast-flex",  # chained suffix
        "openai/gpt-6.1-terra-flex:floor",  # routing variant, not a tier alias
        "openai/gpt-5.6-sol",
    ],
)
def test_parser_leaves_non_aliases_untouched(model):
    assert openrouter_openai_tier_alias(model) is None


@pytest.mark.parametrize("alias,base,tier", VALID_ALIASES)
def test_profile_rewrites_alias_and_pins_tier_endpoint(alias, base, tier):
    p = get_provider_profile("openrouter")
    prefs = {"ignore": ["deepinfra"], "sort": "price"}
    body = p.build_extra_body(model=alias, provider_preferences=prefs)
    assert body["provider"] == {
        "ignore": ["deepinfra"],
        "sort": "price",
        "only": [f"openai/{tier}"],
    }
    assert prefs == {"ignore": ["deepinfra"], "sort": "price"}  # caller dict not mutated

    _, top_level = p.build_api_kwargs_extras(model=alias)
    assert top_level["model"] == base


def test_tier_pin_overrides_conflicting_user_only():
    p = get_provider_profile("openrouter")
    body = p.build_extra_body(
        model="openai/gpt-6.1-terra-flex",
        provider_preferences={"only": ["azure"], "sort": "price"},
    )
    assert body["provider"]["only"] == ["openai/flex"]
    assert body["provider"]["sort"] == "price"


@pytest.mark.parametrize(
    "base,tags",
    [
        ("openai/gpt-6-astra", ("openai", "azure", "azure/us")),
        ("openai/gpt-6-astra-pro", ("openai", "azure", "azure/us")),
    ],
)
def test_existing_base_standard_pins_are_preserved(base, tags):
    p = get_provider_profile("openrouter")
    body = p.build_extra_body(model=base, provider_preferences={"ignore": ["x"]})
    assert body["provider"]["only"] == list(tags)
    _, top_level = p.build_api_kwargs_extras(model=base)
    assert "model" not in top_level

    # An explicit user `only` still wins on a base slug.
    explicit = p.build_extra_body(model=base, provider_preferences={"only": ["custom"]})
    assert explicit["provider"]["only"] == ["custom"]


@pytest.mark.parametrize("model", ["openai/gpt-6.1-terra", "openai/gpt-5.6-sol", "anthropic/claude-opus-5-fast"])
def test_unrelated_models_are_not_rewritten(model):
    p = get_provider_profile("openrouter")
    body = p.build_extra_body(model=model, provider_preferences={"ignore": ["x"]})
    assert "only" not in body.get("provider", {})
    _, top_level = p.build_api_kwargs_extras(model=model)
    assert "model" not in top_level


def test_final_chat_completions_request_uses_base_model_and_flex_endpoint():
    kwargs = ChatCompletionsTransport().build_kwargs(
        "openai/gpt-6.1-terra-flex",
        [{"role": "user", "content": "hi"}],
        tools=None,
        provider_profile=get_provider_profile("openrouter"),
        base_url="https://openrouter.ai/api/v1",
    )
    assert kwargs["model"] == "openai/gpt-6.1-terra"
    assert kwargs["extra_body"]["provider"]["only"] == ["openai/flex"]


def test_reasoning_caps_lookup_uses_base_model(monkeypatch):
    caps = {"supports_reasoning": True, "supported_efforts": ["high"], "mandatory": False}
    monkeypatch.setattr(
        models_module, "_openrouter_reasoning_caps_cache", {"openai/gpt-6.1-terra": caps}
    )
    assert openrouter_model_reasoning_capabilities("openai/gpt-6.1-terra-flex") == caps


def test_picker_keeps_tier_alias_when_live_catalog_lists_only_base(monkeypatch):
    live_items = [
        {
            "id": "openai/gpt-6.1-terra",
            "supported_parameters": ["tools"],
            "pricing": {"prompt": "0.000001", "completion": "0.000005"},
        }
    ]
    monkeypatch.setattr(
        models_module,
        "_fetch_live_catalog_index",
        lambda *args, **kwargs: (live_items, {item["id"]: item for item in live_items}),
    )
    monkeypatch.setattr(
        "hermes_cli.model_catalog.get_curated_openrouter_models",
        lambda: [
            ("openai/gpt-6.1-terra", ""),
            ("openai/gpt-6.1-terra-flex", "0.5x price, flex tier"),
        ],
    )
    ids = dict(models_module.fetch_openrouter_models(force_refresh=True))
    assert ids["openai/gpt-6.1-terra-flex"] == "0.5x price, flex tier"


@pytest.mark.parametrize("suffix", ["fast", "flex"])
def test_tier_alias_returns_base_context_window(monkeypatch, suffix):
    catalog = {"openai/gpt-6.1-terra": {"context_length": 1_000_000}}
    monkeypatch.setattr("agent.model_metadata.get_cached_context_length", lambda *a, **k: None)
    monkeypatch.setattr("agent.models_dev.lookup_models_dev_context", lambda *a, **k: None)
    monkeypatch.setattr("agent.model_metadata.fetch_model_metadata", lambda *a, **k: catalog)
    assert get_model_context_length(f"openai/gpt-6.1-terra-{suffix}", provider="openrouter") == 1_000_000
