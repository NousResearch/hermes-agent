"""Tests for Hindsight's declared config surface."""

from plugins.memory.config_schema import (
    KIND_SECRET,
    KIND_SELECT,
    get_provider_config_schema,
)


def test_hindsight_is_declared():
    provider = get_provider_config_schema("hindsight")

    assert provider is not None
    assert provider.label == "Hindsight"
    assert {field.key for field in provider.fields} == {
        "mode",
        "api_key",
        "api_url",
        "llm_provider",
        "llm_base_url",
        "llm_api_key",
        "llm_model",
        "bank_id",
        "recall_budget",
    }


def test_fields_are_all_inline():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    # Hindsight is simple enough to render fully in the compact panel, so it
    # never grows a Full config… modal.
    assert all(field.inline for field in provider.fields)


def test_mode_gating_is_expressed_as_select_options():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    mode = next(field for field in provider.fields if field.key == "mode")
    assert mode.kind == KIND_SELECT
    assert mode.allowed_values() == {"cloud", "local_external", "local_embedded"}


def test_hermes_is_an_embedded_llm_option():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    llm_provider = next(field for field in provider.fields if field.key == "llm_provider")
    assert llm_provider.kind == KIND_SELECT
    assert llm_provider.allowed_values() == {
        "openai",
        "anthropic",
        "gemini",
        "groq",
        "openrouter",
        "minimax",
        "ollama",
        "lmstudio",
        "openai_compatible",
        "hermes",
    }


def test_mode_specific_fields_declare_visibility_conditions():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    fields = {field.key: field for field in provider.fields}
    assert dict(fields["api_key"].when) == {"mode": "cloud|local_external"}
    assert dict(fields["llm_base_url"].when) == {
        "mode": "local_embedded",
        "llm_provider": "openai_compatible|openrouter",
    }
    assert dict(fields["llm_api_key"].when) == {
        "mode": "local_embedded",
        "llm_provider": "openai|anthropic|gemini|groq|openrouter|minimax|ollama|lmstudio|openai_compatible",
    }


def test_api_key_is_a_secret_bound_to_env():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    api_key = next(field for field in provider.fields if field.key == "api_key")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "HINDSIGHT_API_KEY"
