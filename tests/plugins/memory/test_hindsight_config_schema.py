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
        "llm_api_key",
        "llm_provider",
        "llm_base_url",
        "llm_model",
        "api_url",
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
    assert mode.allowed_values() == {"cloud", "local_embedded", "local_external"}


def test_api_key_is_a_secret_bound_to_env():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    api_key = next(field for field in provider.fields if field.key == "api_key")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "HINDSIGHT_API_KEY"


def test_local_embedded_llm_fields_are_declared():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    fields = {field.key: field for field in provider.fields}
    llm_api_key = fields["llm_api_key"]
    assert llm_api_key.kind == KIND_SECRET
    assert llm_api_key.is_secret is True
    assert llm_api_key.env_key == "HINDSIGHT_LLM_API_KEY"

    llm_provider = fields["llm_provider"]
    assert llm_provider.kind == KIND_SELECT
    assert llm_provider.default == "openai"
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
    }
