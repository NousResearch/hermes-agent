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
        "bank_id",
        "recall_budget",
        "llm_provider",
        "llm_base_url",
        "llm_api_key",
        "llm_model",
        "idle_timeout",
    }


def test_inline_fields_are_the_always_visible_curated_subset():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    # The always-on connection basics render in the compact panel...
    assert {field.key for field in provider.fields if field.inline} == {
        "mode",
        "api_key",
        "api_url",
        "bank_id",
        "recall_budget",
    }
    # ...while the local_embedded-only LLM sub-form is full-config-modal-only,
    # since it only applies once that mode is selected.
    assert {field.key for field in provider.fields if not field.inline} == {
        "llm_provider",
        "llm_base_url",
        "llm_api_key",
        "llm_model",
        "idle_timeout",
    }


def test_mode_gating_is_expressed_as_select_options():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    mode = next(field for field in provider.fields if field.key == "mode")
    assert mode.kind == KIND_SELECT
    assert mode.allowed_values() == {"cloud", "local_embedded", "local_external"}


def test_llm_fields_are_gated_behind_local_embedded_mode():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    llm_fields = {"llm_provider", "llm_api_key", "llm_model", "idle_timeout"}
    for field in provider.fields:
        if field.key in llm_fields:
            assert field.is_visible({"mode": "local_embedded"}) is True
            assert field.is_visible({"mode": "cloud"}) is False
            assert field.is_visible({"mode": "local_external"}) is False

    # llm_base_url carries an additional gate (see the dedicated test below),
    # but still requires local_embedded as one of its conditions.
    base_url = next(field for field in provider.fields if field.key == "llm_base_url")
    assert base_url.is_visible({"mode": "cloud", "llm_provider": "openai_compatible"}) is False


def test_llm_base_url_is_further_gated_to_openai_compatible():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    base_url = next(field for field in provider.fields if field.key == "llm_base_url")
    assert base_url.is_visible({"mode": "local_embedded", "llm_provider": "openai_compatible"}) is True
    assert base_url.is_visible({"mode": "local_embedded", "llm_provider": "openai"}) is False


def test_api_key_is_a_secret_bound_to_env():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    api_key = next(field for field in provider.fields if field.key == "api_key")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "HINDSIGHT_API_KEY"


def test_llm_api_key_is_a_secret_bound_to_its_own_env_var():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    llm_api_key = next(field for field in provider.fields if field.key == "llm_api_key")
    assert llm_api_key.kind == KIND_SECRET
    assert llm_api_key.is_secret is True
    assert llm_api_key.env_key == "HINDSIGHT_LLM_API_KEY"
