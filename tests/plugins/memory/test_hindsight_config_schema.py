"""Tests for Hindsight's declared config surface."""

from plugins.memory.config_schema import (
    KIND_SECRET,
    KIND_SELECT,
    get_provider_config_schema,
)


INLINE_KEYS = {"mode", "api_key", "api_url", "bank_id", "recall_budget"}
LOCAL_EMBEDDED_KEYS = {"llm_provider", "llm_base_url", "llm_api_key", "llm_model"}


def test_hindsight_is_declared():
    provider = get_provider_config_schema("hindsight")

    assert provider is not None
    assert provider.label == "Hindsight"
    keys = [field.key for field in provider.fields]
    assert len(keys) == len(set(keys))
    assert INLINE_KEYS | LOCAL_EMBEDDED_KEYS <= set(keys)


def test_local_embedded_fields_live_in_full_config():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    assert {field.key for field in provider.inline_fields()} == INLINE_KEYS
    assert LOCAL_EMBEDDED_KEYS <= {
        field.key for field in provider.fields if not field.inline
    }


def test_mode_write_gate_keeps_desktop_setup_to_supported_modes():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    mode = next(field for field in provider.fields if field.key == "mode")
    assert mode.kind == KIND_SELECT
    assert mode.allowed_values() == {"cloud", "local_external"}
    # Desktop can display existing Local Embedded config, but setup still owns
    # dependency installation and selecting that mode.
    assert "local_embedded" not in mode.allowed_values()


def test_api_keys_are_bound_to_their_runtime_env_keys():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    by_key = {field.key: field for field in provider.fields}
    assert by_key["api_key"].kind == KIND_SECRET
    assert by_key["api_key"].env_key == "HINDSIGHT_API_KEY"
    assert by_key["llm_api_key"].kind == KIND_SECRET
    assert by_key["llm_api_key"].env_key == "HINDSIGHT_LLM_API_KEY"


def test_local_embedded_llm_provider_accepts_runtime_backends():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    llm_provider = next(
        field for field in provider.fields if field.key == "llm_provider"
    )
    assert "openai_compatible" in llm_provider.allowed_values()
