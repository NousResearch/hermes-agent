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
        "recall_max_results", "recall_live_status_bypass", "recall_simple_budget",
        "recall_simple_max_words", "recall_document_tags", "recall_document_terms",
        "recall_document_tag_routes", "recall_document_types",
    }


def test_basic_fields_stay_inline_and_optional_controls_use_full_config():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    # Keep the normal panel compact; opt-in controls belong in the existing modal.
    assert {field.key for field in provider.inline_fields()} == {
        "mode", "api_key", "api_url", "bank_id", "recall_budget",
    }
    advanced = {field.key: field for field in provider.fields if not field.inline}
    assert all(field.group for field in advanced.values())
    assert advanced["recall_max_results"].default == "0"
    assert advanced["recall_live_status_bypass"].default == "false"
    assert advanced["recall_simple_budget"].default == ""
    assert advanced["recall_simple_max_words"].default == "0"
    assert advanced["recall_document_tags"].default == "[]"
    assert advanced["recall_document_terms"].default == "[]"


def test_mode_gating_is_expressed_as_select_options():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    mode = next(field for field in provider.fields if field.key == "mode")
    assert mode.kind == KIND_SELECT
    assert mode.allowed_values() == {"cloud", "local_external"}
    # local_embedded is intentionally unsupported on desktop.
    assert "local_embedded" not in mode.allowed_values()


def test_api_key_is_a_secret_bound_to_env():
    provider = get_provider_config_schema("hindsight")
    assert provider is not None

    api_key = next(field for field in provider.fields if field.key == "api_key")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "HINDSIGHT_API_KEY"
