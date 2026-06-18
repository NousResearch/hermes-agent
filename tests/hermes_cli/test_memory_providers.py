"""Tests for the declarative memory-provider registry."""

from hermes_cli.memory_providers import (
    KIND_SECRET,
    KIND_SELECT,
    get_memory_provider,
)


def test_hindsight_is_declared():
    provider = get_memory_provider("hindsight")

    assert provider is not None
    assert provider.label == "Hindsight"
    assert {field.key for field in provider.fields} == {
        "mode",
        "api_key",
        "api_url",
        "bank_id",
        "recall_budget",
    }


def test_hindsight_mode_gating_is_expressed_as_select_options():
    provider = get_memory_provider("hindsight")
    assert provider is not None

    mode = next(field for field in provider.fields if field.key == "mode")
    assert mode.kind == KIND_SELECT
    assert mode.allowed_values() == {"cloud", "local_external"}
    # local_embedded is intentionally unsupported on desktop.
    assert "local_embedded" not in mode.allowed_values()


def test_api_key_is_a_secret_bound_to_env():
    provider = get_memory_provider("hindsight")
    assert provider is not None

    api_key = next(field for field in provider.fields if field.key == "api_key")
    assert api_key.kind == KIND_SECRET
    assert api_key.is_secret is True
    assert api_key.env_key == "HINDSIGHT_API_KEY"


def test_unknown_provider_is_none():
    assert get_memory_provider("builtin") is None
