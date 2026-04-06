"""Tests for user-defined provider resolution in switch_model.

Regression tests for the bug where /model <name> with multiple custom
providers would keep the current provider instead of switching to the
provider whose default_model matches the requested model name.

Fixes: https://github.com/NousResearch/hermes-agent/issues/5569
"""

import pytest
from unittest.mock import patch, MagicMock
from hermes_cli.model_switch import switch_model, DIRECT_ALIASES


USER_PROVIDERS = {
    "provider-a": {
        "name": "Provider A",
        "api": "https://api-a.example.com/v1",
        "default_model": "qwen3.5-27b",
    },
    "provider-b": {
        "name": "Provider B",
        "api": "https://api-b.example.com/v1",
        "default_model": "gemma-4-26b",
    },
}


def _make_runtime(provider: str) -> dict:
    return {
        "api_key": f"key-for-{provider}",
        "base_url": f"https://{provider}.example.com/v1",
        "api_mode": "openai_chat",
    }


@pytest.fixture(autouse=True)
def clear_direct_aliases():
    """Ensure DIRECT_ALIASES cache is empty before each test."""
    import hermes_cli.model_switch as msw
    original = dict(msw.DIRECT_ALIASES)
    msw.DIRECT_ALIASES.clear()
    yield
    msw.DIRECT_ALIASES.clear()
    msw.DIRECT_ALIASES.update(original)


@patch("hermes_cli.models.validate_requested_model")
@patch("hermes_cli.runtime_provider.resolve_runtime_provider")
@patch("hermes_cli.models.detect_provider_for_model")
@patch("hermes_cli.model_switch.get_model_info")
def test_switch_to_user_provider_by_default_model(
    mock_get_info, mock_detect, mock_runtime, mock_validate
):
    """Switching by model name should resolve to the user-defined provider
    whose default_model matches, not stay on the current provider."""
    mock_detect.return_value = None
    mock_get_info.return_value = None
    mock_validate.return_value = {"accepted": True, "persist": True, "recognized": False}
    mock_runtime.side_effect = lambda requested: _make_runtime(requested)

    result = switch_model(
        raw_input="gemma-4-26b",
        current_provider="provider-a",
        current_model="qwen3.5-27b",
        user_providers=USER_PROVIDERS,
    )

    assert result.success, result.error_message
    assert result.target_provider == "provider-b"
    assert result.new_model == "gemma-4-26b"
    assert result.provider_changed is True


@patch("hermes_cli.models.validate_requested_model")
@patch("hermes_cli.runtime_provider.resolve_runtime_provider")
@patch("hermes_cli.models.detect_provider_for_model")
@patch("hermes_cli.model_switch.get_model_info")
def test_switch_by_model_name_case_insensitive(
    mock_get_info, mock_detect, mock_runtime, mock_validate
):
    """Model name matching should be case-insensitive."""
    mock_detect.return_value = None
    mock_get_info.return_value = None
    mock_validate.return_value = {"accepted": True, "persist": True, "recognized": False}
    mock_runtime.side_effect = lambda requested: _make_runtime(requested)

    result = switch_model(
        raw_input="Gemma-4-26B",  # different case
        current_provider="provider-a",
        current_model="qwen3.5-27b",
        user_providers=USER_PROVIDERS,
    )

    assert result.success, result.error_message
    assert result.target_provider == "provider-b"


@patch("hermes_cli.models.validate_requested_model")
@patch("hermes_cli.runtime_provider.resolve_runtime_provider")
@patch("hermes_cli.models.detect_provider_for_model")
@patch("hermes_cli.model_switch.get_model_info")
def test_switch_without_user_providers_stays_on_current(
    mock_get_info, mock_detect, mock_runtime, mock_validate
):
    """When user_providers is None, switching an unknown model stays on the
    current provider (no regression in existing behaviour)."""
    mock_detect.return_value = None
    mock_get_info.return_value = None
    mock_validate.return_value = {"accepted": True, "persist": True, "recognized": False}
    mock_runtime.side_effect = lambda requested: _make_runtime(requested)

    result = switch_model(
        raw_input="gemma-4-26b",
        current_provider="provider-a",
        current_model="qwen3.5-27b",
        user_providers=None,
    )

    assert result.success, result.error_message
    assert result.target_provider == "provider-a"
    assert result.provider_changed is False


@patch("hermes_cli.models.validate_requested_model")
@patch("hermes_cli.runtime_provider.resolve_runtime_provider")
@patch("hermes_cli.models.detect_provider_for_model")
@patch("hermes_cli.model_switch.get_model_info")
def test_explicit_provider_overrides_user_provider_lookup(
    mock_get_info, mock_detect, mock_runtime, mock_validate
):
    """When --provider is given explicitly, it takes precedence over
    the user-defined provider default_model lookup."""
    from hermes_cli.providers import ProviderDef
    mock_detect.return_value = None
    mock_get_info.return_value = None
    mock_validate.return_value = {"accepted": True, "persist": True, "recognized": False}
    mock_runtime.side_effect = lambda requested: _make_runtime(requested)

    pdef = ProviderDef(
        id="provider-a",
        name="Provider A",
        transport="openai_chat",
        api_key_env_vars=[],
        base_url="https://api-a.example.com/v1",
    )
    with patch("hermes_cli.model_switch.resolve_provider_full", return_value=pdef):
        result = switch_model(
            raw_input="gemma-4-26b",
            current_provider="provider-b",
            current_model="gemma-4-26b",
            explicit_provider="provider-a",  # explicit override
            user_providers=USER_PROVIDERS,
        )

    assert result.success, result.error_message
    assert result.target_provider == "provider-a"
