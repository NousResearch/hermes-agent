"""Policy tests for restricting MoA to manual interactive use.

Covers:
- MoA cannot be persisted as the default provider/model.
- Explicit session-only MoA switches still work.
- Bare preset names no longer implicitly switch onto the MoA provider.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_MOA_CFG = {
    "moa": {
        "default_preset": "default",
        "presets": {
            "default": {},
            "review": {},
        },
    }
}


def _mock_runtime():
    return {
        "api_key": "moa-virtual-provider",
        "base_url": "moa://local",
        "api_mode": "chat_completions",
        "provider": "moa",
    }


def test_explicit_moa_global_switch_is_rejected():
    with patch("hermes_cli.config.load_config", return_value=_MOA_CFG), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_mock_runtime()), \
         patch("hermes_cli.models.validate_requested_model", return_value={"accepted": True, "persist": True, "recognized": True, "message": None}), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None):
        result = switch_model(
            raw_input="default",
            current_provider="openrouter",
            current_model="anthropic/claude-opus-4.8",
            is_global=True,
            explicit_provider="moa",
        )

    assert result.success is False
    assert "cannot be persisted" in (result.error_message or "")
    assert "--session" in (result.error_message or "")


def test_explicit_moa_session_switch_still_works():
    with patch("hermes_cli.config.load_config", return_value=_MOA_CFG), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_mock_runtime()), \
         patch("hermes_cli.models.validate_requested_model", return_value={"accepted": True, "persist": True, "recognized": True, "message": None}), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None):
        result = switch_model(
            raw_input="default",
            current_provider="openrouter",
            current_model="anthropic/claude-opus-4.8",
            is_global=False,
            explicit_provider="moa",
        )

    assert result.success is True
    assert result.target_provider == "moa"
    assert result.new_model == "default"


def test_bare_preset_name_no_longer_implicitly_switches_to_moa():
    with patch("hermes_cli.config.load_config", return_value=_MOA_CFG), \
         patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={"api_key": "test", "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions", "provider": "openrouter"}), \
         patch("hermes_cli.models.validate_requested_model", return_value={"accepted": False, "persist": False, "recognized": False, "message": "Model `review` was not found."}), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None):
        result = switch_model(
            raw_input="review",
            current_provider="openrouter",
            current_model="anthropic/claude-opus-4.8",
            is_global=False,
            explicit_provider="",
        )

    assert result.success is False
    assert result.target_provider != "moa"
    assert "not found" in (result.error_message or "").lower()
