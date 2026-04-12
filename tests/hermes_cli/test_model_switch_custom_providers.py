"""Regression tests for /model support of config.yaml custom_providers.

The terminal `hermes model` flow already exposes `custom_providers`, but the
shared slash-command pipeline (`/model` in CLI/gateway/Telegram) historically
only looked at `providers:`.
"""

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli.providers import resolve_provider_full


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def test_list_authenticated_providers_includes_custom_providers(monkeypatch):
    """No-args /model menus should include saved custom_providers entries."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={},
        custom_providers=[
            {
                "name": "Local (127.0.0.1:4141)",
                "base_url": "http://127.0.0.1:4141/v1",
                "model": "rotator-openrouter-coding",
                "models": {
                    "rotator-openrouter-coding": {"context_length": 131072},
                    "backup-model": {"context_length": 65536},
                },
            }
        ],
        max_models=50,
    )

    assert any(
        p["slug"] == "custom:local-(127.0.0.1:4141)"
        and p["name"] == "Local (127.0.0.1:4141)"
        and p["models"] == ["rotator-openrouter-coding", "backup-model"]
        and p["api_url"] == "http://127.0.0.1:4141/v1"
        for p in providers
    )


def test_resolve_provider_full_finds_named_custom_provider():
    """Explicit /model --provider should resolve saved custom_providers entries."""
    resolved = resolve_provider_full(
        "custom:local-(127.0.0.1:4141)",
        user_providers={},
        custom_providers=[
            {
                "name": "Local (127.0.0.1:4141)",
                "base_url": "http://127.0.0.1:4141/v1",
            }
        ],
    )

    assert resolved is not None
    assert resolved.id == "custom:local-(127.0.0.1:4141)"
    assert resolved.name == "Local (127.0.0.1:4141)"
    assert resolved.base_url == "http://127.0.0.1:4141/v1"
    assert resolved.source == "user-config"


def test_switch_model_accepts_explicit_named_custom_provider(monkeypatch):
    """Shared /model switch pipeline should accept --provider for custom_providers."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested: {
            "api_key": "no-key-required",
            "base_url": "http://127.0.0.1:4141/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="rotator-openrouter-coding",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        current_base_url="https://chatgpt.com/backend-api/codex",
        current_api_key="",
        explicit_provider="custom:local-(127.0.0.1:4141)",
        user_providers={},
        custom_providers=[
            {
                "name": "Local (127.0.0.1:4141)",
                "base_url": "http://127.0.0.1:4141/v1",
                "model": "rotator-openrouter-coding",
            }
        ],
    )

    assert result.success is True
    assert result.target_provider == "custom:local-(127.0.0.1:4141)"
    assert result.provider_label == "Local (127.0.0.1:4141)"
    assert result.new_model == "rotator-openrouter-coding"
    assert result.base_url == "http://127.0.0.1:4141/v1"
    assert result.api_key == "no-key-required"


def test_switch_model_keeps_named_custom_provider_for_direct_model(monkeypatch):
    """Named custom providers should not auto-switch away for matching bare models."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested: {
            "api_key": "sk-stepfun",
            "base_url": "https://api.stepfun.com/step_plan/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda *a, **k: ("minimax-cn", "MiniMax-M2.7"),
    )

    result = switch_model(
        raw_input="step-3.5-flash",
        current_provider="custom:stepfun-plan",
        current_model="step-3.5-flash",
        current_base_url="https://api.stepfun.com/step_plan/v1",
        current_api_key="sk-stepfun",
        custom_providers=[
            {
                "name": "stepfun-plan",
                "base_url": "https://api.stepfun.com/step_plan/v1",
                "model": "step-3.5-flash",
            }
        ],
    )

    assert result.success is True
    assert result.target_provider == "custom:stepfun-plan"
    assert result.new_model == "step-3.5-flash"
    assert result.base_url == "https://api.stepfun.com/step_plan/v1"


def test_switch_model_resolves_named_custom_provider_from_slash_syntax(monkeypatch):
    """provider/model shorthand should map unique custom-provider prefixes."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested: {
            "api_key": "sk-stepfun",
            "base_url": "https://api.stepfun.com/step_plan/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="stepfun/step-3.5-flash",
        current_provider="openrouter",
        current_model="anthropic/claude-sonnet-4.5",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key="sk-openrouter",
        custom_providers=[
            {
                "name": "stepfun-plan",
                "base_url": "https://api.stepfun.com/step_plan/v1",
                "model": "step-3.5-flash",
            }
        ],
    )

    assert result.success is True
    assert result.target_provider == "custom:stepfun-plan"
    assert result.provider_label == "stepfun-plan"
    assert result.new_model == "step-3.5-flash"
    assert result.base_url == "https://api.stepfun.com/step_plan/v1"


def test_switch_model_auto_detects_model_listed_under_named_custom_provider(monkeypatch):
    """Configured custom-provider models should win before built-in auto-detection."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested: {
            "api_key": "sk-stepfun",
            "base_url": "https://api.stepfun.com/step_plan/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda *a, **k: ("minimax-cn", "MiniMax-M2.7"),
    )

    result = switch_model(
        raw_input="step-3.5-flash",
        current_provider="openrouter",
        current_model="anthropic/claude-sonnet-4.5",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key="sk-openrouter",
        custom_providers=[
            {
                "name": "stepfun-plan",
                "base_url": "https://api.stepfun.com/step_plan/v1",
                "models": {
                    "step-3.5-flash": {"context_length": 262144},
                    "step-3.5v-mini": {"context_length": 131072},
                },
            }
        ],
    )

    assert result.success is True
    assert result.target_provider == "custom:stepfun-plan"
    assert result.new_model == "step-3.5-flash"
    assert result.base_url == "https://api.stepfun.com/step_plan/v1"
