"""Tests for /model provider listing with config.yaml custom_providers."""

import hermes_cli.model_switch as ms
from hermes_cli.model_switch import list_authenticated_providers


def test_list_authenticated_providers_includes_custom_providers():
    providers = list_authenticated_providers(
        current_provider="custom:cpamc",
        user_providers=None,
        custom_providers=[
            {
                "name": "cpamc",
                "base_url": "http://100.93.186.121:8317/v1",
                "models": {
                    "qwen3.6-plus": {"context_length": 1_000_000},
                    "gpt-5.4": {"context_length": 1_000_000},
                    "kimi-k2.5": {"context_length": 256_000},
                },
            }
        ],
        max_models=2,
    )

    cpamc = next(p for p in providers if p["slug"] == "custom:cpamc")
    assert cpamc["name"] == "cpamc"
    assert cpamc["is_current"] is True
    assert cpamc["source"] == "custom-provider"
    assert cpamc["total_models"] == 3
    assert cpamc["models"] == ["qwen3.6-plus", "gpt-5.4"]
    assert cpamc["api_url"] == "100.93.186.121:8317/v1"


def test_list_authenticated_providers_includes_saved_custom_model_first():
    providers = list_authenticated_providers(
        current_provider="openrouter",
        user_providers=None,
        custom_providers=[
            {
                "name": "cpamc",
                "base_url": "http://100.93.186.121:8317/v1",
                "model": "glm-5",
                "models": {
                    "qwen3.6-plus": {"context_length": 1_000_000},
                    "glm-5": {"context_length": 202_752},
                },
            }
        ],
        max_models=3,
    )

    cpamc = next(p for p in providers if p["slug"] == "custom:cpamc")
    assert cpamc["models"][0] == "glm-5"
    assert cpamc["total_models"] == 2


def test_switch_model_accepts_named_custom_provider(monkeypatch):
    monkeypatch.setattr(ms, "resolve_provider_full", lambda provider, user_providers=None: None)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested: {
            "api_key": "test-key",
            "base_url": "http://100.93.186.121:8317/v1",
            "api_mode": "chat_completions",
            "provider": requested,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *a, **kw: {"accepted": True, "persist": True, "recognized": True, "message": None},
    )
    monkeypatch.setattr("hermes_cli.models.opencode_model_api_mode", lambda *a, **kw: "chat_completions")
    monkeypatch.setattr(ms, "get_model_capabilities", lambda *a, **kw: {})
    monkeypatch.setattr(ms, "get_model_info", lambda *a, **kw: None)
    monkeypatch.setattr(ms, "normalize_model_for_provider", lambda model, provider: model)

    result = ms.switch_model(
        raw_input="glm-5",
        current_provider="custom:cpamc",
        current_model="gpt-5.4",
        current_base_url="http://100.93.186.121:8317/v1",
        current_api_key="test-key",
        explicit_provider="custom:cpamc",
        custom_providers=[
            {
                "name": "cpamc",
                "base_url": "http://100.93.186.121:8317/v1",
                "models": {
                    "glm-5": {"context_length": 202_752},
                },
            }
        ],
    )

    assert result.success is True
    assert result.target_provider == "custom:cpamc"
    assert result.new_model == "glm-5"
    assert result.base_url == "http://100.93.186.121:8317/v1"
