from types import SimpleNamespace

import hermes_cli.model_switch as ms


class _RuntimeProviderStub:
    @staticmethod
    def resolve_runtime_provider(requested=None):
        return {
            "api_key": "anthropic-key",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
        }


def test_switch_model_unverified_arms_previous_runtime_as_session_fallback(monkeypatch):
    monkeypatch.setattr(
        ms,
        "resolve_provider_full",
        lambda provider, user_providers=None: SimpleNamespace(
            id="anthropic",
            name="Anthropic",
            base_url="https://api.anthropic.com",
        ),
    )
    monkeypatch.setattr(ms, "resolve_alias", lambda *args, **kwargs: None)
    monkeypatch.setattr(ms, "normalize_model_for_provider", lambda model, provider: model)
    monkeypatch.setattr(ms, "get_model_capabilities", lambda *args, **kwargs: None)
    monkeypatch.setattr(ms, "get_model_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _RuntimeProviderStub.resolve_runtime_provider,
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *args, **kwargs: {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": "Could not reach the Anthropic API to validate `claude-opus-4-6`.",
        },
    )

    result = ms.switch_model(
        raw_input="claude-opus-4-6",
        current_provider="openrouter",
        current_model="openai/gpt-5.4",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key="openrouter-key",
        explicit_provider="anthropic",
        current_fallback_model=[{"provider": "zai", "model": "glm-5"}],
    )

    assert result.success is True
    assert result.fallback_chain == [
        {
            "provider": "openrouter",
            "model": "openai/gpt-5.4",
            "__session_previous_runtime__": True,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "openrouter-key",
        },
        {"provider": "zai", "model": "glm-5"},
    ]
    assert "Session fallback armed" in result.fallback_message
    assert "openai/gpt-5.4" in result.fallback_message


def test_switch_model_validation_exception_still_arms_fallback(monkeypatch):
    monkeypatch.setattr(
        ms,
        "resolve_provider_full",
        lambda provider, user_providers=None: SimpleNamespace(
            id="anthropic",
            name="Anthropic",
            base_url="https://api.anthropic.com",
        ),
    )
    monkeypatch.setattr(ms, "resolve_alias", lambda *args, **kwargs: None)
    monkeypatch.setattr(ms, "normalize_model_for_provider", lambda model, provider: model)
    monkeypatch.setattr(ms, "get_model_capabilities", lambda *args, **kwargs: None)
    monkeypatch.setattr(ms, "get_model_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _RuntimeProviderStub.resolve_runtime_provider,
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("catalog timeout")

    monkeypatch.setattr("hermes_cli.models.validate_requested_model", _boom)

    result = ms.switch_model(
        raw_input="claude-opus-4-6",
        current_provider="openrouter",
        current_model="openai/gpt-5.4",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key="openrouter-key",
        explicit_provider="anthropic",
    )

    assert result.success is True
    assert result.fallback_chain[0]["__session_previous_runtime__"] is True
    assert "catalog timeout" in result.warning_message
    assert "Session fallback armed" in result.fallback_message


def test_switch_model_verified_model_clears_previous_session_fallback(monkeypatch):
    monkeypatch.setattr(
        ms,
        "resolve_provider_full",
        lambda provider, user_providers=None: SimpleNamespace(
            id="anthropic",
            name="Anthropic",
            base_url="https://api.anthropic.com",
        ),
    )
    monkeypatch.setattr(ms, "resolve_alias", lambda *args, **kwargs: None)
    monkeypatch.setattr(ms, "normalize_model_for_provider", lambda model, provider: model)
    monkeypatch.setattr(ms, "get_model_capabilities", lambda *args, **kwargs: None)
    monkeypatch.setattr(ms, "get_model_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _RuntimeProviderStub.resolve_runtime_provider,
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *args, **kwargs: {
            "accepted": True,
            "persist": True,
            "recognized": True,
            "message": None,
        },
    )

    result = ms.switch_model(
        raw_input="claude-opus-4-6",
        current_provider="openrouter",
        current_model="openai/gpt-5.4",
        explicit_provider="anthropic",
        current_fallback_model=[
            {
                "provider": "openrouter",
                "model": "openai/gpt-5.4",
                "__session_previous_runtime__": True,
            },
            {"provider": "zai", "model": "glm-5"},
        ],
    )

    assert result.success is True
    assert result.fallback_chain == [{"provider": "zai", "model": "glm-5"}]
    assert result.fallback_message == ""
