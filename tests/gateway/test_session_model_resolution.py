"""Gateway session model resolution owns mutation orchestration, not model semantics."""

from gateway.session_model_resolution import resolve_session_model


def test_same_custom_provider_keeps_session_endpoint(monkeypatch):
    monkeypatch.setattr(
        "gateway.session_model_resolution._resolve_runtime_credentials",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("credential resolver should not run")),
    )
    result = resolve_session_model(
        config={"model": {"provider": "custom", "default": "original"}},
        raw_model="switched",
        explicit_provider="",
        current_provider="custom",
        current_base_url="http://127.0.0.1:1234/v1",
    )
    assert result.model == "switched"
    assert result.provider == "custom"
    assert result.base_url == "http://127.0.0.1:1234/v1"
    assert result.provider_changed is False


def test_explicit_provider_uses_credential_seam_then_canonical_route(monkeypatch):
    calls = []

    def resolve_runtime_credentials(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
            "runtime_kind": "http",
        }

    monkeypatch.setattr(
        "gateway.session_model_resolution._resolve_runtime_credentials",
        resolve_runtime_credentials,
    )
    result = resolve_session_model(
        config={},
        raw_model="claude-sonnet-4-6",
        explicit_provider="anthropic",
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
    )
    assert calls == [{
        "requested": "anthropic",
        "explicit_api_key": None,
        "explicit_base_url": None,
        "target_model": "claude-sonnet-4-6",
    }]
    assert result.provider == "anthropic"
    assert result.model == "claude-sonnet-4-6"
    assert result.api_mode == "anthropic_messages"
    assert result.provider_changed is True


def test_configured_provider_model_keeps_config_key_for_credentials(monkeypatch):
    calls = []

    def resolve_runtime_credentials(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "custom:named-host",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_mode": "chat_completions",
            "runtime_kind": "http",
        }

    monkeypatch.setattr(
        "gateway.session_model_resolution._resolve_runtime_credentials",
        resolve_runtime_credentials,
    )
    config = {
        "providers": {
            "named-host": {
                "base_url": "http://127.0.0.1:8000/v1",
                "models": ["private-model"],
            }
        }
    }
    result = resolve_session_model(
        config=config,
        raw_model="private-model",
        explicit_provider="",
        current_provider="anthropic",
        current_base_url="https://api.anthropic.com",
    )
    assert calls[0]["requested"] == "named-host"
    assert calls[0]["explicit_base_url"] == "http://127.0.0.1:8000/v1"
    assert result.provider == "custom:named-host"
    assert result.model == "private-model"
    assert result.provider_changed is True


def test_direct_alias_endpoint_uses_alias_key_and_not_session_key(monkeypatch):
    calls = []

    def resolve_runtime_credentials(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "custom",
            "base_url": kwargs["explicit_base_url"],
            "api_mode": "chat_completions",
            "runtime_kind": "http",
        }

    monkeypatch.setattr(
        "gateway.session_model_resolution._resolve_runtime_credentials",
        resolve_runtime_credentials,
    )
    config = {
        "model_aliases": {
            "local": {
                "model": "qwen",
                "provider": "custom",
                "base_url": "http://127.0.0.1:9000/v1",
                "api_key": "alias-key",
            }
        }
    }
    result = resolve_session_model(
        config=config,
        raw_model="local",
        explicit_provider="",
        current_provider="anthropic",
        current_base_url="https://api.anthropic.com",
    )
    assert calls[0]["requested"] == "custom"
    assert calls[0]["explicit_api_key"] == "alias-key"
    assert calls[0]["explicit_base_url"] == "http://127.0.0.1:9000/v1"
    assert result.provider == "custom"
    assert result.model == "qwen"
    assert result.provider_changed is True
