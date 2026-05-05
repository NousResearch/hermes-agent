"""Regression tests for model picker ``configured_only`` mode.

When ``picker_mode: "configured"`` is set, the model picker should only
show providers that are explicitly configured (in auth.json credential pool
or config.yaml providers section), not ambient env-var detections.
"""

import pytest

from hermes_cli.model_switch import list_authenticated_providers


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_configured_only_skips_env_var_detection(monkeypatch):
    """With configured_only=True, env vars alone must not surface a provider.

    Section 1 of list_authenticated_providers checks PROVIDER_TO_MODELS_DEV
    entries.  An env-var like OPENAI_API_KEY would normally surface the
    ``openai`` provider.  With configured_only=True and no credential pool
    entry, it must not appear.
    """
    import hermes_cli.providers as _pmod
    # Prevent HERMES_OVERLAYS from adding providers we don't control.
    monkeypatch.setattr(_pmod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {
        # Simulate a models.dev entry for openai so section 1 processes it.
        "openai": {"name": "OpenAI", "env": ["OPENAI_API_KEY"]},
    })
    from hermes_cli.auth import PROVIDER_REGISTRY
    # Make sure openai has env vars registered so section 1 checks them.
    monkeypatch.setattr(
        "hermes_cli.auth.PROVIDER_REGISTRY",
        {**PROVIDER_REGISTRY, "openai": type("P", (), {
            "api_key_env_vars": ["OPENAI_API_KEY"],
            "auth_type": "api_key",
        })()},
    )
    # No auth store / credential pool — configured_only should skip env var.
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[],
        configured_only=True,
    )

    assert not any(p["slug"] == "openai" for p in providers), (
        "openai should not appear when only env var is set and configured_only=True"
    )


def test_configured_only_false_includes_env_var_provider(monkeypatch):
    """With configured_only=False (default), a provider with a non-empty env var appears."""
    import hermes_cli.providers as _pmod
    monkeypatch.setattr(_pmod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {
        "openai": {"name": "OpenAI", "env": ["OPENAI_API_KEY"]},
    })
    from hermes_cli.auth import PROVIDER_REGISTRY
    monkeypatch.setattr(
        "hermes_cli.auth.PROVIDER_REGISTRY",
        {**PROVIDER_REGISTRY, "openai": type("P", (), {
            "api_key_env_vars": ["OPENAI_API_KEY"],
            "auth_type": "api_key",
        })()},
    )
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[],
        configured_only=False,
    )

    assert any(p["slug"] == "openai" for p in providers), (
        "openai should appear when OPENAI_API_KEY is set and configured_only=False"
    )


def test_configured_only_respects_auth_store(monkeypatch):
    """With configured_only=True, a provider in the credential pool must appear.

    Even without env vars set, a credential pool entry in auth.json should
    still cause the provider to show up.
    """
    import hermes_cli.providers as _pmod
    monkeypatch.setattr(_pmod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {
        "openai": {"name": "OpenAI", "env": ["OPENAI_API_KEY"]},
    })
    from hermes_cli.auth import PROVIDER_REGISTRY
    monkeypatch.setattr(
        "hermes_cli.auth.PROVIDER_REGISTRY",
        {**PROVIDER_REGISTRY, "openai": type("P", (), {
            "api_key_env_vars": ["OPENAI_API_KEY"],
            "auth_type": "api_key",
        })()},
    )
    # No env var, but credential pool has an entry for openai.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from unittest.mock import MagicMock
    fake_store = MagicMock()
    fake_store.get.side_effect = lambda k, d=None: {
        "credential_pool": {"openai": [{"access_token": "***"}]},
    }.get(k, d)
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: fake_store)

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[],
        configured_only=True,
    )

    assert any(p["slug"] == "openai" for p in providers), (
        "openai should appear when in credential pool even without env var"
    )


def test_configured_only_always_shows_user_defined(monkeypatch):
    """User-defined providers from config should always appear regardless of configured_only."""
    import hermes_cli.providers as _pmod
    monkeypatch.setattr(_pmod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: None)
    # No env vars at all.
    for ev in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(ev, "")

    user_providers = {
        "my-endpoint": {
            "name": "My Custom Endpoint",
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-my-key",
            "model": "my-model-v1",
        }
    }

    providers_default = list_authenticated_providers(
        user_providers=user_providers,
        custom_providers=[],
        configured_only=False,
    )
    providers_configured = list_authenticated_providers(
        user_providers=user_providers,
        custom_providers=[],
        configured_only=True,
    )

    assert any(p["slug"] == "my-endpoint" for p in providers_default)
    assert any(p["slug"] == "my-endpoint" for p in providers_configured), (
        "user-defined providers should appear in configured_only mode too"
    )
