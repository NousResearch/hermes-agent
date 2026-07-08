"""Unit tests for the TokenLab provider profile."""

from __future__ import annotations


def test_tokenlab_profile_is_registered():
    import providers

    profile = providers.get_provider_profile("tokenlab")
    assert profile is not None, "tokenlab provider profile must be registered"
    assert profile.display_name == "TokenLab"
    assert profile.base_url == "https://api.tokenlab.sh/v1"
    assert profile.models_url == "https://api.tokenlab.sh/v1/models"
    assert profile.env_vars == ("TOKENLAB_API_KEY", "TOKENLAB_BASE_URL")
    assert profile.auth_type == "api_key"


def test_tokenlab_profile_fallback_models_cover_frontier_families():
    import providers

    profile = providers.get_provider_profile("tokenlab")
    assert profile is not None
    models = set(profile.fallback_models)

    assert {
        "gpt-5.5",
        "gpt-5.4-mini",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "grok-4.3",
        "grok-4-fast",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "kimi-k2.7-code",
        "minimax-m3",
    }.issubset(models)


def test_tokenlab_auto_wires_into_canonical_provider_catalog():
    from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_LABELS
    from hermes_cli.provider_catalog import provider_catalog_by_slug

    slugs = [p.slug for p in CANONICAL_PROVIDERS]
    assert "tokenlab" in slugs
    assert _PROVIDER_LABELS["tokenlab"] == "TokenLab"

    descriptor = provider_catalog_by_slug()["tokenlab"]
    assert descriptor.tab == "keys"
    assert descriptor.api_key_env_vars == ("TOKENLAB_API_KEY",)
    assert descriptor.base_url_env_var == "TOKENLAB_BASE_URL"


def test_tokenlab_env_vars_are_exposed_to_config_ui():
    from hermes_cli.config import OPTIONAL_ENV_VARS

    assert OPTIONAL_ENV_VARS["TOKENLAB_API_KEY"]["category"] == "provider"
    assert OPTIONAL_ENV_VARS["TOKENLAB_API_KEY"]["password"] is True
    assert OPTIONAL_ENV_VARS["TOKENLAB_BASE_URL"]["category"] == "provider"
    assert OPTIONAL_ENV_VARS["TOKENLAB_BASE_URL"]["password"] is False


def test_tokenlab_auto_wires_into_api_key_registry():
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider

    assert resolve_provider("tokenlab") == "tokenlab"
    assert "tokenlab" in PROVIDER_REGISTRY
    config = PROVIDER_REGISTRY["tokenlab"]
    assert config.api_key_env_vars == ("TOKENLAB_API_KEY",)
    assert config.base_url_env_var == "TOKENLAB_BASE_URL"
    assert config.inference_base_url == "https://api.tokenlab.sh/v1"


def test_tokenlab_provider_model_ids_prefers_live_catalog(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": provider_id,
            "api_key": "tokenlab-test-key",
            "base_url": "https://api.tokenlab.sh/v1",
            "source": "TOKENLAB_API_KEY",
        },
    )
    monkeypatch.setattr(
        "providers.base.ProviderProfile.fetch_models",
        lambda self, *, api_key=None, base_url=None, timeout=8.0: [
            "gpt-5.5",
            "claude-sonnet-5",
        ],
    )

    from hermes_cli.models import provider_model_ids

    assert provider_model_ids("tokenlab") == ["gpt-5.5", "claude-sonnet-5"]


def test_tokenlab_provider_model_ids_falls_back_to_profile(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": provider_id,
            "api_key": "",
            "base_url": "https://api.tokenlab.sh/v1",
            "source": "",
        },
    )

    from hermes_cli.models import provider_model_ids
    import providers

    profile = providers.get_provider_profile("tokenlab")
    assert profile is not None
    assert provider_model_ids("tokenlab") == list(profile.fallback_models)
