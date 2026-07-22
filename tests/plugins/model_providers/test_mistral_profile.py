"""Behavioral tests for the bundled Mistral model-provider profile."""

from __future__ import annotations

from types import SimpleNamespace


def test_mistral_profile_core_contract():
    import providers

    profile = providers.get_provider_profile("mistral")

    assert profile is not None
    assert profile.name == "mistral"
    assert profile.auth_type == "api_key"
    assert profile.api_mode == "chat_completions"
    assert profile.base_url == "https://api.mistral.ai/v1"
    assert profile.env_vars == ("MISTRAL_API_KEY", "MISTRAL_BASE_URL")
    assert profile.default_aux_model == "mistral-small-latest"
    assert profile.display_name == "Mistral AI"
    assert profile.description
    assert profile.signup_url.startswith("https://")


def test_mistral_aliases_resolve_through_profile_registry():
    import providers

    for alias in ("mistral-ai", "mistralai"):
        resolved = providers.get_provider_profile(alias)
        assert resolved is not None
        assert resolved.name == "mistral"


def test_mistral_fallback_catalog_is_small_valid_and_deduped():
    import providers

    profile = providers.get_provider_profile("mistral")
    assert profile is not None
    models = list(profile.fallback_models)

    assert models
    assert len(models) == len(set(models))
    assert len(models) <= 8  # intentional fallback set, not an exhaustive snapshot
    assert profile.default_aux_model in models
    assert all(isinstance(model, str) and model.strip() == model and model for model in models)


def test_mistral_generic_api_key_registration_and_catalog_surface():
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.models import CANONICAL_PROVIDERS
    from hermes_cli.provider_catalog import provider_catalog_by_slug

    pconfig = PROVIDER_REGISTRY["mistral"]
    assert pconfig.auth_type == "api_key"
    assert pconfig.inference_base_url == "https://api.mistral.ai/v1"
    assert pconfig.api_key_env_vars == ("MISTRAL_API_KEY",)
    assert pconfig.base_url_env_var == "MISTRAL_BASE_URL"

    assert any(entry.slug == "mistral" for entry in CANONICAL_PROVIDERS)

    descriptor = provider_catalog_by_slug()["mistral"]
    assert descriptor.label == "Mistral AI"
    assert descriptor.tab == "keys"
    assert descriptor.api_key_env_vars == ("MISTRAL_API_KEY",)
    assert descriptor.base_url_env_var == "MISTRAL_BASE_URL"
    assert descriptor.signup_url == "https://console.mistral.ai/"


def test_mistral_aliases_resolve_for_provider_surfaces():
    from hermes_cli.models import normalize_provider, provider_model_ids
    from hermes_cli.providers import resolve_provider_full

    for alias in ("mistral", "mistral-ai", "mistralai"):
        assert normalize_provider(alias) == "mistral"
        pdef = resolve_provider_full(alias)
        assert pdef is not None
        assert pdef.id == "mistral"
        assert pdef.base_url == "https://api.mistral.ai/v1"
        assert pdef.base_url_env_var == "MISTRAL_BASE_URL"
        assert "MISTRAL_API_KEY" in pdef.api_key_env_vars
        ids = provider_model_ids(alias)
        assert ids
        assert len(ids) == len(set(ids))
        assert all(isinstance(model, str) and model for model in ids)


def test_mistral_config_driven_runtime_uses_default_endpoint(monkeypatch):
    from hermes_cli import runtime_provider as rp

    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    monkeypatch.delenv("MISTRAL_BASE_URL", raising=False)
    monkeypatch.setattr(rp, "load_pool", lambda _provider: SimpleNamespace(has_credentials=lambda: False))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "mistral", "default": "mistral-small-latest"},
    )

    resolved = rp.resolve_runtime_provider()

    assert resolved["provider"] == "mistral"
    assert resolved["requested_provider"] == "mistral"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://api.mistral.ai/v1"
    assert resolved["api_key"] == "test-mistral-key"


def test_mistral_base_url_env_override(monkeypatch):
    from hermes_cli import runtime_provider as rp

    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    monkeypatch.setenv("MISTRAL_BASE_URL", "https://mistral-proxy.example/v1/")
    monkeypatch.setattr(rp, "load_pool", lambda _provider: SimpleNamespace(has_credentials=lambda: False))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "mistral", "default": "mistral-small-latest"},
    )

    resolved = rp.resolve_runtime_provider()

    assert resolved["provider"] == "mistral"
    assert resolved["base_url"] == "https://mistral-proxy.example/v1"
    assert resolved["api_key"] == "test-mistral-key"
