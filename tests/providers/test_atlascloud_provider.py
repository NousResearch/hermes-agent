"""Tests for the Atlas Cloud model provider profile."""

from __future__ import annotations


def test_atlascloud_profile_metadata():
    from providers import get_provider_profile

    profile = get_provider_profile("atlas")

    assert profile is not None
    assert profile.name == "atlascloud"
    assert profile.display_name == "Atlas Cloud"
    assert profile.base_url == "https://api.atlascloud.ai/v1"
    assert profile.env_vars == ("ATLASCLOUD_API_KEY", "ATLASCLOUD_BASE_URL")
    assert profile.default_aux_model == "qwen/qwen3.5-flash"
    assert profile.fallback_models == (
        "qwen/qwen3.5-flash",
        "deepseek-ai/deepseek-v4-pro",
    )


def test_atlascloud_wires_auth_catalog_and_picker(monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    monkeypatch.delenv("ATLASCLOUD_BASE_URL", raising=False)

    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.models import CANONICAL_PROVIDERS, provider_model_ids
    from hermes_cli.providers import determine_api_mode, resolve_provider_full

    auth_cfg = PROVIDER_REGISTRY["atlascloud"]
    assert auth_cfg.api_key_env_vars == ("ATLASCLOUD_API_KEY",)
    assert auth_cfg.base_url_env_var == "ATLASCLOUD_BASE_URL"
    assert auth_cfg.inference_base_url == "https://api.atlascloud.ai/v1"
    assert PROVIDER_REGISTRY["atlas"] is auth_cfg

    assert "atlascloud" in {entry.slug for entry in CANONICAL_PROVIDERS}
    assert provider_model_ids("atlascloud") == [
        "qwen/qwen3.5-flash",
        "deepseek-ai/deepseek-v4-pro",
    ]

    provider = resolve_provider_full("atlas-cloud")
    assert provider is not None
    assert provider.id == "atlascloud"
    assert provider.base_url == "https://api.atlascloud.ai/v1"
    assert provider.api_key_env_vars == ("ATLASCLOUD_API_KEY",)
    assert determine_api_mode("atlascloud", provider.base_url) == "chat_completions"
