"""Focused tests for Nebius Token Factory provider wiring."""

from __future__ import annotations

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.model_normalize import normalize_model_for_provider
from hermes_cli.models import (
    CANONICAL_PROVIDERS,
    _PROVIDER_ALIASES,
    _PROVIDER_LABELS,
    normalize_provider,
    provider_model_ids,
)


def test_nebius_provider_profile_loads():
    from providers import get_provider_profile

    profile = get_provider_profile("nebius-token-factory")
    assert profile is not None
    assert profile.name == "nebius-token-factory"
    assert profile.display_name == "Nebius Token Factory"
    assert profile.base_url == "https://api.tokenfactory.nebius.com/v1"
    assert profile.models_url == "https://api.tokenfactory.nebius.com/v1/models"
    assert profile.env_vars[:2] == (
        "NEBIUS_API_KEY",
        "NEBIUS_TOKEN_FACTORY_API_KEY",
    )
    assert profile.default_aux_model == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    assert "Qwen/Qwen3.5-397B-A17B-fast" in profile.fallback_models


def test_nebius_aliases_resolve(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "nebius-test-key")

    for alias in (
        "nebius",
        "nebius-tokenfactory",
        "nebius-tf",
        "token-factory",
        "tokenfactory",
    ):
        assert resolve_provider(alias) == "nebius-token-factory"
        assert normalize_provider(alias) == "nebius-token-factory"
        assert _PROVIDER_ALIASES[alias] == "nebius-token-factory"


def test_nebius_provider_registry_and_credentials(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "nebius-secret")
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://custom.nebius.example/v1")

    pconfig = PROVIDER_REGISTRY["nebius-token-factory"]
    assert pconfig.id == "nebius-token-factory"
    assert pconfig.name == "Nebius Token Factory"
    assert pconfig.auth_type == "api_key"
    assert pconfig.inference_base_url == "https://api.tokenfactory.nebius.com/v1"
    assert pconfig.api_key_env_vars == (
        "NEBIUS_API_KEY",
        "NEBIUS_TOKEN_FACTORY_API_KEY",
    )
    assert pconfig.base_url_env_var == "NEBIUS_BASE_URL"

    creds = resolve_api_key_provider_credentials("nebius-token-factory")
    assert creds["provider"] == "nebius-token-factory"
    assert creds["api_key"] == "nebius-secret"
    assert creds["base_url"] == "https://custom.nebius.example/v1"


def test_nebius_canonical_provider_and_label():
    slugs = [p.slug for p in CANONICAL_PROVIDERS]
    assert "nebius-token-factory" in slugs
    assert _PROVIDER_LABELS["nebius-token-factory"] == "Nebius Token Factory"


def test_nebius_provider_module_overlay():
    from hermes_cli.providers import (
        HERMES_OVERLAYS,
        determine_api_mode,
        get_label,
        get_provider,
        normalize_provider as normalize_provider_in_providers,
    )

    overlay = HERMES_OVERLAYS["nebius-token-factory"]
    assert overlay.transport == "openai_chat"
    assert overlay.base_url_override == "https://api.tokenfactory.nebius.com/v1"
    assert overlay.base_url_env_var == "NEBIUS_BASE_URL"

    provider = get_provider("nebius")
    assert provider is not None
    assert provider.id == "nebius-token-factory"
    assert provider.api_key_env_vars == (
        "NEBIUS_API_KEY",
        "NEBIUS_TOKEN_FACTORY_API_KEY",
    )
    assert provider.base_url == "https://api.tokenfactory.nebius.com/v1"
    assert normalize_provider_in_providers("token-factory") == "nebius-token-factory"
    assert get_label("nebius-token-factory") == "Nebius Token Factory"
    assert determine_api_mode(
        "nebius-token-factory",
        "https://api.tokenfactory.nebius.com/v1",
    ) == "chat_completions"


def test_nebius_model_catalog_prefers_live_profile_fetch(monkeypatch):
    from providers import get_provider_profile

    profile = get_provider_profile("nebius-token-factory")
    assert profile is not None
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": provider_id,
            "api_key": "nebius-live-key",
            "base_url": "https://api.tokenfactory.nebius.com/v1",
            "source": "NEBIUS_API_KEY",
        },
    )
    monkeypatch.setattr(
        profile,
        "fetch_models",
        lambda *, api_key=None, timeout=8.0: [
            "deepseek-ai/DeepSeek-V4-Pro",
            "NousResearch/Hermes-4-70B",
        ],
    )

    assert provider_model_ids("nebius-token-factory") == [
        "deepseek-ai/DeepSeek-V4-Pro",
        "NousResearch/Hermes-4-70B",
    ]


def test_nebius_model_catalog_falls_back_to_profile_models(monkeypatch):
    from providers import get_provider_profile

    profile = get_provider_profile("nebius-token-factory")
    assert profile is not None
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {
            "provider": provider_id,
            "api_key": "nebius-live-key",
            "base_url": "https://api.tokenfactory.nebius.com/v1",
            "source": "NEBIUS_API_KEY",
        },
    )
    monkeypatch.setattr(profile, "fetch_models", lambda *, api_key=None, timeout=8.0: None)

    assert provider_model_ids("nebius") == list(profile.fallback_models)


def test_nebius_reasoning_models_emit_top_level_reasoning_effort():
    from providers import get_provider_profile

    profile = get_provider_profile("nebius-token-factory")
    assert profile is not None

    extra_body, top_level = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "xhigh"},
        model="openai/gpt-oss-120b-fast",
    )
    assert extra_body == {}
    assert top_level == {"reasoning_effort": "high"}


def test_nebius_reasoning_defaults_to_medium_for_known_reasoning_model():
    from providers import get_provider_profile

    profile = get_provider_profile("nebius-token-factory")
    assert profile is not None

    extra_body, top_level = profile.build_api_kwargs_extras(
        reasoning_config=None,
        model="deepseek-ai/DeepSeek-V4-Pro",
    )
    assert extra_body == {}
    assert top_level == {"reasoning_effort": "medium"}


def test_nebius_reasoning_skips_disabled_and_non_reasoning_models():
    from providers import get_provider_profile

    profile = get_provider_profile("nebius-token-factory")
    assert profile is not None

    assert profile.build_api_kwargs_extras(
        reasoning_config={"enabled": False, "effort": "high"},
        model="deepseek-ai/DeepSeek-V4-Pro",
    ) == ({}, {})
    assert profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "high"},
        model="meta-llama/Llama-3.3-70B-Instruct",
    ) == ({}, {})


def test_nebius_transport_emits_top_level_reasoning_effort():
    from agent.transports.chat_completions import ChatCompletionsTransport
    from providers import get_provider_profile

    profile = get_provider_profile("nebius-token-factory")
    assert profile is not None

    kwargs = ChatCompletionsTransport().build_kwargs(
        model="deepseek-ai/DeepSeek-V4-Pro",
        messages=[{"role": "user", "content": "ping"}],
        tools=None,
        provider_profile=profile,
        reasoning_config={"enabled": True, "effort": "low"},
        base_url="https://api.tokenfactory.nebius.com/v1",
        provider_name="nebius-token-factory",
    )
    assert kwargs["reasoning_effort"] == "low"
    assert "extra_body" not in kwargs


def test_nebius_model_normalization_strips_canonical_and_alias_prefixes():
    model = "Qwen/Qwen3.5-397B-A17B-fast"
    assert normalize_model_for_provider(
        f"nebius-token-factory/{model}", "nebius-token-factory"
    ) == model
    assert normalize_model_for_provider(f"nebius/{model}", "nebius-token-factory") == model
    assert normalize_model_for_provider(f"nebius/{model}", "nebius") == model
    assert normalize_model_for_provider(
        "openai/gpt-oss-120b-fast", "nebius-token-factory"
    ) == "openai/gpt-oss-120b-fast"
