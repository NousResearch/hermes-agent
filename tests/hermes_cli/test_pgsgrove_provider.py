"""Focused tests for Phoenix Grove provider wiring."""

from __future__ import annotations

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.models import (
    CANONICAL_PROVIDERS,
    _PROVIDER_ALIASES,
    _PROVIDER_LABELS,
    normalize_provider,
)


def test_pgsgrove_provider_profile_loads():
    from providers import get_provider_profile

    profile = get_provider_profile("pgsgrove")
    assert profile is not None
    assert profile.name == "pgsgrove"
    assert profile.display_name == "Phoenix Grove"
    assert profile.base_url == "https://api.pgsgrove.com/v1"
    assert profile.env_vars == ("PGS_API_KEY", "PGS_BASE_URL")
    assert profile.default_aux_model == "glm-5.3-flash"
    assert "deepseek-v4-pro" in profile.fallback_models
    assert profile.default_headers["User-Agent"].startswith("HermesAgent/")


def test_pgsgrove_aliases_resolve(monkeypatch):
    monkeypatch.setenv("PGS_API_KEY", "pgsk_test")

    for alias in ("phoenix-grove", "phoenixgrove", "pgs"):
        assert resolve_provider(alias) == "pgsgrove"
        assert normalize_provider(alias) == "pgsgrove"
        assert _PROVIDER_ALIASES[alias] == "pgsgrove"


def test_pgsgrove_provider_registry_and_credentials(monkeypatch):
    monkeypatch.setenv("PGS_API_KEY", "pgsk_secret")
    monkeypatch.setenv("PGS_BASE_URL", "https://custom.pgsgrove.example/v1")

    pconfig = PROVIDER_REGISTRY["pgsgrove"]
    assert pconfig.id == "pgsgrove"
    assert pconfig.name == "Phoenix Grove"
    assert pconfig.auth_type == "api_key"
    assert pconfig.inference_base_url == "https://api.pgsgrove.com/v1"
    assert pconfig.api_key_env_vars == ("PGS_API_KEY",)
    assert pconfig.base_url_env_var == "PGS_BASE_URL"

    creds = resolve_api_key_provider_credentials("pgsgrove")
    assert creds["provider"] == "pgsgrove"
    assert creds["api_key"] == "pgsk_secret"
    assert creds["base_url"] == "https://custom.pgsgrove.example/v1"


def test_pgsgrove_canonical_provider_and_label():
    slugs = [p.slug for p in CANONICAL_PROVIDERS]
    assert "pgsgrove" in slugs
    assert _PROVIDER_LABELS["pgsgrove"] == "Phoenix Grove"


def test_pgsgrove_provider_module_overlay():
    from hermes_cli.providers import (
        HERMES_OVERLAYS,
        determine_api_mode,
        get_label,
        get_provider,
        normalize_provider as normalize_provider_in_providers,
    )

    overlay = HERMES_OVERLAYS["pgsgrove"]
    assert overlay.transport == "openai_chat"
    assert overlay.base_url_override == "https://api.pgsgrove.com/v1"
    assert overlay.base_url_env_var == "PGS_BASE_URL"

    provider = get_provider("phoenix-grove")
    assert provider is not None
    assert provider.id == "pgsgrove"
    assert provider.api_key_env_vars == ("PGS_API_KEY",)
    assert provider.base_url == "https://api.pgsgrove.com/v1"
    assert normalize_provider_in_providers("pgs") == "pgsgrove"
    assert get_label("pgsgrove") == "Phoenix Grove"
    assert determine_api_mode("pgsgrove", "https://api.pgsgrove.com/v1") == "chat_completions"


def test_pgsgrove_reasoning_effort_passthrough():
    from providers import get_provider_profile

    profile = get_provider_profile("pgsgrove")
    assert profile is not None

    # Unset stays unset — never invent an effort.
    assert profile.build_api_kwargs_extras(reasoning_config=None, model="glm-5.3") == ({}, {})
    assert profile.build_api_kwargs_extras(reasoning_config={}, model="glm-5.3") == ({}, {})
    # Explicit disable → the wire's own off switch.
    assert profile.build_api_kwargs_extras(
        reasoning_config={"enabled": False, "effort": "high"}, model="glm-5.3",
    ) == ({}, {"reasoning_effort": "none"})
    # Supported levels pass through verbatim.
    assert profile.build_api_kwargs_extras(
        reasoning_config={"effort": "max"}, model="deepseek-v4-pro",
    ) == ({}, {"reasoning_effort": "max"})
    # Hermes-internal levels clamp to the nearest weaker wire level, never escalate.
    assert profile.build_api_kwargs_extras(
        reasoning_config={"effort": "xhigh"}, model="glm-5.3",
    ) == ({}, {"reasoning_effort": "high"})
    assert profile.build_api_kwargs_extras(
        reasoning_config={"effort": "ultra"}, model="glm-5.3",
    ) == ({}, {"reasoning_effort": "max"})
