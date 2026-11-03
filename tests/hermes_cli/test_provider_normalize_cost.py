"""``normalize_provider`` answers built-in ids without reading the provider-profile list (R4 H3).

Profile ``provider_overlay`` lookups made every non-alias name scan the bound home's profiles
(~100x the dict hit it used to be), and registry-wide loops paid that once per row.
"""

from __future__ import annotations

import pytest

from providers.base import ProviderProfile


@pytest.fixture
def counted_profiles(monkeypatch):
    import providers

    overlaid = ProviderProfile(
        name="zz-overlaid", models_dev_id="zz-catalog", overlay_aliases=("zz-alias",),
        provider_overlay={"transport": "openai_chat", "auth_type": "api_key"})
    calls = []

    def list_providers():
        calls.append(1)
        return [overlaid]

    monkeypatch.setattr(providers, "list_providers", list_providers)
    return calls


@pytest.mark.parametrize("name, expected", [
    ("gemini", "gemini"), ("Anthropic", "anthropic"), ("openrouter", "openrouter"), ("custom", "custom"),
    ("nous", "nous"), ("claude", "anthropic"), ("kimi-coding", "kimi-for-coding"),
])
def test_builtin_names_skip_the_profile_scan(counted_profiles, name, expected):
    from hermes_cli.providers import normalize_provider

    assert normalize_provider(name) == expected
    assert counted_profiles == []


def test_overlay_names_still_resolve_through_profiles(counted_profiles):
    from hermes_cli.providers import normalize_provider

    assert normalize_provider("zz-alias") == "zz-catalog"
    assert normalize_provider("zz-overlaid") == "zz-catalog"
    assert normalize_provider("not-a-provider") == "not-a-provider"
    assert counted_profiles


def test_registry_wide_alias_check_reads_profiles_once(counted_profiles):
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.providers import _lossy_alias_registry_pdef

    PROVIDER_REGISTRY.get("kimi-coding")  # a home's first registry read mirrors its plugin scan, once
    counted_profiles.clear()
    assert _lossy_alias_registry_pdef("kimi-coding", "kimi-for-coding") is not None
    assert len(counted_profiles) == 1
