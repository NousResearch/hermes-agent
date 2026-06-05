"""Kiro ACP model-provider plugin profile tests."""

from providers import get_provider_profile


def test_kiro_acp_profile_registered():
    profile = get_provider_profile("kiro-acp")

    assert profile is not None
    assert profile.name == "kiro-acp"
    assert profile.auth_type == "external_process"
    assert profile.base_url == "acp://kiro"
    assert profile.api_mode == "chat_completions"
    assert profile.display_name == "Kiro CLI (ACP)"
    assert profile.fallback_models == ("kiro-cli",)


def test_kiro_acp_aliases_resolve_to_canonical_provider():
    for alias in ("kiro", "kiro-cli", "kiro-cli-acp"):
        profile = get_provider_profile(alias)
        assert profile is not None
        assert profile.name == "kiro-acp"


def test_kiro_acp_fetch_models_returns_none():
    profile = get_provider_profile("kiro-acp")
    assert profile is not None
    assert profile.fetch_models() is None
