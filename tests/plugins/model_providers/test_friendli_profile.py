"""Unit tests for the Friendli provider profile.

Pins the profile's contract without going live: identity, alias
registration, and the curated defaults.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def friendli_profile():
    """Resolve the registered Friendli profile through the real discovery path."""
    # Importing model_tools triggers plugin discovery, registering the profile.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("friendli")
    assert profile is not None, "friendli provider profile must be registered"
    return profile


class TestFriendliIdentity:
    def test_core_fields(self, friendli_profile):
        p = friendli_profile
        assert p.name == "friendli"
        assert p.auth_type == "api_key"
        assert p.base_url == "https://api.friendli.ai/serverless/v1"
        assert "FRIENDLI_API_KEY" in p.env_vars
        assert "FRIENDLI_BASE_URL" in p.env_vars

    def test_display_metadata_present(self, friendli_profile):
        assert friendli_profile.display_name
        assert friendli_profile.description
        assert friendli_profile.signup_url.startswith("https://")


class TestFriendliAliases:
    @pytest.mark.parametrize("alias", ["friendliai", "friendli-ai"])
    def test_alias_resolves_via_registry(self, friendli_profile, alias):
        import providers

        resolved = providers.get_provider_profile(alias)
        assert resolved is not None
        assert resolved.name == "friendli"

    def test_aliases_declared_on_profile(self, friendli_profile):
        assert "friendliai" in friendli_profile.aliases
        assert "friendli-ai" in friendli_profile.aliases


class TestFriendliModelDefaults:
    def test_aux_model_is_a_curated_model(self, friendli_profile):
        assert friendli_profile.default_aux_model in friendli_profile.fallback_models

    def test_fallback_models_are_vendor_slug_form(self, friendli_profile):
        assert friendli_profile.fallback_models, "expected curated fallbacks"
        for model in friendli_profile.fallback_models:
            assert "/" in model, model
