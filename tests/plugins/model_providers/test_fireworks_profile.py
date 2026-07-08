"""Unit tests for the Fireworks AI provider profile.

Pins the profile's contract without going live: identity, attribution headers
(so Fireworks can attribute Hermes traffic for the partnership), alias
registration, and the pay-as-you-go model defaults (direct catalog ``/models/``
IDs, not the router-only tier).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fireworks_profile():
    """Resolve the registered Fireworks profile through the real discovery path."""
    # Importing model_tools triggers plugin discovery, registering the profile.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("fireworks")
    assert profile is not None, "fireworks provider profile must be registered"
    return profile


class TestFireworksIdentity:
    def test_core_fields(self, fireworks_profile):
        p = fireworks_profile
        assert p.name == "fireworks"
        assert p.auth_type == "api_key"
        assert p.base_url == "https://api.fireworks.ai/inference/v1"
        assert "FIREWORKS_API_KEY" in p.env_vars
        assert "FIREWORKS_BASE_URL" in p.env_vars

    def test_display_metadata_present(self, fireworks_profile):
        # Prominence copy is surfaced in the picker; keep it non-empty rather
        # than pinning exact marketing wording (that's expected to change).
        assert fireworks_profile.display_name
        assert fireworks_profile.description
        assert fireworks_profile.signup_url.startswith("https://")


class TestFireworksAttributionHeaders:
    """The profile carries the partner-attribution headers Fireworks captures."""

    def test_referer_and_title_headers(self, fireworks_profile):
        headers = fireworks_profile.default_headers
        assert headers.get("HTTP-Referer") == "https://hermes-agent.nousresearch.com"
        assert headers.get("X-Title") == "Hermes Agent"


class TestFireworksAliases:
    @pytest.mark.parametrize("alias", ["fireworks-ai", "fw"])
    def test_alias_resolves_via_registry(self, fireworks_profile, alias):
        import providers

        resolved = providers.get_provider_profile(alias)
        assert resolved is not None
        assert resolved.name == "fireworks"

    def test_aliases_declared_on_profile(self, fireworks_profile):
        assert "fireworks-ai" in fireworks_profile.aliases
        assert "fw" in fireworks_profile.aliases


class TestFireworksModelDefaults:
    """Defaults must be usable with a standard pay-as-you-go key.

    PAYG keys address ``accounts/fireworks/models/...`` directly; the bundled
    defaults target that (the BYOK motion) so a fresh key works out of the box,
    and use the standard tier rather than turbo as the out-of-box default.
    """

    def test_aux_model_is_payg_model_not_router(self, fireworks_profile):
        aux = fireworks_profile.default_aux_model
        assert aux.startswith("accounts/fireworks/models/"), aux
        assert "/routers/" not in aux
        assert "turbo" not in aux.lower()

    def test_fallback_models_are_payg_models_not_routers(self, fireworks_profile):
        assert fireworks_profile.fallback_models, "expected curated fallbacks"
        for model in fireworks_profile.fallback_models:
            assert model.startswith("accounts/fireworks/models/"), model
            assert "/routers/" not in model
            assert "turbo" not in model.lower(), model
