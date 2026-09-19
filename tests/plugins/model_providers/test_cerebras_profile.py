"""Unit tests for the Cerebras Inference provider profile.

Pins the profile's contract without going live: identity, alias
registration, and the small curated catalog (Cerebras only hosts models
that compile cleanly to the WSE dataflow architecture).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def cerebras_profile():
    """Resolve the registered Cerebras profile through the real discovery path."""
    # Importing model_tools triggers plugin discovery, registering the profile.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("cerebras")
    assert profile is not None, "cerebras provider profile must be registered"
    return profile


class TestCerebrasIdentity:
    def test_core_fields(self, cerebras_profile):
        p = cerebras_profile
        assert p.name == "cerebras"
        assert p.api_mode == "chat_completions"
        assert p.auth_type == "api_key"
        assert p.base_url == "https://api.cerebras.ai/v1"
        assert "CEREBRAS_API_KEY" in p.env_vars

    def test_display_metadata_present(self, cerebras_profile):
        # Prominence copy is surfaced in the picker; keep it non-empty rather
        # than pinning exact marketing wording (that's expected to change).
        assert cerebras_profile.display_name
        assert cerebras_profile.description
        assert cerebras_profile.signup_url.startswith("https://")


class TestCerebrasHeaders:
    def test_no_partner_attribution_headers(self, cerebras_profile):
        # Cerebras doesn't require the OpenRouter-style HTTP-Referer / X-Title
        # attribution pair, and adding them would risk leaking upstream branding
        # in Cloudflare request logs.
        assert "HTTP-Referer" not in cerebras_profile.default_headers
        assert "X-Title" not in cerebras_profile.default_headers


class TestCerebrasAliases:
    @pytest.mark.parametrize("alias", ["cerebras-ai", "cs", "wse"])
    def test_alias_resolves_via_registry(self, cerebras_profile, alias):
        import providers

        resolved = providers.get_provider_profile(alias)
        assert resolved is not None
        assert resolved.name == "cerebras"

    def test_aliases_declared_on_profile(self, cerebras_profile):
        assert "cerebras-ai" in cerebras_profile.aliases
        assert "cs" in cerebras_profile.aliases
        assert "wse" in cerebras_profile.aliases


class TestCerebrasModelDefaults:
    """Cerebras's catalog is small and curated — only models that compile
    cleanly to the WSE dataflow architecture ship. The bundled defaults
    should reference real production models, not router-only IDs.
    """

    def test_aux_model_is_curated_catalog_model(self, cerebras_profile):
        aux = cerebras_profile.default_aux_model
        assert aux, "expected a non-empty aux model"
        assert not aux.startswith("accounts/"), aux
        assert "/routers/" not in aux

    def test_fallback_models_are_curated_catalog_models(self, cerebras_profile):
        assert cerebras_profile.fallback_models, "expected curated fallbacks"
        for model in cerebras_profile.fallback_models:
            assert model, model
            assert not model.startswith("accounts/"), model
            assert "/routers/" not in model

    def test_fallback_models_are_short_plain_ids(self, cerebras_profile):
        # Cerebras's catalog uses short bare model IDs (no org prefix), unlike
        # OpenRouter / HuggingFace style 'org/model' slugs. Pin this so a
        # future contributor doesn't accidentally paste a router-style ID.
        for model in cerebras_profile.fallback_models:
            assert "/" not in model, (
                f"Cerebras model IDs should be bare, not slugs: {model!r}"
            )


class TestCerebrasDiscovery:
    """The profile must be discoverable by both the canonical name and any
    declared alias. The ``_discover_providers()`` path is the production
    entry point; smoke-test that it surfaces Cerebras alongside the other
    bundled providers.
    """

    def test_profile_appears_in_list_providers(self, cerebras_profile):
        import providers

        names = {p.name for p in providers.list_providers()}
        assert "cerebras" in names
