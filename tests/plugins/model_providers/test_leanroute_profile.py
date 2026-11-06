"""Unit tests for the Leanroute provider profile.

Guards against the profile failing to construct at import time: bundled
plugins are loaded inside a try/except, so a bad ProviderProfile kwarg
only logs a warning and the provider silently drops out of the registry.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def leanroute_profile():
    """Resolve the registered Leanroute profile."""
    import model_tools  # noqa: F401  (triggers provider discovery)
    import providers

    profile = providers.get_provider_profile("leanroute")
    assert profile is not None, "leanroute provider profile must be registered"
    return profile


class TestLeanrouteProfileIdentity:
    def test_name_and_display_name(self, leanroute_profile):
        assert leanroute_profile.name == "leanroute"
        assert leanroute_profile.display_name == "Leanroute"

    def test_endpoint_and_auth(self, leanroute_profile):
        assert leanroute_profile.base_url == "https://api.leanroute.dev/v1"
        assert leanroute_profile.api_mode == "chat_completions"
        assert "LEANROUTE_API_KEY" in leanroute_profile.env_vars

    def test_fallback_models_are_provider_prefixed(self, leanroute_profile):
        assert leanroute_profile.fallback_models, "fallback_models must not be empty"
        for model in leanroute_profile.fallback_models:
            assert "/" in model, f"Leanroute model ids are provider/model: got {model}"

    def test_aux_model_is_in_fallbacks(self, leanroute_profile):
        assert leanroute_profile.default_aux_model in leanroute_profile.fallback_models
