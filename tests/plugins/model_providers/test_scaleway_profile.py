"""Unit tests for the Scaleway Generative APIs provider profile.

Pins the profile's contract without going live: identity, alias registration,
endpoint/credential wiring, and the curated fallback catalog. The catalog lists
serverless models that tool-call on the chat-completions wire — gpt-oss-120b
tool-calls only through Scaleway's (beta) Responses API, so it must never be a
curated pick a setup flow could land on.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def scaleway_profile():
    """Resolve the registered Scaleway profile through the real discovery path."""
    # Importing model_tools triggers plugin discovery, registering the profile.
    import model_tools
    import providers

    profile = providers.get_provider_profile("scaleway")
    assert profile is not None, "scaleway provider profile must be registered"
    return profile


class TestScalewayIdentity:
    def test_alias_resolves_to_same_profile(self, scaleway_profile):
        import providers

        assert providers.get_provider_profile("scw") is scaleway_profile

    def test_endpoint_and_credential_wiring(self, scaleway_profile):
        assert scaleway_profile.base_url == "https://api.scaleway.ai/v1"
        assert scaleway_profile.env_vars == ("SCW_SECRET_KEY", "SCALEWAY_BASE_URL")
        assert scaleway_profile.auth_type == "api_key"

    def test_models_dev_catalog_wiring(self):
        """Scaleway models resolve capability metadata from the models.dev catalog
        (context windows, reasoning/tool-call badges) — the id is declared there."""
        from agent.models_dev import models_dev_id_for

        assert models_dev_id_for("scaleway") == "scaleway"


class TestScalewayModelDefaults:
    """The curated list backs the picker when the live /models fetch fails, and
    [0] is the setup default."""

    def test_default_model_leads_fallback_catalog(self, scaleway_profile):
        assert scaleway_profile.fallback_models[0] == "deepseek-v4.1-flash-1017"

    def test_fallback_models_are_serverless_ids(self, scaleway_profile):
        """Serverless catalog ids are bare; the vendor/:quant spellings belong to
        dedicated deployments and are never pickable here."""
        assert scaleway_profile.fallback_models
        for model in scaleway_profile.fallback_models:
            assert "/" not in model, model
            assert ":" not in model, model

    def test_fallback_models_exclude_responses_only_tool_calling(self, scaleway_profile):
        assert "gpt-oss-120b" not in scaleway_profile.fallback_models
