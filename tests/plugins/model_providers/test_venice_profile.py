"""Unit tests for the Venice AI provider profile."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def venice_profile():
    import providers

    profile = providers.get_provider_profile("venice")
    assert profile is not None, "Venice provider profile must be registered"
    return profile


class TestVeniceIdentity:
    def test_profile_contract(self, venice_profile):
        assert venice_profile.name == "venice"
        assert venice_profile.auth_type == "api_key"
        assert venice_profile.base_url == "https://api.venice.ai/api/v1"
        assert "VENICE_API_KEY" in venice_profile.env_vars
        assert "VENICE_BASE_URL" in venice_profile.env_vars
        assert venice_profile.display_name
        assert venice_profile.signup_url.startswith("https://")

    def test_alias_resolves(self, venice_profile):
        import providers

        assert providers.get_provider_profile("venice-ai") is venice_profile


class TestVeniceModelCatalog:
    def test_defaults_are_usable_by_standard_transport(self, venice_profile):
        assert venice_profile.fallback_models
        assert venice_profile.default_aux_model in venice_profile.fallback_models
        assert all(
            not model_id.startswith("e2ee-")
            for model_id in venice_profile.fallback_models
        )

    def test_live_catalog_requests_text_models_and_excludes_e2ee(self, venice_profile):
        with patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["zai-org-glm-5", "e2ee-glm-5", "tee-qwen3-5-122b-a10b"],
        ) as fetch:
            models = venice_profile.fetch_models(api_key="venice-test-key")

        assert models == ["zai-org-glm-5", "tee-qwen3-5-122b-a10b"]
        fetch.assert_called_once_with(
            "venice-test-key",
            "https://api.venice.ai/api/v1",
            timeout=8.0,
            query_params={"type": "text"},
        )
