"""Tests for the Volcengine Ark Agent Plan provider profile.

The Agent Plan endpoint (https://ark.cn-beijing.volces.com/api/plan/v3) has no
``/models`` route — it returns 404 — so the profile declares
``supports_health_check=False`` and carries a minimal ``fallback_models`` floor
while the authoritative catalog comes from models.dev. These tests pin that
contract through the real plugin discovery path.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def ark_profile():
    """Resolve the registered profile through real plugin discovery."""
    import model_tools  # noqa: F401 — triggers plugin discovery
    import providers

    profile = providers.get_provider_profile("volcengine-agent-plan")
    assert profile is not None, "volcengine-agent-plan profile must be registered"
    return profile


class TestVolcengineArkPlanProfile:
    def test_endpoint_is_the_plan_surface(self, ark_profile):
        """Subscription endpoint, not the pay-as-you-go /api/v3."""
        assert ark_profile.base_url == "https://ark.cn-beijing.volces.com/api/plan/v3"

    def test_health_check_disabled_no_models_route(self, ark_profile):
        """The plan endpoint has no /models route (404) — doctor must skip the probe."""
        assert ark_profile.supports_health_check is False

    def test_fetch_models_returns_none_without_models_route(self, ark_profile, monkeypatch):
        """fetch_models maps any HTTP failure to None, never [].

        An empty list would be cached as a valid catalog and blank the picker.
        Mocked at the HTTP layer: the real endpoint has no /models route
        (verified live 2026-09-21), but unit tests must not do network I/O —
        offline CI would stall and the vendor could flip the status code.
        """
        import urllib.error

        def _fake_open(req, timeout=None):
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", hdrs=__import__("email.message").Message(), fp=None
            )

        monkeypatch.setattr(
            "providers.base.open_credentialed_url", _fake_open, raising=False
        )
        result = ark_profile.fetch_models(api_key="invalid-key", timeout=5.0)
        assert result is None

    def test_fetch_models_maps_offline_errors_to_none(self, ark_profile, monkeypatch):
        """Connection failures (offline CI, DNS loss) must also yield None."""
        import urllib.error

        def _fake_open(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(
            "providers.base.open_credentialed_url", _fake_open, raising=False
        )
        assert ark_profile.fetch_models(api_key="k", timeout=5.0) is None

    def test_fallback_models_are_an_entry_floor(self, ark_profile):
        """The static list is a small entry floor (models.dev is the authority):
        only documented, tool-capable entry models, no full roster snapshot."""
        fallback = ark_profile.fallback_models
        assert "ark-code-latest" in fallback
        assert "deepseek-v4-flash" in fallback
        assert len(fallback) <= 4, "fallback floor must stay small — models.dev owns the roster"

    def test_api_key_env_var(self, ark_profile):
        assert "ARK_API_KEY" in ark_profile.env_vars


class TestModelsDevCatalogMapping:
    def test_provider_maps_to_models_dev_slug(self):
        """Hermes provider id → models.dev ``volcengine-agent-plan`` so the
        community-tracked roster reaches the picker."""
        from agent.models_dev import PROVIDER_TO_MODELS_DEV

        assert PROVIDER_TO_MODELS_DEV.get("volcengine-agent-plan") == "volcengine-agent-plan"

    def test_provider_in_models_dev_preferred_set(self):
        """models.dev is the ONLY catalog source for this endpoint (no /models
        route), so it must be in the merge set, not merely consulted."""
        from hermes_cli.models_catalog_static import _MODELS_DEV_PREFERRED

        assert "volcengine-agent-plan" in _MODELS_DEV_PREFERRED
