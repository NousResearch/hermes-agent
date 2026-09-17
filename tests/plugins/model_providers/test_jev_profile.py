"""TypeSafe Jev provider profile: registry, catalog, no /v1/models probe."""

from __future__ import annotations

import pytest


@pytest.fixture
def jev_profile():
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("jev")
    assert profile is not None, "jev provider profile must be registered"
    return profile


class TestJevProfile:
    def test_identity_and_endpoint(self, jev_profile):
        assert jev_profile.name == "jev"
        assert jev_profile.auth_type == "api_key"
        assert jev_profile.base_url == "https://api.typesafe.ai/v1"
        assert jev_profile.get_hostname() == "api.typesafe.ai"
        assert jev_profile.supports_health_check is False
        assert jev_profile.default_aux_model == ""

    def test_env_vars(self, jev_profile):
        assert jev_profile.env_vars[0] == "TYPESAFE_API_KEY"
        assert "TYPESAFE_BASE_URL" in jev_profile.env_vars

    def test_fallback_catalog_is_jev_latest(self, jev_profile):
        assert jev_profile.fallback_models
        assert "jev-latest" in jev_profile.fallback_models

    @pytest.mark.parametrize("alias", ["jev", "typesafe", "typesafe-ai"])
    def test_aliases_resolve(self, alias):
        import model_tools  # noqa: F401
        import providers

        profile = providers.get_provider_profile(alias)
        assert profile is not None
        assert profile.name == "jev"

    def test_fetch_models_is_static_and_offline(self, jev_profile, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise AssertionError("jev fetch_models must not hit the network")

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        models = jev_profile.fetch_models(api_key="unused", timeout=0.1)
        assert models == ["jev-latest"]

    def test_picker_lists_jev(self):
        import model_tools  # noqa: F401
        from hermes_cli.models import CANONICAL_PROVIDERS

        slugs = {p.slug for p in CANONICAL_PROVIDERS}
        assert "jev" in slugs
