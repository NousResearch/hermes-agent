"""Tests for the Jev (TypeSafe) provider profile (#113837)."""

from providers import get_provider_profile


class TestJevProfile:
    def test_name_and_aliases_resolve(self):
        for name in ("jev", "typesafe", "typesafe-ai"):
            p = get_provider_profile(name)
            assert p is not None
            assert p.name == "jev"

    def test_credentials_and_checks(self):
        p = get_provider_profile("jev")
        assert p is not None
        assert "TYPESAFE_API_KEY" in p.env_vars
        assert p.supports_health_check is False

    def test_static_catalog_without_network(self):
        p = get_provider_profile("jev")
        assert p is not None
        assert p.fetch_models() == ["jev-latest"]

    def test_canonical_providers_include_jev(self):
        from hermes_cli.models_catalog_static import CANONICAL_PROVIDERS

        assert "jev" in {p.slug for p in CANONICAL_PROVIDERS}
