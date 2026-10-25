"""Focused tests for Kosmik / KosCompute provider wiring."""

from __future__ import annotations

import sys
import types
import pytest

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv


class TestKosmikResolver:
    """The providers.py resolver must recognise kosmik and its aliases."""

    @pytest.mark.parametrize("slug", ["kosmik", "koscompute", "kosmik-ai", "kos"])
    def test_resolve_provider_full_recognizes_kosmik(self, slug):
        from hermes_cli.providers import resolve_provider_full

        pdef = resolve_provider_full(slug, {}, [])
        assert pdef is not None, f"resolve_provider_full({slug!r}) returned None"
        assert pdef.id == "kosmik"
        assert pdef.base_url == "https://api.koscompute.com/v1"
        assert "KOSMIK_API_KEY" in pdef.api_key_env_vars
        assert "KOSCOMPUTE_API_KEY" in pdef.api_key_env_vars


class TestKosmikAuth:
    """Auth configuration and discovery for Kosmik."""

    def test_auth_provider_registry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        cfg = PROVIDER_REGISTRY.get("kosmik")
        assert cfg is not None, "kosmik must be in PROVIDER_REGISTRY"
        assert cfg.name == "Kosmik"
        assert cfg.inference_base_url == "https://api.koscompute.com/v1"
        assert "KOSMIK_API_KEY" in cfg.api_key_env_vars
        assert "KOSCOMPUTE_API_KEY" in cfg.api_key_env_vars
        assert cfg.base_url_env_var == "KOSMIK_BASE_URL"

    @pytest.mark.parametrize("alias", ["koscompute", "kosmik-ai", "kos"])
    def test_auth_provider_aliases(self, alias):
        from hermes_cli.providers import normalize_provider

        assert normalize_provider(alias) == "kosmik"


class TestKosmikConfigDefaults:
    """Config defaults and optional env var definitions."""

    def test_optional_env_vars(self):
        from hermes_cli.config_defaults import OPTIONAL_ENV_VARS

        assert "KOSMIK_API_KEY" in OPTIONAL_ENV_VARS
        assert "KOSCOMPUTE_API_KEY" in OPTIONAL_ENV_VARS
        assert "KOSMIK_BASE_URL" in OPTIONAL_ENV_VARS

    def test_doctor_connectivity_discovers_kosmik(self):
        from hermes_cli.doctor_connectivity import _build_apikey_providers_list

        entries = [e for e in _build_apikey_providers_list() if e[0] == "Kosmik"]
        assert len(entries) == 1
        label, key_vars, models_url, base_env, supports_check = entries[0]
        assert label == "Kosmik"
        assert "KOSMIK_API_KEY" in key_vars
        assert models_url == "https://api.koscompute.com/v1/models"
        assert base_env == "KOSMIK_BASE_URL"
        assert supports_check is True


class TestKosmikCatalog:
    """Model catalog fallback and live resolution."""

    def test_provider_model_ids_falls_back_without_key(self, monkeypatch):
        from hermes_cli.models import provider_model_ids
        from hermes_cli.models_catalog_static import _PROVIDER_MODELS

        monkeypatch.delenv("KOSMIK_API_KEY", raising=False)
        monkeypatch.delenv("KOSCOMPUTE_API_KEY", raising=False)

        models = provider_model_ids("kosmik")
        assert models == list(_PROVIDER_MODELS["kosmik"])

    def test_provider_model_ids_filters_live_catalog(self, monkeypatch):
        from hermes_cli.models import provider_model_ids
        import providers

        profile = providers.get_provider_profile("kosmik")
        assert profile is not None

        monkeypatch.setenv("KOSMIK_API_KEY", "mock-kosmik-key")
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda pid: {
                "provider": pid,
                "api_key": "mock-kosmik-key",
                "base_url": "https://api.koscompute.com/v1",
                "source": "KOSMIK_API_KEY",
            },
        )
        monkeypatch.setattr(
            profile,
            "fetch_models",
            lambda **kw: ["qwen/qwen3.8-27b"],
        )

        models = provider_model_ids("kosmik")
        assert "qwen/qwen3.8-27b" in models
