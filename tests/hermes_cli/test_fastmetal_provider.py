"""Focused tests for FastMetal first-class provider wiring.

FastMetal is not in models.dev, so every registry that does not auto-wire from the plugin layer
(the providers.py overlay, alias tables, env-var catalog, model normalization) carries an
explicit entry; these pin that an explicit ``provider: fastmetal`` resolves instead of being
dropped to auto-detect (the Upstage regression, #42231 salvage).
"""

from __future__ import annotations

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials, resolve_provider
from hermes_cli.config_defaults import OPTIONAL_ENV_VARS
from hermes_cli.model_normalize import normalize_model_for_provider
from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_ALIASES, _PROVIDER_LABELS, normalize_provider, provider_model_ids
from hermes_cli.providers import (
    HERMES_OVERLAYS,
    determine_api_mode,
    get_label,
    get_provider,
    is_aggregator,
    is_routing_aggregator,
    resolve_provider_full,
)


class TestFastMetalResolver:
    def test_resolve_provider_full_recognizes_fastmetal(self):
        pdef = resolve_provider_full("fastmetal", {}, [])
        assert pdef is not None, "config `provider: fastmetal` would be discarded and auto-detect would win"
        assert pdef.id == "fastmetal"
        assert pdef.base_url == "https://api.fastmetal.ai/v1"
        assert "FASTMETAL_API_KEY" in pdef.api_key_env_vars

    def test_aliases_resolve_everywhere(self, monkeypatch):
        monkeypatch.setenv("FASTMETAL_API_KEY", "sk-test-1234567890")
        for alias in ("fast-metal", "fastmetal-ai"):
            assert resolve_provider(alias) == "fastmetal"
            assert normalize_provider(alias) == "fastmetal"
            assert _PROVIDER_ALIASES[alias] == "fastmetal"
            assert get_provider(alias).id == "fastmetal"

    def test_explicit_config_provider_beats_stray_key(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "junk")
        monkeypatch.setenv("FASTMETAL_API_KEY", "sk-test-1234567890")
        adef = resolve_provider_full("fastmetal", {}, [])
        assert adef is not None and adef.id == "fastmetal"


class TestFastMetalOverlay:
    def test_overlay_is_a_flat_namespace_routing_gateway(self):
        overlay = HERMES_OVERLAYS["fastmetal"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_override == "https://api.fastmetal.ai/v1"
        assert overlay.base_url_env_var == "FASTMETAL_BASE_URL"
        assert overlay.extra_env_vars == ("FASTMETAL_API_KEY",)
        # Bare ids relayed upstream: /model searches the live catalog (aggregator) AND picking a
        # model here re-routes it, so picker dedup must treat it as a routing aggregator.
        assert is_aggregator("fastmetal") and is_routing_aggregator("fastmetal")
        assert determine_api_mode("fastmetal", "https://api.fastmetal.ai/v1") == "chat_completions"

    def test_label_and_canonical_row(self):
        assert get_label("fastmetal") == "FastMetal"
        assert _PROVIDER_LABELS["fastmetal"] == "FastMetal"
        assert "fastmetal" in [p.slug for p in CANONICAL_PROVIDERS]


class TestFastMetalCredentials:
    def test_registry_and_env_credentials(self, monkeypatch):
        monkeypatch.setenv("FASTMETAL_API_KEY", "sk-test-1234567890")
        monkeypatch.setenv("FASTMETAL_BASE_URL", "https://proxy.example.test/v1")
        pconfig = PROVIDER_REGISTRY["fastmetal"]
        assert pconfig.name == "FastMetal"
        assert pconfig.auth_type == "api_key"
        assert pconfig.api_key_env_vars == ("FASTMETAL_API_KEY",)
        assert pconfig.base_url_env_var == "FASTMETAL_BASE_URL"
        creds = resolve_api_key_provider_credentials("fastmetal")
        assert creds["api_key"] == "sk-test-1234567890"
        assert creds["base_url"] == "https://proxy.example.test/v1"

    def test_env_catalog_reaches_setup_and_dashboard(self):
        assert OPTIONAL_ENV_VARS["FASTMETAL_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["FASTMETAL_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["FASTMETAL_API_KEY"]["url"]
        assert OPTIONAL_ENV_VARS["FASTMETAL_BASE_URL"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["FASTMETAL_BASE_URL"]["password"] is False


class TestFastMetalModelIds:
    def test_copied_provider_prefix_is_stripped(self):
        """FastMetal rejects vendor-prefixed ids; a ``fastmetal/`` prefix copied from the aggregator form
        is repaired, while a foreign prefix is left alone (it is the user's mistake to see)."""
        assert normalize_model_for_provider("fastmetal/kimi-k3", "fastmetal") == "kimi-k3"
        assert normalize_model_for_provider("fast-metal/kimi-k3", "fast-metal") == "kimi-k3"
        assert normalize_model_for_provider("moonshotai/kimi-k3", "fastmetal") == "moonshotai/kimi-k3"

    def test_picker_is_live_first_with_curated_backfill(self, monkeypatch):
        """The gateway rotates vendors often, so the live /v1/models list leads and curated ids the
        live API omitted are appended rather than stale curated entries polluting the top."""
        from providers import get_provider_profile

        profile = get_provider_profile("fastmetal")
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {"provider": provider_id, "api_key": "sk-live", "base_url": profile.base_url, "source": "FASTMETAL_API_KEY"},
        )
        live = ["brand-new-model", profile.fallback_models[0]]
        monkeypatch.setattr(profile, "fetch_models", lambda *, api_key=None, base_url=None, timeout=8.0: list(live))
        result = provider_model_ids("fastmetal")
        assert result[: len(live)] == live
        assert set(result) == set(live) | set(profile.fallback_models)

    def test_picker_falls_back_to_curated_when_live_fails(self, monkeypatch):
        from providers import get_provider_profile

        profile = get_provider_profile("fastmetal")
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {"provider": provider_id, "api_key": "sk-live", "base_url": profile.base_url, "source": "FASTMETAL_API_KEY"},
        )
        monkeypatch.setattr(profile, "fetch_models", lambda *, api_key=None, base_url=None, timeout=8.0: None)
        assert provider_model_ids("fast-metal") == list(profile.fallback_models)
