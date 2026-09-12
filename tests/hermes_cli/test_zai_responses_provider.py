"""Behavior contracts for the Z.AI Responses API provider (api.z.ai/api/v1).

Z.AI serves its GLM models on independently-billed endpoints: the built-in
``zai`` profile owns the Chat Completions wire (``/api/paas/v4``), this
plugin profile owns the native OpenAI Responses wire (``/api/v1/responses``).
These tests pin the profile registration, alias resolution, auth/registry
auto-wiring, wire-mode detection, and the ``{"models": [...]}/"slug"``
catalog parsing the generic OpenAI ``{"data": [...]}/"id"`` parser would
miss — all without network access.
"""

import json
import sys

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_LABELS
from hermes_cli.providers import determine_api_mode, get_label, get_provider
from providers import get_provider_profile

# Realistic subset of GET https://api.z.ai/api/v1/models (verified live,
# HTTP 200, 2026-08-30): envelope keyed by "models", entries keyed by
# "slug", with a supported_in_api flag and vendor metadata alongside.
ZAI_MODELS_PAYLOAD = {
    "models": [
        {
            "slug": "glm-5.3",
            "display_name": "glm-5.3",
            "description": "Z.ai's latest flagship model",
            "context_window": 1048576,
            "supported_in_api": True,
        },
        {
            "slug": "glm-5.3-flash",
            "display_name": "glm-5.3-flash",
            "context_window": 1048576,
            "supported_in_api": True,
        },
        {
            "slug": "glm-5-turbo",
            "display_name": "glm-5-turbo",
            "supported_in_api": True,
        },
        {
            "slug": "glm-5.3-preview-internal",
            "display_name": "internal preview",
            "supported_in_api": False,
        },
    ]
}


def _profile():
    profile = get_provider_profile("zai-responses")
    assert profile is not None
    return profile


def _zai_responses_module():
    """The plugin module, via the registered profile (router-test idiom)."""
    profile = _profile()
    return sys.modules[type(profile).__module__]


class TestProfileRegistration:
    def test_profile_loads_with_responses_wire(self):
        profile = _profile()
        assert profile.name == "zai-responses"
        assert profile.display_name == "Z.AI (Responses API)"
        # The entire point of the profile: the Responses endpoint, NOT the
        # Chat Completions endpoint owned by the ``zai`` profile.
        assert profile.base_url == "https://api.z.ai/api/v1"
        assert profile.api_mode == "codex_responses"
        assert profile.auth_type == "api_key"

    def test_env_vars_cover_documented_key_and_aliases(self):
        profile = _profile()
        assert profile.env_vars[:3] == ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY")
        assert "ZAI_RESPONSES_BASE_URL" in profile.env_vars

    def test_fallback_models_match_live_catalog(self):
        # Catalog verified live 2026-08-30: exactly these slugs are served.
        assert _profile().fallback_models == (
            "glm-5.3",
            "glm-5.3-flash",
            "glm-5-turbo",
        )

    def test_canonical_providers_and_label(self):
        assert "zai-responses" in [p.slug for p in CANONICAL_PROVIDERS]
        assert _PROVIDER_LABELS["zai-responses"] == "Z.AI (Responses API)"
        assert get_label("zai-responses") == "Z.AI (Responses API)"


class TestAliasResolution:
    @pytest.mark.parametrize("alias", ["zai-responses", "zai-v1", "glm-responses"])
    def test_aliases_resolve_to_profile(self, alias):
        assert resolve_provider(alias) == "zai-responses"

    def test_built_in_zai_profile_is_not_shadowed(self):
        # Same host, different wire: the Chat Completions profile must keep
        # resolving exactly as before this plugin existed.
        assert resolve_provider("zai") == "zai"
        assert resolve_provider("glm") == "zai"
        assert PROVIDER_REGISTRY["zai"].inference_base_url == (
            "https://api.z.ai/api/paas/v4"
        )
        zai_profile = get_provider_profile("zai")
        assert zai_profile is not None
        assert zai_profile.base_url == "https://api.z.ai/api/paas/v4"


class TestRegistryAutoWiring:
    def test_registry_entry(self):
        pconfig = PROVIDER_REGISTRY["zai-responses"]
        assert pconfig.id == "zai-responses"
        assert pconfig.name == "Z.AI (Responses API)"
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == "https://api.z.ai/api/v1"
        # Base-URL override vars are stripped from the key vars and routed
        # to base_url_env_var by the registry auto-extension.
        assert pconfig.api_key_env_vars == (
            "GLM_API_KEY",
            "ZAI_API_KEY",
            "Z_AI_API_KEY",
        )
        assert pconfig.base_url_env_var == "ZAI_RESPONSES_BASE_URL"

    def test_credentials_resolve_from_glm_api_key(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "zai-responses-test-key")
        creds = resolve_api_key_provider_credentials("zai-responses")
        assert creds["provider"] == "zai-responses"
        assert creds["api_key"] == "zai-responses-test-key"
        assert creds["base_url"] == "https://api.z.ai/api/v1"

    def test_wire_mode_detection(self):
        provider = get_provider("zai-responses")
        assert provider is not None
        assert provider.id == "zai-responses"
        assert provider.base_url == "https://api.z.ai/api/v1"
        assert (
            determine_api_mode("zai-responses", "https://api.z.ai/api/v1")
            == "codex_responses"
        )


class TestCatalogParsing:
    def test_parse_models_slug_envelope(self):
        parsed = _zai_responses_module().parse_zai_responses_models(ZAI_MODELS_PAYLOAD)
        assert parsed == ["glm-5.3", "glm-5.3-flash", "glm-5-turbo"]

    def test_parse_models_dedupes_preserving_order(self):
        payload = {
            "models": [
                {"slug": "glm-5.3-flash", "supported_in_api": True},
                {"slug": "glm-5.3"},
                {"slug": "glm-5.3-flash", "supported_in_api": True},
            ]
        }
        parsed = _zai_responses_module().parse_zai_responses_models(payload)
        assert parsed == ["glm-5.3-flash", "glm-5.3"]

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            {},
            {"models": []},
            {"models": "nope"},
            {"data": [{"id": "openai-shape"}]},
            {"models": [{"id": "no-slug"}, {"slug": ""}, "not-a-dict"]},
        ],
    )
    def test_parse_models_returns_none_without_usable_entries(self, payload):
        parser = _zai_responses_module().parse_zai_responses_models
        assert parser(payload) is None


class TestFetchModels:
    @staticmethod
    def _patch_urlopen(monkeypatch, payload):
        """Stub the credentialed opener; capture the request for assertions."""
        captured = {}

        class _FakeResponse:
            def read(self):
                return json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def _fake_open(req, timeout=None):
            captured["req"] = req
            captured["timeout"] = timeout
            return _FakeResponse()

        monkeypatch.setattr(
            "hermes_cli.urllib_security.open_credentialed_url", _fake_open
        )
        return captured

    def test_fetch_models_parses_zai_envelope_and_sends_bearer(self, monkeypatch):
        captured = self._patch_urlopen(monkeypatch, ZAI_MODELS_PAYLOAD)
        profile = _profile()

        assert profile.fetch_models(api_key="k-123") == [
            "glm-5.3",
            "glm-5.3-flash",
            "glm-5-turbo",
        ]
        req = captured["req"]
        assert req.full_url == "https://api.z.ai/api/v1/models"
        assert req.headers.get("Authorization") == "Bearer k-123"

    def test_fetch_models_returns_none_on_failure(self, monkeypatch):
        def _boom(req, timeout=None):
            raise OSError("network down")

        monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", _boom)
        assert _profile().fetch_models(api_key="k-123") is None
