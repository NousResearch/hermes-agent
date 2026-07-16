"""Unit tests for the Tencent TokenPlan provider profile.

Tencent TokenPlan is a drop-in model-provider plugin
(``plugins/model-providers/tencent-tokenplan/``) that routes Hermes through
Tencent's LKEAP gateway to the Hy3 (Hunyuan) model over an Anthropic
Messages-compatible endpoint.

These tests pin the profile's identity, aliases, transport mode, endpoint,
and auxiliary model so any future drift shows up as a failing test.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def tokenplan_profile():
    """Resolve the registered tencent-tokenplan profile via the public API.

    Importing ``model_tools`` triggers lazy plugin discovery so the plugin
    directory is scanned before we look the profile up.
    """
    import model_tools  # noqa: F401  -- triggers plugin discovery
    import providers

    profile = providers.get_provider_profile("tencent-tokenplan")
    assert profile is not None, "tencent-tokenplan provider profile must be registered"
    return profile


class TestTencentTokenplanIdentity:
    """Canonical id, aliases, and human-readable metadata."""

    def test_name(self, tokenplan_profile):
        assert tokenplan_profile.name == "tencent-tokenplan"

    def test_display_name(self, tokenplan_profile):
        assert tokenplan_profile.display_name == "Tencent TokenPlan"

    @pytest.mark.parametrize("alias", ["tokenplan", "tencent-lkeap"])
    def test_alias_resolves(self, alias):
        import model_tools  # noqa: F401
        import providers

        profile = providers.get_provider_profile(alias)
        assert profile is not None
        assert profile.name == "tencent-tokenplan"


class TestTencentTokenplanTransport:
    """TokenPlan uses the Anthropic Messages protocol over the LKEAP gateway."""

    def test_api_mode_is_anthropic_messages(self, tokenplan_profile):
        assert tokenplan_profile.api_mode == "anthropic_messages"

    def test_base_url(self, tokenplan_profile):
        # Must NOT include the /v1/messages suffix — the Anthropic SDK appends
        # it. Including it would produce a doubled path and a 404.
        assert (
            tokenplan_profile.base_url
            == "https://api.lkeap.cloud.tencent.com/plan/anthropic"
        )

    def test_base_url_has_no_messages_suffix(self, tokenplan_profile):
        assert not tokenplan_profile.base_url.rstrip("/").endswith("/messages")

    def test_auth_type_is_api_key(self, tokenplan_profile):
        assert tokenplan_profile.auth_type == "api_key"

    def test_env_vars(self, tokenplan_profile):
        assert "TOKENPLAN_API_KEY" in tokenplan_profile.env_vars
        assert "TOKENPLAN_BASE_URL" in tokenplan_profile.env_vars


class TestTencentTokenplanModels:
    """Aux model and fallback catalog advertise Hy3."""

    def test_default_aux_model(self, tokenplan_profile):
        assert tokenplan_profile.default_aux_model == "hy3"

    def test_hy3_in_fallback_models(self, tokenplan_profile):
        assert "hy3" in tokenplan_profile.fallback_models

    def test_consumer_aux_api_returns_hy3(self, tokenplan_profile):
        from agent.auxiliary_client import _get_aux_model_for_provider

        resolved = _get_aux_model_for_provider("tencent-tokenplan")
        assert resolved == tokenplan_profile.default_aux_model
