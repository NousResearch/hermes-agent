"""Unit tests for the Baseten provider profile.

Pins the profile's contract without going live: identity, attribution headers,
and the Model APIs catalog defaults (shared ``inference.baseten.co`` endpoint,
``vendor/Model`` ids — not dedicated-deployment URLs).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def baseten_profile():
    """Resolve the registered Baseten profile through the real discovery path."""
    # Importing model_tools triggers plugin discovery, registering the profile.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("baseten")
    assert profile is not None, "baseten provider profile must be registered"
    return profile


class TestBasetenIdentity:
    def test_endpoint_is_the_shared_model_apis_route(self, baseten_profile):
        """Model APIs, not a dedicated deployment: dedicated routes are
        per-model hosts (``model-<id>.api.baseten.co/environments/...``) and
        cannot back a provider-wide catalog."""
        assert baseten_profile.base_url == "https://inference.baseten.co/v1"
        assert baseten_profile.get_hostname() == "inference.baseten.co"

    def test_api_key_env_var_is_declared(self, baseten_profile):
        assert "BASETEN_API_KEY" in baseten_profile.env_vars
        assert baseten_profile.auth_type == "api_key"


class TestBasetenHeaders:
    def test_attribution_matches_canonical_hermes_values(self, baseten_profile):
        """Asserted against the shared constant rather than the literals so a
        rebrand can't leave one provider on a stale referer/title."""
        from agent.auxiliary_client import _OR_HEADERS_BASE

        headers = baseten_profile.default_headers
        assert headers["HTTP-Referer"] == _OR_HEADERS_BASE["HTTP-Referer"]
        assert headers["X-Title"] == _OR_HEADERS_BASE["X-Title"]

    def test_user_agent_identifies_hermes(self, baseten_profile):
        from hermes_cli.version_info import get_version_info
        assert baseten_profile.default_headers["User-Agent"] == (
            f"HermesAgent/{get_version_info().base_version}"
        )


class TestBasetenModelDefaults:
    """Baseten's Model APIs catalog is exclusively ``vendor/Model``; the curated
    fallbacks must be addressable with a plain BASETEN_API_KEY."""

    def test_fallback_models_are_vendor_qualified(self, baseten_profile):
        assert baseten_profile.fallback_models, "expected curated fallbacks"
        for model in baseten_profile.fallback_models:
            assert "/" in model, model
            assert not model.startswith("http"), model

    def test_aux_model_is_a_cheap_catalog_entry(self, baseten_profile):
        aux = baseten_profile.default_aux_model
        assert "/" in aux, aux
        assert aux in baseten_profile.fallback_models, (
            "aux model should be one of the curated ids so the picker and the "
            "auxiliary path cannot drift apart"
        )


class TestBasetenReasoning:
    """The wire contract here was verified against the live Model APIs endpoint, not inferred:
    Baseten validates ``reasoning_effort`` server-side and returns 400 for anything outside
    ``none, minimal, low, medium, high, xhigh, max``."""

    def test_effort_goes_out_as_top_level_reasoning_effort(self, baseten_profile):
        extra_body, top_level = baseten_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "low"},
            model="zai-org/GLM-5.3",
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "low"}

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max"])
    def test_natively_supported_levels_are_not_downgraded(self, baseten_profile, effort):
        """Regression: clamping to a three-level ladder silently turned a user's ``max`` into
        ``high``. Baseten takes all six natively, so each must reach the wire unchanged."""
        _, top = baseten_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="zai-org/GLM-5.3",
        )
        assert top == {"reasoning_effort": effort}

    def test_effort_above_the_ladder_clamps_down_not_up(self, baseten_profile):
        _, top = baseten_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "ultra"}, model="zai-org/GLM-5.3",
        )
        assert top == {"reasoning_effort": "max"}

    def test_unset_effort_leaves_the_route_default_in_charge(self, baseten_profile):
        """A bare request is accepted by every live model; Hermes does not invent a default."""
        assert baseten_profile.build_api_kwargs_extras(
            reasoning_config=None, model="zai-org/GLM-5.3",
        ) == ({}, {})

    @pytest.mark.parametrize("reasoning_config", [
        {"enabled": False},
        {"enabled": False, "effort": "high"},
        {"enabled": True, "effort": "none"},
        {"enabled": True, "effort": "off"},
    ])
    def test_disabled_maps_to_minimal_never_none(self, baseten_profile, reasoning_config):
        """``none`` is documented but not portable: zai-org/GLM-5.3-Fast returns 400 for it while
        accepting every other level, and on models that do take it the reasoning-token count
        matches ``minimal`` rather than dropping to zero. ``minimal`` is accepted by every live
        model, so it is the weakest level Hermes can ask for without risking a hard failure."""
        _, top = baseten_profile.build_api_kwargs_extras(
            reasoning_config=reasoning_config, model="zai-org/GLM-5.3-Fast",
        )
        assert top == {"reasoning_effort": "minimal"}
        assert top["reasoning_effort"] != "none"

    def test_vocabulary_excludes_none_so_it_can_never_be_clamped_onto(self, baseten_profile):
        from agent.reasoning_effort import BASETEN_EFFORTS

        assert "none" not in BASETEN_EFFORTS
        assert BASETEN_EFFORTS == ("minimal", "low", "medium", "high", "xhigh", "max")

    def test_explicit_non_reasoning_model_is_skipped(self, baseten_profile):
        assert baseten_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="some-vendor/Not-A-Thinker",
            supports_reasoning=False,
        ) == ({}, {})
