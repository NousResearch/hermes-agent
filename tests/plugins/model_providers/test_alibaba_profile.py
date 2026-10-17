"""Unit tests for the Alibaba DashScope provider profiles' reasoning wiring.

DashScope's OpenAI-compatible mode reads thinking from two fields that must agree:
``extra_body.enable_thinking`` plus a top-level ``reasoning_effort``. The server's
default for Qwen thinking models is *full* thinking, so a profile that never
translates the user's setting leaves ``--reasoning none`` and
``agent.reasoning_effort`` silent no-ops while the config surfaces still record
them as applied (#137174).

These tests pin the profile's wire-shape contract so DashScope requests stay
correctly shaped without going live.
"""

from __future__ import annotations

import pytest

PROFILE_NAMES = ("alibaba", "alibaba-cn", "alibaba-token-plan", "alibaba-token-plan-cn")


@pytest.fixture(scope="module")
def profiles():
    """Resolve the registered DashScope profiles.

    Going through ``providers.get_provider_profile`` keeps the test honest —
    if someone later replaces the registered class with a plain
    ``ProviderProfile``, every assertion below collapses. ``model_tools``
    triggers plugin discovery on import, which is what registers the Alibaba
    profiles in the global provider registry.
    """
    import model_tools
    import providers

    out = {}
    for name in PROFILE_NAMES:
        profile = providers.get_provider_profile(name)
        assert profile is not None, f"{name} provider profile must be registered"
        out[name] = profile
    return out


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
class TestDashScopeReasoningWireShape:
    """``build_api_kwargs_extras`` produces DashScope's exact wire format."""

    def test_no_config_leaves_the_server_default(self, profiles, profile_name):
        extra_body, top_level = profiles[profile_name].build_api_kwargs_extras(
            reasoning_config=None, model="qwen3.8-flash"
        )
        assert extra_body == {"preserve_thinking": True}
        assert top_level == {}

    def test_enabled_without_effort_omits_the_knob(self, profiles, profile_name):
        extra_body, top_level = profiles[profile_name].build_api_kwargs_extras(
            reasoning_config={"enabled": True}, model="qwen3.8-flash"
        )
        assert extra_body == {"preserve_thinking": True}
        assert top_level == {}

    def test_disabled_sends_both_fields_agreeing_off(self, profiles, profile_name):
        extra_body, top_level = profiles[profile_name].build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="qwen3.8-flash"
        )
        assert extra_body["enable_thinking"] is False
        assert top_level == {"reasoning_effort": "none"}

    @pytest.mark.parametrize("effort", ["none", "false", "disabled", "None", " FALSE "])
    def test_off_words_are_an_explicit_disable(self, profiles, profile_name, effort):
        extra_body, top_level = profiles[profile_name].build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="qwen3.8-flash"
        )
        assert extra_body["enable_thinking"] is False
        assert top_level == {"reasoning_effort": "none"}

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "HIGH", " Medium "])
    def test_standard_efforts_pass_through(self, profiles, profile_name, effort):
        extra_body, top_level = profiles[profile_name].build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="qwen3.8-flash"
        )
        assert extra_body["enable_thinking"] is True
        assert top_level == {"reasoning_effort": effort.strip().lower()}

    @pytest.mark.parametrize("effort", ["xhigh", "max", "ultra", "MAX"])
    def test_above_high_clamps_down_to_high(self, profiles, profile_name, effort):
        extra_body, top_level = profiles[profile_name].build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="qwen3.8-flash"
        )
        assert extra_body["enable_thinking"] is True
        assert top_level == {"reasoning_effort": "high"}

    def test_minimal_clamps_down_to_low(self, profiles, profile_name):
        extra_body, top_level = profiles[profile_name].build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "minimal"},
            model="qwen3.8-flash",
        )
        assert extra_body["enable_thinking"] is True
        assert top_level == {"reasoning_effort": "low"}

    def test_unrecognized_effort_is_not_guessed(self, profiles, profile_name):
        extra_body, top_level = profiles[profile_name].build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "turbo"}, model="qwen3.8-flash"
        )
        assert "enable_thinking" not in extra_body
        assert top_level == {}


class TestDashScopeDisableKeepsReplayExtras:
    """The reasoning translation must compose with the replay-preservation extras."""

    def test_disable_on_non_preserving_model_sends_only_the_pair(self, profiles):
        extra_body, top_level = profiles["alibaba"].build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="qwen-turbo"
        )
        assert extra_body == {"enable_thinking": False}
        assert top_level == {"reasoning_effort": "none"}

    def test_disable_on_preserving_model_merges_both_extras(self, profiles):
        extra_body, top_level = profiles["alibaba-cn"].build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="qwen3.8-flash"
        )
        assert extra_body == {"preserve_thinking": True, "enable_thinking": False}
        assert top_level == {"reasoning_effort": "none"}
