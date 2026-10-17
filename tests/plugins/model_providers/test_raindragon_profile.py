"""Unit tests for the RainDragon provider profile.

RainDragon is an OpenAI-compatible api-key provider serving Gemma 4. These check
the registered identity / endpoint / auth / catalog fields every downstream layer
reads, and that ``/reasoning`` maps onto the Gemma template's on/off toggle
(``chat_template_kwargs.enable_thinking`` in extra_body).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def raindragon_profile():
    import model_tools  # triggers plugin discovery
    import providers

    profile = providers.get_provider_profile("raindragon")
    assert profile is not None, "raindragon provider profile must be registered"
    return profile


class TestRainDragonProfile:

    def test_identity_and_endpoint(self, raindragon_profile):
        assert raindragon_profile.name == "raindragon"
        assert raindragon_profile.base_url == "https://inference.raindragon.ai/v1"
        assert raindragon_profile.auth_type == "api_key"
        assert raindragon_profile.env_vars[0] == "RAINDRAGON_API_KEY"
        assert raindragon_profile.fallback_models[0] == "google/gemma-4-26B-A4B-it"

    @pytest.mark.parametrize("alias", ["raindragon-ai"])
    def test_aliases_resolve(self, alias):
        import model_tools
        import providers

        assert providers.get_provider_profile(alias).name == "raindragon"


class TestRainDragonReasoning:

    def test_unset_leaves_server_default(self, raindragon_profile):
        assert raindragon_profile.build_api_kwargs_extras() == ({}, {})
        assert raindragon_profile.build_api_kwargs_extras(reasoning_config={}) == ({}, {})

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
    def test_any_enabled_effort_turns_thinking_on(self, raindragon_profile, effort):
        extra_body, top_level = raindragon_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort})
        assert extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
        assert top_level == {}

    def test_disabled_turns_thinking_off(self, raindragon_profile):
        extra_body, top_level = raindragon_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"})
        assert extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
        assert top_level == {}

    @pytest.mark.parametrize("effort,on", [("none", False), ("minimal", False), ("high", True)])
    def test_effort_without_enabled_flag(self, raindragon_profile, effort, on):
        extra_body, _ = raindragon_profile.build_api_kwargs_extras(reasoning_config={"effort": effort})
        assert extra_body == {"chat_template_kwargs": {"enable_thinking": on}}
