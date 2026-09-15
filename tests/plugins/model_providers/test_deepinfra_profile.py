"""Regression coverage for DeepInfra's top-level reasoning_effort wire field."""

from __future__ import annotations

import pytest


@pytest.fixture
def deepinfra_profile():
    """Resolve the registered profile through the normal plugin discovery path."""
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("deepinfra")
    assert profile is not None, "deepinfra provider profile must be registered"
    return profile


class TestDeepInfraReasoningEffort:
    def test_default_preserves_the_model_default(self, deepinfra_profile):
        assert deepinfra_profile.build_api_kwargs_extras(reasoning_config=None) == (
            {},
            {},
        )

    @pytest.mark.parametrize(
        "effort", ("minimal", "low", "medium", "high", "xhigh", "max")
    )
    def test_explicit_efforts_are_sent_verbatim(self, deepinfra_profile, effort):
        assert deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        ) == ({}, {"reasoning_effort": effort})

    def test_ultra_clamps_to_deepinfra_maximum(self, deepinfra_profile):
        assert deepinfra_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "ultra"}
        ) == ({}, {"reasoning_effort": "max"})

    @pytest.mark.parametrize(
        "reasoning_config", ({"enabled": False}, {"enabled": False, "effort": "high"})
    )
    def test_disabled_sends_the_explicit_off_value(
        self, deepinfra_profile, reasoning_config
    ):
        assert deepinfra_profile.build_api_kwargs_extras(
            reasoning_config=reasoning_config
        ) == ({}, {"reasoning_effort": "none"})

    @pytest.mark.parametrize(
        "reasoning_config",
        ({}, {"enabled": True}, {"enabled": True, "effort": "future-tier"}),
    )
    def test_missing_or_unknown_effort_preserves_the_model_default(
        self, deepinfra_profile, reasoning_config
    ):
        assert deepinfra_profile.build_api_kwargs_extras(
            reasoning_config=reasoning_config
        ) == ({}, {})

    def test_transport_includes_top_level_reasoning_effort_without_capability_gate(
        self, deepinfra_profile
    ):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="deepseek-ai/DeepSeek-V4.1-Flash",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=deepinfra_profile,
            provider_name="deepinfra",
            reasoning_config={"enabled": True, "effort": "high"},
            supports_reasoning=False,
        )
        assert kwargs["reasoning_effort"] == "high"
