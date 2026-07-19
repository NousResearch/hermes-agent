"""Unit tests for the StepFun provider profile's reasoning_effort wiring.

Step-3.7-flash (and future reasoning-capable StepFun models) use top-level
``reasoning_effort`` (low/medium/high) on the Chat Completions path.
The profile must emit this correctly and leave legacy 3.5-flash models
completely untouched.

These tests pin the profile's wire-shape contract so StepFun requests stay
correctly shaped without going live.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def stepfun_profile():
    """Resolve the registered StepFun profile.

    Going through ``providers.get_provider_profile`` keeps the test honest —
    if someone later replaces the registered class with a plain
    ``ProviderProfile``, every assertion below collapses.
    """
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the StepFun profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("stepfun")
    assert profile is not None, "stepfun provider profile must be registered"
    return profile


class TestStepFunReasoningWireShape:
    """``build_api_kwargs_extras`` produces StepFun's exact wire format."""

    def test_37_flash_default_effort_is_medium(self, stepfun_profile):
        """No reasoning_config or no effort → medium (good agent default)."""
        extra_body, top_level = stepfun_profile.build_api_kwargs_extras(
            reasoning_config=None, model="step-3.7-flash"
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "medium"}

    def test_37_flash_explicit_high_effort(self, stepfun_profile):
        extra_body, top_level = stepfun_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="step-3.7-flash",
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "high"}

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_standard_efforts_pass_through(self, stepfun_profile, effort):
        _, top_level = stepfun_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="step-3.7-flash",
        )
        assert top_level == {"reasoning_effort": effort}

    @pytest.mark.parametrize("effort", ["xhigh", "max", "MAX", "  Max  "])
    def test_xhigh_and_max_normalize_to_high(self, stepfun_profile, effort):
        _, top_level = stepfun_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="step-3.7-flash",
        )
        assert top_level == {"reasoning_effort": "high"}

    def test_explicitly_disabled_omits_effort(self, stepfun_profile):
        """``reasoning_config.enabled=False`` → omit reasoning_effort entirely.

        StepFun will then use its server default (low for flash-tier models).
        We deliberately do not send a "disabled" marker.
        """
        extra_body, top_level = stepfun_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="step-3.7-flash"
        )
        assert extra_body == {}
        assert top_level == {}

    def test_disabled_ignores_effort_field(self, stepfun_profile):
        """Effort silently dropped when reasoning is turned off."""
        _, top_level = stepfun_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"},
            model="step-3.7-flash",
        )
        assert top_level == {}

    def test_unknown_effort_falls_back_to_medium(self, stepfun_profile):
        """Garbage effort → still emit medium (safe default)."""
        _, top_level = stepfun_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "garbage"},
            model="step-3.7-flash",
        )
        assert top_level == {"reasoning_effort": "medium"}

    def test_empty_effort_falls_back_to_medium(self, stepfun_profile):
        _, top_level = stepfun_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": ""},
            model="step-3.7-flash",
        )
        assert top_level == {"reasoning_effort": "medium"}


class TestStepFunModelGating:
    """Only 3.7+ / reasoning models get the parameter; legacy models stay clean."""

    @pytest.mark.parametrize(
        "model",
        [
            "step-3.7-flash",
            "step-3.7-flash-something",
            "stepfun/step-3.7-flash",
            "STEP-3.7-FLASH",  # case-insensitive
            "some-reasoning-model",
            "step-3.7-reasoning",
        ],
    )
    def test_reasoning_capable_models_emit_effort(self, stepfun_profile, model):
        extra_body, top_level = stepfun_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model=model
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "high"}

    @pytest.mark.parametrize(
        "model",
        [
            "step-3.5-flash",
            "step-3.5-flash-2603",
            "step-3.5-flash-something",
            "",                       # bare/unknown
            None,                     # missing
            "step-3-unknown",         # unrecognized
            "some-other-model",
            "step-13.7-flash",        # bare-substring over-match guard
        ],
    )
    def test_non_reasoning_models_emit_nothing(self, stepfun_profile, model):
        extra_body, top_level = stepfun_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model=model
        )
        assert extra_body == {}
        assert top_level == {}


class TestStepFunFullKwargsIntegration:
    """End-to-end: the transport produces correct top-level reasoning_effort."""

    def test_full_kwargs_include_reasoning_effort(self, stepfun_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="step-3.7-flash",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=stepfun_profile,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://api.stepfun.ai/step_plan/v1",
            provider_name="stepfun",
        )
        assert kwargs["model"] == "step-3.7-flash"
        assert kwargs["reasoning_effort"] == "high"
        # No extra_body.reasoning pollution for StepFun
        assert "reasoning" not in kwargs.get("extra_body", {})
