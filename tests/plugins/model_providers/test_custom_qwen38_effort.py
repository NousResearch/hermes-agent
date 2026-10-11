"""Qwen3.8 chat-template effort vocabulary on the custom/llamacpp route (#136324).

The GGUF catalog models (``Qwen3.8-27B-UD-Q4_K_M`` …) embed a chat template that validates
``reasoning_effort`` against exactly low/medium/xhigh; ``high`` and ``max`` raise a Jinja
exception and llama-server answers 500 on every retry, after which Hermes falls back to a
cloud provider. These tests pin the clamp so no picker level can produce a guaranteed-500
request on that family, while every other custom-endpoint model keeps the widest wire set.
"""

from __future__ import annotations

import pytest

from agent.reasoning_effort import (
    OPENAI_COMPAT_WIRE_EFFORTS,
    QWEN38_EFFORTS,
    QWEN38_OVERRIDES,
    clamp_effort,
    is_qwen38_model,
)


@pytest.fixture
def custom_profile():
    import model_tools
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


class TestQwen38SlugDetection:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("Qwen3.8-27B-UD-Q4_K_M", True),
            ("qwen3.8-14b", True),
            ("unsloth/Qwen3.8-27B-UD-Q4_K_M", True),  # vendor/ prefix stripped
            ("Qwen3.8", True),
            ("qwen3.85-27b", False),  # later minor must not match
            ("qwen3-32b", False),
            ("Qwen3.5-32B", False),
            ("", False),
            (None, False),
        ],
    )
    def test_slug_detection(self, model, expected):
        assert is_qwen38_model(model) is expected


class TestQwen38Clamp:
    @pytest.mark.parametrize(
        "requested,expected",
        [
            ("low", "low"),
            ("medium", "medium"),
            ("xhigh", "xhigh"),
            # The template rejects both; its high tier is *named* xhigh.
            ("high", "xhigh"),
            ("max", "xhigh"),
            # Hermes-internal top step clamps down to the template's top tier.
            ("ultra", "xhigh"),
            # Nothing weaker than the template floor exists — policy floor applies.
            ("minimal", "low"),
        ],
    )
    def test_levels(self, requested, expected):
        assert clamp_effort(requested, QWEN38_EFFORTS, QWEN38_OVERRIDES) == expected


class TestCustomProfileQwen38Wire:
    def test_rejected_levels_round_to_the_template_top_tier(self, custom_profile):
        """high/max on a Qwen3.8 slug must never reach llama-server verbatim (Jinja 500)."""
        for effort in ("high", "max", "ultra"):
            _, top_level = custom_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": effort},
                model="Qwen3.8-27B-UD-Q4_K_M",
            )
            assert top_level == {"reasoning_effort": "xhigh"}, effort

    def test_template_levels_pass_verbatim(self, custom_profile):
        for effort in ("low", "medium", "xhigh"):
            _, top_level = custom_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": effort},
                model="Qwen3.8-27B-UD-Q4_K_M",
            )
            assert top_level == {"reasoning_effort": effort}, effort

    def test_other_models_keep_the_widest_wire_set(self, custom_profile):
        """A non-Qwen3.8 model on the same endpoint keeps max verbatim (#114249)."""
        _, top_level = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"},
            model="qwen3-32b",
        )
        assert top_level == {"reasoning_effort": "max"}

    def test_supported_reasoning_efforts_declares_template_set(self, custom_profile):
        assert (
            custom_profile.supported_reasoning_efforts("Qwen3.8-27B-UD-Q4_K_M")
            == QWEN38_EFFORTS
        )
        assert (
            custom_profile.supported_reasoning_efforts("glm-5.2")
            == OPENAI_COMPAT_WIRE_EFFORTS
        )
