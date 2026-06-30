"""Unit tests for the Z.AI / GLM provider profile's thinking-mode wiring.

Z.AI's GLM-4.5-and-later chat models default to thinking-mode ON when the
request omits ``thinking``.  GLM-5.2+ additionally supports an ``effort``
field (``high`` / ``max``) inside the thinking object.

These tests pin the profile's wire-shape contract so Z.AI requests stay
correctly shaped without going live.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def zai_profile():
    """Resolve the registered Z.AI profile through the real discovery path."""
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the Z.AI profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("zai")
    assert profile is not None, "zai provider profile must be registered"
    return profile


class TestZaiThinkingWireShape:
    """``build_api_kwargs_extras`` produces Z.AI's exact wire format."""

    def test_no_preference_omits_thinking(self, zai_profile):
        """No reasoning_config → omit ``thinking`` so the server default
        applies (matches prior behavior for users with no preference)."""
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config=None, model="glm-5"
        )
        assert extra_body == {}
        assert top_level == {}

    def test_enabled_sends_enabled_marker(self, zai_profile):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"}, model="glm-5"
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    def test_explicitly_disabled_sends_disabled_marker(self, zai_profile):
        """``reasoning_config.enabled=False`` → ``thinking.type=disabled``.

        The crucial bit is that the parameter is *sent* at all — GLM defaults
        to thinking-on when ``thinking`` is absent, so an unsent disable
        burns thinking tokens forever.
        """
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5"
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    def test_no_effort_levels_leak_to_top_level(self, zai_profile):
        """GLM has no effort knob — never emit ``reasoning_effort``."""
        for effort in ("minimal", "low", "medium", "high", "xhigh"):
            _, top_level = zai_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": effort}, model="glm-5.2"
            )
            assert top_level == {}


class TestZaiEffortMapping:
    """GLM-5.2 effort field — xhigh→max, high→high, lower→omit."""

    def test_glm52_xhigh_maps_to_max(self, zai_profile):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"},
            model="glm-5.2",
        )
        assert extra_body == {"thinking": {"type": "enabled", "effort": "max"}}
        assert top_level == {}

    def test_glm52_high_maps_to_high(self, zai_profile):
        extra_body, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="glm-5.2",
        )
        assert extra_body == {"thinking": {"type": "enabled", "effort": "high"}}

    @pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "", "garbage"])
    def test_glm52_lower_efforts_omit_effort_field(self, zai_profile, effort):
        """Lower or unknown efforts → omit effort, GLM uses server default."""
        extra_body, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="glm-5.2",
        )
        assert extra_body == {"thinking": {"type": "enabled"}}

    def test_glm52_disabled_ignores_effort(self, zai_profile):
        """Effort silently dropped when thinking is off."""
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "xhigh"},
            model="glm-5.2",
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    @pytest.mark.parametrize("effort", ["xhigh", "max", "XHIGH", "  Max  "])
    def test_xhigh_and_max_normalize_to_max(self, zai_profile, effort):
        """All max-level efforts produce thinking.effort=max."""
        extra_body, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="glm-5.2",
        )
        assert extra_body["thinking"]["effort"] == "max"

    def test_case_insensitive_model_and_effort(self, zai_profile):
        extra_body, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "XHigh"},
            model="GLM-5.2",
        )
        assert extra_body == {"thinking": {"type": "enabled", "effort": "max"}}

    def test_vendor_prefixed_model(self, zai_profile):
        extra_body, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"},
            model="zai/glm-5.2",
        )
        assert extra_body == {"thinking": {"type": "enabled", "effort": "max"}}


class TestZaiModelGating:
    """GLM 4.5+ get thinking; GLM-5.2+ additionally gets effort."""

    @pytest.mark.parametrize(
        "model",
        [
            "glm-4.5",
            "glm-4.5-air",
            "glm-4.5-flash",
            "glm-4.6",
            "glm-5",
            "glm-5.2",
            "GLM-5",  # case-insensitive
        ],
    )
    def test_thinking_capable_models_emit_thinking(self, zai_profile, model):
        extra_body, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model=model
        )
        assert extra_body == {"thinking": {"type": "disabled"}}

    @pytest.mark.parametrize(
        "model",
        [
            "glm-4-9b",   # pre-4.5, no thinking param
            "glm-4",
            "glm-3-turbo",
            "",            # bare/unknown
            None,          # missing
            "charglm-3",  # non-GLM-versioned id
        ],
    )
    def test_non_thinking_models_emit_nothing(self, zai_profile, model):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model=model
        )
        assert extra_body == {}
        assert top_level == {}

    @pytest.mark.parametrize("model", ["glm-5.2", "GLM-5.2", "zai/glm-5.2", "zhipu/glm-5.2"])
    def test_effort_capable_models_get_effort(self, zai_profile, model):
        extra_body, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"}, model=model
        )
        assert extra_body["thinking"]["effort"] == "max"

    @pytest.mark.parametrize(
        "model",
        ["glm-5.1", "glm-5", "glm-4.6", "glm-4.5"],
    )
    def test_older_models_omit_effort(self, zai_profile, model):
        """Older GLM models get thinking.type but no effort field."""
        extra_body, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"}, model=model
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert "effort" not in extra_body.get("thinking", {})

    @pytest.mark.parametrize("model", ["glm-4-9b", "unknown-model", None, ""])
    def test_non_thinking_models_return_empty(self, zai_profile, model):
        """Pre-4.5 or unknown models → no thinking at all."""
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"}, model=model
        )
        assert extra_body == {}
        assert top_level == {}

    def test_glm_52_variant_matches(self, zai_profile):
        """GLM-5.2-preview, glm-5.2-turbo etc. should also match."""
        extra_body, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"},
            model="glm-5.2-preview",
        )
        assert extra_body["thinking"]["effort"] == "max"

    def test_glm_520_does_not_match(self, zai_profile):
        """Hypothetical glm-5.20 must NOT match glm-5.2."""
        extra_body, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"},
            model="glm-5.20",
        )
        assert "effort" not in extra_body.get("thinking", {})


class TestZaiBackwardCompat:
    """Ensure the profile doesn't break existing GLM users."""

    def test_none_reasoning_config_preserves_wire_format(self, zai_profile):
        """The critical backward-compat test: no reasoning_config → no-op."""
        for model in ["glm-5.1", "glm-5", "glm-4.6", "glm-5.2"]:
            extra_body, top_level = zai_profile.build_api_kwargs_extras(
                reasoning_config=None, model=model
            )
            assert extra_body == {}, f"model={model} should get empty extra_body"
            assert top_level == {}, f"model={model} should get empty top_level"


class TestZaiFullKwargsIntegration:
    """End-to-end: the transport's full kwargs carry the thinking marker."""

    def test_disabled_reaches_the_wire(self, zai_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="glm-5",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=zai_profile,
            reasoning_config={"enabled": False},
            base_url="https://api.z.ai/api/paas/v4",
            provider_name="zai",
        )
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    def test_no_preference_keeps_wire_clean(self, zai_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="glm-5",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=zai_profile,
            reasoning_config=None,
            base_url="https://api.z.ai/api/paas/v4",
            provider_name="zai",
        )
        assert "thinking" not in kwargs.get("extra_body", {})

    def test_glm52_xhigh_produces_correct_wire_shape(self, zai_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="glm-5.2",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=zai_profile,
            reasoning_config={"enabled": True, "effort": "xhigh"},
            base_url="https://api.z.ai/api/paas/v4",
            provider_name="zai",
        )
        assert kwargs["model"] == "glm-5.2"
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled", "effort": "max"}

    def test_glm52_disabled_thinking(self, zai_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="glm-5.2",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=zai_profile,
            reasoning_config={"enabled": False},
            base_url="https://api.z.ai/api/paas/v4",
            provider_name="zai",
        )
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
