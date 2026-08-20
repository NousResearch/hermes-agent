"""Unit tests for the Z.AI / GLM provider profile's reasoning wiring.

Z.AI's GLM-4.5-and-later chat models default to thinking-mode ON when the
request omits ``thinking``.  Before the profile emitted the parameter,
``reasoning_config = {"enabled": False}`` was a silent no-op on the direct
Z.AI route — users who turned thinking off kept burning thinking tokens on
every turn (the desktop "thinking reverts to medium" report).

GLM-5.2 additionally exposes a native ``reasoning_effort`` knob with two
enabled levels (high / max) on the OpenAI-compatible ``/api/paas/v4``
endpoint; the Hermes effort scale is collapsed onto those.

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


class TestZaiGLM52ReasoningEffort:
    """GLM-5.2's native ``reasoning_effort`` knob (two enabled levels)."""

    def test_high_maps_to_high(self, zai_profile):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="glm-5.2",
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {"reasoning_effort": "high"}

    @pytest.mark.parametrize("effort", ["low", "medium", "minimal"])
    def test_lower_efforts_clamp_up_to_high(self, zai_profile, effort):
        """GLM-5.2's minimum thinking level is high — lower Hermes levels
        clamp onto it."""
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="glm-5.2",
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {"reasoning_effort": "high"}

    @pytest.mark.parametrize("effort", ["xhigh", "max"])
    def test_strong_efforts_map_to_max(self, zai_profile, effort):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="glm-5.2",
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {"reasoning_effort": "max"}

class TestZaiGlm53:
    """GLM-5.3: effort control via reasoning_effort, no thinking param.

    Same 743B base as GLM-5.2 (post-training only), 1M context window.
    z.ai silently ignores ``thinking.disabled`` for 5.3 — thinking still
    runs — so the no-op marker must not be emitted; ``reasoning_effort``
    is 5.3's actual effort dial.
    """

    @pytest.mark.parametrize(
        "model",
        ["glm-5.3", "glm-5-3", "glm-5p3", "z-ai/glm-5.3"],
    )
    def test_alias_spellings_recognized(self, zai_profile, model):
        _, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model=model,
        )
        assert top_level == {"reasoning_effort": "high"}

    def test_disabled_reasoning_omits_noop_thinking_marker(self, zai_profile):
        """5.3 ignores thinking.disabled — don't send it.
        (No clear_thinking either: z.ai's OpenAI-compat wire drops replayed
        reasoning_content, so preserved mode is not emitted at all.)"""
        extra_body, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"},
            model="glm-5.3",
        )
        assert extra_body == {}

    def test_disabled_on_5_3_logs_noop(self, zai_profile, caplog):
        """The silent no-op is surfaced at debug level so a config that
        appears to do nothing isn't mistaken for a wiring bug
        (review point on PR #86433)."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="plugins.model_providers.zai"):
            zai_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": False},
                model="glm-5.3",
            )
        assert any(
            "ignores thinking.disabled" in r.message for r in caplog.records
        ), caplog.records

    def test_no_spurious_log_when_thinking_emitted(self, zai_profile, caplog):
        """Control: the debug line fires only on the 5.3 no-op path —
        normal emits (disabled on a model that honors it) stay quiet."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="plugins.model_providers.zai"):
            extra_body, _ = zai_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": False},
                model="glm-4.5",
            )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert not any(
            "ignores thinking.disabled" in r.message for r in caplog.records
        ), caplog.records

    def test_context_length_registered(self):
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS

        assert DEFAULT_CONTEXT_LENGTHS["glm-5.3"] == 1_048_576


class TestZaiAliasBoundaries:
    """Version-boundary matching: plausible FUTURE ids must not classify as
    known variants (a bare substring check would match glm-5.30 as glm-5.3)."""

    @pytest.mark.parametrize(
        "model",
        [
            "glm-5.30",       # future minor
            "glm-5.3x",       # suffixed variant
            "notglm-5.3",     # embedded in another word
            "glm-5.35",
            "glm-5p30",
        ],
    )
    def test_future_and_embedded_ids_not_matched(self, zai_profile, model):
        _, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model=model,
        )
        assert top_level == {}  # no reasoning_effort for unknown variants

    @pytest.mark.parametrize(
        "model",
        [
            "glm-5.2",
            "glm-5-2",
            "glm-5p2",
            "z-ai/glm-5.2",
            "accounts/fireworks/models/glm-5p2",
            "zai-org-glm-5-2",
            "glm-5.3",
            "glm-5-3",
            "glm-5p3",
            "GLM-5.2",  # case-insensitive
        ],
    )
    def test_known_spellings_still_matched(self, zai_profile, model):
        _, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model=model,
        )
        assert top_level == {"reasoning_effort": "high"}



class TestZaiModelGating:
    """GLM 4.5+ get thinking; earlier GLM models are left untouched."""

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


class TestZaiFullKwargsIntegration:
    """End-to-end: the transport's full kwargs carry the reasoning wiring."""


    def test_glm_5_2_effort_reaches_top_level(self, zai_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="glm-5.2",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=zai_profile,
            reasoning_config={"enabled": True, "effort": "max"},
            base_url="https://api.z.ai/api/paas/v4",
            provider_name="zai",
        )
        assert kwargs["reasoning_effort"] == "max"
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}
