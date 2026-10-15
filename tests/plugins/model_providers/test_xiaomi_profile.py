"""Unit tests for the Xiaomi MiMo provider profile's thinking-toggle wiring.

MiMo thinking-capable chat models take a binary ``extra_body.thinking``
toggle on Chat Completions; Xiaomi documents no graded ``reasoning_effort``
on that wire, and the server default when the field is omitted is not
documented as stable, so the toggle is sent explicitly on every
thinking-capable request.  asr/tts ids never think and must stay untouched.

These tests pin the profile's wire-shape contract so MiMo requests stay
correctly shaped without going live.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def xiaomi_profile():
    """Resolve the registered Xiaomi profile.

    Going through ``providers.get_provider_profile`` keeps the test honest -
    if someone later replaces the registered class with a plain
    ``ProviderProfile``, every assertion below collapses.
    """
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the Xiaomi profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("xiaomi")
    assert profile is not None, "xiaomi provider profile must be registered"
    return profile


class TestXiaomiThinkingWireShape:
    """``build_api_kwargs_extras`` produces MiMo's exact wire format."""

    def test_no_config_defaults_to_enabled(self, xiaomi_profile):
        """No reasoning_config -> toggle sent enabled (server default not documented)."""
        extra_body, top_level = xiaomi_profile.build_api_kwargs_extras(
            reasoning_config=None, model="mimo-v2.6-pro"
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max", "garbage"])
    def test_any_effort_stays_binary_enabled(self, xiaomi_profile, effort):
        """MiMo's wire has only the toggle; a graded effort never becomes reasoning_effort."""
        extra_body, top_level = xiaomi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="mimo-v2.6-pro"
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    def test_explicitly_disabled_sends_disabled_marker(self, xiaomi_profile):
        """``reasoning_config.enabled=False`` -> ``thinking.type=disabled``.

        The crucial bit is that the parameter is *sent* at all - the MiMo
        server default when ``thinking`` is absent is not documented as stable.
        """
        extra_body, top_level = xiaomi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="mimo-v2.6-pro"
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    def test_disabled_ignores_effort_field(self, xiaomi_profile):
        """Effort silently dropped when thinking is off."""
        extra_body, top_level = xiaomi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"}, model="mimo-v2.6-pro"
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    @pytest.mark.parametrize(
        "reasoning_config",
        [
            {"effort": "none"},
            {"effort": "false"},
            {"effort": "disabled"},
            {"effort": "NONE"},
            {"enabled": True, "effort": "none"},
        ],
    )
    def test_effort_none_without_enabled_false_sends_disabled(
        self, xiaomi_profile, reasoning_config
    ):
        """UI/CLI ``none`` can arrive as ``{effort: "none"}`` without ``enabled: False``.

        Session surfaces already report that as Off, but a bare toggle profile
        would fall through to ``thinking: enabled`` (#107238, DeepSeek parity).
        """
        extra_body, top_level = xiaomi_profile.build_api_kwargs_extras(
            reasoning_config=reasoning_config, model="mimo-v2.6-pro"
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}
        assert "reasoning_effort" not in top_level


class TestXiaomiModelGating:
    """Thinking-capable ids get the toggle; asr/tts and unknown stay untouched."""

    @pytest.mark.parametrize(
        "model",
        [
            "mimo-v2.6-pro",
            "mimo-v2.6-flash",
            "mimo-v2.6-pro-ultraspeed",
            "mimo-v2.5-pro",
            "mimo-v2-pro",
            "mimo-v2-flash",
            "MIMO-V2.6-PRO",  # case-insensitive
        ],
    )
    def test_thinking_capable_models_emit_toggle(self, xiaomi_profile, model):
        extra_body, _ = xiaomi_profile.build_api_kwargs_extras(
            reasoning_config=None, model=model
        )
        assert extra_body == {"thinking": {"type": "enabled"}}

    @pytest.mark.parametrize(
        "model",
        [
            "mimo-v2.5-asr",
            "mimo-v2.5-tts",
            "mimo-v2.5-tts-voiceclone",
            "mimo-v2.5-tts-voicedesign",
            "",
            None,
            "mimo-unknown",
        ],
    )
    def test_non_thinking_models_emit_nothing(self, xiaomi_profile, model):
        extra_body, top_level = xiaomi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model=model
        )
        assert extra_body == {}
        assert top_level == {}


class TestXiaomiFullKwargsIntegration:
    """End-to-end: the transport's built kwargs carry the toggle on the MiMo wire."""

    def test_full_kwargs_effort_high_enables_thinking(self, xiaomi_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="mimo-v2.6-pro",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=xiaomi_profile,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://api.xiaomimimo.com/v1",
            provider_name="xiaomi",
        )
        assert kwargs["model"] == "mimo-v2.6-pro"
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in kwargs  # binary wire: no graded effort

    def test_full_kwargs_disabled_disables_thinking(self, xiaomi_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="mimo-v2.6-pro",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=xiaomi_profile,
            reasoning_config={"enabled": False},
            base_url="https://api.xiaomimimo.com/v1",
            provider_name="xiaomi",
        )
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in kwargs

    def test_full_kwargs_none_effort_disables_thinking(self, xiaomi_profile):
        """CLI/UI ``none`` as ``{effort: "none"}`` (no enabled:False) must disable."""
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="mimo-v2.6-pro",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=xiaomi_profile,
            reasoning_config={"effort": "none"},
            base_url="https://api.xiaomimimo.com/v1",
            provider_name="xiaomi",
        )
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in kwargs

    def test_full_kwargs_tts_model_untouched(self, xiaomi_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="mimo-v2.5-tts",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=xiaomi_profile,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://api.xiaomimimo.com/v1",
            provider_name="xiaomi",
        )
        assert "thinking" not in kwargs.get("extra_body", {})
        assert "reasoning_effort" not in kwargs
