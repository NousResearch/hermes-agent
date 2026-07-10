"""Unit tests for the Kimi/Moonshot provider profile's reasoning wiring.

Moonshot's OpenAI-compat endpoint (``api.moonshot.ai/v1``) treats
``extra_body.thinking`` and a top-level ``reasoning_effort`` as mutually
exclusive. The profile must send at most one of them — never both — so a
request can't trip "cannot specify both 'thinking' and 'reasoning_effort'".

This mirrors the kimi-k2 handling already shipped for the opencode-go relay
(see ``tests/plugins/model_providers/test_opencode_go_profile.py``).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def kimi_profile():
    """Resolve the registered Kimi profile via the provider registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    Kimi profile. Going through ``get_provider_profile`` keeps the test honest:
    if the registered class is ever swapped for a plain ``ProviderProfile`` the
    assertions below collapse.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("kimi-coding")
    assert profile is not None, "kimi-coding provider profile must be registered"
    return profile


class TestKimiReasoningWireShape:
    """``build_api_kwargs_extras`` never emits thinking + reasoning_effort together."""

    def test_no_config_enables_thinking_without_effort(self, kimi_profile):
        """No reasoning_config → thinking on, server picks the depth.

        Regression guard: this path previously also sent
        ``reasoning_effort="medium"``, pairing thinking + effort on every
        default call.
        """
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(reasoning_config=None)
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_explicit_effort_sends_effort_only(self, kimi_profile, effort):
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )
        assert top_level == {"reasoning_effort": effort}
        assert "thinking" not in extra_body

    def test_enabled_without_effort_falls_back_to_thinking(self, kimi_profile):
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True}
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    @pytest.mark.parametrize("effort", ["", "garbage"])
    def test_unrecognized_effort_falls_back_to_thinking(self, kimi_profile, effort):
        """Unknown efforts use the binary thinking toggle."""
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    @pytest.mark.parametrize("effort", ["xhigh", "max", "ultra"])
    def test_strong_effort_projects_to_high(self, kimi_profile, effort):
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )

        assert extra_body == {}
        assert top_level == {"reasoning_effort": "high"}

    def test_disabled_sends_thinking_disabled_only(self, kimi_profile):
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    def test_disabled_ignores_effort(self, kimi_profile):
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"}
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    @pytest.mark.parametrize(
        "reasoning_config",
        [
            None,
            {"enabled": True},
            {"enabled": True, "effort": "high"},
            {"enabled": True, "effort": "garbage"},
            {"enabled": False},
            {"enabled": False, "effort": "low"},
        ],
    )
    def test_never_emits_both(self, kimi_profile, reasoning_config):
        """The core invariant: thinking and reasoning_effort are never both set."""
        extra_body, top_level = kimi_profile.build_api_kwargs_extras(
            reasoning_config=reasoning_config
        )
        assert not ("thinking" in extra_body and "reasoning_effort" in top_level)


class TestKimiFullKwargsIntegration:
    """The transport's full kwargs carry at most one reasoning knob."""

    def _build(self, kimi_profile, reasoning_config):
        from agent.transports.chat_completions import ChatCompletionsTransport

        return ChatCompletionsTransport().build_kwargs(
            model="kimi-k2-turbo-preview",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=kimi_profile,
            reasoning_config=reasoning_config,
            base_url="https://api.moonshot.ai/v1",
            provider_name="kimi-coding",
        )

    def test_explicit_effort_omits_thinking(self, kimi_profile):
        kwargs = self._build(kimi_profile, {"enabled": True, "effort": "high"})
        assert kwargs["reasoning_effort"] == "high"
        assert "thinking" not in kwargs.get("extra_body", {})

    def test_high_end_effort_projects_without_thinking(self, kimi_profile):
        kwargs = self._build(kimi_profile, {"enabled": True, "effort": "ultra"})

        assert kwargs["reasoning_effort"] == "high"
        assert "thinking" not in kwargs.get("extra_body", {})

    def test_no_config_omits_effort(self, kimi_profile):
        kwargs = self._build(kimi_profile, None)
        assert "reasoning_effort" not in kwargs
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
