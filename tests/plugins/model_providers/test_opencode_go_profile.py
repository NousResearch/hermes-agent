"""Unit tests for OpenCode Go reasoning-control wiring."""

from __future__ import annotations

import pytest


@pytest.fixture
def opencode_go_profile():
    """Resolve the registered OpenCode Go provider profile."""
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("opencode-go")
    assert profile is not None, "opencode-go provider profile must be registered"
    return profile


class TestOpenCodeGoKimiReasoning:
    """Kimi K2 models use Moonshot's thinking + reasoning_effort shape on OpenCode Go."""

    def test_high_effort_emits_thinking_and_effort(self, opencode_go_profile):
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="kimi-k2.6",
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "high"}

    def test_disabled_emits_thinking_disabled_without_effort(self, opencode_go_profile):
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="kimi-k2.6",
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    def test_minimal_effort_enables_thinking_without_effort(self, opencode_go_profile):
        # "minimal" is not a Moonshot-supported value — drop it, keep thinking on.
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "minimal"},
            model="kimi-k2.6",
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    @pytest.mark.parametrize(
        "effort",
        [
            "xhigh",
            "max",
        ],
    )
    def test_strong_efforts_clamp_to_high(self, opencode_go_profile, effort):
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="moonshotai/kimi-k2.6",
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "high"}

    def test_low_and_medium_pass_through(self, opencode_go_profile):
        for effort in ("low", "medium"):
            extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": effort},
                model="kimi-k2.5",
            )
            assert extra_body == {}
            assert top_level == {"reasoning_effort": effort}

    def test_no_config_preserves_server_default(self, opencode_go_profile):
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config=None,
            model="kimi-k2.6",
        )
        assert extra_body == {}
        assert top_level == {}


class TestOpenCodeGoDeepSeekThinking:
    """DeepSeek V4 models use DeepSeek-style thinking controls on OpenCode Go."""

    def test_high_effort_emits_thinking_and_effort(self, opencode_go_profile):
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="deepseek-v4-pro",
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "high"}

    def test_disabled_emits_thinking_disabled_without_effort(self, opencode_go_profile):
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"},
            model="deepseek-v4-pro",
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    def test_no_config_emits_thinking_enabled_without_effort(self, opencode_go_profile):
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config=None,
            model="deepseek-v4-pro",
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    def test_minimal_effort_enables_thinking_without_effort(self, opencode_go_profile):
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "minimal"},
            model="deepseek-v4-pro",
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    def test_xhigh_and_max_normalize_to_max(self, opencode_go_profile):
        for effort in ("xhigh", "max"):
            extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": effort},
                model="deepseek/deepseek-v4-pro",
            )
            assert extra_body == {}
            assert top_level == {"reasoning_effort": "max"}


class TestOpenCodeGoModelGating:
    """Other OpenCode Go models must not receive Kimi/DeepSeek controls."""

    @pytest.mark.parametrize(
        "model",
        [
            "glm-5.1",
            "qwen3.6-plus",
            "minimax-m2.7",
            "deepseek-v3.1",
            "deepseek-chat",
            "",
            None,
        ],
    )
    def test_non_target_models_emit_nothing(self, opencode_go_profile, model):
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model=model,
        )
        assert extra_body == {}
        assert top_level == {}


class TestOpenCodeGoFullKwargsIntegration:
    """End-to-end transport kwargs include the profile-provided controls."""

    def test_kimi_reasoning_reaches_extra_body_and_top_level(self, opencode_go_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="kimi-k2.6",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=opencode_go_profile,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://opencode.ai/zen/go/v1",
        )
        assert "extra_body" not in kwargs
        assert kwargs["reasoning_effort"] == "high"

    def test_deepseek_thinking_reaches_extra_body_and_top_level(
        self, opencode_go_profile
    ):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=opencode_go_profile,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://opencode.ai/zen/go/v1",
        )
        assert "extra_body" not in kwargs
        assert kwargs["reasoning_effort"] == "high"
