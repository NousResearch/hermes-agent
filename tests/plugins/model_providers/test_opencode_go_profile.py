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


@pytest.fixture
def opencode_zen_profile():
    """Resolve the registered OpenCode Zen provider profile."""
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("opencode-zen")
    assert profile is not None, "opencode-zen provider profile must be registered"
    return profile


class TestOpenCodeZenOxReasoning:
    """Ox Alpha Free uses OpenCode Zen's native reasoning_effort control."""

    def test_max_effort_is_emitted(self, opencode_zen_profile):
        extra_body, top_level = opencode_zen_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"},
            model="x-preview-f-free",
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "max"}

    @pytest.mark.parametrize("reasoning_config", [None, {"enabled": False}])
    def test_unset_or_disabled_preserves_server_default(
        self, opencode_zen_profile, reasoning_config
    ):
        extra_body, top_level = opencode_zen_profile.build_api_kwargs_extras(
            reasoning_config=reasoning_config,
            model="x-preview-f-free",
        )
        assert extra_body == {}
        assert top_level == {}

    def test_other_zen_models_are_untouched(self, opencode_zen_profile):
        extra_body, top_level = opencode_zen_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"},
            model="gemini-3-flash",
        )
        assert extra_body == {}
        assert top_level == {}

    def test_max_reaches_chat_completions_request(self, opencode_zen_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="x-preview-f-free",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=opencode_zen_profile,
            reasoning_config={"enabled": True, "effort": "max"},
            base_url="https://opencode.ai/zen/v1",
        )
        assert "extra_body" not in kwargs
        assert kwargs["reasoning_effort"] == "max"

    def test_unsupported_efforts_clamp_to_wire_vocabulary(self, opencode_zen_profile):
        """medium/xhigh are not on Ox Alpha's wire (400 raw); they must clamp
        to the nearest supported level, never pass through."""
        for requested, expected in (("medium", "low"), ("xhigh", "max")):
            _, top_level = opencode_zen_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": requested},
                model="x-preview-f-free",
            )
            assert top_level == {"reasoning_effort": expected}, requested

    def test_opencode_free_profile_shares_the_translation(self):
        """Ox Alpha is reachable via the keyless opencode-free provider too;
        its profile must emit the identical clamped reasoning_effort."""
        import model_tools  # noqa: F401
        import providers
        from providers.base import ProviderProfile

        profile = providers.get_provider_profile("opencode-free")
        assert profile is not None
        assert (
            type(profile).build_api_kwargs_extras
            is not ProviderProfile.build_api_kwargs_extras
        ), "opencode-free must override build_api_kwargs_extras (aux gate)"
        _, top_level = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model="x-preview-f-free",
        )
        assert top_level == {"reasoning_effort": "low"}
        _, other = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"},
            model="big-pickle",
        )
        assert other == {}


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

    def test_minimal_effort_clamps_to_low(self, opencode_go_profile):
        # "minimal" is below Moonshot's floor — the shared clamp degrades it
        # to "low" (nearest supported) instead of dropping the ask and
        # leaving the server default (which was MORE thinking than asked).
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "minimal"},
            model="kimi-k2.6",
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "low"}

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


    def test_xhigh_and_max_normalize_to_max(self, opencode_go_profile):
        for effort in ("xhigh", "max"):
            extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": effort},
                model="deepseek/deepseek-v4-pro",
            )
            assert extra_body == {}
            assert top_level == {"reasoning_effort": "max"}


class TestOpenCodeGoGLM52Reasoning:
    """GLM-5.2 uses its native high/max reasoning_effort knob on OpenCode Go."""


    @pytest.mark.parametrize("model", ["glm-5-2", "glm-5p2"])
    def test_alias_spellings_recognized(self, opencode_go_profile, model):
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"},
            model=model,
        )
        assert top_level == {"reasoning_effort": "max"}


class TestOpenCodeGoModelGating:
    """Other OpenCode Go models must not receive Kimi/DeepSeek/GLM controls."""

    @pytest.mark.parametrize(
        "model",
        [
            "glm-5.1",
            "glm-5",
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
