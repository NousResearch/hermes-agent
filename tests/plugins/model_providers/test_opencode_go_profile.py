"""Unit tests for OpenCode Go reasoning-control wiring."""

from __future__ import annotations

from types import SimpleNamespace

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


class TestOpenCodeGoSessionAffinityHeader:
    """OpenCode Go's chat_completions relay 400s some backends (e.g.
    deepseek-v4-flash) without a stable per-conversation session-affinity
    header (#81584)."""

    def test_session_id_sets_affinity_header(self, opencode_go_profile):
        _, top_level = opencode_go_profile.build_api_kwargs_extras(
            model="deepseek-v4-flash",
            session_id="sess-abc123",
        )
        assert top_level["extra_headers"]["x-opencode-session"] == "sess-abc123"

    def test_header_normalizes_cron_timestamp(self, opencode_go_profile):
        _, first = opencode_go_profile.build_api_kwargs_extras(
            model="deepseek-v4-flash", session_id="cron_job42_20260801_090000",
        )
        _, second = opencode_go_profile.build_api_kwargs_extras(
            model="deepseek-v4-flash", session_id="cron_job42_20260802_090000",
        )
        assert first["extra_headers"]["x-opencode-session"] == "cron_job42"
        assert (
            first["extra_headers"]["x-opencode-session"]
            == second["extra_headers"]["x-opencode-session"]
        )

    def test_no_session_id_omits_header(self, opencode_go_profile):
        _, top_level = opencode_go_profile.build_api_kwargs_extras(
            model="deepseek-v4-flash",
        )
        assert "extra_headers" not in top_level

    def test_header_coexists_with_reasoning_top_level_kwargs(self, opencode_go_profile):
        _, top_level = opencode_go_profile.build_api_kwargs_extras(
            model="kimi-k2.6",
            reasoning_config={"enabled": True, "effort": "high"},
            session_id="sess-xyz",
        )
        assert top_level["reasoning_effort"] == "high"
        assert top_level["extra_headers"]["x-opencode-session"] == "sess-xyz"


class TestOpenCodeGoSessionHeaderOtherTransports:
    """MiniMax/Qwen on OpenCode Go route through anthropic_messages, and (for
    parity) codex_responses is covered too — build_api_kwargs_extras is only
    consulted by the chat_completions transport, so build_api_kwargs must
    merge the header itself on those other routes (#81584)."""

    def _build_anthropic_kwargs(self, session_id="sess-abc123"):
        from agent.chat_completion_helpers import build_api_kwargs
        from agent.transports.anthropic import AnthropicTransport

        transport = AnthropicTransport()
        agent = SimpleNamespace(
            api_mode="anthropic_messages",
            provider="opencode-go",
            model="minimax-m2.7",
            session_id=session_id,
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            request_overrides={},
            context_compressor=None,
            _ephemeral_max_output_tokens=None,
            _is_anthropic_oauth=False,
            _anthropic_base_url="https://opencode.ai/zen/go/v1",
            _oauth_1m_beta_disabled=False,
            _get_transport=lambda: transport,
            _prepare_anthropic_messages_for_api=lambda msgs: msgs,
            _anthropic_preserve_dots=lambda: False,
        )
        return build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    def _build_codex_kwargs(self, session_id="sess-abc123"):
        from agent.chat_completion_helpers import build_api_kwargs
        from agent.transports.codex import ResponsesApiTransport

        transport = ResponsesApiTransport()
        agent = SimpleNamespace(
            api_mode="codex_responses",
            provider="opencode-go",
            model="glm-5",
            session_id=session_id,
            tools=None,
            base_url="https://opencode.ai/zen/go/v1",
            _base_url_hostname="opencode.ai",
            _base_url_lower="https://opencode.ai/zen/go/v1",
            max_tokens=1024,
            reasoning_config=None,
            request_overrides={},
            _get_transport=lambda: transport,
            _prepare_messages_for_non_vision_model=lambda msgs: msgs,
            _resolved_api_call_timeout=lambda: 30.0,
            _github_models_reasoning_extra_body=lambda: None,
            _codex_reasoning_replay_enabled=True,
        )
        return build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    def test_anthropic_messages_route_gets_the_affinity_header(self):
        kwargs = self._build_anthropic_kwargs(session_id="sess-abc123")
        assert kwargs["extra_headers"]["x-opencode-session"] == "sess-abc123"

    def test_codex_responses_route_gets_the_affinity_header(self):
        kwargs = self._build_codex_kwargs(session_id="sess-abc123")
        assert kwargs["extra_headers"]["x-opencode-session"] == "sess-abc123"

    def test_non_opencode_go_provider_is_untouched_on_anthropic_messages(self):
        from agent.chat_completion_helpers import _merge_opencode_go_session_header

        kwargs = {"model": "claude-opus-4-8"}
        agent = SimpleNamespace(provider="nous", model="claude-opus-4-8", session_id="s")
        assert _merge_opencode_go_session_header(agent, kwargs) is kwargs
        assert "extra_headers" not in kwargs


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
