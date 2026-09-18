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

    @pytest.mark.parametrize("model", ["deepseek-flash", "deepseek/deepseek-flash"])
    def test_version_less_canonical_id_gets_the_controls(self, opencode_go_profile, model):
        """The canonical version-less Flash id must reach the wire like the versioned ids:
        the asked effort, and an explicit thinking-off when reasoning is disabled."""
        _, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model=model,
        )
        assert top_level == {"reasoning_effort": "high"}
        extra_body, top_level = opencode_go_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model=model,
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    def test_version_less_flash_effort_reaches_the_wire(self, opencode_go_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=opencode_go_profile,
            reasoning_config={"enabled": True, "effort": "max"},
            base_url="https://opencode.ai/zen/go/v1",
        )
        assert "extra_body" not in kwargs
        assert kwargs["reasoning_effort"] == "max"


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


# Chat Completions' optional top-level ``name`` on a tool result is no longer accepted by the
# Console Go upstream: every tool-using conversation 400s with
# ``messages[N]: "name" is not supported by this endpoint``. The tool_call_id already carries
# the association, so the profile drops the field copy-on-write.
_TOOL_TURN = [
    {"role": "system", "content": "You are a bot."},
    {"role": "user", "content": "hi"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "ok"},
]


def _wire_tool_messages(messages):
    return [m for m in messages if m.get("role") == "tool"]


class TestOpenCodeGoToolResultNameStripping:
    """Console Go rejects the top-level ``name`` on tool results (issue repro payload)."""

    def test_prepare_messages_strips_tool_result_name(self, opencode_go_profile):
        prepared = opencode_go_profile.prepare_messages(_TOOL_TURN)

        assert prepared[3] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "ok",
        }
        # Copy-on-write: the caller's history still holds the name for the transports
        # that do consume it (gemini native, kanban stop), and untouched rows are shared.
        assert _TOOL_TURN[3]["name"] == "terminal"
        assert prepared[2] is _TOOL_TURN[2]

    def test_prepare_messages_passthrough_without_tool_result_names(
        self, opencode_go_profile
    ):
        msgs = [
            {"role": "user", "content": "ping"},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        assert opencode_go_profile.prepare_messages(msgs) is msgs

    def test_assistant_and_user_names_are_not_touched(self, opencode_go_profile):
        """Only the tool role is rewritten; a look-alike key elsewhere is none of our business."""
        msgs = [
            {"role": "system", "content": "You are a bot.", "name": "bot"},
            {"role": "assistant", "content": "hi"},
        ]
        assert opencode_go_profile.prepare_messages(msgs) is msgs

    def test_stub_and_compression_tool_results_lose_the_name_too(
        self, opencode_go_profile
    ):
        """Unanswered-call stubs are assembled with a name as well; same 400 without this."""
        msgs = [
            {
                "role": "tool",
                "name": "terminal",
                "tool_call_id": "call_9",
                "content": "[Result unavailable — see context summary above]",
            }
        ]
        out = opencode_go_profile.prepare_messages(msgs)
        assert "name" not in out[0]
        assert out[0]["tool_call_id"] == "call_9"

    def test_transport_kwargs_carry_no_tool_message_name(self, opencode_go_profile):
        """End-to-end: what ``build_kwargs`` hands the OpenAI client is name-free."""
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="omen-alpha",
            messages=[dict(m) if m.get("role") == "tool" else m for m in _TOOL_TURN],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "terminal", "parameters": {"type": "object"}},
                }
            ],
            provider_profile=opencode_go_profile,
            base_url="https://opencode.ai/zen/go/v1",
        )

        tool_msgs = _wire_tool_messages(kwargs["messages"])
        assert tool_msgs, "the tool result must survive message assembly"
        assert "name" not in tool_msgs[0], tool_msgs[0]
        # The association the endpoint does accept is preserved.
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert tool_msgs[0]["content"] == "ok"

    def test_other_provider_is_untouched(self, opencode_go_profile):
        """Protection: the guard is opencode-go-scoped — a sibling profile keeps the
        identity contract, and both providers emit the identical wire row."""
        import model_tools  # noqa: F401
        import providers
        from agent.transports.chat_completions import ChatCompletionsTransport

        zen = providers.get_provider_profile("opencode-zen")
        assert zen is not None
        assert zen.prepare_messages(_TOOL_TURN) is _TOOL_TURN
        assert _TOOL_TURN[3]["name"] == "terminal"

        transport = ChatCompletionsTransport()

        def wire(profile, base_url):
            return transport.build_kwargs(
                model="omen-alpha",
                messages=[dict(m) if m.get("role") == "tool" else m for m in _TOOL_TURN],
                tools=None,
                provider_profile=profile,
                base_url=base_url,
            )["messages"]

        go_rows = wire(opencode_go_profile, "https://opencode.ai/zen/go/v1")
        zen_rows = wire(zen, "https://opencode.ai/zen/v1")
        assert go_rows == zen_rows
        assert "name" not in zen_rows[3]

    def test_user_and_assistant_names_survive_the_wire(self, opencode_go_profile):
        """Protection: ``name`` is schema-valid off the tool role — only tool rows lose it."""
        from agent.transports.chat_completions import ChatCompletionsTransport

        msgs = [
            {"role": "system", "content": "You are a bot.", "name": "hermes"},
            {"role": "user", "content": "hi", "name": "sylvain"},
            _TOOL_TURN[2],
            dict(_TOOL_TURN[3]),
        ]
        kwargs = ChatCompletionsTransport().build_kwargs(
            model="omen-alpha",
            messages=msgs,
            tools=None,
            provider_profile=opencode_go_profile,
            base_url="https://opencode.ai/zen/go/v1",
        )
        wire = kwargs["messages"]
        assert [m.get("name") for m in wire[:2]] == ["hermes", "sylvain"]
        assert "name" not in wire[3]

    def test_nvidia_profile_still_strips_its_own_fields(self):
        """Protection: the sibling copy-on-write profile keeps stripping name + tool_name."""
        import model_tools  # noqa: F401
        import providers

        nvidia = providers.get_provider_profile("nvidia")
        msgs = [dict(_TOOL_TURN[3], tool_name="terminal")]
        out = nvidia.prepare_messages(msgs)
        assert out[0] == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}

    def test_plain_conversation_is_untouched(self, opencode_go_profile):
        """Protection: the tool-free path (which never 400'd) stays byte-identical."""
        from agent.transports.chat_completions import ChatCompletionsTransport

        msgs = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "hi"},
        ]
        kwargs = ChatCompletionsTransport().build_kwargs(
            model="omen-alpha",
            messages=msgs,
            tools=None,
            provider_profile=opencode_go_profile,
            base_url="https://opencode.ai/zen/go/v1",
        )
        assert kwargs["messages"] == msgs
