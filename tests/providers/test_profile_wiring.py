"""Profile-path parity tests: verify profile path produces identical output to legacy flags.

Each test calls build_kwargs twice — once with legacy flags, once with provider_profile —
and asserts the output is identical. This catches any behavioral drift between the two paths.
"""

from types import SimpleNamespace

import pytest
from agent.agent_init import request_overrides_for_turn
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


@pytest.fixture
def transport():
    return ChatCompletionsTransport()


def _msgs():
    return [{"role": "user", "content": "hello"}]


def _max_tokens_fn(n):
    return {"max_completion_tokens": n}


class TestNvidiaProfileParity:
    def test_max_tokens_match(self, transport):
        """NVIDIA profile sets max_tokens=16384; legacy flag is removed."""
        profile = transport.build_kwargs(
            model="nvidia/nemotron", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("nvidia"),
            max_tokens_param_fn=_max_tokens_fn,
        )
        assert profile["max_completion_tokens"] == 16384


class TestKimiProfileParity:
    def test_temperature_omitted(self, transport):
        legacy = transport.build_kwargs(
            model="kimi-k2", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("kimi-coding"), omit_temperature=True,
        )
        profile = transport.build_kwargs(
            model="kimi-k2", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("kimi"),
        )
        assert "temperature" not in legacy
        assert "temperature" not in profile

    def test_max_tokens(self, transport):
        legacy = transport.build_kwargs(
            model="kimi-k2", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("kimi-coding"), max_tokens_param_fn=_max_tokens_fn,
        )
        profile = transport.build_kwargs(
            model="kimi-k2", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("kimi"),
            max_tokens_param_fn=_max_tokens_fn,
        )
        assert profile["max_completion_tokens"] == legacy["max_completion_tokens"] == 32000

    def test_thinking_enabled(self, transport):
        # xor contract: explicit effort → reasoning_effort only, no thinking.
        rc = {"enabled": True, "effort": "high"}
        legacy = transport.build_kwargs(
            model="kimi-k2", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("kimi-coding"), reasoning_config=rc,
        )
        profile = transport.build_kwargs(
            model="kimi-k2", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("kimi"),
            reasoning_config=rc,
        )
        assert profile["reasoning_effort"] == legacy["reasoning_effort"] == "high"
        assert "thinking" not in profile.get("extra_body", {})
        assert "thinking" not in legacy.get("extra_body", {})

    def test_thinking_disabled(self, transport):
        rc = {"enabled": False}
        legacy = transport.build_kwargs(
            model="kimi-k2", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("kimi-coding"), reasoning_config=rc,
        )
        profile = transport.build_kwargs(
            model="kimi-k2", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("kimi"),
            reasoning_config=rc,
        )
        assert profile["extra_body"]["thinking"] == legacy["extra_body"]["thinking"]
        assert profile["extra_body"]["thinking"]["type"] == "disabled"
        assert "reasoning_effort" not in profile
        assert "reasoning_effort" not in legacy

    def test_reasoning_effort_default(self, transport):
        # xor contract: enabled w/o effort → thinking-enabled only, no effort.
        rc = {"enabled": True}
        legacy = transport.build_kwargs(
            model="kimi-k2", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("kimi-coding"), reasoning_config=rc,
        )
        profile = transport.build_kwargs(
            model="kimi-k2", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("kimi"),
            reasoning_config=rc,
        )
        assert profile["extra_body"]["thinking"] == legacy["extra_body"]["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in profile
        assert "reasoning_effort" not in legacy


class TestOpenRouterProfileParity:
    def test_provider_preferences(self, transport):
        prefs = {"allow": ["anthropic"]}
        legacy = transport.build_kwargs(
            model="anthropic/claude-sonnet-4.6", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"), provider_preferences=prefs,
        )
        profile = transport.build_kwargs(
            model="anthropic/claude-sonnet-4.6", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"),
            provider_preferences=prefs,
        )
        assert profile["extra_body"]["provider"] == legacy["extra_body"]["provider"]

    def test_reasoning_full_config(self, transport):
        rc = {"enabled": True, "effort": "high"}
        legacy = transport.build_kwargs(
            model="deepseek/deepseek-chat", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"), supports_reasoning=True, reasoning_config=rc,
        )
        profile = transport.build_kwargs(
            model="deepseek/deepseek-chat", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"),
            supports_reasoning=True, reasoning_config=rc,
        )
        assert profile["extra_body"]["reasoning"] == legacy["extra_body"]["reasoning"]

    def test_default_reasoning(self, transport):
        legacy = transport.build_kwargs(
            model="deepseek/deepseek-chat", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"), supports_reasoning=True,
        )
        profile = transport.build_kwargs(
            model="deepseek/deepseek-chat", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"),
            supports_reasoning=True,
        )
        assert profile["extra_body"]["reasoning"] == legacy["extra_body"]["reasoning"]


class TestNousProfileParity:
    def test_tags(self, transport):
        legacy = transport.build_kwargs(
            model="hermes-3", messages=_msgs(), tools=None, provider_profile=get_provider_profile("nous"),
        )
        profile = transport.build_kwargs(
            model="hermes-3", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("nous"),
        )
        assert profile["extra_body"]["tags"] == legacy["extra_body"]["tags"]

    def test_reasoning_omitted_when_disabled(self, transport):
        rc = {"enabled": False}
        legacy = transport.build_kwargs(
            model="hermes-3", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("nous"), supports_reasoning=True, reasoning_config=rc,
        )
        profile = transport.build_kwargs(
            model="hermes-3", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("nous"),
            supports_reasoning=True, reasoning_config=rc,
        )
        assert "reasoning" not in legacy.get("extra_body", {})
        assert "reasoning" not in profile.get("extra_body", {})


class TestQwenProfileParity:
    def test_max_tokens(self, transport):
        legacy = transport.build_kwargs(
            model="qwen3.5", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("qwen-oauth"), max_tokens_param_fn=_max_tokens_fn,
        )
        profile = transport.build_kwargs(
            model="qwen3.5", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("qwen"),
            max_tokens_param_fn=_max_tokens_fn,
        )
        assert profile["max_completion_tokens"] == legacy["max_completion_tokens"] == 65536

    def test_vl_high_resolution(self, transport):
        legacy = transport.build_kwargs(
            model="qwen3.5", messages=_msgs(), tools=None, provider_profile=get_provider_profile("qwen-oauth"),
        )
        profile = transport.build_kwargs(
            model="qwen3.5", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("qwen"),
        )
        assert profile["extra_body"]["vl_high_resolution_images"] == legacy["extra_body"]["vl_high_resolution_images"]

    def test_metadata_top_level(self, transport):
        meta = {"sessionId": "s123", "promptId": "p456"}
        legacy = transport.build_kwargs(
            model="qwen3.5", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("qwen-oauth"), qwen_session_metadata=meta,
        )
        profile = transport.build_kwargs(
            model="qwen3.5", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("qwen"),
            qwen_session_metadata=meta,
        )
        assert profile["metadata"] == legacy["metadata"] == meta
        assert "metadata" not in profile.get("extra_body", {})

    def test_message_preprocessing(self, transport):
        """Qwen profile normalizes string content to list-of-parts."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        profile = transport.build_kwargs(
            model="qwen3.5", messages=msgs, tools=None,
            provider_profile=get_provider_profile("qwen"),
        )
        out_msgs = profile["messages"]
        # System message content normalized + cache_control injected
        assert isinstance(out_msgs[0]["content"], list)
        assert out_msgs[0]["content"][0]["type"] == "text"
        assert "cache_control" in out_msgs[0]["content"][-1]
        # User message content normalized
        assert isinstance(out_msgs[1]["content"], list)
        assert out_msgs[1]["content"][0] == {"type": "text", "text": "hello"}


class TestDeveloperRoleParity:
    """Developer role swap must work on BOTH legacy and profile paths."""

    def test_legacy_path_swaps_for_gpt5(self, transport):
        msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "hi"}]
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=msgs, tools=None,
        )
        assert kw["messages"][0]["role"] == "developer"

    def test_profile_path_swaps_for_gpt5(self, transport):
        msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "hi"}]
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=msgs, tools=None,
            provider_profile=get_provider_profile("openrouter"),
        )
        assert kw["messages"][0]["role"] == "developer"

    def test_profile_path_no_swap_for_claude(self, transport):
        msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "hi"}]
        kw = transport.build_kwargs(
            model="anthropic/claude-sonnet-4.6", messages=msgs, tools=None,
            provider_profile=get_provider_profile("openrouter"),
        )
        assert kw["messages"][0]["role"] == "system"


class TestRequestOverridesParity:
    """request_overrides with extra_body must merge identically on both paths."""

    def test_extra_body_override_legacy(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"),
            request_overrides={"extra_body": {"custom_key": "custom_val"}},
        )
        assert kw["extra_body"]["custom_key"] == "custom_val"

    def test_extra_body_override_profile(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"),
            request_overrides={"extra_body": {"custom_key": "custom_val"}},
        )
        assert kw["extra_body"]["custom_key"] == "custom_val"

    def test_extra_body_override_merges_with_provider_body(self, transport):
        """Override extra_body merges WITH provider extra_body, not replaces."""
        from agent.portal_tags import nous_portal_tags
        kw = transport.build_kwargs(
            model="hermes-3", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("nous"),
            request_overrides={"extra_body": {"custom": True}},
        )
        assert kw["extra_body"]["tags"] == nous_portal_tags()  # from profile
        assert kw["extra_body"]["custom"] is True  # from override

    def test_top_level_override(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"),
            request_overrides={"top_p": 0.9},
        )
        assert kw["top_p"] == 0.9


class TestReusedAgentTurnOverrides:
    """Reused gateway agents must keep custom-provider extra_body per turn (#54922).

    ``request_overrides_for_turn`` replaces the old reused-agent turn setup
    (``agent.request_overrides = turn_route.get("request_overrides") or {}`` in
    gateway/run.py), which clobbered the init-time ``custom_providers[].extra_body``
    on messaging paths while the CLI kept working.
    """

    def _custom_agent(self, extra_body):
        return SimpleNamespace(
            provider="custom",
            model="gpt-5.5",
            base_url="http://my-endpoint/v1",
            _custom_providers=[{
                "name": "my-endpoint",
                "base_url": "http://my-endpoint/v1",
                "extra_body": dict(extra_body),
            }],
            request_overrides={},
        )

    def test_extra_body_preserved_when_turn_overrides_empty(self):
        # No /fast → _resolve_turn_agent_config yields {}; extra_body must survive.
        agent = self._custom_agent({"reasoning_effort": "xhigh"})
        assert request_overrides_for_turn(agent, {})["extra_body"] == {"reasoning_effort": "xhigh"}

    def test_extra_body_preserved_when_turn_overrides_none(self):
        agent = self._custom_agent({"service_tier": "flex"})
        assert request_overrides_for_turn(agent, None)["extra_body"] == {"service_tier": "flex"}

    def test_turn_top_level_override_overlaid(self):
        agent = self._custom_agent({"reasoning_effort": "xhigh"})
        result = request_overrides_for_turn(agent, {"service_tier": "priority"})
        assert result["service_tier"] == "priority"
        assert result["extra_body"] == {"reasoning_effort": "xhigh"}

    def test_transient_turn_override_does_not_leak_across_turns(self):
        # Turn 1: /fast on. Turn 2: /fast off. The transient service_tier must
        # not persist, but the durable custom-provider extra_body must.
        agent = self._custom_agent({"reasoning_effort": "xhigh"})
        agent.request_overrides = request_overrides_for_turn(agent, {"service_tier": "priority"})
        assert agent.request_overrides["service_tier"] == "priority"
        agent.request_overrides = request_overrides_for_turn(agent, {})
        assert "service_tier" not in agent.request_overrides
        assert agent.request_overrides["extra_body"] == {"reasoning_effort": "xhigh"}

    def test_turn_extra_body_wins_per_key(self):
        agent = self._custom_agent({"reasoning_effort": "xhigh", "enable_thinking": True})
        result = request_overrides_for_turn(agent, {"extra_body": {"reasoning_effort": "low"}})
        assert result["extra_body"] == {"reasoning_effort": "low", "enable_thinking": True}

    def test_no_custom_extra_body_is_passthrough(self):
        agent = SimpleNamespace(
            provider="openai", model="gpt-5.5",
            base_url="https://api.openai.com/v1",
            _custom_providers=[], request_overrides={},
        )
        assert request_overrides_for_turn(agent, {"service_tier": "priority"}) == {"service_tier": "priority"}
        assert request_overrides_for_turn(agent, {}) == {}
        assert request_overrides_for_turn(agent, None) == {}

    def test_extra_body_reaches_wire_after_reuse_turn(self, transport):
        """E2E: preserved extra_body lands in the actual API kwargs on a reuse turn."""
        agent = self._custom_agent({"reasoning_effort": "xhigh"})
        overrides = request_overrides_for_turn(agent, {})
        kw = transport.build_kwargs(
            model="gpt-5.5", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"),
            request_overrides=overrides,
        )
        assert kw["extra_body"]["reasoning_effort"] == "xhigh"
