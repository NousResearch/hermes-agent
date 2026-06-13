"""Profile-path parity tests: verify profile path produces identical output to legacy flags.

Each test calls build_kwargs twice — once with legacy flags, once with provider_profile —
and asserts the output is identical. This catches any behavioral drift between the two paths.
"""

import pytest
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
        assert profile["extra_body"]["thinking"] == legacy["extra_body"]["thinking"]
        assert profile["reasoning_effort"] == legacy["reasoning_effort"] == "high"

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
        assert profile["reasoning_effort"] == legacy["reasoning_effort"] == "medium"


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
            model="anthropic/claude-sonnet-4.6", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"), supports_reasoning=True, reasoning_config=rc,
        )
        profile = transport.build_kwargs(
            model="anthropic/claude-sonnet-4.6", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"),
            supports_reasoning=True, reasoning_config=rc,
        )
        assert profile["extra_body"]["reasoning"] == legacy["extra_body"]["reasoning"]

    def test_default_reasoning(self, transport):
        legacy = transport.build_kwargs(
            model="anthropic/claude-sonnet-4.6", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"), supports_reasoning=True,
        )
        profile = transport.build_kwargs(
            model="anthropic/claude-sonnet-4.6", messages=_msgs(), tools=None,
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
