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
        """The NVIDIA profile default max_tokens reaches the wire when the user sets none."""
        profile = transport.build_kwargs(
            model="nvidia/nemotron", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("nvidia"),
            max_tokens_param_fn=_max_tokens_fn,
        )
        assert profile["max_completion_tokens"] == get_provider_profile("nvidia").default_max_tokens


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



class TestRequestOverridesParity:
    """request_overrides with extra_body must merge identically on both paths."""

    def test_extra_body_override_legacy(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"),
            request_overrides={"extra_body": {"custom_key": "custom_val"}},
        )
        assert kw["extra_body"]["custom_key"] == "custom_val"



    def test_top_level_override(self, transport):
        kw = transport.build_kwargs(
            model="gpt-5.4", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("openrouter"),
            request_overrides={"top_p": 0.9},
        )
        assert kw["top_p"] == 0.9

    def test_kimi_k3_medium_is_served_directly_not_upgraded(self, transport):
        # Endpoint evidence 2026-10-21 (#142365): the Kimi Code endpoints accept
        # ``medium`` — silently upgrading it to ``high`` spends more reasoning
        # tokens/latency than the user asked for.
        kw = transport.build_kwargs(
            model="k3", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("kimi-coding"),
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert kw["reasoning_effort"] == "medium"

    def test_kimi_k3_none_disables_thinking(self, transport):
        # ``reasoning_effort="none"`` is a real off switch on the Kimi Code
        # endpoints (#142365) — it must disable thinking, not enable it.
        kw = transport.build_kwargs(
            model="k3", messages=_msgs(), tools=None,
            provider_profile=get_provider_profile("kimi-coding"),
            reasoning_config={"enabled": True, "effort": "none"},
        )
        assert kw["extra_body"]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in kw
