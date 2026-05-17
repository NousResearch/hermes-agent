"""Tests for dynamic reasoning_content echo-back detection.

When the API response includes ``reasoning_content`` (either as a top-level
SDK attribute or in ``model_extra``), the session flag
``_requires_reasoning_echo`` is set.  This makes ``_needs_thinking_reasoning_pad()``
return True for *any* thinking-mode provider — including those not covered by
the static provider-name checks.

Refs #27297.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from run_agent import AIAgent


def _make_agent(provider: str = "", model: str = "", base_url: str = "") -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


def _sdk_tool_call(call_id: str = "c1", name: str = "terminal", arguments: str = "{}"):
    return SimpleNamespace(
        id=call_id,
        call_id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
        extra_content=None,
    )


def _build_sdk_message(reasoning_content=None, **extra):
    kwargs = {"content": "", **extra}
    if reasoning_content is not None:
        kwargs["reasoning_content"] = reasoning_content
    return SimpleNamespace(**kwargs)


_ATTR_ABSENT = object()


class TestDynamicEchoDetection:
    """_requires_reasoning_echo is auto-set when API returns reasoning_content."""

    def test_flag_set_on_sdk_reasoning_content(self) -> None:
        """When SDK exposes reasoning_content, flag is set on agent."""
        agent = _make_agent(provider="unknown-custom", model="my-model")
        assert not getattr(agent, "_requires_reasoning_echo", False)

        msg_obj = _build_sdk_message(
            reasoning_content="Let me think about this...",
            tool_calls=[_sdk_tool_call()],
        )
        result = agent._build_assistant_message(msg_obj, finish_reason="tool_calls")

        assert agent._requires_reasoning_echo is True
        assert "reasoning_content" in result

    def test_flag_set_on_model_extra_reasoning_content(self) -> None:
        """When reasoning_content comes via model_extra, flag is set."""
        agent = _make_agent(provider="qwen-tp", model="qwen3.6-plus")
        msg_obj = SimpleNamespace(
            content="",
            tool_calls=[_sdk_tool_call()],
            model_extra={"reasoning_content": "Step 1: analyze..."},
        )
        result = agent._build_assistant_message(msg_obj, finish_reason="tool_calls")

        assert agent._requires_reasoning_echo is True
        assert "reasoning_content" in result

    def test_flag_not_set_without_reasoning_content(self) -> None:
        """Normal (non-thinking) response does not set the flag."""
        agent = _make_agent(provider="openrouter", model="gpt-4o")
        msg_obj = _build_sdk_message(content="Hello!")
        agent._build_assistant_message(msg_obj, finish_reason="stop")

        assert not getattr(agent, "_requires_reasoning_echo", False)

    def test_flag_idempotent(self) -> None:
        """Setting the flag multiple times is safe (no error)."""
        agent = _make_agent(provider="custom")
        for _ in range(3):
            msg_obj = _build_sdk_message(
                reasoning_content="thinking...",
                tool_calls=[_sdk_tool_call()],
            )
            agent._build_assistant_message(msg_obj, finish_reason="tool_calls")

        assert agent._requires_reasoning_echo is True


class TestNeedsThinkingPadWithDynamicFlag:
    """_needs_thinking_reasoning_pad uses dynamic flag as primary check."""

    def test_dynamic_flag_overrides_unknown_provider(self) -> None:
        """Unknown provider returns True once dynamic flag is set."""
        agent = _make_agent(provider="qwen-tp", model="qwen3.6-plus")
        # Static checks should be False for unknown provider
        assert agent._needs_deepseek_tool_reasoning() is False
        assert agent._needs_kimi_tool_reasoning() is False
        assert agent._needs_mimo_tool_reasoning() is False

        # Before dynamic flag
        assert agent._needs_thinking_reasoning_pad() is False

        # Set dynamic flag
        agent._requires_reasoning_echo = True
        assert agent._needs_thinking_reasoning_pad() is True

    def test_static_fallback_still_works(self) -> None:
        """Static checks work when dynamic flag is not set."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-pro")
        # No dynamic flag set
        assert not getattr(agent, "_requires_reasoning_echo", False)
        # Static check should still detect DeepSeek
        assert agent._needs_thinking_reasoning_pad() is True

    def test_dynamic_flag_wins_over_static(self) -> None:
        """Even with wrong provider, dynamic flag makes pad return True."""
        agent = _make_agent(provider="custom-gateway", model="my-model")
        agent._requires_reasoning_echo = True
        assert agent._needs_thinking_reasoning_pad() is True


class TestEndToEndDynamicDetection:
    """Full flow: API response → flag set → pad returns True → echo applied."""

    def test_unknown_provider_tool_call_gets_reasoning_content(self) -> None:
        """Unknown provider with thinking mode gets reasoning_content on tool calls."""
        agent = _make_agent(provider="aliyun-qwen", model="qwen3.6-plus")
        # First response has reasoning_content → sets flag
        msg1 = _build_sdk_message(
            reasoning_content="I need to use a tool...",
            tool_calls=[_sdk_tool_call()],
        )
        agent._build_assistant_message(msg1, finish_reason="tool_calls")
        assert agent._requires_reasoning_echo is True

        # Second tool-call message WITHOUT reasoning_content should still get padded
        msg2 = _build_sdk_message(
            tool_calls=[_sdk_tool_call()],
        )
        result = agent._build_assistant_message(msg2, finish_reason="tool_calls")
        assert "reasoning_content" in result
        assert result["reasoning_content"]  # non-empty
