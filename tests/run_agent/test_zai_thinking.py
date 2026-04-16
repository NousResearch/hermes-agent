"""
Tests for z.ai/GLM preserved thinking support in AIAgent.

Covers:
  - _is_zai_direct() detection (provider + URL)
  - _build_api_kwargs() injecting ``thinking`` parameter for GLM models
  - reasoning_config gating (enabled/disabled)
  - Multi-turn reasoning_content re-injection on message sanitization
"""

import pytest
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_defs(*names: str) -> list:
    """Build minimal tool definition list accepted by AIAgent.__init__."""
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_zai_agent(model="glm-5.1", base_url="https://api.z.ai/api/paas/v4",
                     provider="zai", reasoning_config=None):
    """Create a minimal AIAgent wired to z.ai."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-glm-key",
            model=model,
            base_url=base_url,
            provider=provider,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a.reasoning_config = reasoning_config
        a.api_mode = "chat_completions"
        return a


# ===========================================================================
# _is_zai_direct()
# ===========================================================================

class TestIsZaiDirect:

    def test_detected_by_provider(self):
        a = _make_zai_agent(provider="zai")
        assert a._is_zai_direct() is True

    def test_detected_by_bigmodel_cn_url(self):
        a = _make_zai_agent(base_url="https://open.bigmodel.cn/api/paas/v4")
        assert a._is_zai_direct() is True

    def test_detected_by_api_z_ai_url(self):
        a = _make_zai_agent(base_url="https://api.z.ai/api/paas/v4")
        assert a._is_zai_direct() is True

    def test_not_detected_for_openrouter(self):
        a = _make_zai_agent(
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
        )
        assert a._is_zai_direct() is False

    def test_not_detected_for_openai(self):
        a = _make_zai_agent(
            base_url="https://api.openai.com/v1",
            provider="openai",
        )
        assert a._is_zai_direct() is False

    def test_not_detected_for_empty_provider(self):
        a = _make_zai_agent(provider="")
        a.base_url = "https://api.unknown.com/v1"
        a._base_url_lower = a.base_url.lower()
        assert a._is_zai_direct() is False


# ===========================================================================
# _build_api_kwargs() — thinking parameter injection
# ===========================================================================

class TestZaiThinkingParam:

    def test_glm51_gets_thinking_enabled(self):
        """GLM-5.1 should get thinking=enabled by default."""
        a = _make_zai_agent(model="glm-5.1")
        kwargs = a._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["thinking"] == {
            "type": "enabled",
            "compact_history": False,
        }

    def test_glm5_gets_thinking_enabled(self):
        a = _make_zai_agent(model="glm-5")
        kwargs = a._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["thinking"]["type"] == "enabled"
        assert kwargs["extra_body"]["thinking"]["compact_history"] is False

    def test_glm5_turbo_gets_thinking_enabled(self):
        a = _make_zai_agent(model="glm-5-turbo")
        kwargs = a._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["thinking"]["type"] == "enabled"

    def test_glm47_gets_thinking_enabled(self):
        a = _make_zai_agent(model="glm-4.7")
        kwargs = a._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["thinking"]["type"] == "enabled"

    def test_glm46_no_thinking_param(self):
        """GLM-4.6 auto-determines thinking, no parameter injected."""
        a = _make_zai_agent(model="glm-4.6")
        kwargs = a._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "thinking" not in kwargs.get("extra_body", {})

    def test_glm45_no_thinking_param(self):
        a = _make_zai_agent(model="glm-4.5")
        kwargs = a._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "thinking" not in kwargs.get("extra_body", {})

    def test_reasoning_config_disabled(self):
        """reasoning_config enabled=False should disable thinking."""
        a = _make_zai_agent(model="glm-5.1", reasoning_config={"enabled": False})
        kwargs = a._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    def test_reasoning_config_enabled_with_effort(self):
        """reasoning_config with effort should still enable thinking."""
        a = _make_zai_agent(
            model="glm-5.1",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        kwargs = a._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["thinking"]["type"] == "enabled"
        # z.ai doesn't use "effort" — it's always compulsory when enabled.
        # The effort key is not forwarded for z.ai.
        assert "effort" not in kwargs["extra_body"]["thinking"]

    def test_no_thinking_for_non_zai_provider(self):
        """OpenAI provider should never get z.ai thinking param."""
        a = _make_zai_agent(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            provider="openai",
        )
        kwargs = a._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "thinking" not in kwargs.get("extra_body", {})

    def test_bigmodel_cn_url_gets_thinking(self):
        """China endpoint URL should trigger thinking too."""
        a = _make_zai_agent(
            model="glm-5.1",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            provider="",
        )
        kwargs = a._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["extra_body"]["thinking"]["type"] == "enabled"


# ===========================================================================
# Multi-turn reasoning_content re-injection
# ===========================================================================

class TestZaiMultiTurnReasoning:

    def test_reasoning_content_injected_on_assistant_msg(self):
        """Assistant messages with reasoning should get reasoning_content
        when z.ai direct is active."""
        a = _make_zai_agent(model="glm-5.1")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there", "reasoning": "I should greet the user."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        # Re-use the sanitization loop from the agent loop.
        # We simulate what happens in run_conversation().
        api_messages = []
        for msg in messages:
            api_msg = msg.copy()
            for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                api_msg.pop(internal_field, None)
            if a._is_zai_direct() and msg.get("role") == "assistant" and msg.get("reasoning"):
                api_msg["reasoning_content"] = msg["reasoning"]
            api_messages.append(api_msg)

        # System message: no reasoning
        assert "reasoning" not in api_messages[0]
        assert "reasoning_content" not in api_messages[0]

        # User message: no reasoning
        assert "reasoning" not in api_messages[1]
        assert "reasoning_content" not in api_messages[1]

        # Assistant message: reasoning stripped, reasoning_content injected
        assert "reasoning" not in api_messages[2]
        assert api_messages[2]["reasoning_content"] == "I should greet the user."

        # Next user message: untouched
        assert api_messages[3]["content"] == "What is 2+2?"

    def test_reasoning_content_not_injected_for_non_zai(self):
        """Non-z.ai providers should NOT get reasoning_content re-injected."""
        a = _make_zai_agent(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            provider="openai",
        )
        messages = [
            {"role": "assistant", "content": "Hi", "reasoning": "thinking..."},
        ]
        api_messages = []
        for msg in messages:
            api_msg = msg.copy()
            for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                api_msg.pop(internal_field, None)
            if a._is_zai_direct() and msg.get("role") == "assistant" and msg.get("reasoning"):
                api_msg["reasoning_content"] = msg["reasoning"]
            api_messages.append(api_msg)

        # reasoning stripped, reasoning_content NOT injected
        assert "reasoning" not in api_messages[0]
        assert "reasoning_content" not in api_messages[0]

    def test_assistant_without_reasoning_untouched(self):
        """Assistant messages without reasoning should not get empty
        reasoning_content."""
        a = _make_zai_agent(model="glm-5.1")
        messages = [
            {"role": "assistant", "content": "Hi"},
        ]
        api_messages = []
        for msg in messages:
            api_msg = msg.copy()
            for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                api_msg.pop(internal_field, None)
            if a._is_zai_direct() and msg.get("role") == "assistant" and msg.get("reasoning"):
                api_msg["reasoning_content"] = msg["reasoning"]
            api_messages.append(api_msg)

        assert "reasoning_content" not in api_messages[0]
