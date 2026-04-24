"""Verify _copy_reasoning_content_for_api injects reasoning_content for DeepSeek.

DeepSeek v4/v4-flash/v4-pro in thinking mode requires `reasoning_content` on
every assistant message in replayed history, otherwise the API returns:

    HTTP 400: The reasoning_content in the thinking mode must be passed back to the API.

This fails on compressed-summary assistant messages and on replays of sessions
whose assistant messages were persisted before the fix (the "poisoned history"
case). Real reasoning is preserved by the earlier `explicit_reasoning` and
`normalized_reasoning` branches; the new DeepSeek branch only fills in the
empty-string placeholder when no reasoning is available.

Related upstream: PR #15228, Issue #15213.
"""

from __future__ import annotations

import pytest

from run_agent import AIAgent


def _make_agent(
    *,
    model: str,
    reasoning_config: dict | None,
    provider: str = "openrouter",
    base_url: str = "https://openrouter.ai/api/v1",
) -> AIAgent:
    """Construct a minimal AIAgent stub for _copy_reasoning_content_for_api.

    The method reads self.model, self.provider, self.base_url, and
    self.reasoning_config. We bypass __init__ and set only those attrs.
    """
    agent = AIAgent.__new__(AIAgent)
    agent.model = model
    agent.provider = provider
    agent.base_url = base_url
    agent.reasoning_config = reasoning_config
    return agent


class TestDeepSeekReasoningContentInjection:
    """Cover the new DeepSeek branch in _copy_reasoning_content_for_api."""

    def test_deepseek_enabled_injects_empty_when_no_reasoning(self):
        """DeepSeek model + thinking on + no stored reasoning → inject ''."""
        agent = _make_agent(
            model="deepseek/deepseek-v4-flash",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        source_msg = {"role": "assistant", "content": "hi"}
        api_msg = {"role": "assistant", "content": "hi"}
        agent._copy_reasoning_content_for_api(source_msg, api_msg)
        assert api_msg["reasoning_content"] == ""

    def test_deepseek_disabled_skips_injection(self):
        """reasoning_effort: none → reasoning_config.enabled=False → do nothing."""
        agent = _make_agent(
            model="deepseek/deepseek-v4-flash",
            reasoning_config={"enabled": False},
        )
        source_msg = {"role": "assistant", "content": "hi"}
        api_msg = {"role": "assistant", "content": "hi"}
        agent._copy_reasoning_content_for_api(source_msg, api_msg)
        assert "reasoning_content" not in api_msg

    def test_non_deepseek_skips_injection(self):
        """Non-DeepSeek models must not get the empty-string injection."""
        agent = _make_agent(
            model="anthropic/claude-sonnet-4.6",
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://api.anthropic.com",
        )
        source_msg = {"role": "assistant", "content": "hi"}
        api_msg = {"role": "assistant", "content": "hi"}
        agent._copy_reasoning_content_for_api(source_msg, api_msg)
        assert "reasoning_content" not in api_msg

    def test_deepseek_preserves_real_reasoning(self):
        """Real `reasoning` must win over the empty-string default."""
        agent = _make_agent(
            model="deepseek/deepseek-v4-flash",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        source_msg = {
            "role": "assistant",
            "content": "hi",
            "reasoning": "chain of thought text",
        }
        api_msg = {"role": "assistant", "content": "hi"}
        agent._copy_reasoning_content_for_api(source_msg, api_msg)
        assert api_msg["reasoning_content"] == "chain of thought text"

    def test_deepseek_reasoning_config_none_defaults_enabled(self):
        """reasoning_config=None means default-enabled (chat_completions.py:256)."""
        agent = _make_agent(
            model="deepseek/deepseek-v4-flash",
            reasoning_config=None,
        )
        source_msg = {"role": "assistant", "content": "hi"}
        api_msg = {"role": "assistant", "content": "hi"}
        agent._copy_reasoning_content_for_api(source_msg, api_msg)
        assert api_msg["reasoning_content"] == ""

    def test_deepseek_preserves_explicit_empty_reasoning(self):
        """Pre-existing reasoning_content='' must be preserved (explicit branch)."""
        agent = _make_agent(
            model="deepseek/deepseek-v4-flash",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        source_msg = {
            "role": "assistant",
            "content": "hi",
            "reasoning_content": "",
        }
        api_msg = {"role": "assistant", "content": "hi"}
        agent._copy_reasoning_content_for_api(source_msg, api_msg)
        # Empty string is explicitly preserved by the explicit_reasoning
        # isinstance(str) branch, before the new DeepSeek branch can run.
        assert api_msg["reasoning_content"] == ""

    def test_deepseek_injects_on_tool_call_message(self):
        """The whole point of dropping the tool_calls gate (per #15228).

        DeepSeek requires reasoning_content on EVERY assistant message in
        history, including tool-call turns. The Kimi branch gates on
        tool_calls; the DeepSeek branch intentionally does not.
        """
        agent = _make_agent(
            model="deepseek/deepseek-v4-flash",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        source_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "some_tool", "arguments": "{}"},
            }],
        }
        api_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": source_msg["tool_calls"],
        }
        agent._copy_reasoning_content_for_api(source_msg, api_msg)
        assert api_msg["reasoning_content"] == ""

    def test_deepseek_reasoning_config_without_enabled_key(self):
        """reasoning_config={"effort": "high"} (no `enabled` key) → enabled.

        Common real-world shape. The guard explicitly checks
        `enabled is False`; a missing key falls through to default-enabled,
        so injection should happen.
        """
        agent = _make_agent(
            model="deepseek/deepseek-v4-flash",
            reasoning_config={"effort": "high"},
        )
        source_msg = {"role": "assistant", "content": "hi"}
        api_msg = {"role": "assistant", "content": "hi"}
        agent._copy_reasoning_content_for_api(source_msg, api_msg)
        assert api_msg["reasoning_content"] == ""

    def test_deepseek_reasoning_config_non_dict_is_safe(self):
        """Defensive: non-dict reasoning_config must not crash.

        The isinstance guard (mirrored from line 7342) prevents AttributeError
        if reasoning_config is ever an unexpected type (Mock, string, etc.).
        Falls through to default-enabled because the guard only disables on
        explicit dict with enabled=False.
        """
        agent = _make_agent(
            model="deepseek/deepseek-v4-flash",
            reasoning_config="high",  # intentionally wrong type
        )
        source_msg = {"role": "assistant", "content": "hi"}
        api_msg = {"role": "assistant", "content": "hi"}
        agent._copy_reasoning_content_for_api(source_msg, api_msg)
        assert api_msg["reasoning_content"] == ""

    def test_direct_deepseek_api_injects(self):
        """api.deepseek.com path is the other supported host (upstream PR #15228).

        Direct DeepSeek API uses model names WITHOUT a `deepseek/` prefix
        (e.g. `deepseek-chat`, `deepseek-v4`). Detection relies on base_url,
        not model name — we trust the endpoint.
        """
        agent = _make_agent(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        source_msg = {"role": "assistant", "content": "hi"}
        api_msg = {"role": "assistant", "content": "hi"}
        agent._copy_reasoning_content_for_api(source_msg, api_msg)
        assert api_msg["reasoning_content"] == ""

    def test_bedrock_deepseek_skips_injection(self):
        """Regression: Bedrock-hosted DeepSeek must NOT get reasoning_content injected.

        Model slug contains 'deepseek' but the endpoint is AWS Bedrock, which
        has not been validated to accept the reasoning_content field. A broad
        substring match on model name would wrongly fire here.
        """
        agent = _make_agent(
            model="deepseek.v3.2",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            provider="bedrock",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        source_msg = {"role": "assistant", "content": "hi"}
        api_msg = {"role": "assistant", "content": "hi"}
        agent._copy_reasoning_content_for_api(source_msg, api_msg)
        assert "reasoning_content" not in api_msg

    def test_nvidia_nim_deepseek_skips_injection(self):
        """Regression: NVIDIA NIM-hosted DeepSeek must NOT get injected."""
        agent = _make_agent(
            model="deepseek-ai/deepseek-v3.2",
            base_url="https://integrate.api.nvidia.com/v1",
            provider="nvidia",
            reasoning_config={"enabled": True, "effort": "high"},
        )
        source_msg = {"role": "assistant", "content": "hi"}
        api_msg = {"role": "assistant", "content": "hi"}
        agent._copy_reasoning_content_for_api(source_msg, api_msg)
        assert "reasoning_content" not in api_msg
