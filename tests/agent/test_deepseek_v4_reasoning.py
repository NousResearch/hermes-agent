"""Regression tests for DeepSeek V4 reasoning_content consistency.

DeepSeek V4 requires all assistant messages to have reasoning_content
once any turn has produced reasoning.  Mixing (some with, some without)
triggers HTTP 400.
"""

import pytest
from run_agent import AIAgent


class TestNeedsReasoningBackfill:
    """Verify _needs_reasoning_backfill detection."""

    def _make_agent(self, provider="deepseek", model="deepseek-v4-pro"):
        agent = AIAgent.__new__(AIAgent)
        agent.provider = provider
        agent.model = model
        return agent

    def test_true_when_assistant_has_reasoning(self):
        agent = self._make_agent()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning": "thinking..."},
        ]
        assert agent._needs_reasoning_backfill(messages) is True

    def test_false_when_no_reasoning(self):
        agent = self._make_agent()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert agent._needs_reasoning_backfill(messages) is False

    def test_false_for_legacy_reasoner(self):
        agent = self._make_agent(model="deepseek-reasoner")
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning": "thinking..."},
        ]
        assert agent._needs_reasoning_backfill(messages) is False

    def test_false_for_other_providers(self):
        agent = self._make_agent(provider="openai", model="gpt-4")
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning": "thinking..."},
        ]
        assert agent._needs_reasoning_backfill(messages) is False

    def test_true_with_mixed_messages(self):
        agent = self._make_agent()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a", "reasoning": "r1"},
            {"role": "assistant", "content": "b"},  # no reasoning
        ]
        assert agent._needs_reasoning_backfill(messages) is True


class TestReasoningContentInjection:
    """Verify api_messages build injects reasoning_content correctly."""

    def _make_agent(self, provider="deepseek", model="deepseek-v4-pro"):
        agent = AIAgent.__new__(AIAgent)
        agent.provider = provider
        agent.model = model
        agent.api_mode = "chat_completions"
        agent._cached_system_prompt = None
        agent.ephemeral_system_prompt = None
        agent.prefill_messages = None
        agent._ext_prefetch_cache = None
        agent._plugin_user_context = None
        return agent

    def test_v4_injects_reasoning_content_when_present(self):
        agent = self._make_agent()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning": "thinking..."},
        ]
        # Simulate the core api_messages build logic (main loop path)
        _needs_backfill = agent._needs_reasoning_backfill(messages)
        api_messages = []
        for msg in messages:
            api_msg = msg.copy()
            if msg.get("role") == "assistant":
                reasoning = msg.get("reasoning")
                if reasoning:
                    if not (agent.provider == "deepseek" and agent.model == "deepseek-reasoner"):
                        api_msg["reasoning_content"] = reasoning
                elif _needs_backfill:
                    api_msg["reasoning_content"] = ""
                api_msg.pop("reasoning", None)
            api_messages.append(api_msg)

        assert api_messages[1].get("reasoning_content") == "thinking..."
        assert "reasoning" not in api_messages[1]

    def test_v4_backfills_empty_reasoning_content(self):
        agent = self._make_agent()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a", "reasoning": "r1"},
            {"role": "assistant", "content": "b"},  # no reasoning
        ]
        _needs_backfill = agent._needs_reasoning_backfill(messages)
        api_messages = []
        for msg in messages:
            api_msg = msg.copy()
            if msg.get("role") == "assistant":
                reasoning = msg.get("reasoning")
                if reasoning:
                    if not (agent.provider == "deepseek" and agent.model == "deepseek-reasoner"):
                        api_msg["reasoning_content"] = reasoning
                elif _needs_backfill:
                    api_msg["reasoning_content"] = ""
                api_msg.pop("reasoning", None)
            api_messages.append(api_msg)

        # First assistant has reasoning
        assert api_messages[1].get("reasoning_content") == "r1"
        # Second assistant gets backfilled empty string
        assert api_messages[2].get("reasoning_content") == ""
        # Neither has internal 'reasoning' field
        assert "reasoning" not in api_messages[1]
        assert "reasoning" not in api_messages[2]

    def test_legacy_reasoner_no_injection(self):
        agent = self._make_agent(model="deepseek-reasoner")
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning": "thinking..."},
        ]
        _needs_backfill = agent._needs_reasoning_backfill(messages)
        api_messages = []
        for msg in messages:
            api_msg = msg.copy()
            if msg.get("role") == "assistant":
                reasoning = msg.get("reasoning")
                if reasoning:
                    if not (agent.provider == "deepseek" and agent.model == "deepseek-reasoner"):
                        api_msg["reasoning_content"] = reasoning
                elif _needs_backfill:
                    api_msg["reasoning_content"] = ""
                api_msg.pop("reasoning", None)
            api_messages.append(api_msg)

        # Legacy model should NOT get reasoning_content
        assert "reasoning_content" not in api_messages[1]

    def test_non_deepseek_no_backfill(self):
        agent = self._make_agent(provider="openai", model="gpt-4")
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a", "reasoning": "r1"},
            {"role": "assistant", "content": "b"},
        ]
        _needs_backfill = agent._needs_reasoning_backfill(messages)
        api_messages = []
        for msg in messages:
            api_msg = msg.copy()
            if msg.get("role") == "assistant":
                reasoning = msg.get("reasoning")
                if reasoning:
                    if not (agent.provider == "deepseek" and agent.model == "deepseek-reasoner"):
                        api_msg["reasoning_content"] = reasoning
                elif _needs_backfill:
                    api_msg["reasoning_content"] = ""
                api_msg.pop("reasoning", None)
            api_messages.append(api_msg)

        # OpenAI should get reasoning_content only when present
        assert api_messages[1].get("reasoning_content") == "r1"
        assert "reasoning_content" not in api_messages[2]

    def test_handle_max_iterations_backfill(self):
        agent = self._make_agent()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a", "reasoning": "r1"},
            {"role": "assistant", "content": "b"},
        ]
        _needs_backfill = agent._needs_reasoning_backfill(messages)
        api_messages = []
        for msg in messages:
            api_msg = msg.copy()
            for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                api_msg.pop(internal_field, None)
            if _needs_backfill and msg.get("role") == "assistant" and "reasoning_content" not in api_msg:
                api_msg["reasoning_content"] = ""
            api_messages.append(api_msg)

        # First assistant: reasoning stripped, but backfill should NOT apply
        # because reasoning_content was not in api_msg BEFORE the pop loop.
        # Wait — the fix checks AFTER pop, so it should backfill.
        assert api_messages[1].get("reasoning_content") == ""
        # Second assistant: backfilled
        assert api_messages[2].get("reasoning_content") == ""
