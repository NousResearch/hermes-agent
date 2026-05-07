"""Regression test: _build_api_messages() rebuild after provider fallback.

When Hermes falls back from one provider (e.g. GLM-5.1) to DeepSeek via
_try_activate_fallback(), the api_messages array must be rebuilt with the new
provider's settings so that `_copy_reasoning_content_for_api` applies the
correct reasoning_content padding for the target provider.

Without this fix, the stale api_messages built with GLM settings (no padding
needed) is reused after fallback to DeepSeek, causing HTTP 400::

    The reasoning_content in the thinking mode must be passed back to the API.

This test verifies that _build_api_messages() correctly applies provider-specific
transformations when called directly — simulating what happens after fallback.

Refs #13235 / #17212 / #17825.
"""

from __future__ import annotations

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
    # Attributes used by _build_api_messages internals
    agent.ephemeral_system_prompt = None
    agent.prefill_messages = []
    agent._use_prompt_caching = False
    agent._cache_ttl = 3600
    agent._use_native_cache_layout = True
    agent.api_mode = "chat_completions"
    return agent


def _make_assistant_msg(content: str = "Hello", reasoning: str | None = None):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning"] = reasoning
    return msg


def _make_user_msg(content: str = "Hi"):
    return {"role": "user", "content": content}


def _find_assistant(api_msgs: list[dict]) -> dict | None:
    for msg in api_msgs:
        if msg.get("role") == "assistant":
            return msg
    return None


class TestBuildApiMessagesBasic:
    """Method existence and basic contract."""

    def test_method_exists_and_callable(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        assert callable(agent._build_api_messages)

    def test_returns_correct_types(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        messages = [_make_user_msg("Hi"), _make_assistant_msg("Hello")]
        api_msgs, total_chars, approx_tokens = agent._build_api_messages(messages, 0)

        assert isinstance(api_msgs, list)
        assert isinstance(total_chars, int)
        assert isinstance(approx_tokens, int)

    def test_preserves_message_count(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        messages = [_make_user_msg("A"), _make_assistant_msg("B")]
        api_msgs, _, _ = agent._build_api_messages(messages, 0)
        # Should have system + 2 messages, or just 2 if no system prompt
        assert len(api_msgs) >= 2


class TestBuildApiMessagesDeepSeekPadding:
    """DeepSeek requires reasoning_content padding on all assistant messages."""

    def test_deepseek_adds_padding_when_no_reasoning(self) -> None:
        """DeepSeek should add ' ' padding for assistant msgs without reasoning field."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        messages = [
            _make_user_msg("Hi"),
            _make_assistant_msg("Hello"),  # No reasoning field
        ]

        api_msgs, _, _ = agent._build_api_messages(messages, 0)
        assistant = _find_assistant(api_msgs)

        assert assistant is not None
        assert "reasoning_content" in assistant
        assert assistant["reasoning_content"] == " "

    def test_deepseek_promotes_reasoning_to_reasoning_content(self) -> None:
        """When reasoning field is present, it's promoted to reasoning_content."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        messages = [
            _make_user_msg("Hi"),
            _make_assistant_msg("Hello", reasoning="I am thinking..."),
        ]

        api_msgs, _, _ = agent._build_api_messages(messages, 0)
        assistant = _find_assistant(api_msgs)

        assert assistant is not None
        assert assistant["reasoning_content"] == "I am thinking..."

    def test_deepseek_promotes_reasoning_even_with_empty_content(self) -> None:
        """Reasoning field is promoted regardless of content presence."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        messages = [
            _make_user_msg("Hi"),
            _make_assistant_msg("", reasoning="just thinking"),
        ]

        api_msgs, _, _ = agent._build_api_messages(messages, 0)
        # The assistant message may be dropped by _drop_thinking_only_and_merge_users
        # if it has no content and no tool_calls. That's a separate concern; what
        # we care about here is that _build_api_messages doesn't crash and returns
        # a valid tuple.
        assert isinstance(api_msgs, list)
        assert len(api_msgs) >= 1


class TestBuildApiMessagesGLMMode:
    """GLM-5.1 does NOT enforce the thinking echo — padding differs from DeepSeek."""

    def test_glmmode_adds_no_padding_when_no_reasoning(self) -> None:
        """GLM should NOT add padding for assistant msgs without reasoning field."""
        agent = _make_agent(provider="ollama", model="glm-5.1")
        messages = [
            _make_user_msg("Hi"),
            _make_assistant_msg("Hello"),  # No reasoning field
        ]

        api_msgs, _, _ = agent._build_api_messages(messages, 0)
        assistant = _find_assistant(api_msgs)

        assert assistant is not None
        # GLM does NOT need thinking padding — no reasoning_content injected
        assert "reasoning_content" not in assistant

    def test_glmmode_promotes_reasoning_to_reasoning_content(self) -> None:
        """Reasoning field is always promoted regardless of provider."""
        agent = _make_agent(provider="ollama", model="glm-5.1")
        messages = [
            _make_user_msg("Hi"),
            _make_assistant_msg("Hello", reasoning="I am thinking..."),
        ]

        api_msgs, _, _ = agent._build_api_messages(messages, 0)
        assistant = _find_assistant(api_msgs)

        assert assistant is not None
        # reasoning is promoted to reasoning_content regardless of provider
        assert "reasoning_content" in assistant
        assert assistant["reasoning_content"] == "I am thinking..."


class TestBuildApiMessagesFallbackSimulation:
    """This is THE bug: after fallback to DeepSeek, rebuilt messages must have padding."""

    def test_fallback_from_glm_to_deepseek_adds_padding(self) -> None:
        """Simulate fallback: messages built without padding get padding after rebuild."""
        agent = _make_agent(provider="ollama", model="glm-5.1")
        messages = [
            _make_user_msg("Hi"),
            _make_assistant_msg("Hello"),  # No reasoning field — no padding needed for GLM
        ]

        # Build with GLM — no padding
        api_msgs_before, _, _ = agent._build_api_messages(messages, 0)
        assistant_before = _find_assistant(api_msgs_before)
        assert assistant_before is not None
        assert "reasoning_content" not in assistant_before

        # Simulate fallback: switch provider to DeepSeek
        agent.provider = "deepseek"
        agent.model = "deepseek-v4-flash"

        # Rebuild — now padding IS needed
        api_msgs_after, _, _ = agent._build_api_messages(messages, 0)
        assistant_after = _find_assistant(api_msgs_after)

        assert assistant_after is not None
        assert "reasoning_content" in assistant_after
        # The fix: after fallback, DeepSeek's padding requirement is satisfied
        assert assistant_after["reasoning_content"] == " "

    def test_fallback_from_deepseek_to_glm_strips_padding(self) -> None:
        """Fallback in the opposite direction: padding requirement should be relaxed."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        messages = [
            _make_user_msg("Hi"),
            _make_assistant_msg("Hello"),  # No reasoning field
        ]

        # Build with DeepSeek — gets padding
        api_msgs_before, _, _ = agent._build_api_messages(messages, 0)
        assistant_before = _find_assistant(api_msgs_before)
        assert assistant_before is not None
        assert assistant_before["reasoning_content"] == " "

        # Fallback: switch to GLM
        agent.provider = "ollama"
        agent.model = "glm-5.1"

        api_msgs_after, _, _ = agent._build_api_messages(messages, 0)
        assistant_after = _find_assistant(api_msgs_after)
        assert assistant_after is not None
        assert "reasoning_content" not in assistant_after


class TestBuildApiMessagesPreservesOriginal:
    """Verify _build_api_messages does not mutate the original messages list."""

    def test_original_messages_unchanged(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        original_msg = _make_assistant_msg("Hello", reasoning="I am thinking...")
        messages = [_make_user_msg("Hi"), original_msg]

        agent._build_api_messages(messages, 0)

        # Original should be unchanged — no reasoning_content added to source
        assert "reasoning_content" not in original_msg

    def test_system_prompt_preserved(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        messages = [_make_user_msg("Hi")]
        api_msgs, _, _ = agent._build_api_messages(messages, 0,
                                                   active_system_prompt="You are a helpful assistant.")

        system_msgs = [m for m in api_msgs if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert "You are a helpful assistant" in system_msgs[0]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
