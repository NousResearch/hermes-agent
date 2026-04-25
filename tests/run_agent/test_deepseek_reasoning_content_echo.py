"""Regression test: DeepSeek V4 thinking mode reasoning_content echo.

DeepSeek V4-flash / V4-pro thinking mode requires ``reasoning_content`` on
every assistant message that carries ``tool_calls``. When a persisted
session replays an assistant tool-call turn that was recorded without the
field, DeepSeek rejects the next request with HTTP 400::

    The reasoning_content in the thinking mode must be passed back to the API.

Fix covers three paths:

1. ``_build_assistant_message`` — new tool-call messages without raw
   reasoning_content get ``""`` pinned at creation time so nothing gets
   persisted poisoned.
2. ``_copy_reasoning_content_for_api`` — already-poisoned history replays
   with ``reasoning_content=""`` injected defensively.
3. Detection covers three signals: ``provider == "deepseek"``,
   ``"deepseek" in model``, and ``api.deepseek.com`` host match. The third
   catches custom-provider setups pointing at DeepSeek.

Refs #15250 / #15353.
"""

from __future__ import annotations

import pytest

from run_agent import AIAgent


def _make_agent(provider: str = "", model: str = "", base_url: str = "") -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    return agent


class TestNeedsDeepSeekToolReasoning:
    """_needs_deepseek_tool_reasoning() recognises all three detection signals."""

    def test_provider_deepseek(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        assert agent._needs_deepseek_tool_reasoning() is True

    def test_model_substring(self) -> None:
        # Custom provider pointing at DeepSeek with provider='custom'
        agent = _make_agent(provider="custom", model="deepseek-v4-pro")
        assert agent._needs_deepseek_tool_reasoning() is True

    def test_base_url_host(self) -> None:
        agent = _make_agent(
            provider="custom",
            model="some-aliased-name",
            base_url="https://api.deepseek.com/v1",
        )
        assert agent._needs_deepseek_tool_reasoning() is True

    def test_provider_case_insensitive(self) -> None:
        agent = _make_agent(provider="DeepSeek", model="")
        assert agent._needs_deepseek_tool_reasoning() is True

    def test_non_deepseek_provider(self) -> None:
        agent = _make_agent(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
        )
        assert agent._needs_deepseek_tool_reasoning() is False

    def test_empty_everything(self) -> None:
        agent = _make_agent()
        assert agent._needs_deepseek_tool_reasoning() is False


class TestCopyReasoningContentForApi:
    """_copy_reasoning_content_for_api pads reasoning_content for DeepSeek tool-calls."""

    def test_deepseek_tool_call_poisoned_history_gets_empty_string(self) -> None:
        """Already-poisoned history (no reasoning_content, no reasoning) gets ''."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        source = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == ""

    def test_deepseek_assistant_no_tool_call_gets_padded(self) -> None:
        """DeepSeek thinking mode pads ALL assistant turns, even without tool_calls."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        source = {"role": "assistant", "content": "hello"}
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == ""

    def test_deepseek_explicit_reasoning_content_preserved(self) -> None:
        """When reasoning_content is already set, it's copied verbatim."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        source = {
            "role": "assistant",
            "reasoning_content": "<think>real chain of thought</think>",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == "<think>real chain of thought</think>"

    def test_deepseek_reasoning_field_promoted(self) -> None:
        """When only 'reasoning' is set, it gets promoted to reasoning_content."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        source = {
            "role": "assistant",
            "reasoning": "thought trace",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == "thought trace"

    def test_kimi_path_still_works(self) -> None:
        """Existing Kimi detection still pads reasoning_content."""
        agent = _make_agent(provider="kimi-coding", model="kimi-k2.5")
        source = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == ""

    def test_kimi_moonshot_base_url(self) -> None:
        agent = _make_agent(
            provider="custom", model="kimi-k2", base_url="https://api.moonshot.ai/v1"
        )
        source = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == ""

    def test_non_thinking_provider_not_padded(self) -> None:
        """Providers that don't require the echo are untouched."""
        agent = _make_agent(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
        )
        source = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert "reasoning_content" not in api_msg

    def test_deepseek_custom_base_url(self) -> None:
        """Custom provider pointing at api.deepseek.com is detected via host."""
        agent = _make_agent(
            provider="custom",
            model="whatever",
            base_url="https://api.deepseek.com/v1",
        )
        source = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == ""

    def test_non_assistant_role_ignored(self) -> None:
        """User/tool messages are left alone."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        source = {"role": "user", "content": "hi"}
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert "reasoning_content" not in api_msg


class TestNeedsKimiToolReasoning:
    """The extracted _needs_kimi_tool_reasoning() helper keeps Kimi behavior intact."""

    @pytest.mark.parametrize(
        "provider,base_url",
        [
            ("kimi-coding", ""),
            ("kimi-coding-cn", ""),
            ("custom", "https://api.kimi.com/v1"),
            ("custom", "https://api.moonshot.ai/v1"),
            ("custom", "https://api.moonshot.cn/v1"),
        ],
    )
    def test_kimi_signals(self, provider: str, base_url: str) -> None:
        agent = _make_agent(provider=provider, model="kimi-k2", base_url=base_url)
        assert agent._needs_kimi_tool_reasoning() is True

    def test_non_kimi_provider(self) -> None:
        agent = _make_agent(
            provider="openrouter",
            model="moonshotai/kimi-k2",
            base_url="https://openrouter.ai/api/v1",
        )
        # model name contains 'moonshot' but host is openrouter — should be False
        assert agent._needs_kimi_tool_reasoning() is False
