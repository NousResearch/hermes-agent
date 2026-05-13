"""Regression test: Xiaomi MiMo thinking mode reasoning_content echo.

Xiaomi MiMo (mimo-v2.5-pro, mimo-v2.5, …) is a DeepSeek-derived
thinking-capable model family. Both its OpenAI-compatible
``api.xiaomimimo.com/v1`` endpoint and its Anthropic-compatible
``token-plan-*.xiaomimimo.com/anthropic`` endpoints enforce the same
reasoning_content echo-back contract as DeepSeek. When a persisted session
replays an assistant tool-call turn that was recorded without
``reasoning_content``, Xiaomi rejects the next request with HTTP 400::

    The reasoning_content in the thinking mode must be passed back to the API.

Coverage mirrors the DeepSeek regression test (see
``test_deepseek_reasoning_content_echo.py``):

1. ``_needs_xiaomi_tool_reasoning`` recognises three signals — provider name,
   model prefix, and base URL host.
2. ``_copy_reasoning_content_for_api`` pads / preserves reasoning_content for
   Xiaomi assistant turns the same way it does for DeepSeek.
3. ``_build_assistant_message`` pins reasoning_content on tool-call turns so
   nothing gets persisted poisoned.
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


_ATTR_ABSENT = object()
_EXPECT_NOT_PRESENT = object()


def _sdk_tool_call(call_id: str = "c1", name: str = "terminal", arguments: str = "{}"):
    return SimpleNamespace(
        id=call_id,
        call_id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
        extra_content=None,
    )


def _build_sdk_message(reasoning_content=_ATTR_ABSENT, **extra):
    kwargs = {"content": "", **extra}
    if reasoning_content is not _ATTR_ABSENT:
        kwargs["reasoning_content"] = reasoning_content
    return SimpleNamespace(**kwargs)


class TestNeedsXiaomiToolReasoning:
    """_needs_xiaomi_tool_reasoning() recognises all three detection signals."""

    def test_provider_xiaomi(self) -> None:
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        assert agent._needs_xiaomi_tool_reasoning() is True

    def test_model_prefix(self) -> None:
        # Custom provider pointing at Xiaomi with provider='custom'
        agent = _make_agent(provider="custom", model="mimo-v2.5-pro")
        assert agent._needs_xiaomi_tool_reasoning() is True

    def test_model_prefix_lowercase(self) -> None:
        # Model names get lowercased before matching — mixed case input still matches.
        agent = _make_agent(provider="custom", model="MiMo-V2.5-Pro")
        assert agent._needs_xiaomi_tool_reasoning() is True

    def test_base_url_host_v1(self) -> None:
        agent = _make_agent(
            provider="custom",
            model="some-aliased-name",
            base_url="https://api.xiaomimimo.com/v1",
        )
        assert agent._needs_xiaomi_tool_reasoning() is True

    def test_base_url_host_anthropic(self) -> None:
        agent = _make_agent(
            provider="custom",
            model="some-aliased-name",
            base_url="https://token-plan-cn.xiaomimimo.com/anthropic",
        )
        assert agent._needs_xiaomi_tool_reasoning() is True

    def test_provider_case_insensitive(self) -> None:
        agent = _make_agent(provider="Xiaomi", model="")
        assert agent._needs_xiaomi_tool_reasoning() is True

    def test_non_xiaomi_provider(self) -> None:
        agent = _make_agent(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
        )
        assert agent._needs_xiaomi_tool_reasoning() is False

    def test_empty_everything(self) -> None:
        agent = _make_agent()
        assert agent._needs_xiaomi_tool_reasoning() is False


class TestCopyReasoningContentForApiXiaomi:
    """_copy_reasoning_content_for_api pads reasoning_content for Xiaomi tool-calls."""

    def test_xiaomi_tool_call_poisoned_history_gets_space_placeholder(self) -> None:
        """Already-poisoned history (no reasoning_content, no reasoning) gets ' '."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        source = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == " "

    def test_xiaomi_assistant_no_tool_call_gets_padded(self) -> None:
        """Xiaomi thinking mode pads ALL assistant turns, even without tool_calls."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        source = {"role": "assistant", "content": "hello"}
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == " "

    def test_xiaomi_explicit_reasoning_content_preserved(self) -> None:
        """When reasoning_content is already set, it's copied verbatim."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        source = {
            "role": "assistant",
            "reasoning_content": "<think>real chain of thought</think>",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == "<think>real chain of thought</think>"

    def test_xiaomi_stale_empty_placeholder_upgraded_to_space(self) -> None:
        """Sessions persisted with ``reasoning_content=""`` get upgraded to " "
        on replay because Xiaomi rejects empty-string reasoning_content the
        same way DeepSeek V4 Pro does.
        """
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        source = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == " "

    def test_xiaomi_reasoning_field_promoted(self) -> None:
        """When only 'reasoning' is set, it gets promoted to reasoning_content."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        source = {
            "role": "assistant",
            "content": "",
            "reasoning": "thought trace",
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == "thought trace"

    def test_xiaomi_poisoned_cross_provider_history_padded(self) -> None:
        """Cross-provider tool-call turn: if the source has tool_calls AND a
        'reasoning' field but NO 'reasoning_content' key, it's from a prior
        provider. Inject ' ' instead of forwarding the prior chain of thought.
        """
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        source = {
            "role": "assistant",
            "content": "",
            "reasoning": "MiniMax chain of thought from a prior turn",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == " "

    def test_xiaomi_anthropic_base_url(self) -> None:
        """Custom provider pointing at Xiaomi /anthropic is detected via host."""
        agent = _make_agent(
            provider="custom",
            model="whatever",
            base_url="https://token-plan-cn.xiaomimimo.com/anthropic",
        )
        source = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == " "

    def test_xiaomi_v1_base_url(self) -> None:
        """Custom provider pointing at Xiaomi /v1 is detected via host too."""
        agent = _make_agent(
            provider="custom",
            model="whatever",
            base_url="https://api.xiaomimimo.com/v1",
        )
        source = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == " "


class TestBuildAssistantMessageXiaomiReasoningContent:
    """_build_assistant_message pins replay-safe Xiaomi tool-call state."""

    def test_xiaomi_tool_call_reasoning_is_backfilled_into_reasoning_content(self) -> None:
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        assistant_message = SimpleNamespace(
            content=None,
            reasoning="Xiaomi tool-call reasoning",
            reasoning_content=None,
            reasoning_details=None,
            codex_reasoning_items=None,
            codex_message_items=None,
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    call_id=None,
                    response_item_id=None,
                    type="function",
                    function=SimpleNamespace(name="terminal", arguments="{}"),
                )
            ],
        )

        msg = agent._build_assistant_message(assistant_message, "tool_calls")

        assert msg["reasoning_content"] == "Xiaomi tool-call reasoning"
        assert msg["tool_calls"][0]["id"] == "call_1"

    def test_xiaomi_tool_call_without_raw_reasoning_content_gets_space_placeholder(self) -> None:
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        assistant_message = SimpleNamespace(
            content=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
            codex_reasoning_items=None,
            codex_message_items=None,
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    call_id=None,
                    response_item_id=None,
                    type="function",
                    function=SimpleNamespace(name="terminal", arguments="{}"),
                )
            ],
        )

        msg = agent._build_assistant_message(assistant_message, "tool_calls")

        assert msg["reasoning_content"] == " "
        assert msg["tool_calls"][0]["id"] == "call_1"

    @pytest.mark.parametrize(
        "provider,model,base_url,sdk_reasoning_content,expected",
        [
            pytest.param(
                "xiaomi", "mimo-v2.5-pro", "",
                None, " ",
                id="xiaomi-attr-none",
            ),
            pytest.param(
                "xiaomi", "mimo-v2.5-pro", "",
                _ATTR_ABSENT, " ",
                id="xiaomi-attr-absent",
            ),
            pytest.param(
                "custom", "mimo-v2.5", "https://token-plan-cn.xiaomimimo.com/anthropic",
                _ATTR_ABSENT, " ",
                id="xiaomi-anthropic-base-url",
            ),
            pytest.param(
                "openrouter", "anthropic/claude-sonnet-4.6", "https://openrouter.ai/api/v1",
                _ATTR_ABSENT, _EXPECT_NOT_PRESENT,
                id="openrouter-no-pad",
            ),
        ],
    )
    def test_tool_call_reasoning_content_pad(
        self, provider, model, base_url, sdk_reasoning_content, expected,
    ) -> None:
        agent = _make_agent(provider=provider, model=model, base_url=base_url)
        msg_in = _build_sdk_message(
            reasoning_content=sdk_reasoning_content,
            tool_calls=[_sdk_tool_call()],
        )
        msg = agent._build_assistant_message(msg_in, finish_reason="tool_calls")
        if expected is _EXPECT_NOT_PRESENT:
            assert "reasoning_content" not in msg
        else:
            assert msg["reasoning_content"] == expected
