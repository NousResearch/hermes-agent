"""Regression test: Xiaomi MiMo thinking mode reasoning_content echo.

MiMo's OpenAI-compatible API requires ``reasoning_content`` on every prior
assistant message in thinking mode. When a persisted session replays an
assistant turn that was recorded without the field, MiMo rejects the next
request with HTTP 400::

    The reasoning_content in the thinking mode must be passed back to the API.

Fix covers three paths (mirroring the DeepSeek / Kimi fix):

1. ``_build_assistant_message`` — new tool-call messages without raw
   ``reasoning_content`` get ``" "`` pinned at creation time so nothing gets
   persisted poisoned.
2. ``_copy_reasoning_content_for_api`` — already-poisoned history replays
   with ``reasoning_content=" "`` injected defensively, ``""`` placeholders
   upgrade to ``" "``, and cross-provider history doesn't leak another
   provider's chain of thought.
3. Detection covers four signals: ``provider == "xiaomi"``, the
   ``xiaomimimo.com`` API host, model names starting with ``xiaomi/`` (catalog
   form), and bare names starting with ``mimo-`` (config form).

Refs #24443.
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


class TestNeedsMimoToolReasoning:
    """_needs_mimo_tool_reasoning() recognises all four detection signals."""

    def test_provider_xiaomi(self) -> None:
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        assert agent._needs_mimo_tool_reasoning() is True

    def test_provider_case_insensitive(self) -> None:
        agent = _make_agent(provider="Xiaomi", model="")
        assert agent._needs_mimo_tool_reasoning() is True

    def test_base_url_host(self) -> None:
        agent = _make_agent(
            provider="custom",
            model="some-aliased-name",
            base_url="https://api.xiaomimimo.com/v1",
        )
        assert agent._needs_mimo_tool_reasoning() is True

    def test_catalog_model_prefix(self) -> None:
        agent = _make_agent(provider="openrouter", model="xiaomi/mimo-v2.5-pro")
        assert agent._needs_mimo_tool_reasoning() is True

    def test_bare_model_prefix(self) -> None:
        agent = _make_agent(provider="custom", model="mimo-v2-pro")
        assert agent._needs_mimo_tool_reasoning() is True

    def test_bare_model_prefix_dotted(self) -> None:
        agent = _make_agent(provider="custom", model="mimo-v2.5")
        assert agent._needs_mimo_tool_reasoning() is True

    def test_third_party_catalog_form(self) -> None:
        """Some hosted catalogs surface MiMo as ``vendor/mimo-…``."""
        agent = _make_agent(provider="openrouter", model="nous/mimo-v2-flash")
        assert agent._needs_mimo_tool_reasoning() is True

    def test_non_mimo_provider(self) -> None:
        agent = _make_agent(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
        )
        assert agent._needs_mimo_tool_reasoning() is False

    def test_empty_everything(self) -> None:
        agent = _make_agent()
        assert agent._needs_mimo_tool_reasoning() is False

    def test_deepseek_not_flagged_as_mimo(self) -> None:
        """Cross-provider guard: DeepSeek must not trip MiMo detection."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-pro")
        assert agent._needs_mimo_tool_reasoning() is False

    @pytest.mark.parametrize(
        "provider,model,base_url",
        [
            # MiniMax — name contains "mi" but is not MiMo
            ("minimax", "minimax/abab6-chat", ""),
            ("minimax-cn", "MiniMax-M2.7", ""),
            # Mistral models — must not be confused with MiMo
            ("custom", "mistral/mistral-large", ""),
            # Microsoft Phi-4 mini — startswith "mi" but no "mimo-" segment
            ("openrouter", "microsoft/phi-4-mini-instruct", ""),
            # Plain ``mimo`` substring without dash separator must not match
            ("custom", "vendor-mimowave-1", ""),
        ],
    )
    def test_lookalike_model_names_not_flagged(
        self, provider: str, model: str, base_url: str
    ) -> None:
        """Substring-based detection must not over-match neighbouring vendors."""
        agent = _make_agent(provider=provider, model=model, base_url=base_url)
        assert agent._needs_mimo_tool_reasoning() is False


class TestThinkingPadGateIncludesMimo:
    """_needs_thinking_reasoning_pad() must OR in the MiMo detector."""

    def test_xiaomi_provider_enables_pad(self) -> None:
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        assert agent._needs_thinking_reasoning_pad() is True

    def test_mimo_base_url_enables_pad(self) -> None:
        agent = _make_agent(
            provider="custom",
            model="whatever",
            base_url="https://api.xiaomimimo.com/v1",
        )
        assert agent._needs_thinking_reasoning_pad() is True

    def test_existing_deepseek_path_unaffected(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-pro")
        assert agent._needs_thinking_reasoning_pad() is True

    def test_existing_kimi_path_unaffected(self) -> None:
        agent = _make_agent(provider="kimi-coding", model="kimi-k2.6")
        assert agent._needs_thinking_reasoning_pad() is True

    def test_unrelated_provider_still_false(self) -> None:
        agent = _make_agent(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
        )
        assert agent._needs_thinking_reasoning_pad() is False


class TestCopyReasoningContentForApiMimo:
    """_copy_reasoning_content_for_api pads reasoning_content for MiMo turns."""

    def test_mimo_tool_call_poisoned_history_gets_space_placeholder(self) -> None:
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        source = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == " "

    def test_mimo_assistant_no_tool_call_gets_padded(self) -> None:
        """MiMo thinking mode pads ALL assistant turns, mirroring DeepSeek."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        source = {"role": "assistant", "content": "hello"}
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == " "

    def test_mimo_explicit_reasoning_content_preserved(self) -> None:
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        source = {
            "role": "assistant",
            "reasoning_content": "<think>real chain of thought</think>",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == "<think>real chain of thought</think>"

    def test_mimo_stale_empty_placeholder_upgraded_to_space(self) -> None:
        """Pre-fix sessions with reasoning_content="" upgrade to " " on replay."""
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

    def test_mimo_reasoning_field_promoted(self) -> None:
        """When only 'reasoning' is set, promote to reasoning_content."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        source = {
            "role": "assistant",
            "content": "",
            "reasoning": "thought trace",
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == "thought trace"

    def test_mimo_poisoned_cross_provider_history_padded(self) -> None:
        """Cross-provider tool-call turn: prior provider's reasoning must NOT
        leak into the MiMo request; inject " " instead.
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

    def test_mimo_base_url_path(self) -> None:
        agent = _make_agent(
            provider="custom",
            model="mimo-v2-flash",
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

    def test_non_thinking_provider_not_padded(self) -> None:
        """Identical inputs under a non-enforcing provider stay untouched."""
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


class TestBuildAssistantMessageMimo:
    """_build_assistant_message pins replay-safe MiMo tool-call state."""

    def test_mimo_tool_call_reasoning_is_backfilled_into_reasoning_content(self) -> None:
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        assistant_message = SimpleNamespace(
            content=None,
            reasoning="MiMo tool-call reasoning",
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
        assert msg["reasoning_content"] == "MiMo tool-call reasoning"
        assert msg["tool_calls"][0]["id"] == "call_1"

    def test_mimo_model_extra_reasoning_content_preserved(self) -> None:
        """OpenAI SDK stores unknown provider fields in model_extra."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        assistant_message = SimpleNamespace(
            content=None,
            reasoning=None,
            reasoning_content=None,
            model_extra={"reasoning_content": "MiMo model_extra reasoning"},
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
        assert msg["reasoning_content"] == "MiMo model_extra reasoning"

    def test_mimo_tool_call_without_raw_reasoning_content_gets_space_placeholder(self) -> None:
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
                "custom", "mimo-v2-pro", "https://api.xiaomimimo.com/v1",
                _ATTR_ABSENT, " ",
                id="mimo-base-url",
            ),
            pytest.param(
                "openrouter", "xiaomi/mimo-v2.5-pro", "https://openrouter.ai/api/v1",
                _ATTR_ABSENT, " ",
                id="openrouter-xiaomi-catalog",
            ),
            pytest.param(
                "openrouter", "anthropic/claude-sonnet-4.6", "https://openrouter.ai/api/v1",
                _ATTR_ABSENT, _EXPECT_NOT_PRESENT,
                id="openrouter-non-mimo-no-pad",
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

    def test_streamed_reasoning_text_promoted_over_pad(self) -> None:
        """When .reasoning carries streamed thinking, promote it rather than
        overwriting with the empty pad."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        msg_in = _build_sdk_message(
            reasoning="streamed thoughts",
            tool_calls=[_sdk_tool_call()],
        )
        msg = agent._build_assistant_message(msg_in, finish_reason="tool_calls")
        assert msg["reasoning_content"] == "streamed thoughts"

    def test_build_assistant_message_text_only_turn_no_pad_at_creation_time(self) -> None:
        """Plain-text turns must NOT be padded at creation time — the
        tool-call pad branch in ``_build_assistant_message`` only fires when
        ``assistant_tool_calls`` is truthy. Replay-time padding for text
        turns happens later in ``_copy_reasoning_content_for_api`` Tier 4
        (covered by ``test_mimo_assistant_no_tool_call_gets_padded``); the
        boundary between creation-time and replay-time pad is intentional.
        """
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5-pro")
        msg_in = SimpleNamespace(content="hello", tool_calls=None)
        msg = agent._build_assistant_message(msg_in, finish_reason="stop")
        assert "tool_calls" not in msg
        assert "reasoning_content" not in msg
