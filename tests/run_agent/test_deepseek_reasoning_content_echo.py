"""Regression test: DeepSeek V4 thinking mode reasoning_content echo.

DeepSeek V4-flash / V4-pro thinking mode requires ``reasoning_content`` on
every assistant message that carries ``tool_calls``. When a persisted
session replays an assistant tool-call turn that was recorded without the
field, DeepSeek rejects the next request with HTTP 400::

    The reasoning_content in the thinking mode must be passed back to the API.

Fix covers three paths:

1. ``_build_assistant_message`` — new tool-call messages without raw
   reasoning_content get ``" "`` pinned at creation time so nothing gets
   persisted poisoned.
2. ``_copy_reasoning_content_for_api`` — already-poisoned history replays
   with ``reasoning_content=" "`` injected defensively.
3. Detection covers three signals: ``provider == "deepseek"``,
   ``"deepseek" in model``, and ``api.deepseek.com`` host match. The third
   catches custom-provider setups pointing at DeepSeek.

The placeholder is a single space (not empty string) because DeepSeek V4 Pro
tightened validation and rejects empty-string reasoning_content with a
400 ("The reasoning content in the thinking mode must be passed back to
the API"). A space satisfies non-empty checks everywhere without leaking
fabricated reasoning.

Refs #15250 / #15353 / #17341.
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
    """Minimal SDK-shaped tool_call object that satisfies the builder's iteration."""
    return SimpleNamespace(
        id=call_id,
        call_id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
        extra_content=None,
    )


def _build_sdk_message(reasoning_content=_ATTR_ABSENT, **extra):
    """SDK-shaped assistant message; ``reasoning_content`` defaults to absent."""
    kwargs = {"content": "", **extra}
    if reasoning_content is not _ATTR_ABSENT:
        kwargs["reasoning_content"] = reasoning_content
    return SimpleNamespace(**kwargs)


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

    def test_deepseek_tool_call_poisoned_history_gets_space_placeholder(self) -> None:
        """Already-poisoned history (no reasoning_content, no reasoning) gets ' '."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        source = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == " "

    def test_deepseek_assistant_no_tool_call_gets_padded(self) -> None:
        """DeepSeek thinking mode pads ALL assistant turns, even without tool_calls."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        source = {"role": "assistant", "content": "hello"}
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg.get("reasoning_content") == " "

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

    def test_deepseek_stale_empty_placeholder_upgraded_to_space(self) -> None:
        """Sessions persisted before #17341 have ``reasoning_content=""`` pinned
        at creation time. DeepSeek V4 Pro rejects "" with HTTP 400. When the
        active provider enforces the thinking-mode echo, the replay path
        upgrades "" → " " so stale history doesn't break the next turn.
        """
        agent = _make_agent(provider="deepseek", model="deepseek-v4-pro")
        source = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == " "

    def test_non_thinking_provider_preserves_empty_reasoning_content_verbatim(self) -> None:
        """The stale-placeholder upgrade ONLY fires when the active provider
        enforces thinking-mode echo. On non-thinking providers, an empty
        reasoning_content must still round-trip verbatim.
        """
        agent = _make_agent(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
        )
        source = {
            "role": "assistant",
            "content": "hi",
            "reasoning_content": "",
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == ""

    def test_deepseek_reasoning_field_promoted(self) -> None:
        """When only 'reasoning' is set, it gets promoted to reasoning_content."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        source = {
            "role": "assistant",
            "content": "",
            "reasoning": "thought trace",
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == "thought trace"

    def test_deepseek_poisoned_cross_provider_history_padded(self) -> None:
        """Cross-provider tool-call turn (#15748): MiniMax reasoning leaks
        to DeepSeek/Kimi request.

        If the source turn has tool_calls AND a 'reasoning' field but NO
        'reasoning_content' key, it's from a prior provider (the DeepSeek
        build path pins reasoning_content at creation). Inject " " instead
        of forwarding the prior provider's chain of thought.
        """
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        source = {
            "role": "assistant",
            "content": "",
            "reasoning": "MiniMax chain of thought from a prior turn",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == " "

    def test_kimi_poisoned_cross_provider_history_padded(self) -> None:
        """Kimi path of #15748 — same rule as DeepSeek."""
        agent = _make_agent(provider="kimi-coding", model="kimi-k2.5")
        source = {
            "role": "assistant",
            "content": "",
            "reasoning": "DeepSeek chain of thought from a prior turn",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == " "

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
        assert api_msg.get("reasoning_content") == " "

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
        assert api_msg.get("reasoning_content") == " "

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
        assert api_msg.get("reasoning_content") == " "

    def test_non_assistant_role_ignored(self) -> None:
        """User/tool messages are left alone."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        source = {"role": "user", "content": "hi"}
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert "reasoning_content" not in api_msg


class TestBuildAssistantMessageDeepSeekReasoningContent:
    """_build_assistant_message pins replay-safe DeepSeek tool-call state."""

    def test_deepseek_tool_call_reasoning_is_backfilled_into_reasoning_content(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        assistant_message = SimpleNamespace(
            content=None,
            reasoning="DeepSeek tool-call reasoning",
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

        assert msg["reasoning_content"] == "DeepSeek tool-call reasoning"
        assert msg["tool_calls"][0]["id"] == "call_1"

    def test_deepseek_model_extra_reasoning_content_is_preserved(self) -> None:
        """OpenAI SDK stores unknown provider fields in model_extra."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        assistant_message = SimpleNamespace(
            content=None,
            reasoning=None,
            reasoning_content=None,
            model_extra={"reasoning_content": "DeepSeek model_extra reasoning"},
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

        assert msg["reasoning_content"] == "DeepSeek model_extra reasoning"

    def test_deepseek_tool_call_without_raw_reasoning_content_gets_space_placeholder(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
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


class TestBuildAssistantMessagePadsStrictProviders:
    """Regression for #17400: _build_assistant_message must pin reasoning_content
    on tool-call turns when the active provider enforces echo-back, regardless
    of whether the SDK exposed reasoning_content as None, omitted it entirely,
    or returned an empty thinking block.

    Prior to the fix, the pad branch was guarded by ``msg.get("tool_calls")``,
    which was always falsy because tool_calls were assigned later in the same
    method. Persisted history accumulated assistant tool-call turns with no
    reasoning_content; the next replay 400'd on DeepSeek/Kimi.
    """

    @pytest.mark.parametrize(
        "provider,model,base_url,sdk_reasoning_content,expected",
        [
            pytest.param(
                "deepseek", "deepseek-v4-pro", "",
                None, " ",
                id="deepseek-attr-none",
            ),
            pytest.param(
                "deepseek", "deepseek-v4-pro", "",
                _ATTR_ABSENT, " ",
                id="deepseek-attr-absent",
            ),
            pytest.param(
                "kimi-coding", "kimi-k2.6", "",
                None, " ",
                id="kimi-attr-none",
            ),
            pytest.param(
                "custom", "kimi-k2", "https://api.moonshot.ai/v1",
                _ATTR_ABSENT, " ",
                id="moonshot-base-url",
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

    def test_tool_call_preserves_real_reasoning_content(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-pro")
        msg_in = _build_sdk_message(
            reasoning_content="actual chain of thought",
            tool_calls=[_sdk_tool_call()],
        )
        msg = agent._build_assistant_message(msg_in, finish_reason="tool_calls")
        assert msg["reasoning_content"] == "actual chain of thought"

    def test_text_only_turn_not_padded_by_tool_call_branch(self) -> None:
        """Plain-text turns rely on _copy_reasoning_content_for_api at replay
        time, not on this builder's tool-call pad."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-pro")
        msg_in = SimpleNamespace(content="hello", tool_calls=None)
        msg = agent._build_assistant_message(msg_in, finish_reason="stop")
        assert "tool_calls" not in msg
        assert "reasoning_content" not in msg

    def test_streamed_reasoning_text_promoted_over_pad(self) -> None:
        """When ``.reasoning`` carries streamed thinking, it must be promoted
        to reasoning_content rather than overwritten with the empty pad."""
        agent = _make_agent(provider="deepseek", model="deepseek-v4-pro")
        msg_in = _build_sdk_message(
            reasoning="streamed thoughts",
            tool_calls=[_sdk_tool_call()],
        )
        msg = agent._build_assistant_message(msg_in, finish_reason="tool_calls")
        assert msg["reasoning_content"] == "streamed thoughts"


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
