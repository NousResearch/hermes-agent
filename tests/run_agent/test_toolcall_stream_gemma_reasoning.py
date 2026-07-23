"""Tool-call streams must route suppressed Gemma reasoning to the sink.

Regression for the sweeper-flagged gap on the streaming tool-call path:
``interruptible_streaming_api_call`` suppresses ordinary content deltas
once ``tool_calls_acc`` is populated (to avoid chatty "I'll use the
tool..." text next to tool calls).  Before the fix, that branch invoked
``stream_delta_callback`` directly, bypassing ``_fire_stream_delta`` and
therefore the shared ``StreamingThinkScrubber(on_reasoning=...)`` — so a
Gemma 4 channel-thought block (``<|channel>thought…<channel|>``) inside a
tool-call turn never produced ``reasoning.delta`` for the TUI gateway.

The fix feeds tool-suppressed content through the shared scrubber solely
to extract reasoning, while keeping the visible commentary suppressed.

End-to-end assertions:
  * visible stream callback receives NO tool-suppressed content;
  * the reasoning callback receives the Gemma block exactly once;
  * trailing non-reasoning commentary stays suppressed everywhere.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None, model=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=None)


def _make_tool_call_delta(index=0, tc_id=None, name=None, arguments=None):
    func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=func)


def _make_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


class TestToolCallStreamGemmaReasoning:
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_gemma_block_reaches_reasoning_sink_once_and_stays_invisible(
        self, mock_close, mock_create
    ):
        """Gemma reasoning inside a tool-call stream fires the reasoning
        callback (once, tags stripped) and never reaches the visible
        stream; trailing commentary stays fully suppressed."""
        chunks = [
            # Tool call arrives first, so every later content delta is
            # tool-suppressed.
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_1", name="run_go"),
            ]),
            # Gemma channel-thought block, split across deltas, with the
            # open tag itself split mid-token (real providers do this).
            _make_stream_chunk(content="<|chan"),
            _make_stream_chunk(content="nel>thoughtLet me check whether "),
            _make_stream_chunk(content="Go is installed.<chan"),
            _make_stream_chunk(content="nel|>"),
            # Ordinary commentary after the block — must stay suppressed
            # everywhere (this is the "chatty text" the branch exists to
            # hide) and must NOT leak into the reasoning callback.
            _make_stream_chunk(content="I'll run the tool now."),
            _make_stream_chunk(
                tool_calls=[_make_tool_call_delta(index=0, arguments="{}")],
                finish_reason="tool_calls",
                model="gemma-4",
            ),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = _make_agent()

        visible: list[str] = []
        reasoning: list[str] = []
        agent.stream_delta_callback = visible.append
        agent.reasoning_callback = reasoning.append

        response = agent._interruptible_streaming_api_call({})

        # Tool call accumulated correctly.
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls
        first = tool_calls[0]
        name = (
            first["function"]["name"]
            if isinstance(first, dict)
            else first.function.name
        )
        assert name == "run_go"

        # 1. Nothing tool-suppressed reached the visible stream.
        assert visible == [], f"tool-suppressed content leaked: {visible!r}"

        # 2. The reasoning sink received the block content, once.
        joined = "".join(reasoning)
        assert "Let me check whether Go is installed." in joined
        assert joined.count("Let me check whether Go is installed.") == 1

        # 3. No raw tags and no post-block commentary in the sink.
        assert "<|channel>" not in joined and "<channel|>" not in joined
        assert "I'll run the tool now." not in joined

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_plain_commentary_without_reasoning_fires_nothing(
        self, mock_close, mock_create
    ):
        """Tool-suppressed commentary with no reasoning tags stays
        suppressed and produces no reasoning deltas at all."""
        chunks = [
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_1", name="run_go"),
            ]),
            _make_stream_chunk(content="Using the tool to check."),
            _make_stream_chunk(
                tool_calls=[_make_tool_call_delta(index=0, arguments="{}")],
                finish_reason="tool_calls",
            ),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = _make_agent()

        visible: list[str] = []
        reasoning: list[str] = []
        agent.stream_delta_callback = visible.append
        agent.reasoning_callback = reasoning.append

        agent._interruptible_streaming_api_call({})

        assert visible == []
        assert reasoning == []
