"""Streaming regression for internal context/control packet leakage."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
        model="test-model",
        usage=None,
    )


def _make_tool_call_delta(index=0, tc_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_suppressed_tool_text_uses_context_stream_scrubber(mock_close, mock_create):
    """Tool-turn suppressed text must not bypass memory/guard scrubbers."""
    from run_agent import AIAgent

    chunks = [
        _make_stream_chunk(content="thinking..."),
        _make_stream_chunk(tool_calls=[
            _make_tool_call_delta(index=0, tc_id="call_abc", name="read_file")
        ]),
        _make_stream_chunk(content="\n<ship_mode_guard>\nsecret route metadata"),
        _make_stream_chunk(content="</ship_mode_guard>\n visible after"),
        _make_stream_chunk(finish_reason="tool_calls"),
    ]

    deltas = []
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_create.return_value = mock_client

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        stream_delta_callback=lambda t: deltas.append(t),
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False

    agent._interruptible_streaming_api_call({})

    streamed = "".join(deltas)
    assert "thinking..." in streamed
    assert " visible after" in streamed
    assert "secret route metadata" not in streamed
    assert "ship_mode_guard" not in streamed
