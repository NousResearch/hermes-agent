"""A streamed chunk carrying both ``delta.content`` and ``delta.tool_calls`` keeps the tool call (#126758).

The text-hold branch (possible echoed SSE / router timeout shim prefix) used to ``continue`` past the
tool-call accumulator for that same chunk, so a gateway that packs a text preamble and the first
``tool_calls`` delta into one frame lost the action silently: ``finish_reason`` still said
``tool_calls``, the turn finalized as prose, nothing executed.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _tc_delta(name=None, arguments=None, tc_id=None):
    return SimpleNamespace(index=0, id=tc_id, function=SimpleNamespace(name=name, arguments=arguments))


def _chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
                           model="m", usage=None)


@pytest.mark.parametrize(
    "preamble",
    [
        # Prefix of the router-timeout shim sentinel: held back until it can be judged.
        "Connect",
        # Looks like an SSE control block: held back as possibly echoed transport framing.
        ": keepalive\n",
    ],
)
@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_held_text_chunk_still_feeds_tool_calls(mock_close, mock_create, preamble):
    from run_agent import AIAgent

    chunks = [
        _chunk(content=preamble, tool_calls=[_tc_delta(name="write_file", arguments='{"path": "a",', tc_id="call_1")]),
        _chunk(tool_calls=[_tc_delta(arguments=' "content": "x"}')]),
        _chunk(finish_reason="tool_calls"),
    ]
    deltas = []
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_create.return_value = mock_client
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, stream_delta_callback=deltas.append)
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False

    response = agent._interruptible_streaming_api_call({})

    tool_calls = response.choices[0].message.tool_calls
    assert [(tc.function.name, tc.function.arguments) for tc in tool_calls] == [
        ("write_file", '{"path": "a", "content": "x"}'),
    ]
    assert response.choices[0].finish_reason == "tool_calls"
