"""A chunk whose text is held back must still deliver its tool-call delta (#158731).

Text that may be an echoed SSE block, or a prefix of the router timeout shim
(whitespace included), is held back until it can be judged. The chunk that
carries that text can also carry a tool call's first delta (id + name); that
delta must reach the accumulator, or the call is assembled with an empty name.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
                           model=None, usage=None)


def _tc(tc_id=None, name=None, arguments=None):
    return SimpleNamespace(index=0, id=tc_id, function=SimpleNamespace(name=name, arguments=arguments))


def _agent():
    from run_agent import AIAgent
    agent = AIAgent(api_key="test-key", base_url="https://example.com/v1", model="test/model",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


# The issue's stream: text starting with ":" is held as a possible SSE comment, and the
# next chunk ends the text and opens the tool call.
_SSE_LIKE = [
    _chunk(content="<think>\nLet me verify"),
    _chunk(content=": the state."),
    _chunk(content="\n</think>\n\n", tool_calls=[_tc("call_00_a", "terminal", "")]),
    _chunk(tool_calls=[_tc(arguments='{"command": "ls"}')]),
    _chunk(finish_reason="tool_calls"),
]
# No text shown yet: whitespace is a prefix of the router timeout shim, so it is held too.
_LEADING_WHITESPACE = [
    _chunk(content="\n\n", tool_calls=[_tc("call_00_a", "terminal", "")]),
    _chunk(tool_calls=[_tc(arguments='{"command": "ls"}')]),
    _chunk(finish_reason="tool_calls"),
]


@pytest.mark.parametrize("chunks", [_SSE_LIKE, _LEADING_WHITESPACE], ids=["sse_like_text", "leading_whitespace"])
@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_tool_call_in_held_text_chunk_keeps_its_name(_close, create, chunks):
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda *a, **kw: iter(chunks)
    create.return_value = client
    agent = _agent()
    shown: list[str] = []
    agent._fire_stream_delta = shown.append

    response = agent._interruptible_streaming_api_call({})

    held = "".join(c.choices[0].delta.content or "" for c in chunks)
    if held.strip():
        assert "Let me verify: the state." in "".join(shown)  # the held text is still shown

    calls = response.choices[0].message.tool_calls
    assert calls is not None and len(calls) == 1
    assert calls[0].id == "call_00_a"
    assert calls[0].function.name == "terminal"
    assert json.loads(calls[0].function.arguments) == {"command": "ls"}
    assert response.choices[0].finish_reason == "tool_calls"
