"""Regression test for #119389 — a tool call whose arguments never parse as
valid JSON must not be dispatched just because the provider reported a terminal
``finish_reason``.

``_assemble_tool_calls`` flags an unrepairable tool-call payload
(``_repair_tool_call_arguments`` gives up and returns ``"{}"``) by setting
``has_truncated_tool_args``. The recovery branch in ``_finish_chat_stream``
previously fired only when ``finish_reason is None``; when the provider sent a
normal ``finish_reason`` (e.g. ``"stop"``) alongside the corrupt payload the
call fell through, was stamped ``"length"`` and dispatched with empty/corrupt
arguments — a ``write_file`` that silently never landed, with no signal to the
agent. The fix returns a partial-stream stub so the loop retries instead, the
same recovery ``interruptible_streaming_api_call`` uses for #74798.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_constants import PARTIAL_STREAM_STUB_ID, FINISH_REASON_LENGTH


def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content, tool_calls=tool_calls,
        reasoning_content=None, reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=None, usage=None)


def _tc_delta(index=0, tc_id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=tc_id,
                           function=SimpleNamespace(name=name, arguments=arguments))


def _make_agent():
    from run_agent import AIAgent
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_unrepairable_tool_args_with_finish_stop_returns_stub(_mock_close, mock_create, monkeypatch):
    """finish_reason="stop" + unrepairable write_file args → partial-stream stub,
    not a dispatched tool call with corrupt arguments (#119389)."""

    def _stream():
        # A multi-KB markdown payload with embedded quotes/newlines that never
        # closes its JSON string — exactly the shape the issue reproduced.
        yield _make_stream_chunk(tool_calls=[
            _tc_delta(0, "call_1", "write_file", '{"path": "/opt/data/CURRENT.md", "content": "# State')])
        yield _make_stream_chunk(tool_calls=[
            _tc_delta(0, None, None, '\n\n## Objective\nkeep `going` without a closing quote')])
        yield _make_stream_chunk(finish_reason="stop")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = lambda *a, **kw: _stream()
    mock_create.return_value = mock_client

    agent = _make_agent()
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    response = agent._interruptible_streaming_api_call({})

    assert response.id == PARTIAL_STREAM_STUB_ID, (
        "An unrepairable tool call with a terminal finish_reason must return a "
        "partial-stream stub so the loop retries, not a completed turn that "
        "dispatches the corrupt call (#119389).")
    assert response.choices[0].finish_reason == FINISH_REASON_LENGTH
    assert response.choices[0].message.tool_calls is None
    assert response._dropped_tool_names == ["write_file"]


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_valid_tool_args_with_finish_stop_still_dispatch(_mock_close, mock_create, monkeypatch):
    """Control: well-formed arguments with finish_reason="stop" must still be
    dispatched normally — the new guard must not over-trigger."""

    def _stream():
        yield _make_stream_chunk(tool_calls=[
            _tc_delta(0, "call_1", "write_file", '{"path": "/opt/data/CURRENT.md",')])
        yield _make_stream_chunk(tool_calls=[
            _tc_delta(0, None, None, ' "content": "# State"}')])
        yield _make_stream_chunk(finish_reason="stop")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = lambda *a, **kw: _stream()
    mock_create.return_value = mock_client

    agent = _make_agent()
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    response = agent._interruptible_streaming_api_call({})

    assert response.id != PARTIAL_STREAM_STUB_ID
    assert response.choices[0].message.tool_calls is not None
    assert response.choices[0].message.tool_calls[0].function.name == "write_file"
