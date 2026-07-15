"""Regression tests for the live-/steer stream break.

A /steer that arrives while the model is generating a text-only final
answer has no tool result to piggyback on, so it used to sit pending
until the whole answer finished and only then ran as a queued follow-up
turn ("responds, then keeps going"). The streaming loop now breaks at
the steer point once enough of the answer has accumulated: the partial
answer commits normally and the leftover steer is delivered as the
immediate next turn via result["pending_steer"] (see
turn_finalizer.finalize_turn -> gateway/run.py and tui_gateway/server.py
for the caller-side delivery).

Pins the contract:

- pending steer + text-only stream past the 120-char floor → the stream
  ends with finish_reason="stop" and the tail is never consumed;
- pending steer + short answer → no break, the answer completes intact
  (the steer lands through the leftover path instead);
- pending steer + tool-call generation → never breaks, the injection
  path owns delivery;
- no steer → streaming is untouched.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers (mirrors test_partial_stream_finish_reason.py) ────────────────

def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content, tool_calls=tool_calls,
        reasoning_content=None, reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=None, usage=None)


def _make_tool_call_delta(index=0, tc_id=None, name=None, arguments=None):
    func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=func)


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


def _run_stream(agent, chunks):
    consumed = {"n": 0}

    def _stream():
        for c in chunks:
            consumed["n"] += 1
            yield c

    with patch("run_agent.AIAgent._create_request_openai_client") as mock_create, \
            patch("run_agent.AIAgent._close_request_openai_client"):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _stream()
        mock_create.return_value = mock_client
        response = agent._interruptible_streaming_api_call({})
    return response, consumed["n"]


class TestSteerStreamBreak:
    def test_pending_steer_breaks_text_stream_past_floor(self):
        """Once ≥120 chars of answer have streamed, a pending steer ends the
        stream: the partial answer keeps finish_reason=stop and the tail is
        never pulled from the wire."""
        agent = _make_agent()
        agent.steer("stop the essay and check the logs instead")

        chunks = [
            _make_stream_chunk(content="x" * 50),
            _make_stream_chunk(content="y" * 50),
            _make_stream_chunk(content="z" * 50),   # crosses the 120 floor
            _make_stream_chunk(content="TAIL_MARKER"),
            _make_stream_chunk(content=None, finish_reason="stop"),
        ]
        response, consumed = _run_stream(agent, chunks)

        content = response.choices[0].message.content
        assert "TAIL_MARKER" not in content
        assert content == "x" * 50 + "y" * 50 + "z" * 50
        assert response.choices[0].finish_reason == "stop"
        # The break happens at the top of the loop: the marker chunk is the
        # last one pulled, the stop chunk is never consumed.
        assert consumed == 4
        # The steer itself is NOT consumed here — the turn finalizer drains it
        # into result["pending_steer"] for immediate redelivery.
        assert agent._pending_steer == "stop the essay and check the logs instead"

    def test_pending_steer_below_floor_lets_short_answer_finish(self):
        """A short answer never reaches the floor: it completes intact and
        the steer waits for the leftover-delivery path — breaking on the
        first tokens committed bare sentence fragments as real messages."""
        agent = _make_agent()
        agent.steer("and then check the logs")

        chunks = [
            _make_stream_chunk(content="Sure — done."),
            _make_stream_chunk(content=" Anything else?"),
            _make_stream_chunk(content=None, finish_reason="stop"),
        ]
        response, consumed = _run_stream(agent, chunks)

        assert response.choices[0].message.content == "Sure — done. Anything else?"
        assert consumed == len(chunks)

    def test_pending_steer_never_breaks_tool_call_generation(self):
        """Tool-call turns are owned by the injection path: once tool-call
        deltas accumulate, the stream must run to completion so the call
        arrives whole."""
        agent = _make_agent()
        agent.steer("prefer the staging database")

        chunks = [
            _make_stream_chunk(content="Let me check."),
            _make_stream_chunk(tool_calls=[_make_tool_call_delta(
                index=0, tc_id="call_1", name="terminal", arguments='{"comman')]),
            _make_stream_chunk(tool_calls=[_make_tool_call_delta(
                index=0, arguments='d": "ls"}')]),
            _make_stream_chunk(content=None, finish_reason="tool_calls"),
        ]
        response, consumed = _run_stream(agent, chunks)

        assert consumed == len(chunks)
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls and tool_calls[0].function.name == "terminal"
        assert tool_calls[0].function.arguments == '{"command": "ls"}'

    def test_no_steer_streams_untouched(self):
        agent = _make_agent()

        chunks = [
            _make_stream_chunk(content="x" * 200),
            _make_stream_chunk(content="TAIL_MARKER"),
            _make_stream_chunk(content=None, finish_reason="stop"),
        ]
        response, consumed = _run_stream(agent, chunks)

        assert consumed == len(chunks)
        assert response.choices[0].message.content.endswith("TAIL_MARKER")
