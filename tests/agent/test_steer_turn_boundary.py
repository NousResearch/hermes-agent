"""A /steer never lands out of order at a turn boundary.

The steer row is appended to the live transcript and flushed append-only, so the only position
where it may appear in a request is where state.db will also hold it: after a tool-result tail.
A steer left pending when a turn ends (accepted after the last drain, or on an exit that skips
finalize_turn) is handed back as ``pending_steer`` instead of being spliced into history the
next turn already replays.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.prompt_builder import STEER_MARKER_OPEN
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_run_agent import _mock_response
from tools.registry import registry

_TOOL = "steer_turn_boundary_probe"
_TOOL_SCHEMA = {
    "name": _TOOL,
    "description": "probe tool for the steer turn-boundary tests",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
registry.register(
    name=_TOOL, toolset="utility", schema=_TOOL_SCHEMA,
    handler=lambda args, **_kw: "probe ok", override=True,
)

STEER = "STEER_NOTE_TEXT"


def _agent(db: SessionDB) -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[{"type": "function", "function": _TOOL_SCHEMA}]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, session_db=db, session_id="steer-boundary",
        )
    agent.client = MagicMock()
    agent._disable_streaming = True
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _tool_call_response():
    call = SimpleNamespace(id="call_1", type="function", function=SimpleNamespace(name=_TOOL, arguments="{}"))
    return _mock_response(content=None, finish_reason="tool_calls", tool_calls=[call])


def _shape(rows):
    return [
        (m.get("role"), STEER_MARKER_OPEN in str(m.get("content") or ""))
        for m in rows if m.get("role") != "system"
    ]


def _late_steer_after_turn_end(agent, payloads):
    replies = iter([_tool_call_response(), _mock_response(content="done one"), _mock_response(content="done two")])
    agent._interruptible_api_call = lambda kw: payloads.append([dict(m) for m in kw["messages"]]) or next(replies)
    first = agent.run_conversation("first task")
    accepted = agent.steer(STEER)  # sent after the turn's final drain, while the surface still showed it busy
    return agent.run_conversation("second question", conversation_history=first["messages"]), accepted


def _steer_then_redirect_during_call(agent, payloads):
    accepted = []

    def call(kw):
        payloads.append([dict(m) for m in kw["messages"]])
        if len(payloads) == 1:
            return _tool_call_response()
        if len(payloads) == 2:
            accepted.append(agent.steer(STEER))
            assert agent.redirect("REDIRECT_TEXT")
            raise InterruptedError("redirect cancelled the in-flight request")
        return _mock_response(content="rebuilt reply")

    agent._interruptible_api_call = call
    return agent.run_conversation("start something"), accepted == [True]


@pytest.mark.parametrize("scenario", [_late_steer_after_turn_end, _steer_then_redirect_during_call])
def test_live_request_is_a_prefix_of_the_durable_transcript(tmp_path, scenario):
    db = SessionDB(tmp_path / "state.db")
    try:
        agent, payloads = _agent(db), []
        result, accepted = scenario(agent, payloads)
        request = _shape(payloads[-1])
        durable = _shape(db.get_messages("steer-boundary"))
        assert durable[: len(request)] == request
        # An accepted steer is delivered exactly once (in the request, or handed back as the next
        # user turn); a rejected one is left to the surface, which queues it as a normal message.
        in_request = sum(steer for _role, steer in request)
        assert in_request + (STEER in (result.get("pending_steer") or "")) == int(accepted)
    finally:
        db.close()


def test_turn_end_hands_back_the_steer_and_closes_acceptance(tmp_path):
    """Thinking-exhausted truncation returns without finalize_turn; the steer accepted during that
    call must come back as ``pending_steer``. The same drain closes acceptance: a steer sent after it
    is rejected (the surface queues it) rather than acknowledged and stranded, until the next turn."""
    db = SessionDB(tmp_path / "state.db")
    try:
        agent = _agent(db)

        def call(kw):
            if kw["messages"][-1].get("role") != "tool":
                return _tool_call_response()
            agent.steer(STEER)
            return _mock_response(content="<think>" + "reasoning " * 40 + "</think>", finish_reason="length")

        agent._interruptible_api_call = call
        result = agent.run_conversation("first task")
        assert result["completed"] is False
        assert result.get("pending_steer") == STEER
        assert agent.steer("sent after the turn ended") is False

        accepted = []
        agent._interruptible_api_call = lambda kw: accepted.append(agent.steer("NEXT_TURN_STEER")) or _mock_response(
            content="second answer")
        second = agent.run_conversation("second question", conversation_history=result["messages"])
        assert accepted == [True]
        assert second.get("pending_steer") == "NEXT_TURN_STEER"
    finally:
        db.close()
