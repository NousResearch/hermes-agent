"""delegation.busy_result_mode=steer: a finished subagent result rides the running turn's next tool
result instead of waiting for a new turn."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.background_results import BACKGROUND_RESULT_MARKER, offer, take_outcome
from tools.registry import registry

_PROBE_TOOL = "background_result_probe"
_RESULT_TEXT = "[ASYNC DELEGATION COMPLETE — deleg_x] reviewer verdict: BLOCKER in step 3"

registry.register(
    name=_PROBE_TOOL,
    toolset="utility",
    schema={
        "name": _PROBE_TOOL,
        "description": "probe tool for the background-result carrier test",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=lambda args, **_kw: "probe ok",
    override=True,
)


def _loop_agent():
    from run_agent import AIAgent

    tool_schema = {"type": "function", "function": {
        "name": _PROBE_TOOL, "description": "probe",
        "parameters": {"type": "object", "properties": {}, "required": []}}}
    with (
        patch("model_tools.get_tool_definitions", return_value=[tool_schema]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._disable_streaming = True
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _run(agent, model_call):
    agent._interruptible_api_call = model_call
    with (
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("start the work")


def _tool_call_response():
    from tests.agent.test_run_agent import _mock_response
    call = SimpleNamespace(id="call_1", type="function",
                           function=SimpleNamespace(name=_PROBE_TOOL, arguments="{}"))
    return _mock_response(content=None, finish_reason="tool_calls", tool_calls=[call])


def test_result_offered_mid_turn_reaches_the_next_request_on_the_tool_result():
    from tests.agent.test_run_agent import _mock_response

    agent = _loop_agent()
    payloads = []

    def model_call(api_kwargs):
        payloads.append([dict(m) for m in api_kwargs["messages"]])
        if len(payloads) == 1:
            # The subagent finishes while the model is thinking about its first tool call.
            offer(agent, "deleg_x", _RESULT_TEXT)
            return _tool_call_response()
        return _mock_response(content="adjusted plan", finish_reason="stop")

    result = _run(agent, model_call)

    second = payloads[1]
    tool_rows = [m for m in second if m.get("role") == "tool"]
    assert tool_rows and _RESULT_TEXT in str(tool_rows[-1]["content"])
    assert BACKGROUND_RESULT_MARKER in str(tool_rows[-1]["content"])
    # Tool-channel evidence, never a user-authority row and never a second user turn.
    assert not any(m.get("role") == "user" and _RESULT_TEXT in str(m.get("content")) for m in second)
    assert result["completed"] is True
    assert take_outcome(agent) == (["deleg_x"], [])


def test_a_tool_row_that_never_persisted_does_not_acknowledge_the_result(monkeypatch):
    """A settlement tick must never complete a claim for a result only RAM has seen.

    ``attach_to_tool_tail`` appends to the tool row in memory; the flush that persists that row can
    still fail (or the process can die before it runs). Staging until the flush confirms keeps the
    result deliverable instead of settling it behind an unwritten receipt."""
    from agent import conversation_loop, tool_executor, turn_iteration_prep
    from agent.background_results import attach_to_tool_tail, confirm_attached, take_delivered

    agent = SimpleNamespace()
    offer(agent, "deleg_race", _RESULT_TEXT)
    monkeypatch.setattr(conversation_loop, "_maybe_inject_run_budget_wrapup", lambda a, m: None)
    monkeypatch.setattr(turn_iteration_prep, "_maybe_inject_iteration_budget_warning", lambda a, m: None)

    failed = [{"role": "tool", "tool_call_id": "c1", "content": "tool output"}]
    agent._flush_messages_to_session_db = lambda messages: False
    assert tool_executor._flush_session_db_after_tool_progress(agent, failed, stage="probe") is False
    assert _RESULT_TEXT in failed[-1]["content"]  # the running turn still sees it
    assert take_delivered(agent) == []  # but nothing is acknowledged on a failed write
    assert take_outcome(agent) == ([], [("deleg_race", _RESULT_TEXT)])  # handed back, at-least-once

    # Positive control: the same attach acknowledges once the flush persisted the row.
    offer(agent, "deleg_ok", _RESULT_TEXT)
    persisted = [{"role": "tool", "tool_call_id": "c2", "content": "tool output"}]
    agent._flush_messages_to_session_db = lambda messages: None  # a real flush returns None on success
    assert tool_executor._flush_session_db_after_tool_progress(agent, persisted, stage="probe") is True
    assert take_delivered(agent) == ["deleg_ok"]
    assert take_outcome(agent) == ([], [])


def test_an_unconfirmed_attach_is_never_settled_by_a_watcher_tick():
    """The other half of the race: staged offers are invisible to the non-final settle path."""
    from agent.background_results import attach_to_tool_tail, confirm_attached, take_delivered

    agent = SimpleNamespace()
    offer(agent, "deleg_staged", _RESULT_TEXT)
    attached = attach_to_tool_tail(agent, [{"role": "tool", "tool_call_id": "c3", "content": "out"}])
    assert len(attached) == 1
    assert take_delivered(agent) == []  # a watcher tick here would otherwise settle the claim
    confirm_attached(agent, attached, True)
    assert take_delivered(agent) == ["deleg_staged"]

