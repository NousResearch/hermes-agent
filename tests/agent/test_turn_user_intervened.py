"""A turn reports ``user_intervened`` when the user steered, redirected or interrupted it, even when
the correction was fully absorbed and the turn still ``completed`` (first-task-done detection)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.agent.test_run_agent import _mock_response
from tools.registry import registry

_PROBE = "intervention_probe"
_DURING_TOOL: list = []  # callables run inside the tool, i.e. while the turn is executing tools


def _probe(args, **_kw):
    for fn in _DURING_TOOL:
        fn()
    return "probe ok"


registry.register(
    name=_PROBE, toolset="utility",
    schema={"name": _PROBE, "description": "probe", "parameters": {"type": "object", "properties": {}}},
    handler=_probe, override=True)


@pytest.fixture(autouse=True)
def _clear_tool_hooks():
    _DURING_TOOL.clear()
    yield
    _DURING_TOOL.clear()


def _loop_agent():
    from run_agent import AIAgent
    schema = {"type": "function", "function": {
        "name": _PROBE, "description": "probe", "parameters": {"type": "object", "properties": {}}}}
    with (patch("model_tools.get_tool_definitions", return_value=[schema]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
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


def _run(agent, model_call, text="do the task"):
    agent._interruptible_api_call = model_call
    with (patch.object(agent, "_flush_messages_to_session_db"), patch.object(agent, "_persist_session"),
          patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources")):
        return agent.run_conversation(text)


def _tool_then_reply(agent, intervene):
    calls = []

    def model_call(api_kwargs):
        calls.append(1)
        if len(calls) == 1:
            call = SimpleNamespace(id="c1", type="function", function=SimpleNamespace(name=_PROBE, arguments="{}"))
            return _mock_response(content=None, finish_reason="tool_calls", tool_calls=[call])
        if len(calls) == 2 and intervene == "redirect":
            assert agent.redirect("actually, do it differently") is True
            raise InterruptedError("redirect cancelled the in-flight request")
        return _mock_response(content="all done", finish_reason="stop")

    if intervene == "steer":
        # Lands while the tool runs; the steer drain absorbs it fully before the next request.
        _DURING_TOOL.append(lambda: agent.steer("also mention the weather"))
    return model_call


@pytest.mark.parametrize("intervene", ["steer", "redirect"])
def test_absorbed_correction_still_marks_the_completed_turn(intervene):
    agent = _loop_agent()
    result = _run(agent, _tool_then_reply(agent, intervene))
    assert result["completed"] is True and not result.get("pending_steer")
    assert result["user_intervened"] is True


def test_next_untouched_turn_starts_clean():
    agent = _loop_agent()
    assert _run(agent, _tool_then_reply(agent, "redirect"))["user_intervened"] is True
    result = _run(agent, lambda api_kwargs: _mock_response(content="hi", finish_reason="stop"), "hello")
    assert result["completed"] is True
    assert result["user_intervened"] is False
