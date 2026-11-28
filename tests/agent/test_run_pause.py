"""AIAgent.pause() / resume(): suspend a running turn at its safe boundary.

A pause asked while a tool runs lets that tool finish, then blocks the turn before its next
model request; nothing in flight is aborted. These drive a real ``run_conversation`` on a
worker thread with a scripted model and a probe tool that requests the pause mid-call.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.agent.test_run_agent import _mock_response
from tests.waiting import wait_until
from tools.registry import registry

_PAUSE_TOOL = "run_pause_probe"
_ACTIVE: dict[str, object] = {}  # the agent under test, so the module-level handler can reach it
_SCHEMA = {
    "name": _PAUSE_TOOL,
    "description": "probe tool that pauses the run while it is executing",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _pause_probe(args, **_kwargs):
    pause = _ACTIVE["pause"]
    assert pause(), "pause() refused while the turn was running"
    return "probe ok"


registry.register(name=_PAUSE_TOOL, toolset="utility", schema=_SCHEMA, handler=_pause_probe, override=True)


def _loop_agent():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[{"type": "function", "function": _SCHEMA}]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="dummy", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent.client = MagicMock()
    agent._disable_streaming = True
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


class _Run:
    """One turn on a worker thread: first model call asks for the probe tool (which pauses the
    run), the next answers. ``payloads`` holds every request's messages."""

    def __init__(self, agent, **pause_kwargs):
        self.agent, self.payloads, self.events, self.result = agent, [], [], None
        _ACTIVE["pause"] = lambda: agent.pause(listener=self._listen, **pause_kwargs)
        agent._interruptible_api_call = self._model_call
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _listen(self, name, fields):
        self.events.append((name, fields))

    def _model_call(self, api_kwargs):
        self.payloads.append([dict(m) for m in api_kwargs["messages"]])
        if len(self.payloads) == 1:
            call = SimpleNamespace(id="call_1", type="function",
                                   function=SimpleNamespace(name=_PAUSE_TOOL, arguments="{}"))
            return _mock_response(content=None, finish_reason="tool_calls", tool_calls=[call])
        return _mock_response(content="done", finish_reason="stop")

    def _run(self):
        agent = self.agent
        with (
            patch.object(agent, "_flush_messages_to_session_db"), patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources"),
        ):
            self.result = agent.run_conversation("start something")

    def start(self):
        self._thread.start()
        wait_until(lambda: self.agent.pause_state() == "paused", what="the turn to block at its boundary")
        return self

    def join(self):
        self._thread.join(timeout=30)
        assert not self._thread.is_alive(), "turn never finished"
        return self.result


@pytest.fixture
def agent():
    yield _loop_agent()
    _ACTIVE.clear()


def _blob(payload) -> str:
    return "\n".join(str(m.get("content")) for m in payload if isinstance(m, dict))


def test_pause_blocks_after_the_tool_batch_and_resume_finishes_the_turn(agent):
    run = _Run(agent).start()
    # The in-flight tool finished (its result is persisted), and no next model request went out.
    assert len(run.payloads) == 1
    assert [name for name, _ in run.events] == ["run.paused"]
    assert agent.resume() is True
    result = run.join()
    assert result["completed"] is True and not result.get("interrupted")
    assert len(run.payloads) == 2
    assert "probe ok" in _blob(run.payloads[1])
    assert [name for name, _ in run.events] == ["run.paused", "run.resumed"]
    assert run.events[1][1]["previous_status"] == "paused"
    assert agent.pause_state() is None


def test_steer_while_paused_is_delivered_after_resume(agent):
    run = _Run(agent).start()
    assert agent.steer("STEERED_WHILE_PAUSED") is True
    assert len(run.payloads) == 1
    agent.resume()
    result = run.join()
    assert "STEERED_WHILE_PAUSED" in _blob(run.payloads[1])
    assert result.get("pending_steer") is None


def test_stop_while_paused_wakes_the_turn_and_ends_it(agent):
    from agent.interrupt_compat import request_hard_interrupt

    run = _Run(agent).start()
    assert request_hard_interrupt(agent, "Stop requested via API") is True
    result = run.join()
    assert result.get("interrupted") is True
    assert len(run.payloads) == 1  # no model request after the stop
    assert agent.pause_state() is None
    assert agent.resume() is False


def test_max_pause_guard_stops_a_forgotten_pause(agent):
    run = _Run(agent, max_seconds=0.3).start()
    result = run.join()
    assert result.get("interrupted") is True
    assert len(run.payloads) == 1
    assert "run_pause_timeout" in str(result.get("turn_exit_reason"))


def test_pause_and_resume_refuse_the_wrong_state(agent):
    assert agent.resume() is False  # nothing to resume
    run = _Run(agent).start()
    assert agent.pause() is False  # already paused
    agent.resume()
    run.join()
    assert agent.resume() is False


def test_a_pause_from_an_ended_turn_does_not_pause_the_next(agent):
    """A pause asked too late to reach a boundary is dropped with its turn: the gateway caches
    agents across turns, and the next message must not start paused."""
    agent._current_turn_id = "turn-that-already-ended"
    assert agent.pause() is True
    agent._interruptible_api_call = lambda api_kwargs: _mock_response(content="hi", finish_reason="stop")
    with (
        patch.object(agent, "_flush_messages_to_session_db"), patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("next message")
    assert result["completed"] is True
    assert agent.pause_state() is None
