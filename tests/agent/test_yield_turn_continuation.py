"""The originating turn continues after a foreground command yields to the background.

#144499 reported a gateway turn that only resumed once a later user message arrived. The yield
itself is caused by the message that redirects the turn: ``redirect()`` during tool execution
queues the steer and asks the live tool worker to yield, the terminal backend adopts the
still-running command, and the tool result comes back at once. What must then hold is that the
turn keeps going on its own — the steer rides the same turn's next model call, and no further
user input is needed.

The probe tool below stands in for the foreground terminal wait at its precise seam: it blocks
on ``consume_yield`` for the worker tid the executor registered, then returns the
``yielded_to_background`` result the terminal tool returns. Everything else is the real
executor, batch finalizer, steer injection and conversation loop.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from tests.agent.test_run_agent import _mock_response
from tools import interrupt as interrupt_mod
from tools.registry import registry

PROBE_TOOL = "yield_resume_probe"
YIELD_NOTE = "user sent a message while this command was running"

_started = threading.Event()
_yielded = threading.Event()


def _yield_resume_probe(_args, **_kwargs):
    """Block like the foreground terminal wait and yield when the redirect asks us to."""
    tid = threading.current_thread().ident
    _started.set()
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if interrupt_mod.consume_yield(tid):
            _yielded.set()
            return json.dumps({
                "output": "started\n", "exit_code": None, "error": None,
                "status": "yielded_to_background", "session_id": "proc_probe", "pid": 4242,
                "notify_on_complete": True, "note": YIELD_NOTE,
            })
        time.sleep(0.02)
    return json.dumps({"output": "the redirect never asked this tool to yield", "exit_code": -1})


registry.register(
    name=PROBE_TOOL,
    toolset="utility",
    schema={
        "name": PROBE_TOOL,
        "description": "probe tool standing in for a yielded foreground terminal command",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=_yield_resume_probe,
    override=True,
)


def _loop_agent() -> AIAgent:
    tool_schema = {
        "type": "function",
        "function": {
            "name": PROBE_TOOL,
            "description": "probe tool standing in for a yielded foreground terminal command",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    with (
        patch("model_tools.get_tool_definitions", return_value=[tool_schema]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._disable_streaming = True
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def test_a_redirected_foreground_command_yields_and_the_turn_continues_without_another_message():
    agent = _loop_agent()
    payloads = []
    result = {}

    def model_call(api_kwargs):
        payloads.append([dict(m) for m in api_kwargs["messages"]])
        if len(payloads) == 1:
            tool_call = SimpleNamespace(
                id="call_1", type="function",
                function=SimpleNamespace(name=PROBE_TOOL, arguments="{}"),
            )
            return _mock_response(content=None, finish_reason="tool_calls", tool_calls=[tool_call])
        return _mock_response(content="resumed after the yield", finish_reason="stop")

    agent._interruptible_api_call = model_call

    def turn():
        try:
            result["value"] = agent.run_conversation("run the long thing")
        except Exception as exc:  # surfaced below, never swallowed
            result["error"] = exc

    _started.clear()
    _yielded.clear()
    worker = threading.Thread(target=turn, daemon=True)
    worker.start()
    try:
        assert _started.wait(20), "the foreground tool never started"
        assert agent.redirect("继续执行") is True
        worker.join(60)
        assert not worker.is_alive(), "the turn never resumed after the yielded tool result"
        assert "error" not in result, result.get("error")
        assert _yielded.is_set(), "the yield request never reached the tool worker"

        run = result["value"]
        assert run["completed"] is True, run
        # One call to request the tool, one continuation after its yielded result: no follow-up
        # user message was needed to wake the turn.
        assert run["api_calls"] == 2, run
        assert len(payloads) == 2, "the turn must make its next model call on its own"

        continuation = payloads[1]
        tool_rows = [m for m in continuation if m.get("role") == "tool"]
        assert json.loads(tool_rows[-1]["content"])["status"] == "yielded_to_background"
        steer_rows = [
            m for m in continuation
            if m.get("role") == "user" and "继续执行" in str(m.get("content"))
        ]
        assert len(steer_rows) == 1, "the redirect's text must ride this turn exactly once"
        # The first request predates the redirect and must not already carry it.
        assert not any("继续执行" in str(m.get("content")) for m in payloads[0])
        # Consumed, not left for a separate follow-up turn.
        assert run.get("pending_steer") is None
    finally:
        if worker.is_alive():
            agent.interrupt("test cleanup")
            worker.join(20)
        agent.close()
