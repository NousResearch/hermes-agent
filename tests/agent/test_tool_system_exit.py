"""A tool that raises ``SystemExit`` fails as a tool; the turn goes on.

Plugin tools dispatch through ``ToolRegistry.dispatch``, memory-provider tools through
``MemoryManager.handle_tool_call`` and context-engine tools through the sequential executor's
error boundary. All three caught only ``Exception``, so a handler (or a library it calls)
running ``sys.exit()`` unwound ``AIAgent.run_conversation`` before the model saw a tool result;
on the gateway the same exception stopped the event loop for every session. In a parallel-safe
batch the turn survived but the model got "thread did not return a result" instead of the
error. ``KeyboardInterrupt`` is still the user's Ctrl+C and keeps propagating.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from tests.agent.test_run_agent import _mock_response
from tools.registry import registry

_EXITS, _OK, _ENGINE = "system_exit_probe", "system_exit_peer", "system_exit_engine_probe"
_EXIT_TEXT = "plugin dependency called sys.exit(): auth failed"


def _exits(args, **_kw):
    raise SystemExit(_EXIT_TEXT)


def _ok(args, **_kw):
    return json.dumps({"ok": True})


@pytest.fixture(autouse=True)
def _probe_tools():
    for name, handler in ((_EXITS, _exits), (_OK, _ok)):
        registry.register(
            name=name, toolset="utility", handler=handler, override=True,
            schema={"name": name, "description": "probe", "parameters": {"type": "object", "properties": {}}})
    yield
    for name in (_EXITS, _OK):
        registry.deregister(name)


def _loop_agent():
    from run_agent import AIAgent
    schemas = [{"type": "function", "function": {
        "name": name, "description": "probe", "parameters": {"type": "object", "properties": {}}}}
        for name in (_EXITS, _OK, _ENGINE)]
    with (patch("model_tools.get_tool_definitions", return_value=schemas),
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


def _run_turn(agent, names):
    """One tool round calling ``names``, then a final answer; returns (result, second request)."""
    requests = []

    def model_call(api_kwargs):
        requests.append(api_kwargs)
        if len(requests) == 1:
            calls = [SimpleNamespace(id=f"c{i}", type="function", function=SimpleNamespace(name=n, arguments="{}"))
                     for i, n in enumerate(names)]
            return _mock_response(content=None, finish_reason="tool_calls", tool_calls=calls)
        return _mock_response(content="reported the failure", finish_reason="stop")

    agent._interruptible_api_call = model_call
    with (patch.object(agent, "_flush_messages_to_session_db"), patch.object(agent, "_persist_session"),
          patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources")):
        result = agent.run_conversation("go")
    return result, requests[1]


def _tool_result(request, call_id):
    return next(str(m["content"]) for m in request["messages"]
                if m.get("role") == "tool" and m.get("tool_call_id") == call_id)


@pytest.mark.parametrize("parallel", [False, True], ids=["single_call", "parallel_batch"])
def test_turn_survives_a_registry_tool_that_raises_system_exit(parallel):
    agent = _loop_agent()
    names = [_EXITS, _OK] if parallel else [_EXITS]
    # Probe tools are not parallel-safe by default; mark them so the batch really runs concurrently.
    parallel_safe = patch("agent.tool_dispatch_helpers._PARALLEL_SAFE_TOOLS", frozenset(names)) if parallel else nullcontext()
    with parallel_safe:
        result, request = _run_turn(agent, names)
    assert result["completed"] is True and result["final_response"] == "reported the failure"
    failed = _tool_result(request, "c0")
    assert "SystemExit" in failed and "auth failed" in failed
    if parallel:
        assert json.loads(_tool_result(request, "c1")) == {"ok": True}


def test_turn_survives_a_context_engine_tool_that_raises_system_exit():
    agent = _loop_agent()
    agent._context_engine_tool_names = {_ENGINE}
    with patch.object(agent.context_compressor, "handle_tool_call", side_effect=SystemExit(_EXIT_TEXT)):
        result, request = _run_turn(agent, [_ENGINE])
    assert result["completed"] is True and result["final_response"] == "reported the failure"
    assert "auth failed" in _tool_result(request, "c0")


class _Provider(MemoryProvider):
    def __init__(self, exc: BaseException):
        self._exc = exc

    @property
    def name(self) -> str:
        return "external"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return [{"name": "recall", "description": "recall", "parameters": {}}]

    def handle_tool_call(self, tool_name, args, **kwargs):
        raise self._exc


def test_memory_provider_tool_system_exit_is_a_tool_error():
    mgr = MemoryManager()
    mgr.add_provider(_Provider(SystemExit("provider auth failed")))
    result = json.loads(mgr.handle_tool_call("recall", {}))
    assert "recall" in result["error"] and "provider auth failed" in result["error"]


def test_memory_provider_tool_keyboard_interrupt_still_propagates():
    mgr = MemoryManager()
    mgr.add_provider(_Provider(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        mgr.handle_tool_call("recall", {})
