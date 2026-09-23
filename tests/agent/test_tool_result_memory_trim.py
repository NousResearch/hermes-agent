"""A >=1 MB tool result hands allocator pages back via ``trim_memory`` — after the batch (#70684).

Compaction already trims after it frees the compressed-away messages; a huge tool result
(raw stdout, file dumps) is the other allocation a turn drops. The commit point only flags
it, because the string is still referenced by the publish frames there; the trim runs once
``AIAgent._execute_tool_calls`` has unwound every executor frame, so ``gc.collect`` +
``malloc_trim`` actually see the allocation as garbage.
"""

from unittest.mock import MagicMock

from tests.agent.test_start_order_gate import (  # noqa: F401 — autouse fixture rides along
    _FakeAssistantMsg,
    _FakeToolCall,
    _isolate_hermes,
    _make_agent,
)


def _agent_returning(monkeypatch, payload):
    agent = _make_agent(monkeypatch)
    agent._trim_after_tool_batch = False  # _set_defaults (agent_init._CONTROL_STATE) does this on a real AIAgent
    agent._tool_guardrails = MagicMock()
    agent._tool_guardrails.before_call = lambda name, args: MagicMock(allows_execution=True)
    agent._invoke_tool = MagicMock(return_value=payload)
    agent._append_guardrail_observation = lambda name, args, result, *a, **kw: result
    return agent


def _trim_recorder(monkeypatch, agent):
    seen = []

    def trim(*, reason):
        seen.append((reason, agent._executing_tools))
        return True

    monkeypatch.setattr("hermes_cli.mem_trim.trim_memory", trim)
    return seen




def test_execute_tool_calls_trims_once_after_every_executor_frame_unwound(monkeypatch):
    import run_agent as _ra

    agent = _agent_returning(monkeypatch, "x" * 1_000_000)
    agent._execute_tool_calls = _ra.AIAgent._execute_tool_calls.__get__(agent)
    # stand-in for the sequential executor: publishes through the real concurrent commit path
    agent._execute_tool_calls_sequential = agent._execute_tool_calls_concurrent
    seen = _trim_recorder(monkeypatch, agent)

    messages: list = []
    agent._execute_tool_calls(_FakeAssistantMsg([_FakeToolCall("terminal", "tc_big")]), messages, "task")

    assert [m["role"] for m in messages] == ["tool"]
    # one trim per batch, issued only after the tool-execution scope closed (frames holding the 1 MB str are gone)
    assert seen == [("large tool result", False)]
    assert agent._trim_after_tool_batch is False
