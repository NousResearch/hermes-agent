"""Publishing a >=1 MB tool result hands allocator pages back via ``trim_memory`` (#70684).

Compaction already trims after it frees the compressed-away messages; a huge tool
result (raw stdout, file dumps) is the other allocation a turn drops, and both publish
paths (sequential and concurrent) commit through the same point.
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
    agent._tool_guardrails = MagicMock()
    agent._tool_guardrails.before_call = lambda name, args: MagicMock(allows_execution=True)
    agent._invoke_tool = MagicMock(return_value=payload)
    agent._append_guardrail_observation = lambda name, args, result, *a, **kw: result
    return agent


def test_large_sequential_result_trims_memory_once(monkeypatch):
    import agent.tool_executor as te

    trim = MagicMock(return_value=True)
    monkeypatch.setattr(te, "trim_memory", trim)
    agent = _agent_returning(monkeypatch, "x" * 1_000_000)

    messages: list = []
    ref = te._ToolCallRef("terminal", {"command": "cat big.log"}, "task", "tc_big", [])
    managed = te._ManagedToolResult("x" * 1_000_000, ref.args, [], blocked=False, dispatched=True)
    assert te._publish_sequential_result(
        agent, messages, ref, managed, tool_duration=0.1, index=1, budget=te.DEFAULT_BUDGET,
    )

    assert [m["role"] for m in messages] == ["tool"]
    trim.assert_called_once_with(reason="large tool result")


def test_small_concurrent_result_does_not_trim(monkeypatch):
    import agent.tool_executor as te

    trim = MagicMock(return_value=True)
    monkeypatch.setattr(te, "trim_memory", trim)
    agent = _agent_returning(monkeypatch, "x" * 999_999)

    messages: list = []
    agent._execute_tool_calls_concurrent(_FakeAssistantMsg([_FakeToolCall("terminal", "tc_small")]), messages, "task")

    assert [m["role"] for m in messages] == ["tool"]
    trim.assert_not_called()
