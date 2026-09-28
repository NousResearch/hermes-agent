import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from agent.turn_resource_budget import TurnResourceBudget
from tools.budget_config import configured_max_tool_executions


def test_turn_resource_budget_is_atomic_across_parallel_calls():
    budget = TurnResourceBudget(8)
    barrier = threading.Barrier(32)

    def consume(index):
        barrier.wait(timeout=5)
        return budget.try_consume_tool_execution(
            tool_name="terminal", tool_call_id=f"call-{index}"
        ).allowed

    with ThreadPoolExecutor(max_workers=32) as pool:
        allowed = list(pool.map(consume, range(32)))

    assert sum(allowed) == 8
    assert budget.used_tool_executions == 8
    assert budget.exhausted is True


def test_tool_execution_config_defaults_to_existing_iteration_cap(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    assert configured_max_tool_executions(90) == 90

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"tool_budget": {"max_tool_executions": 17}},
    )
    assert configured_max_tool_executions(90) == 17

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"tool_budget": {"max_tool_executions": 0}},
    )
    assert configured_max_tool_executions(90) is None


def _executor_agent(budget):
    class Guardrail:
        @staticmethod
        def before_call(_name, _args):
            return SimpleNamespace(allows_execution=True)

    return SimpleNamespace(
        turn_resource_budget=budget,
        _tool_guardrails=Guardrail(),
        _turns_since_memory=0,
        _iters_since_skill=0,
        _emit_diagnostic_status=lambda _message: None,
    )


def test_dispatch_boundary_counts_only_actual_live_execution(monkeypatch):
    from agent import tool_executor

    budget = TurnResourceBudget(1)
    agent = _executor_agent(budget)
    posts = []
    executed = []

    monkeypatch.setattr(tool_executor, "_pre_tool_block", lambda _agent, ref: (None, ref.args))
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_a, **_k: None)
    monkeypatch.setattr(
        tool_executor, "_run_with_activity_heartbeat",
        lambda _agent, _name, fn: fn(),
    )
    monkeypatch.setattr(
        tool_executor._ToolCallRef,
        "emit_post",
        lambda self, _agent, result, **kwargs: posts.append((result, kwargs)),
    )
    monkeypatch.setattr(
        "agent.terminal_approval_batch.prepare_current_terminal",
        lambda _ref: None,
    )

    def dispatch(call_id):
        ref = tool_executor._ToolCallRef("terminal", {}, "task", call_id, [])
        state = tool_executor._ManagedToolResult(
            result=None, args={}, middleware_trace=[], blocked=False, dispatched=True
        )
        return tool_executor._dispatch_authorized_once(
            agent,
            state,
            ref,
            execute=lambda _args: executed.append(call_id) or "ok",
            scope_block=None,
            display_index=None,
            begin_execution=None,
            authorization_gate=None,
        )

    assert dispatch("one") == "ok"
    denied = json.loads(dispatch("two"))

    assert executed == ["one"]
    assert denied["code"] == "turn_tool_execution_budget_exhausted"
    assert denied["used_tool_executions"] == 1
    assert denied["max_tool_executions"] == 1
    assert budget.used_tool_executions == 1
    assert posts[-1][1]["status"] == "blocked"


def test_policy_block_does_not_consume_turn_resource_budget(monkeypatch):
    from agent import tool_executor

    budget = TurnResourceBudget(1)
    agent = _executor_agent(budget)
    monkeypatch.setattr(tool_executor._ToolCallRef, "emit_post", lambda *_a, **_k: None)

    ref = tool_executor._ToolCallRef("terminal", {}, "task", "blocked", [])
    state = tool_executor._ManagedToolResult(
        result=None, args={}, middleware_trace=[], blocked=False, dispatched=True
    )
    result = tool_executor._dispatch_authorized_once(
        agent,
        state,
        ref,
        execute=lambda _args: (_ for _ in ()).throw(AssertionError("must not dispatch")),
        scope_block="not allowed",
        display_index=None,
        begin_execution=None,
        authorization_gate=None,
    )

    assert json.loads(result) == {"error": "not allowed"}
    assert budget.used_tool_executions == 0


def test_dispatch_emits_one_near_limit_status(monkeypatch):
    from agent import tool_executor

    budget = TurnResourceBudget(5)
    agent = _executor_agent(budget)
    statuses = []
    agent._emit_diagnostic_status = statuses.append

    monkeypatch.setattr(tool_executor, "_pre_tool_block", lambda _agent, ref: (None, ref.args))
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_a, **_k: None)
    monkeypatch.setattr(
        tool_executor, "_run_with_activity_heartbeat",
        lambda _agent, _name, fn: fn(),
    )
    monkeypatch.setattr(tool_executor._ToolCallRef, "emit_post", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "agent.terminal_approval_batch.prepare_current_terminal",
        lambda _ref: None,
    )

    for index in range(4):
        ref = tool_executor._ToolCallRef("terminal", {}, "task", f"call-{index}", [])
        state = tool_executor._ManagedToolResult(
            result=None, args={}, middleware_trace=[], blocked=False, dispatched=True
        )
        assert tool_executor._dispatch_authorized_once(
            agent,
            state,
            ref,
            execute=lambda _args: "ok",
            scope_block=None,
            display_index=None,
            begin_execution=None,
            authorization_gate=None,
        ) == "ok"

    assert statuses == ["⚠️ Turn tool budget nearing limit (4/5 executions used)."]
