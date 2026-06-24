import json
from types import SimpleNamespace

from agent.delegation_router_lock import (
    LOCK_MODEL_CONFIG_KEY,
    STATUS_ACTIVE,
    STATUS_VERIFICATION_REQUIRED,
    activate_after_delegate_result,
    clear_same_turn_lock,
    empty_state,
    mark_async_completion_from_text,
    prepare_same_turn_lock,
    restore_agent_state_from_session,
    should_block_tool,
)
from hermes_state import SessionDB


class _ToolFunction:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = json.dumps(arguments)


class _ToolCall:
    def __init__(self, name: str, arguments: dict):
        self.function = _ToolFunction(name, arguments)


def _agent(*, state=None, db=None, session_id="session-router-lock"):
    return SimpleNamespace(
        session_id=session_id,
        _session_db=db,
        _session_init_model_config={"preserve": "yes"},
        _delegation_router_lock_state=state if state is not None else empty_state(),
        _delegation_router_lock_same_turn_scopes=[],
    )


def _state_with_active_lock(delegation_id="deleg_ok"):
    return {
        "version": 1,
        "locks": [
            {
                "version": 1,
                "delegation_id": delegation_id,
                "status": STATUS_ACTIVE,
                "scope": {"match": "delegated_work"},
            }
        ],
    }


def test_successful_non_trivial_delegation_activates_and_persists_router_lock(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    session_id = "session-successful-delegation"
    db.create_session(session_id, "cli", model_config={"existing": "kept"})
    agent = _agent(db=db, session_id=session_id)

    activated = activate_after_delegate_result(
        agent,
        {"goal": "Implement the production fix", "context": "repo path"},
        json.dumps({"status": "dispatched", "delegation_id": "deleg_1234"}),
    )

    assert activated is True
    decision = should_block_tool(agent, "terminal", {"command": "touch should-not-run"})
    assert decision.blocks_execution is True
    assert decision.delegation_id == "deleg_1234"

    row = db.get_session(session_id)
    assert row is not None
    stored = json.loads(row["model_config"])
    assert stored["existing"] == "kept"
    locks = stored[LOCK_MODEL_CONFIG_KEY]["locks"]
    assert len(locks) == 1
    assert locks[0]["delegation_id"] == "deleg_1234"
    assert locks[0]["status"] == STATUS_ACTIVE

    fresh_agent = _agent(db=db, session_id=session_id, state=empty_state())
    restore_agent_state_from_session(fresh_agent)
    restored_decision = should_block_tool(fresh_agent, "write_file", {"path": str(tmp_path / "x")})
    assert restored_decision.blocks_execution is True
    assert restored_decision.delegation_id == "deleg_1234"


def test_failed_or_trivial_delegation_does_not_activate_lock(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    session_id = "session-failed-delegation"
    db.create_session(session_id, "cli", model_config={"existing": "kept"})
    agent = _agent(db=db, session_id=session_id)

    failed_activated = activate_after_delegate_result(
        agent,
        {"goal": "Implement the production fix"},
        json.dumps({"status": "rejected", "error": "pool full"}),
    )
    trivial_activated = activate_after_delegate_result(
        agent,
        {"goal": "   "},
        json.dumps({"status": "dispatched", "delegation_id": "deleg_nope"}),
    )

    assert failed_activated is False
    assert trivial_activated is False
    assert should_block_tool(agent, "terminal", {"command": "true"}).allows_execution is True
    row = db.get_session(session_id)
    assert row is not None
    stored = json.loads(row["model_config"])
    assert LOCK_MODEL_CONFIG_KEY not in stored


def test_same_turn_mixed_delegate_batch_blocks_implementation_before_execution():
    agent = _agent()
    prepare_same_turn_lock(
        agent,
        [
            _ToolCall("terminal", {"command": "touch should-not-run"}),
            _ToolCall("delegate_task", {"goal": "Implement the production fix"}),
        ],
    )

    terminal_decision = should_block_tool(agent, "terminal", {"command": "touch should-not-run"})
    delegate_decision = should_block_tool(agent, "delegate_task", {"goal": "Re-dispatch safely"})
    read_decision = should_block_tool(agent, "read_file", {"path": "run_agent.py"})

    assert terminal_decision.blocks_execution is True
    assert terminal_decision.delegation_id == "same_turn_delegate_task"
    assert delegate_decision.allows_execution is True
    assert read_decision.allows_execution is True

    clear_same_turn_lock(agent)
    assert should_block_tool(agent, "terminal", {"command": "true"}).allows_execution is True


def test_router_mode_allows_routing_monitoring_and_read_only_verification_but_blocks_implementation():
    agent = _agent(state=_state_with_active_lock())

    allowed_calls = [
        ("delegate_task", {"goal": "Route follow-up to child"}),
        ("clarify", {"question": "Which branch?"}),
        ("read_file", {"path": "run_agent.py"}),
        ("search_files", {"pattern": "delegate_task"}),
        ("process", {"action": "poll", "session_id": "p1"}),
        ("process", {"action": "kill", "session_id": "p1"}),
        ("kanban_show", {"id": "TASK-1"}),
    ]
    blocked_calls = [
        ("terminal", {"command": "touch should-not-run"}),
        ("write_file", {"path": "x", "content": "y"}),
        ("patch", {"mode": "replace", "path": "x", "old_string": "a", "new_string": "b"}),
        ("process", {"action": "submit", "session_id": "p1", "data": "yes"}),
        ("kanban_comment", {"id": "TASK-1", "comment": "mutates board"}),
        ("unknown_plugin_tool", {}),
    ]

    for tool_name, args in allowed_calls:
        assert should_block_tool(agent, tool_name, args).allows_execution is True, tool_name
    for tool_name, args in blocked_calls:
        decision = should_block_tool(agent, tool_name, args)
        assert decision.blocks_execution is True, tool_name
        assert decision.delegation_id == "deleg_ok"


def test_async_completion_moves_lock_to_verification_required_without_unlocking():
    agent = _agent(state=_state_with_active_lock("deleg_done"))

    changed = mark_async_completion_from_text(
        agent,
        "[ASYNC DELEGATION COMPLETE — deleg_done]\nThe child returned evidence.",
    )

    assert changed is True
    decision = should_block_tool(agent, "terminal", {"command": "touch should-not-run"})
    assert decision.blocks_execution is True
    assert decision.status == STATUS_VERIFICATION_REQUIRED
    assert agent._delegation_router_lock_state["locks"][0]["status"] == STATUS_VERIFICATION_REQUIRED
    assert agent._delegation_router_lock_state["locks"][0]["completed_at"] is not None


def test_invoke_tool_returns_synthetic_block_before_terminal_dispatch(tmp_path):
    from agent.agent_runtime_helpers import invoke_tool

    marker = tmp_path / "should-not-exist"
    agent = _agent(state=_state_with_active_lock("deleg_block"))

    result = invoke_tool(
        agent,
        "terminal",
        {"command": f"touch {marker}"},
        "task-router-lock",
    )

    payload = json.loads(result)
    assert payload["status"] == "blocked"
    assert payload["code"] == "delegation_router_lock"
    assert payload["delegation_id"] == "deleg_block"
    assert marker.exists() is False
