"""Central pre-dispatch enforcement for typed Goal tool constraints."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.goals import GoalContract, GoalManager, GoalToolConstraints
from model_tools import handle_function_call


class _Registry:
    def __init__(self, capabilities=("read",)):
        self.entry = SimpleNamespace(capabilities=capabilities)
        self.calls = []

    def get_entry(self, name):
        return self.entry

    def dispatch(self, name, args, **kwargs):
        self.calls.append((name, args, kwargs))
        return json.dumps({"result": "executed"})


def _set_goal(session_id: str, **constraints) -> GoalManager:
    manager = GoalManager(session_id)
    manager.set(
        "Exercise typed constraints",
        contract=GoalContract(tool_constraints=GoalToolConstraints(**constraints)),
    )
    return manager


def _call(registry, session_id, name="goal_test_tool", args=None, **kwargs):
    with patch("model_tools.registry", registry):
        return json.loads(
            handle_function_call(
                name,
                args or {},
                session_id=session_id,
                task_id=f"task-{session_id}",
                **kwargs,
            )
        )


def test_allowed_read_tool_executes_when_admitted():
    session_id = "goal-constraints-allowed"
    _set_goal(session_id, allowed_tools=["goal_test_tool"])
    registry = _Registry(("read",))

    result = _call(registry, session_id)

    assert result["result"] == "executed"
    assert len(registry.calls) == 1
    assert GoalManager(session_id).state.status == "active"


def test_disallowed_tool_is_blocked_before_handler():
    session_id = "goal-constraints-disallowed"
    _set_goal(session_id, allowed_tools=["some_other_tool"])
    registry = _Registry(("read",))

    result = _call(registry, session_id)

    assert "error" in result
    assert "Goal tool constraint" in result["error"]
    assert registry.calls == []
    assert GoalManager(session_id).state.status == "active"


def test_irreversible_denial_persists_blocked_pause():
    session_id = "goal-constraints-delete"
    manager = _set_goal(session_id, denied_capabilities=["delete"])
    manager.wait_for_seconds(60, "pending work")
    registry = _Registry(("delete",))

    result = _call(registry, session_id)

    assert "error" in result
    assert registry.calls == []
    reloaded = GoalManager(session_id).state
    assert reloaded.status == "paused"
    assert reloaded.last_verdict == "blocked"
    assert reloaded.last_reason and "goal_test_tool" in reloaded.last_reason
    assert reloaded.paused_reason == reloaded.last_reason
    assert reloaded.waiting_until == 0.0
    assert reloaded.waiting_reason is None


def test_reversible_read_denial_blocks_without_pausing():
    session_id = "goal-constraints-read-denied"
    _set_goal(session_id, denied_capabilities=["read"])
    registry = _Registry(("read",))

    result = _call(registry, session_id)

    assert "error" in result
    assert registry.calls == []
    reloaded = GoalManager(session_id).state
    assert reloaded.status == "active"
    assert reloaded.last_verdict is None


def test_unknown_capability_fails_closed_and_pauses():
    session_id = "goal-constraints-unknown"
    _set_goal(session_id, allowed_tools=["goal_test_tool"])
    registry = _Registry(("unknown",))

    result = _call(registry, session_id)

    assert "error" in result
    assert registry.calls == []
    reloaded = GoalManager(session_id).state
    assert reloaded.status == "paused"
    assert reloaded.last_verdict == "blocked"


def test_missing_registry_entry_fails_closed_and_pauses():
    session_id = "goal-constraints-missing-entry"
    _set_goal(session_id, allowed_tools=["goal_test_tool"])
    registry = _Registry(("read",))
    registry.entry = None

    result = _call(registry, session_id)

    assert "error" in result
    assert registry.calls == []
    assert GoalManager(session_id).state.status == "paused"


@pytest.mark.parametrize("capabilities", [("bogus",), (123,), ()])
def test_malformed_capability_metadata_fails_closed(capabilities):
    session_id = f"goal-constraints-malformed-{len(str(capabilities))}"
    _set_goal(session_id, allowed_tools=["goal_test_tool"])
    registry = _Registry(capabilities)

    result = _call(registry, session_id)

    assert "error" in result
    assert registry.calls == []
    assert GoalManager(session_id).state.status == "paused"


def test_registry_lookup_error_fails_closed():
    session_id = "goal-constraints-registry-error"
    _set_goal(session_id, allowed_tools=["goal_test_tool"])
    registry = _Registry(("read",))
    registry.get_entry = lambda _name: (_ for _ in ()).throw(RuntimeError("lookup failed"))

    result = _call(registry, session_id)

    assert "error" in result
    assert registry.calls == []
    assert GoalManager(session_id).state.status == "paused"


@pytest.mark.parametrize(
    "args",
    [
        {"path": "C:/allowed/file.txt"},
        {"urls": ["https://allowed.example/a", "https://allowed.example/b"]},
    ],
)
def test_target_prefix_accepts_scalar_and_list_targets(args):
    session_id = f"goal-constraints-target-ok-{len(str(args))}"
    prefix = "C:/allowed/" if "path" in args else "https://allowed.example/"
    _set_goal(session_id, target_prefixes=[prefix])
    registry = _Registry(("read",))

    result = _call(registry, session_id, args=args)

    assert result["result"] == "executed"
    assert len(registry.calls) == 1


@pytest.mark.parametrize(
    "args",
    [
        {"path": "C:/outside/file.txt"},
        {"query": "has no target"},
    ],
)
def test_out_of_scope_or_missing_target_fails_closed(args):
    session_id = f"goal-constraints-target-bad-{len(str(args))}"
    _set_goal(session_id, target_prefixes=["C:/allowed/"])
    registry = _Registry(("read",))

    result = _call(registry, session_id, args=args)

    assert "error" in result
    assert registry.calls == []


def test_plugin_modified_arguments_are_enforced():
    session_id = "goal-constraints-plugin-modified"
    _set_goal(session_id, target_prefixes=["C:/allowed/"])
    registry = _Registry(("read",))

    with (
        patch("model_tools.registry", registry),
        patch(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            return_value=(None, {"path": "C:/outside/changed.txt"}),
        ),
    ):
        result = json.loads(
            handle_function_call(
                "goal_test_tool",
                {"path": "C:/allowed/original.txt"},
                session_id=session_id,
                task_id="task-plugin-modified",
            )
        )

    assert "error" in result
    assert registry.calls == []


def test_skip_pre_tool_hook_does_not_bypass_constraints():
    session_id = "goal-constraints-skip-hook"
    _set_goal(session_id, denied_capabilities=["read"])
    registry = _Registry(("read",))

    result = _call(registry, session_id, skip_pre_tool_call_hook=True)

    assert "error" in result
    assert registry.calls == []


def test_skip_execution_middleware_does_not_bypass_constraints():
    session_id = "goal-constraints-skip-middleware"
    _set_goal(session_id, denied_capabilities=["read"])
    registry = _Registry(("read",))

    result = _call(registry, session_id, skip_tool_execution_middleware=True)

    assert "error" in result
    assert registry.calls == []


@pytest.mark.parametrize("mode", ["no-goal", "empty-constraints"])
def test_no_goal_or_empty_constraints_preserves_dispatch(mode):
    session_id = f"goal-constraints-preserve-{mode}"
    if mode == "empty-constraints":
        _set_goal(session_id)
    registry = _Registry(("unknown",))

    result = _call(registry, session_id)

    assert result["result"] == "executed"
    assert len(registry.calls) == 1
