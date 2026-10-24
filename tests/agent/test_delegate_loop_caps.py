"""Delegation caps count activations, not retained-session inspection or cleanup."""

import pytest

from agent.tool_guardrails import (
    LoopCapConfig,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
)


@pytest.mark.parametrize("action", ["list", "observe", "steer", "stop", "close"])
@pytest.mark.parametrize("hard_stop_enabled", [False, True])
def test_lifecycle_actions_preserve_spawn_budget_before_and_at_cap(action, hard_stop_enabled):
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=hard_stop_enabled, loop_caps=LoopCapConfig(max_subagents=50),
    ))

    def controls():
        for i in range(66):
            args = {"action": action, "session_id": f"child-{i}"}
            assert controller.before_call("delegate_task", args).action == "allow"
            assert controller.after_call("delegate_task", args, '{"status": "ok"}').action == "allow"

    controls()
    for i in range(50):
        assert controller.before_call("delegate_task", {"tasks": [{"goal": f"task-{i}"}]}).action == "allow"
    controls()
    blocked = controller.before_call("delegate_task", {"tasks": [{"goal": "over cap"}]})
    assert blocked.action == "block"
    assert blocked.code == "loop_subagent_cap"
    assert blocked.count == 50
    # Hitting the cap must still leave inspection/cleanup available.
    controls()


@pytest.mark.parametrize("action,batch_size", [(None, 5), ("spawn", 5), ("resume", 1)])
def test_activations_still_consume_budget_and_reset_each_turn(action, batch_size):
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        loop_caps=LoopCapConfig(max_subagents=50),
    ))
    args: dict = ({"session_id": "child", "message": "Continue"} if action == "resume"
            else {"tasks": [{"goal": f"task-{i}"} for i in range(batch_size)]})
    if action is not None:
        args["action"] = action
    for _ in range(50 // batch_size):
        assert controller.before_call("delegate_task", args).action == "allow"
    blocked = controller.before_call("delegate_task", args)
    assert blocked.action == "block"
    assert blocked.code == "loop_subagent_cap"
    assert blocked.count == 50
    controller.reset_for_turn()
    assert controller.before_call("delegate_task", args).action == "allow"
