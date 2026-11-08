"""Batched delegation must fit the remaining per-turn spawn allowance."""

from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


def test_batch_cannot_cross_remaining_spawn_budget():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig.from_mapping({"loop_caps": {"max_subagents": 3}})
    )
    assert controller.before_call("delegate_task", {"tasks": [{"goal": "first"}]}).action == "allow"
    blocked = controller.before_call(
        "delegate_task", {"tasks": [{"goal": "second"}, {"goal": "third"}, {"goal": "fourth"}]}
    )
    assert blocked.action == "block"
    # A refused prospective batch consumes nothing; exact-fit work remains admissible.
    assert controller.before_call(
        "delegate_task", {"tasks": [{"goal": "second"}, {"goal": "third"}]}
    ).action == "allow"
    assert controller.before_call("delegate_task", {"action": "list"}).action == "allow"
