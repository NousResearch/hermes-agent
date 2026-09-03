"""Regression test: guardrails must detect identical-tool spam across turns."""

from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


def test_identical_terminal_call_across_turns_halts_under_hard_stop():
    """Regression: a model calling the same terminal command each turn
    and getting the same result must be halted, not loop forever.

    Before the fix, reset_for_turn() cleared the identical-call streak
    every turn, so the model could spam the same tail command indefinitely
    without ever hitting the stall-guard threshold.
    """
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=3)
    )
    args = {"command": "tail -5 /tmp/build.log"}
    result = "still building\\n"

    # Turn 1: first call, no halt.
    controller.reset_for_turn()
    controller.after_call("terminal", args, result, failed=False)
    controller.observe_call("terminal", args, result, failed=False)
    assert controller.halt_decision is None

    # Turn 2: second identical call — streak persists across turns.
    controller.reset_for_turn()
    controller.after_call("terminal", args, result, failed=False)
    controller.observe_call("terminal", args, result, failed=False)
    assert controller.halt_decision is None

    # Turn 3: third identical call — now at threshold, must halt.
    controller.reset_for_turn()
    controller.after_call("terminal", args, result, failed=False)
    controller.observe_call("terminal", args, result, failed=False)
    halt = controller.halt_decision
    assert halt is not None and halt.should_halt
    assert halt.code == "identical_call_streak_halt"
    assert halt.tool_name == "terminal"
    assert halt.count == 3
