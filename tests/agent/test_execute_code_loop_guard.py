"""Loop-guard coverage for execute_code: identical cells must be detected.

Regression: a flash subagent replayed ~140 byte-identical execute_code calls and the guard
never fired, because (a) execute_code is mutating so the no-progress branch bailed out early,
and (b) the result hash covered the volatile kernel wrapper (execution_count, duration, ...),
so every replay looked like a fresh result.
"""

import json

from agent.tool_guardrails import (
    STALL_GUARD_IDENTICAL_CALL_THRESHOLD,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
)

_ARGS = {"code": "print(open('a.py').read())"}


def _execute_code_result(execution_count: int, output: str = "same output\n") -> str:
    """Shape of a real execute_code result: stable output plus a volatile kernel wrapper."""
    return json.dumps(
        {
            "status": "success",
            "output": output,
            "exit_code": 0,
            "tool_calls_made": 0,
            "duration_seconds": 0.1 * execution_count,
            "kernel": {
                "mode": "session",
                "reused": True,
                "execution_count": execution_count,
                "state_reset": False,
            },
        }
    )


def test_identical_execute_code_results_differing_only_in_execution_count_fire_identical_notice():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig())

    observations = [
        controller.observe_call("execute_code", _ARGS, _execute_code_result(i), failed=False)
        for i in range(1, STALL_GUARD_IDENTICAL_CALL_THRESHOLD + 1)
    ]

    assert [o.notice for o in observations[: STALL_GUARD_IDENTICAL_CALL_THRESHOLD - 1]] == [None, None]
    notice = observations[-1].notice
    assert notice is not None
    assert "consecutive identical call" in notice
    assert "execute_code" in notice


def test_execute_code_no_progress_halts_when_hard_stop_enabled():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=3)
    )

    for i in range(1, 3):
        assert controller.before_call("execute_code", _ARGS).action == "allow"
        controller.after_call("execute_code", _ARGS, _execute_code_result(i), failed=False)
        controller.observe_call("execute_code", _ARGS, _execute_code_result(i), failed=False)
        assert controller.halt_decision is None, f"halted early at {i}"

    controller.after_call("execute_code", _ARGS, _execute_code_result(3), failed=False)
    controller.observe_call("execute_code", _ARGS, _execute_code_result(3), failed=False)

    halt = controller.halt_decision
    assert halt is not None and halt.should_halt
    assert halt.code == "identical_call_streak_halt"
    assert halt.tool_name == "execute_code"
    assert halt.count == 3


def test_plain_mutating_tool_still_skips_the_no_progress_counter():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_warn_after=2, no_progress_block_after=2)
    )
    args = {"command": "ls"}

    for _ in range(4):
        assert controller.before_call("terminal", args).action == "allow"
        assert controller.after_call("terminal", args, "same\n", failed=False).action == "allow"

    assert controller._no_progress == {}
    assert controller.halt_decision is None
