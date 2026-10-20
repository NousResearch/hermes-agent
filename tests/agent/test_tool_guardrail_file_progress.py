"""Landed file edits make rereads new work while preserving turn ceilings (PR #56929)."""

import pytest

from agent.tool_guardrails import (
    LoopCapConfig,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
)


@pytest.mark.parametrize("tool,result", [
    ("patch", '{"success":true}'),
    ("write_file", '{"bytes_written":12,"lint":{"error":"invalid syntax"}}'),
])
def test_landed_file_edits_restart_reread_streak_without_resetting_turn_caps(tool, result):
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True, no_progress_block_after=2,
        loop_caps=LoopCapConfig(max_web_searches=2, max_subagents=2),
    ))
    for i in range(2):
        assert controller.before_call("web_search", {"query": str(i)}).allows_execution
        assert controller.before_call("delegate_task", {"goal": str(i)}).allows_execution
    args = {"path": "/tmp/result.txt"}
    for _ in range(3):
        for _ in range(2):
            assert controller.before_call("read_file", args).allows_execution
            controller.after_call("read_file", args, "same content", failed=False)
        # Repair before the next read would halt the turn.
        assert controller.before_call(tool, args).allows_execution
        controller.after_call(tool, args, result)
        assert controller.before_call("read_file", args).allows_execution
        assert controller.halt_decision is None
    for _ in range(2):
        controller.after_call("read_file", args, "same content", failed=False)
    assert controller.before_call("read_file", args).code == "idempotent_no_progress_block"
    assert controller.before_call("web_search", {"query": "third"}).code == "loop_web_search_cap"
    assert controller.before_call("delegate_task", {"goal": "third"}).code == "loop_subagent_cap"


@pytest.mark.parametrize("tool,result", [
    ("patch", '{"success":false}'),
    ("patch", '{"success":true,"error":"write failed"}'),
    ("write_file", '{"error":"permission denied"}'),
    ("patch", "not JSON"),
    ("read_file", '{"success":true}'),
])
def test_unlanded_or_read_only_results_do_not_restart_reread_streak(tool, result):
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True, no_progress_block_after=2,
    ))
    args = {"path": "/tmp/result.txt"}
    for _ in range(2):
        controller.after_call("read_file", args, "same content", failed=False)
    controller.after_call(tool, {"path": "/tmp/other.txt"}, result)
    assert controller.before_call("read_file", args).code == "idempotent_no_progress_block"
