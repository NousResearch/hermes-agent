"""Deterministic lifecycle-loop guard for kanban run-lifecycle tools.

Root cause (observed attended worker runs, messages 3262–3299): the model
replayed the identical summary-less `kanban_complete` call 19 times. The failure
counted correctly, but on a warn-only config (`hard_stop_enabled=False`, the default
for attended platforms) `before_call` unconditionally allowed every replay and the
only defence was appended warning text the model ignored.

Contracts:
  - after 3 identical FAILED lifecycle calls the 4th identical call is BLOCKED,
    deterministically, even with hard stops disabled;
  - the block decision halts the turn (no 4th execution, clear auditable reason);
  - changed arguments, a successful mutating call in between, other tools and the
    normal successful toolloop are unaffected.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_guardrails import (
    KANBAN_LIFECYCLE_TOOL_NAMES,
    LIFECYCLE_IDENTICAL_FAILURE_BLOCK_AFTER,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
)
from run_agent import AIAgent


FAULTY_COMPLETE_ARGS = {
    "artifacts": ["/tmp/hermes-test.XXXXXX/report.json"],
    "board": "default",
}


def _seed_identical_failures(controller: ToolCallGuardrailController, tool_name: str, args: dict, count: int) -> None:
    for _ in range(count):
        controller.after_call(
            tool_name, args, json.dumps({"error": "kanban_complete: summary is required"}), failed=True,
        )


def test_lifecycle_replays_block_at_three_even_with_hard_stops_disabled():
    """The observed config: warn-only (interactive platform default). The 4th
    identical replay must still be blocked — warnings alone proved ignorable."""
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=False))
    assert controller.config.hard_stop_enabled is False

    _seed_identical_failures(controller, "kanban_complete", FAULTY_COMPLETE_ARGS, count=3)

    decision = controller.before_call("kanban_complete", FAULTY_COMPLETE_ARGS)
    assert decision.action == "block"
    assert decision.code == "lifecycle_identical_failure_block"
    assert decision.count == LIFECYCLE_IDENTICAL_FAILURE_BLOCK_AFTER
    assert decision.should_halt is True
    # The block reason must be auditable and must state the card was not completed.
    assert "NOT completed" in decision.message
    assert controller.halt_decision is decision


def test_repro_observed_worker_config_platform_cli_still_blocks_lifecycle_replays():
    """RCA repro (observed attended worker runs): a kanban worker runs with
    platform='cli', so ToolCallGuardrailConfig.from_mapping({}, platform='cli') has
    hard stops disabled — the generic exact-failure rule never fires there. The
    lifecycle rule must block the 4th identical replay anyway."""
    config = ToolCallGuardrailConfig.from_mapping({}, platform="cli")
    assert config.hard_stop_enabled is False, "the observed worker config is warn-only"
    controller = ToolCallGuardrailController(config)

    args = {"artifacts": ["/tmp/hermes-test.XXXXXX/report.json"], "board": "default"}
    for i in range(1, 4):
        assert controller.before_call("kanban_complete", args).allows_execution is True, i
        controller.after_call("kanban_complete", args, json.dumps({"error": "boom"}), failed=True)

    # The generic rule would allow this call forever on this config; the lifecycle
    # rule stops the replay deterministically.
    blocked = controller.before_call("kanban_complete", args)
    assert blocked.action == "block"
    assert blocked.code == "lifecycle_identical_failure_block"


def test_two_identical_lifecycle_failures_still_allow_the_third():
    """Calls 1–3 execute (and fail visibly); the guard fires on the 4th, not earlier."""
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=False))
    _seed_identical_failures(controller, "kanban_complete", FAULTY_COMPLETE_ARGS, count=2)
    assert controller.before_call("kanban_complete", FAULTY_COMPLETE_ARGS).allows_execution is True


def test_lifecycle_guard_applies_to_all_lifecycle_tools():
    from tools.kanban_tools import _RUN_LIFECYCLE_TOOLS

    assert KANBAN_LIFECYCLE_TOOL_NAMES == _RUN_LIFECYCLE_TOOLS
    for tool in sorted(KANBAN_LIFECYCLE_TOOL_NAMES):
        controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=False))
        args = {"task_id": "t_x", "reason": "stuck"} if tool != "kanban_complete" else FAULTY_COMPLETE_ARGS
        _seed_identical_failures(controller, tool, args, count=3)
        decision = controller.before_call(tool, args)
        assert decision.action == "block", tool
        assert decision.code == "lifecycle_identical_failure_block", tool


def test_changed_lifecycle_args_are_never_blocked():
    """A legitimately changed recovery attempt (different arguments) is a new
    signature and must never be blocked by the replay guard."""
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=False))
    _seed_identical_failures(controller, "kanban_complete", FAULTY_COMPLETE_ARGS, count=5)
    changed = dict(FAULTY_COMPLETE_ARGS, summary="now with a real summary")
    assert controller.before_call("kanban_complete", changed).action == "allow"


def test_successful_mutation_between_retries_resets_the_streak():
    """Shared progress semantics: a successful mutating call between identical
    lifecycle failures makes the next retry a new experiment, not a replay."""
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=False))
    _seed_identical_failures(controller, "kanban_complete", FAULTY_COMPLETE_ARGS, count=3)
    controller.after_call("write_file", {"path": "/a", "content": "fix"}, json.dumps({"ok": True}), failed=False)
    decision = controller.before_call("kanban_complete", FAULTY_COMPLETE_ARGS)
    assert decision.action == "allow"


def test_non_lifecycle_tools_keep_their_configured_thresholds():
    """Other tools must be untouched: on a warn-only config three identical
    failures do NOT block; with hard stops enabled the configured threshold
    (default 5) still governs, not the lifecycle threshold 3."""
    soft = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=False))
    _seed_identical_failures(soft, "terminal", {"command": "make"}, count=3)
    assert soft.before_call("terminal", {"command": "make"}).action == "allow"

    hard = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    _seed_identical_failures(hard, "terminal", {"command": "make"}, count=3)
    assert hard.before_call("terminal", {"command": "make"}).action == "allow"
    _seed_identical_failures(hard, "terminal", {"command": "make"}, count=2)
    blocked = hard.before_call("terminal", {"command": "make"})
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"


def test_successful_lifecycle_call_never_blocks_and_clears_counts():
    """The normal toolloop case: a successful lifecycle call is allowed and clears
    the failure accounting for its signature."""
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=False))
    _seed_identical_failures(controller, "kanban_complete", FAULTY_COMPLETE_ARGS, count=2)
    ok_args = dict(FAULTY_COMPLETE_ARGS, summary="done")
    controller.after_call("kanban_complete", ok_args, json.dumps({"ok": True}), failed=False)
    assert controller.before_call("kanban_complete", FAULTY_COMPLETE_ARGS).action == "allow"


# --- Runtime wiring: the 4th identical call never executes, the turn halts ---

def _make_agent(tmp_path, monkeypatch, *tool_names: str, max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir(exist_ok=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.chdir(work_dir)
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in tool_names
    ]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("hermes_cli.config.load_config_readonly", return_value=config or {}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="cli",  # attended default: hard stops OFF — the observed worker config
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _mock_tool_call(name, arguments, call_id):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(tool_calls=None, content=""):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls" if tool_calls else "stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_runtime_fourth_identical_lifecycle_call_is_not_executed(tmp_path, monkeypatch):
    """End-to-end at the executor: 3 identical faulty kanban_complete calls run,
    the 4th is refused pre-dispatch and sets the turn-halting decision."""
    agent = _make_agent(tmp_path, monkeypatch, "kanban_complete")
    for _ in range(3):
        agent._tool_guardrails.after_call(
            "kanban_complete", FAULTY_COMPLETE_ARGS,
            json.dumps({"error": "kanban_complete: summary is required"}), failed=True,
        )
    starts = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    tc = _mock_tool_call("kanban_complete", json.dumps(FAULTY_COMPLETE_ARGS), "c-4th")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("model_tools.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert starts == []
    assert len(messages) == 1
    assert "lifecycle_identical_failure_block" in messages[0]["content"]
    assert agent._tool_guardrail_halt_decision is not None
    assert agent._tool_guardrail_halt_decision.code == "lifecycle_identical_failure_block"


def test_runtime_loop_halts_after_exactly_three_executions(tmp_path, monkeypatch):
    """Full conversation loop replaying the observed failure: the identical call
    executes exactly 3 times, then the run stops with a guardrail_halt and a clear
    final response — never a 4th execution, never a silent burn of the budget."""
    agent = _make_agent(tmp_path, monkeypatch, "kanban_complete", max_iterations=50)
    # Ten identical tool-call responses — far more than the guard tolerates.
    responses = [
        _mock_response(tool_calls=[_mock_tool_call(
            "kanban_complete", json.dumps(FAULTY_COMPLETE_ARGS), f"c{i}")])
        for i in range(10)
    ]
    agent.client.chat.completions.create.side_effect = responses
    agent._disable_streaming = True

    with (
        patch("model_tools.handle_function_call",
              return_value=json.dumps({"error": "kanban_complete: summary is required"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("complete the task")

    assert mock_hfc.call_count == 3, "the 4th identical lifecycle call must never execute"
    assert result["turn_exit_reason"] == "guardrail_halt"
    halt = result["guardrail"]
    assert halt["code"] == "lifecycle_identical_failure_block"
    assert halt["tool_name"] == "kanban_complete"
    # Auditable block reason: the 4th call's synthetic result states the loop stop
    # and that the card was NOT completed.
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("lifecycle_identical_failure_block" in c and "NOT completed" in c for c in tool_contents)
    assert result["final_response"]


def test_runtime_normal_lifecycle_call_is_unaffected(tmp_path, monkeypatch):
    """The normal toolloop: a summary-bearing kanban_complete succeeds and the
    conversation ends normally — no guardrail interference."""
    agent = _make_agent(tmp_path, monkeypatch, "kanban_complete", max_iterations=5)
    ok_args = dict(FAULTY_COMPLETE_ARGS, summary="done, tests green")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(tool_calls=[_mock_tool_call("kanban_complete", json.dumps(ok_args), "c-ok")]),
        _mock_response(content="all done"),
    ]

    with (
        patch("model_tools.handle_function_call", return_value=json.dumps({"ok": True})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("complete the task")

    assert result["turn_exit_reason"].startswith("text_response")
    assert result["final_response"] == "all done"
    assert agent._tool_guardrail_halt_decision is None
