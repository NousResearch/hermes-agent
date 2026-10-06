"""Agent-harness accuracy shared metrics: file edits, loop guards, tool-error recovery, terminal
command outcomes and model reply issues."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_harness as harness
from tests.hermes_cli.test_relay_shared_metrics_runtime import (  # noqa: F401 - fixture
    _stored_values,
    direct_runtime,
)


def _flush() -> None:
    runtime = relay_shared_metrics._get_runtime(retry_failed=True)
    assert runtime is not None
    runtime.relay.subscribers.flush()


def _rows(tmp_path, metric, *fields):
    _flush()
    return sorted((*(d[f] for f in fields), v) for d, v in _stored_values(tmp_path, metric))


def _agent(provider="openrouter", model="anthropic/claude-sonnet", **extra):
    return SimpleNamespace(**{
        "provider": provider, "model": model, "_memory_write_origin": "assistant_tool",
        "_has_content_after_think_block": lambda text: bool(text.replace("<think>x</think>", "").strip()),
        "_extract_reasoning": lambda message: getattr(message, "reasoning", None),
        **extra,
    })


# ---- file edits ------------------------------------------------------------------------------

def test_patch_and_write_file_count_outcome_and_the_strategy_that_landed(direct_runtime, tmp_path):
    from tools.file_tools import _handle_patch, _handle_write_file

    target = tmp_path / "work" / "a.py"
    _handle_write_file({"path": str(target), "content": "def f():\n    return 1\n\nx = 1\nx = 1\n"}, task_id="t")
    _handle_patch({"path": str(target), "old_string": "return 1", "new_string": "return 2"}, task_id="t")
    # Wrong indentation: only a whitespace-tolerant strategy lands it.
    _handle_patch({"path": str(target), "old_string": "def f():\n  return 2", "new_string": "def f():\n    return 3"},
                  task_id="t")
    _handle_patch({"path": str(target), "old_string": "nothing like this", "new_string": "y"}, task_id="t")
    _handle_patch({"path": str(target), "old_string": "x = 1", "new_string": "x = 2"}, task_id="t")
    _handle_patch({"mode": "patch", "patch": (
        f"*** Begin Patch\n*** Update File: {target}\n@@\n-def f():\n+def g():\n*** End Patch\n")}, task_id="t")
    assert "return 3" in target.read_text() and "def g" in target.read_text()

    rows = _rows(tmp_path, contract.FILE_EDIT_METRIC, "tool", "mode", "outcome", "match_strategy")
    fuzzy = [r for r in rows if r[3] not in {"exact", "none"}]
    assert len(fuzzy) == 1 and fuzzy[0][:3] == ("patch", "replace", "applied")
    assert fuzzy[0][3] in contract.FILE_EDIT_STRATEGIES
    assert [r for r in rows if r not in fuzzy] == sorted([
        ("patch", "replace", "ambiguous", "none", 1),
        ("patch", "replace", "applied", "exact", 1),
        ("patch", "replace", "no_match", "none", 1),
        ("patch", "v4a", "applied", "exact", 1),
        ("write_file", "whole_file", "applied", "none", 1),
    ])


def test_matcher_calls_outside_an_edit_tool_call_are_not_counted(direct_runtime, tmp_path):
    from tools.fuzzy_match import fuzzy_find_and_replace

    fuzzy_find_and_replace("a = 1\n", "a = 1", "a = 2")  # e.g. a skill patch, internal callers
    assert _rows(tmp_path, contract.FILE_EDIT_METRIC, "outcome") == []


def test_v4a_patch_reports_its_loosest_hunk_strategy():
    probe = harness._EditProbe(strategies=["exact", "line_trimmed", "exact"])
    fields = harness.file_edit_fields(tool="patch", mode="patch", result='{"success": true}', probe=probe)
    assert fields == {"match_strategy": "line_trimmed", "mode": "v4a", "outcome": "applied", "tool": "patch"}
    probe = harness._EditProbe(misses=["no_match"])
    assert harness.file_edit_fields(
        tool="patch", mode="patch", result='{"success": true, "no_change": true}', probe=probe,
    )["outcome"] == "already_applied"


# ---- terminal ----------------------------------------------------------------------------------

@pytest.mark.parametrize(("command", "kind"), [
    ("git status", "git"), ("FOO=1 sudo -E pip install x", "package_manager"), ("/usr/bin/python3.12 -c 1", "python"),
    ("pytest -q tests/", "test_runner"), ("  cd /tmp && make", "shell_builtin"), ("npx tsc", "node"),
    ("docker ps", "container"), ("curl -s http://x", "network"), ("ls -la", "file_ops"), ("bash -lc 'x'", "shell"),
    ("cargo build", "build"), ("my-private-tool --secret", "other"), ("", "other"), (None, "other"),
])
def test_command_kind_comes_from_the_first_program_word(command, kind):
    assert harness.command_kind(command) == kind
    assert kind in contract.TERMINAL_COMMAND_KINDS


def test_terminal_outcome_trusts_hermes_flags_not_exit_124():
    assert harness.terminal_outcome({"returncode": 124}) == "nonzero"  # the command's own `exit 124`
    assert harness.terminal_outcome({"returncode": 124, "hermes_timed_out": True}) == "timeout"
    assert harness.terminal_outcome({"returncode": 130, "hermes_interrupted": True}) == "killed"
    assert harness.terminal_outcome({"returncode": 137}) == "killed"
    assert harness.terminal_outcome({"returncode": -9}) == "killed"
    assert harness.terminal_outcome({"returncode": 0}) == "ok"
    assert harness.terminal_outcome({"output": "x"}) is None


def test_foreground_terminal_commands_count_by_kind_and_outcome(direct_runtime, tmp_path, monkeypatch):
    from tools.terminal_tool import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "local")
    results = [json.loads(terminal_tool(command, task_id="harness", timeout=timeout)) for command, timeout in (
        ("ls /definitely/not/here", 30), ("echo ok", 30), ("sh -c 'exit 124'", 30), ("sleep 5", 1),
    )]
    assert [r["exit_code"] for r in results] == [2, 0, 124, 124]
    terminal_tool("true", task_id="harness", _host_local=True)  # Hermes' own control plane: never counted
    rows = _rows(tmp_path, contract.TERMINAL_OUTCOME_METRIC, "backend", "command_kind", "outcome")
    assert rows == sorted([
        ("local", "file_ops", "nonzero", 1), ("local", "shell_builtin", "ok", 1),
        ("local", "shell", "nonzero", 1), ("local", "other", "timeout", 1),
    ])
    assert "definitely" not in json.dumps(_stored_values(tmp_path, contract.TERMINAL_OUTCOME_METRIC))


def test_backend_exception_mentioning_timeout_is_not_a_terminal_timeout(direct_runtime, tmp_path, monkeypatch):
    from tools.environments.base import BaseEnvironment
    from tools.terminal_tool import terminal_tool

    def connect_timeout(self, *a, **k):
        raise RuntimeError("ssh: connect to host example port 22: Connection timeout")

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(BaseEnvironment, "execute", connect_timeout)
    assert json.loads(terminal_tool("echo hi", task_id="harness-exc"))["exit_code"] == 124  # user-facing result unchanged
    assert not _stored_values(tmp_path, contract.TERMINAL_OUTCOME_METRIC)


# ---- loop guards -------------------------------------------------------------------------------

def test_loop_guards_count_once_per_turn_per_signal_and_detector(direct_runtime, tmp_path):
    agent = _agent()
    for _ in range(3):
        harness.record_guardrail_decision(agent, "warn", "same_tool_failure_warning")
    harness.record_guardrail_decision(agent, "halt", "same_tool_failure_halt")
    harness.record_guardrail_decision(agent, "warn", "identical_call_streak")
    harness.record_guardrail_decision(agent, "block", "loop_web_search_cap")
    harness.record_guardrail_decision(agent, "allow", "allow")  # not a guard verdict
    harness.finish_turn(agent, "max_iterations_reached(90/90)", "", interrupted=False, failed=False)
    agent._harness_metrics_turn = None  # next turn: the latch resets
    harness.record_guardrail_decision(agent, "warn", "same_tool_failure_warning")
    harness.record_guardrail_decision(_agent(_memory_write_origin="background_review"), "halt", "identical_cycle_halt")

    assert _rows(tmp_path, contract.LOOP_GUARD_METRIC, "provider", "model", "signal", "detector") == sorted([
        ("openrouter", "anthropic/claude-sonnet", "iteration_cap", "iteration_budget", 1),
        ("openrouter", "anthropic/claude-sonnet", "loop_detected", "same_tool_failure", 1),
        ("openrouter", "anthropic/claude-sonnet", "loop_detected", "web_search_cap", 1),
        ("openrouter", "anthropic/claude-sonnet", "repeated_tool_call", "identical_call_streak", 1),
        ("openrouter", "anthropic/claude-sonnet", "repeated_tool_call", "same_tool_failure", 2),
    ])


def test_real_guardrail_halt_reaches_the_metric(direct_runtime, tmp_path):
    from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController
    from run_agent import AIAgent

    agent = _agent()
    agent._tool_guardrails = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    agent._tool_guardrail_halt_decision = None
    agent._stall_guards_enabled = lambda: True
    agent._set_tool_guardrail_halt = lambda decision: AIAgent._set_tool_guardrail_halt(agent, decision)
    for i in range(8):
        AIAgent._append_guardrail_observation(
            agent, "terminal", {"command": "false"}, '{"exit_code": 1, "error": "boom"}', failed=True,
            tool_call_id=f"c{i}",
        )
    halt = agent._tool_guardrail_halt_decision
    assert halt is not None
    signals = {(s, d) for s, d, _v in _rows(tmp_path, contract.LOOP_GUARD_METRIC, "signal", "detector")}
    assert ("loop_detected", harness._GUARD_DETECTORS[halt.code]) in signals
    assert any(s == "repeated_tool_call" for s, _d in signals)


# ---- tool-error recovery -----------------------------------------------------------------------

def _round(agent, *outcomes):
    for name, failed in outcomes:
        harness.observe_tool_outcome(agent, name, failed)
    harness.finish_tool_round(agent)


def test_each_failed_call_resolves_against_the_models_next_call(direct_runtime, tmp_path):
    agent = _agent()
    _round(agent, ("terminal", True), ("read_file", False))
    _round(agent, ("read_file", False), ("terminal", False))    # retried the same tool: success
    _round(agent, ("patch", True), ("mcp_acme_secret", True))
    _round(agent, ("search_files", True))                       # patch: switched tools; mcp: switched
    harness.finish_turn(agent, "text_response(stop)", "done", interrupted=False, failed=False)
    agent._harness_metrics_turn = None
    _round(agent, ("web_search", True))
    harness.finish_turn(agent, "guardrail_halt", "stopped", interrupted=False, failed=True)

    rows = _rows(tmp_path, contract.TOOL_RECOVERY_METRIC, "tool", "next_tool", "next_outcome")
    assert rows == sorted([
        ("terminal", "same", "success", 1),
        ("patch", "different", "error", 1),
        ("mcp", "different", "error", 1),
        ("search_files", "none", "no_tool_call", 1),
        ("web_search", "none", "gave_up", 1),
    ])
    assert "acme" not in json.dumps(_stored_values(tmp_path, contract.TOOL_RECOVERY_METRIC))


# ---- model reply issues ------------------------------------------------------------------------

def test_every_primary_reply_counts_once_with_its_issue(direct_runtime, tmp_path):
    agent = _agent()

    def reply(finish_reason, **message):
        response = object()
        harness.record_reply_finish(agent, response, finish_reason)
        harness.record_reply_content(agent, response, SimpleNamespace(**{"content": None, "tool_calls": None, **message}))

    reply("stop", content="hello")
    reply("tool_calls", tool_calls=[object()])
    reply("stop", content="   ")
    reply("stop", content="<think>x</think>", reasoning="x")
    reply("length", content="half an ans")       # counted once, at the finish reason
    harness.record_reply_finish(agent, object(), "content_filter")  # refusal never reaches intake
    harness.record_reply_content(_agent(_memory_write_origin="background_review"), object(),
                                 SimpleNamespace(content="", tool_calls=None))

    assert _rows(tmp_path, contract.MODEL_REPLY_ISSUE_METRIC, "issue") == sorted([
        ("empty", 1), ("none", 2), ("reasoning_only", 1), ("refusal", 1), ("truncated_length", 1),
    ])


def test_everything_is_a_no_op_while_disabled(direct_runtime, tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.read_raw_config_readonly", lambda: {})
    agent = _agent()
    harness.record_reply_finish(agent, object(), "length")
    _round(agent, ("terminal", True))
    harness.record_guardrail_decision(agent, "halt", "identical_cycle_halt")
    assert getattr(agent, "_harness_metrics_turn", None) is None
    assert harness.record_file_edit("patch", "replace", lambda: '{"success": true}') == '{"success": true}'


# ---- schema ------------------------------------------------------------------------------------

def test_v3_schema_accepts_exactly_the_contract_values():
    import hermes_cli.observability as observability

    schema = json.loads((Path(observability.__file__).parent / "schemas/hermes.shared_metrics.v4.schema.json").read_text())
    by_name = {d["properties"]["name"]["const"]: d for d in schema["$defs"].values() if "properties" in d}
    for metric in (contract.FILE_EDIT_METRIC, contract.LOOP_GUARD_METRIC, contract.TOOL_RECOVERY_METRIC,
                   contract.TERMINAL_OUTCOME_METRIC, contract.MODEL_REPLY_ISSUE_METRIC):
        dims = by_name[metric]["properties"]["dimensions"]["properties"]
        closed = {field: set(spec["enum"]) for field, spec in dims.items() if "enum" in spec}
        assert closed == {
            field: set(values) for field, values in contract._COUNTER_DIMENSION_VALUES[metric].items()
            if values is not contract.TOOL_NAMES
        }
        assert ({"provider", "model"} <= set(dims)) == (metric in contract._IDENTIFIER_FIELDS)
