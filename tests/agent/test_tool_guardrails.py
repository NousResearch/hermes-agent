"""Pure tool-call guardrail primitive tests."""

import json

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    canonical_tool_args,
    classify_tool_failure,
)


def test_tool_call_signature_hashes_canonical_nested_unicode_args_without_exposing_raw_args():
    args_a = {
        "z": [{"β": "☤", "a": 1}],
        "a": {"y": 2, "x": "secret-token-value"},
    }
    args_b = {
        "a": {"x": "secret-token-value", "y": 2},
        "z": [{"a": 1, "β": "☤"}],
    }

    assert canonical_tool_args(args_a) == canonical_tool_args(args_b)
    sig_a = ToolCallSignature.from_call("web_search", args_a)
    sig_b = ToolCallSignature.from_call("web_search", args_b)

    assert sig_a == sig_b
    assert len(sig_a.args_hash) == 64
    metadata = sig_a.to_metadata()
    assert metadata == {"tool_name": "web_search", "args_hash": sig_a.args_hash}
    assert "secret-token-value" not in json.dumps(metadata)
    assert "☤" not in json.dumps(metadata)


def test_default_config_is_soft_warning_only_with_hard_stop_disabled():
    cfg = ToolCallGuardrailConfig()

    assert cfg.warnings_enabled is True
    assert cfg.hard_stop_enabled is False
    assert cfg.exact_failure_warn_after == 2
    assert cfg.exact_failure_selfcheck_after == 3
    assert cfg.same_tool_failure_warn_after == 3
    assert cfg.same_tool_failure_selfcheck_after == 3
    assert cfg.no_progress_warn_after == 2
    assert cfg.no_progress_selfcheck_after == 3
    assert cfg.exact_failure_block_after == 5
    assert cfg.same_tool_failure_halt_after == 8
    assert cfg.no_progress_block_after == 5


def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
            },
            "hard_stop_after": {
                "exact_failure": 6,
                "same_tool_failure": 7,
                "idempotent_no_progress": 8,
            },
        }
    )

    assert cfg.warnings_enabled is False
    assert cfg.hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 3
    assert cfg.same_tool_failure_warn_after == 4
    assert cfg.no_progress_warn_after == 5
    assert cfg.exact_failure_block_after == 6
    assert cfg.same_tool_failure_halt_after == 7
    assert cfg.no_progress_block_after == 8


def test_default_repeated_identical_failed_call_warns_without_blocking():
    controller = ToolCallGuardrailController()
    args = {"query": "same"}

    decisions = []
    for _ in range(5):
        assert controller.before_call("web_search", args).action == "allow"
        decisions.append(
            controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
        )

    assert decisions[0].action == "allow"
    assert [d.action for d in decisions[1:]] == ["warn", "warn", "warn", "warn"]
    # count=2: repeated_exact_failure_warning, count>=3: tool_call_self_check_required
    assert decisions[1].code == "repeated_exact_failure_warning"
    assert decisions[2].code == "tool_call_self_check_required"
    assert decisions[3].code == "tool_call_self_check_required"
    assert decisions[4].code == "tool_call_self_check_required"
    assert controller.before_call("web_search", args).action == "allow"
    assert controller.halt_decision is None


def test_hard_stop_enabled_blocks_repeated_exact_failure_before_next_execution():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_warn_after=2,
            exact_failure_block_after=2,
            same_tool_failure_halt_after=99,
        )
    )
    args = {"query": "same"}

    assert controller.before_call("web_search", args).action == "allow"
    first = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert first.action == "allow"

    assert controller.before_call("web_search", args).action == "allow"
    second = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert second.action == "warn"
    assert second.code == "repeated_exact_failure_warning"

    blocked = controller.before_call("web_search", args)
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"
    assert blocked.count == 2


def test_success_resets_exact_signature_failure_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, exact_failure_block_after=2, same_tool_failure_halt_after=99)
    )
    args = {"query": "same"}

    controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    controller.after_call("web_search", args, '{"ok":true}', failed=False)

    assert controller.before_call("web_search", args).action == "allow"
    controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert controller.before_call("web_search", args).action == "allow"


def test_file_mutation_lint_error_result_is_not_a_tool_failure():
    write_result = json.dumps({
        "bytes_written": 12,
        "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
    })
    patch_result = json.dumps({
        "success": True,
        "diff": "--- a/tmp.py\n+++ b/tmp.py\n",
        "lsp_diagnostics": "<diagnostics>ERROR [1:1] type mismatch</diagnostics>",
    })

    assert classify_tool_failure("write_file", write_result) == (False, "")
    assert classify_tool_failure("patch", patch_result) == (False, "")


def test_same_tool_varying_args_warns_by_default_without_halting():
    """same_tool_failure warns at count=2 (warn_after=2), self-checks at count=3 (default selfcheck_after=3)."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(same_tool_failure_warn_after=2, same_tool_failure_halt_after=5,
                                same_tool_failure_selfcheck_after=5)
    )

    first = controller.after_call("terminal", {"command": "cmd-1"}, '{"exit_code":1}', failed=True)
    second = controller.after_call("terminal", {"command": "cmd-2"}, '{"exit_code":1}', failed=True)
    third = controller.after_call("terminal", {"command": "cmd-3"}, '{"exit_code":1}', failed=True)
    fourth = controller.after_call("terminal", {"command": "cmd-4"}, '{"exit_code":1}', failed=True)

    assert first.action == "allow"
    assert [second.action, third.action, fourth.action] == ["warn", "warn", "warn"]
    assert {second.code, third.code, fourth.code} == {"same_tool_failure_warning"}
    assert "Do not switch to text-only replies" in second.message
    assert "keep using tools" in second.message
    assert "diagnose before retrying" in second.message
    assert "different tool" in second.message
    assert controller.halt_decision is None


def test_hard_stop_enabled_halts_same_tool_varying_args_failure_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_block_after=99,
            same_tool_failure_warn_after=2,
            same_tool_failure_halt_after=3,
        )
    )

    first = controller.after_call("terminal", {"command": "cmd-1"}, '{"exit_code":1}', failed=True)
    assert first.action == "allow"
    second = controller.after_call("terminal", {"command": "cmd-2"}, '{"exit_code":1}', failed=True)
    assert second.action == "warn"
    assert second.code == "same_tool_failure_warning"
    third = controller.after_call("terminal", {"command": "cmd-3"}, '{"exit_code":1}', failed=True)
    assert third.action == "halt"
    assert third.code == "same_tool_failure_halt"
    assert third.count == 3


def test_idempotent_no_progress_repeated_result_warns_without_blocking_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2,
                                no_progress_selfcheck_after=5)
    )
    args = {"path": "/tmp/same.txt"}
    result = "same file contents"

    for _ in range(4):
        assert controller.before_call("read_file", args).action == "allow"
        decision = controller.after_call("read_file", args, result, failed=False)

    assert decision.action == "warn"
    assert decision.code == "no_progress_warning"
    assert controller.before_call("read_file", args).action == "allow"
    assert controller.halt_decision is None


def test_hard_stop_enabled_blocks_idempotent_no_progress_future_repeat():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=2,
        )
    )
    args = {"path": "/tmp/same.txt"}
    result = "same file contents"

    assert controller.before_call("read_file", args).action == "allow"
    assert controller.after_call("read_file", args, result, failed=False).action == "allow"
    assert controller.before_call("read_file", args).action == "allow"
    warn = controller.after_call("read_file", args, result, failed=False)
    assert warn.action == "warn"
    assert warn.code == "no_progress_warning"

    blocked = controller.before_call("read_file", args)
    assert blocked.action == "block"
    assert blocked.code == "no_progress_block"


def test_only_execution_mutating_tools_trigger_no_progress_when_repeated():
    """terminal/execute_code trigger no_progress on repeated identical success output.
    Other mutating tools (write_file, browser_click) do NOT — repeated writes/clicks
    can be intentional and should not be blocked."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=2,
        )
    )

    # terminal does trigger no_progress
    args = {"command": "python script.py"}
    assert controller.before_call("terminal", args).action == "allow"
    d1 = controller.after_call("terminal", args, "same output", failed=False)
    assert d1.action == "allow"

    assert controller.before_call("terminal", args).action == "allow"
    d2 = controller.after_call("terminal", args, "same output", failed=False)
    assert d2.action == "warn"
    assert d2.code == "no_progress_warning"

    blocked = controller.before_call("terminal", args)
    assert blocked.action == "block"
    assert blocked.code == "no_progress_block"

    # write_file does NOT trigger no_progress
    write_args = {"path": "/tmp/x", "content": "x"}
    for _ in range(3):
        assert controller.before_call("write_file", write_args).action == "allow"
        assert controller.after_call("write_file", write_args, "ok", failed=False).action == "allow"


def test_reset_for_turn_clears_bounded_guardrail_state():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, exact_failure_block_after=2, no_progress_block_after=2)
    )
    controller.after_call("web_search", {"query": "same"}, '{"error":"boom"}', failed=True)
    controller.after_call("web_search", {"query": "same"}, '{"error":"boom"}', failed=True)
    controller.after_call("read_file", {"path": "/tmp/x"}, "same", failed=False)
    controller.after_call("read_file", {"path": "/tmp/x"}, "same", failed=False)

    assert controller.before_call("web_search", {"query": "same"}).action == "block"
    assert controller.before_call("read_file", {"path": "/tmp/x"}).action == "block"

    controller.reset_for_turn()

    # Per-turn state is cleared: exact_failure block is gone
    assert controller.before_call("web_search", {"query": "same"}).action == "allow"
    # read_file per-turn no_progress is also cleared after reset
    assert controller.before_call("read_file", {"path": "/tmp/x"}).action == "allow"
    # Cross-turn tracking does NOT apply to read_file (read-only tool)
    # — only terminal/execute_code are tracked cross-turn


def test_terminal_exit_code_zero_with_task_failure_output_is_detected():
    """Terminal with exit_code=0 but output containing error patterns is a failure."""
    # HTTP 400 in output
    result = json.dumps({"exit_code": 0, "output": "❌ Error: 400 Bad Request"})
    is_fail, suffix = classify_tool_failure("terminal", result)
    assert is_fail is True
    assert "task_failure" in suffix

    # Traceback in output
    result = json.dumps({"exit_code": 0, "output": "Traceback (most recent call last):\n  File ..."})
    is_fail, suffix = classify_tool_failure("terminal", result)
    assert is_fail is True

    # Normal success output should NOT be flagged
    result = json.dumps({"exit_code": 0, "output": "File written successfully"})
    is_fail, suffix = classify_tool_failure("terminal", result)
    assert is_fail is False

    # "0 failed" should NOT be flagged
    result = json.dumps({"exit_code": 0, "output": "pytest summary: 10 passed, 0 failed"})
    is_fail, _ = classify_tool_failure("terminal", result)
    assert is_fail is False

    # "404 page test passed" should NOT be flagged
    result = json.dumps({"exit_code": 0, "output": "404 page test passed"})
    is_fail, _ = classify_tool_failure("terminal", result)
    assert is_fail is False

    # Long output with traceback at the end should be caught
    result = json.dumps({"exit_code": 0, "output": "x" * 3000 + "\nTraceback (most recent call last):\nboom"})
    is_fail, _ = classify_tool_failure("terminal", result)
    assert is_fail is True


def test_execute_code_success_with_task_failure_output_is_detected():
    """execute_code with status=success but output containing error patterns is a failure."""
    # Error in output
    result = json.dumps({"status": "success", "output": "❌ Error: connection refused"})
    is_fail, suffix = classify_tool_failure("execute_code", result)
    assert is_fail is True
    assert "task_failure" in suffix

    # Normal success output should NOT be flagged
    result = json.dumps({"status": "success", "output": "All tests passed"})
    is_fail, suffix = classify_tool_failure("execute_code", result)
    assert is_fail is False

    # status=error is still caught
    result = json.dumps({"status": "error", "error": "Script failed"})
    is_fail, suffix = classify_tool_failure("execute_code", result)
    assert is_fail is True


def test_mutating_tool_task_failure_triggers_exact_failure_counting():
    """When terminal/execute_code output indicates task failure, repeated identical
    calls should trigger the exact_failure counter (not just no_progress)."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_warn_after=2,
            exact_failure_block_after=2,
            same_tool_failure_halt_after=99,
        )
    )
    args = {"command": "curl -s http://api.example.com/data"}
    # exit_code=0 but output has error pattern → classified as failure
    result = json.dumps({"exit_code": 0, "output": "❌ Error: 400 Bad Request"})

    # First call: allow
    assert controller.before_call("terminal", args).action == "allow"
    d1 = controller.after_call("terminal", args, result, failed=True)
    assert d1.action == "allow"

    # Second call: warn (exact_failure_warn_after=2)
    assert controller.before_call("terminal", args).action == "allow"
    d2 = controller.after_call("terminal", args, result, failed=True)
    assert d2.action == "warn"
    assert d2.code == "repeated_exact_failure_warning"

    # Third call: block (exact_failure_block_after=2, already failed 2 times)
    blocked = controller.before_call("terminal", args)
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"


def test_cross_turn_no_progress_catches_one_call_per_turn_loop():
    """One identical terminal call per turn should still be caught by cross-turn tracking."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=3, no_progress_warn_after=2)
    )

    # Turn 1
    controller.after_call("terminal", {"command": "curl http://example.com"}, "Page not found", failed=False)
    assert controller.before_call("terminal", {"command": "curl http://example.com"}).action == "allow"

    # Turn 2 (reset clears per-turn, but cross-turn persists)
    controller.reset_for_turn()
    d = controller.after_call("terminal", {"command": "curl http://example.com"}, "Page not found", failed=False)
    # Cross-turn count is 2 — should warn
    assert d.action == "warn"
    assert d.code == "no_progress_cross_turn_warning"
    assert controller.before_call("terminal", {"command": "curl http://example.com"}).action == "allow"

    # Turn 3
    controller.reset_for_turn()
    d = controller.after_call("terminal", {"command": "curl http://example.com"}, "Page not found", failed=False)
    # Cross-turn count is 3 — still warn (warn_after=2, already warned)
    # before_call should now block
    assert controller.before_call("terminal", {"command": "curl http://example.com"}).action == "block"
    assert controller.before_call("terminal", {"command": "curl http://example.com"}).code == "no_progress_cross_turn_block"


def test_cross_turn_no_progress_resets_on_different_result():
    """A different result should reset the cross-turn counter."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=2, no_progress_warn_after=2)
    )

    # Turn 1
    controller.after_call("terminal", {"command": "curl http://example.com"}, "Page not found", failed=False)

    # Turn 2 — same result
    controller.reset_for_turn()
    controller.after_call("terminal", {"command": "curl http://example.com"}, "Page not found", failed=False)

    # Turn 3 — different result (streak broken)
    controller.reset_for_turn()
    controller.after_call("terminal", {"command": "curl http://example.com"}, "OK success", failed=False)
    assert controller.before_call("terminal", {"command": "curl http://example.com"}).action == "allow"

    # Turn 4 — even if we go back to "Page not found", it starts fresh
    controller.reset_for_turn()
    controller.after_call("terminal", {"command": "curl http://example.com"}, "Page not found", failed=False)
    assert controller.before_call("terminal", {"command": "curl http://example.com"}).action == "allow"


def test_read_file_cross_turn_should_not_block():
    """read_file across turns should NOT be blocked by cross-turn tracking."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=2, no_progress_warn_after=2)
    )

    # Turn 1
    controller.after_call("read_file", {"path": "/tmp/config.yaml"}, "key: value", failed=False)

    # Turn 2 — same result, should NOT block cross-turn
    controller.reset_for_turn()
    controller.after_call("read_file", {"path": "/tmp/config.yaml"}, "key: value", failed=False)
    assert controller.before_call("read_file", {"path": "/tmp/config.yaml"}).action == "allow"

    # Turn 3 — still should NOT block
    controller.reset_for_turn()
    controller.after_call("read_file", {"path": "/tmp/config.yaml"}, "key: value", failed=False)
    assert controller.before_call("read_file", {"path": "/tmp/config.yaml"}).action == "allow"


def test_read_file_same_turn_still_triggers_no_progress():
    """read_file within the SAME turn should still trigger no-progress warning/block."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=2, no_progress_warn_after=2)
    )

    # Same turn: read_file repeated with same result
    controller.after_call("read_file", {"path": "/tmp/x"}, "same content", failed=False)
    assert controller.before_call("read_file", {"path": "/tmp/x"}).action == "allow"

    d = controller.after_call("read_file", {"path": "/tmp/x"}, "same content", failed=False)
    assert d.action == "warn"
    assert d.code == "no_progress_warning"

    blocked = controller.before_call("read_file", {"path": "/tmp/x"})
    assert blocked.action == "block"
    assert blocked.code == "no_progress_block"


def test_output_indicates_task_failure_true_positives():
    """Patterns that SHOULD be detected as task failures."""
    from agent.tool_failure_detection import output_indicates_task_failure

    assert output_indicates_task_failure("Traceback (most recent call last):\n  File ...")
    assert output_indicates_task_failure("  Error: 400 Bad Request")
    assert output_indicates_task_failure("HTTP/1.1 404 Not Found")
    assert output_indicates_task_failure("HTTP 500 Internal Server Error")
    assert output_indicates_task_failure("status_code=500")
    assert output_indicates_task_failure("status: 404")
    assert output_indicates_task_failure("bash: command not found")
    assert output_indicates_task_failure("No such file or directory")
    assert output_indicates_task_failure("Permission denied")
    assert output_indicates_task_failure("Connection refused")
    assert output_indicates_task_failure("Connection reset by peer")
    assert output_indicates_task_failure("Connection timed out")
    assert output_indicates_task_failure("Timeout expired")
    assert output_indicates_task_failure("Fatal: something broke")


def test_output_indicates_task_failure_false_positives():
    """Patterns that should NOT be detected as task failures."""
    from agent.tool_failure_detection import output_indicates_task_failure

    assert not output_indicates_task_failure("Search result: no matching files found")
    assert not output_indicates_task_failure("Expected Not Found response was returned")
    assert not output_indicates_task_failure("404 page test passed")
    assert not output_indicates_task_failure("pytest summary: 10 passed, 0 failed")
    assert not output_indicates_task_failure("expected error path passed")
    assert not output_indicates_task_failure("Testing that unauthorized access returns 401")
    assert not output_indicates_task_failure("The forbidden fruit was delicious")
    assert not output_indicates_task_failure("File not found in cache, generating fresh")
    assert not output_indicates_task_failure("No errors found in the build")
    assert not output_indicates_task_failure("All 404 tests passed successfully")


def test_output_indicates_task_failure_traceback_at_end_of_long_output():
    """Traceback at the end of a long output should still be detected."""
    from agent.tool_failure_detection import output_indicates_task_failure

    long_output = "Some normal log output\n" * 300 + "Traceback (most recent call last):\n  File 'app.py', line 1"
    assert output_indicates_task_failure(long_output)


def test_selfcheck_observation_at_count_3_with_structured_failure_history():
    """At count=3, after_call returns a structured self-check observation with failure history."""
    from agent.tool_guardrails import (
        ToolCallGuardrailController,
        append_toolguard_guidance,
    )
    import json as _json

    controller = ToolCallGuardrailController()
    args = {"command": "python3 server.py --port 8189"}

    # First 2 failures: allow + warn
    d1 = controller.after_call("terminal", args, '{"output":"","exit_code":124}', failed=True)
    assert d1.action == "allow"

    d2 = controller.after_call("terminal", args, '{"output":"timeout","exit_code":124}', failed=True)
    assert d2.action == "warn"
    assert d2.code == "repeated_exact_failure_warning"

    # Count=3: self-check observation
    d3 = controller.after_call("terminal", args, '{"output":"Agent Service Gateway :8189","exit_code":124}', failed=True)
    assert d3.action == "warn"
    assert d3.code == "tool_call_self_check_required"
    assert d3.count == 3
    assert d3.last_tool_call_args == args
    assert d3.recent_failures is not None
    assert len(d3.recent_failures) == 3

    # Verify failure history contains structured data
    assert all("exit_code" in f for f in d3.recent_failures)
    assert d3.recent_failures[0]["exit_code"] == 124
    assert d3.recent_failures[0]["exit_code_meaning"] == "Command timed out"

    # Verify the appended guidance contains structured JSON
    result = '{"output":"timeout","exit_code":124}'
    guided = append_toolguard_guidance(result, d3)
    # The guidance should contain the self-check JSON block
    assert "Tool-call self-check required before retrying" in guided
    parsed_guidance = _json.loads(guided.split("\n\n")[1])
    assert parsed_guidance["guardrail"]["code"] == "tool_call_self_check_required"
    assert parsed_guidance["last_tool_call"]["tool_name"] == "terminal"
    assert parsed_guidance["last_tool_call"]["args"] == args
    assert len(parsed_guidance["recent_failures"]) == 3
    assert "required_self_check" in parsed_guidance
    assert "required_next_step" in parsed_guidance


def test_stronger_selfcheck_warning_at_count_4():
    """At count=4, self-check message escalates to stronger warning."""
    controller = ToolCallGuardrailController()
    args = {"command": "python3 server.py"}

    for i in range(4):
        controller.after_call("terminal", args, '{"output":"timeout","exit_code":124}', failed=True)

    d4 = controller.after_call("terminal", args, '{"output":"timeout","exit_code":124}', failed=True)
    assert d4.action == "warn"
    assert d4.code == "tool_call_self_check_required"
    assert d4.count == 5
    assert "confirmed loop" in d4.message
    assert "MUST change strategy" in d4.message


def test_selfcheck_configurable_threshold():
    """exact_failure_selfcheck_after can be configured to delay self-check."""
    from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

    cfg = ToolCallGuardrailConfig(exact_failure_selfcheck_after=4)
    controller = ToolCallGuardrailController(cfg)
    args = {"command": "cmd"}

    # count=1: allow
    d1 = controller.after_call("terminal", args, '{"exit_code":1}', failed=True)
    assert d1.action == "allow"

    # count=2: repeated_exact_failure_warning (warn_after=2)
    d2 = controller.after_call("terminal", args, '{"exit_code":1}', failed=True)
    assert d2.action == "warn"
    assert d2.code == "repeated_exact_failure_warning"

    # count=3: repeated_exact_failure_warning (selfcheck_after=4, so not yet)
    d3 = controller.after_call("terminal", args, '{"exit_code":1}', failed=True)
    assert d3.action == "warn"
    assert d3.code == "repeated_exact_failure_warning"

    # count=4: tool_call_self_check_required (selfcheck_after=4)
    d4 = controller.after_call("terminal", args, '{"exit_code":1}', failed=True)
    assert d4.action == "warn"
    assert d4.code == "tool_call_self_check_required"


def test_failure_history_capped_at_5_entries():
    """Recent failure history is capped at 5 entries to avoid unbounded growth."""
    controller = ToolCallGuardrailController()
    args = {"command": "cmd"}

    for i in range(8):
        controller.after_call("terminal", args, '{"exit_code":124}', failed=True)

    d = controller.after_call("terminal", args, '{"exit_code":124}', failed=True)
    if d.recent_failures is not None:
        assert len(d.recent_failures) <= 5


def test_failure_history_records_execute_code_status():
    """Failure history captures execute_code status field."""
    controller = ToolCallGuardrailController()
    args = {"code": "print(1/0)"}

    d1 = controller.after_call("execute_code", args, '{"status":"error","output":"ZeroDivisionError"}', failed=True)
    d2 = controller.after_call("execute_code", args, '{"status":"error","output":"ZeroDivisionError"}', failed=True)
    d3 = controller.after_call("execute_code", args, '{"status":"error","output":"ZeroDivisionError"}', failed=True)

    assert d3.code == "tool_call_self_check_required"
    assert d3.recent_failures is not None
    assert d3.recent_failures[0]["status"] == "error"
    assert "ZeroDivisionError" in d3.recent_failures[0]["output_tail"]


def test_selfcheck_guidance_includes_required_steps():
    """Self-check guidance includes concrete required self-check steps."""
    from agent.tool_guardrails import (
        ToolCallGuardrailController,
        append_toolguard_guidance,
    )
    import json as _json

    controller = ToolCallGuardrailController()
    args = {"command": "python3 server.py"}

    for _ in range(3):
        controller.after_call("terminal", args, '{"exit_code":124}', failed=True)

    d = controller.after_call("terminal", args, '{"exit_code":124}', failed=True)
    guided = append_toolguard_guidance('{"exit_code":124}', d)
    parsed = _json.loads(guided.split("\n\n")[1])

    required = parsed["required_self_check"]
    assert any("concrete change" in step for step in required)
    assert any("args.background" in step for step in required)
    assert any("unchanged" in step for step in required)

    assert "materially different" in parsed["required_next_step"]


def test_same_tool_varying_args_triggers_selfcheck():
    """same_tool_failure with varying args triggers self-check at count >= same_tool_failure_selfcheck_after."""
    from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController, append_toolguard_guidance

    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            same_tool_failure_selfcheck_after=3,
        )
    )

    # Three different commands, all failing — same tool, different args
    d1 = controller.after_call("terminal", {"command": "python3 server.py"}, '{"exit_code":124}', failed=True)
    d2 = controller.after_call("terminal", {"command": "exec python3 server.py"}, '{"exit_code":124}', failed=True)
    d3 = controller.after_call("terminal", {"command": "curl http://localhost:8189/health"}, '{"exit_code":7}', failed=True)

    # d1: count=1, allow
    assert d1.action == "allow"
    # d2: same_count=2, exact_count=1 — allow (below selfcheck_after=3)
    assert d2.action == "allow"
    # d3: same_count=3 — self-check!
    assert d3.action == "warn"
    assert d3.code == "tool_call_self_check_required"
    assert d3.failure_scope == "same_tool"
    assert d3.count == 3
    assert d3.recent_failures is not None
    assert len(d3.recent_failures) == 3  # all 3 failures aggregated by tool name
    # Check that failures include different exit codes
    exit_codes = [f.get("exit_code") for f in d3.recent_failures if "exit_code" in f]
    assert 124 in exit_codes
    assert 7 in exit_codes


def test_no_progress_triggers_selfcheck():
    """no_progress with repeated identical result triggers self-check at count >= no_progress_selfcheck_after."""
    from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            no_progress_selfcheck_after=3,
            no_progress_warn_after=2,
        )
    )
    args = {"path": "/tmp/same.txt"}
    result = "same file contents"

    d1 = controller.after_call("read_file", args, result, failed=False)
    d2 = controller.after_call("read_file", args, result, failed=False)
    d3 = controller.after_call("read_file", args, result, failed=False)

    assert d1.action == "allow"
    assert d2.action == "warn"
    assert d2.code == "no_progress_warning"
    # d3: repeat_count=3, self-check kicks in
    assert d3.action == "warn"
    assert d3.code == "tool_call_self_check_required"
    assert d3.failure_scope == "no_progress"
    assert d3.count == 3


def test_background_true_is_materially_different_args():
    """Adding background=True to terminal args should be a different signature, not blocked by exact failure."""
    from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_block_after=5,
        )
    )

    # Foreground call fails 5 times
    fg_args = {"command": "python3 server.py", "notify_on_complete": False}
    for i in range(5):
        d = controller.after_call("terminal", fg_args, '{"exit_code":124}', failed=True)

    # Foreground should be blocked now
    fg_decision = controller.before_call("terminal", fg_args)
    assert fg_decision.action == "block"
    assert fg_decision.code == "repeated_exact_failure_block"

    # Background call with same command should NOT be blocked — different signature
    bg_args = {"command": "python3 server.py", "background": True, "notify_on_complete": True}
    bg_decision = controller.before_call("terminal", bg_args)
    assert bg_decision.action == "allow"
    assert bg_decision.code != "repeated_exact_failure_block"


def test_selfcheck_json_includes_failure_scope():
    """Self-check JSON output includes failure_scope field."""
    from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController, append_toolguard_guidance

    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            exact_failure_selfcheck_after=3,
        )
    )
    args = {"command": "python3 server.py"}

    d = None
    for i in range(3):
        d = controller.after_call("terminal", args, '{"exit_code":124}', failed=True)

    assert d is not None
    assert d.code == "tool_call_self_check_required"
    assert d.failure_scope == "exact_signature"

    guided = append_toolguard_guidance('{"exit_code":124}', d)
    import json
    parsed = json.loads(guided.split("\n\n")[1])
    assert parsed["guardrail"]["failure_scope"] == "exact_signature"


def test_exit_code_7_triggers_selfcheck():
    """Self-check triggers for non-timeout exit codes (e.g., exit_code=7 connection refused)."""
    from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(exact_failure_selfcheck_after=3)
    )
    args = {"command": "curl http://localhost:8189/health"}

    d1 = controller.after_call("terminal", args, '{"exit_code":7}', failed=True)
    d2 = controller.after_call("terminal", args, '{"exit_code":7}', failed=True)
    d3 = controller.after_call("terminal", args, '{"exit_code":7}', failed=True)

    assert d1.action == "allow"
    assert d2.code == "repeated_exact_failure_warning"
    assert d3.code == "tool_call_self_check_required"
    assert d3.failure_scope == "exact_signature"
    # Check that exit_code_meaning is captured
    assert any(f.get("exit_code") == 7 for f in d3.recent_failures)
    assert any(f.get("exit_code_meaning") == "Failed to connect to host" for f in d3.recent_failures)


def test_exit_code_1_triggers_selfcheck():
    """Self-check triggers for general error exit codes (e.g., exit_code=1)."""
    from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(exact_failure_selfcheck_after=3)
    )
    args = {"command": "python3 script.py"}

    d1 = controller.after_call("terminal", args, '{"exit_code":1}', failed=True)
    d2 = controller.after_call("terminal", args, '{"exit_code":1}', failed=True)
    d3 = controller.after_call("terminal", args, '{"exit_code":1}', failed=True)

    assert d3.code == "tool_call_self_check_required"
    assert d3.failure_scope == "exact_signature"
    assert any(f.get("exit_code") == 1 for f in d3.recent_failures)
    assert any(f.get("exit_code_meaning") == "General error" for f in d3.recent_failures)
