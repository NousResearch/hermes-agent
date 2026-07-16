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
    assert cfg.same_tool_failure_warn_after == 3
    assert cfg.no_progress_warn_after == 2
    assert cfg.exact_failure_block_after == 5
    assert cfg.same_tool_failure_halt_after == 8
    assert cfg.no_progress_block_after == 5
    assert cfg.repeated_success_warn_after == 4
    assert cfg.turn_volume_warn_after == 25


def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
                "repeated_success": 6,
                "turn_volume": 30,
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
    assert cfg.repeated_success_warn_after == 6
    assert cfg.turn_volume_warn_after == 30


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
    assert {d.code for d in decisions[1:]} == {"repeated_exact_failure_warning"}
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
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(same_tool_failure_warn_after=2, same_tool_failure_halt_after=3)
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
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )
    args = {"path": "/tmp/same.txt"}
    result = "same file contents"

    for _ in range(4):
        assert controller.before_call("read_file", args).action == "allow"
        decision = controller.after_call("read_file", args, result, failed=False)

    assert decision.action == "warn"
    assert decision.code == "idempotent_no_progress_warning"
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
    assert warn.code == "idempotent_no_progress_warning"

    blocked = controller.before_call("read_file", args)
    assert blocked.action == "block"
    assert blocked.code == "idempotent_no_progress_block"


def test_mutating_or_unknown_tools_are_not_blocked_for_repeated_identical_success_output_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )

    for _ in range(3):
        assert controller.before_call("write_file", {"path": "/tmp/x", "content": "x"}).action == "allow"
        assert controller.after_call("write_file", {"path": "/tmp/x", "content": "x"}, "ok", failed=False).action == "allow"
        assert controller.before_call("custom_tool", {"x": 1}).action == "allow"
        assert controller.after_call("custom_tool", {"x": 1}, "ok", failed=False).action == "allow"


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

    assert controller.before_call("web_search", {"query": "same"}).action == "allow"
    assert controller.before_call("read_file", {"path": "/tmp/x"}).action == "allow"

def test_repeated_identical_success_on_mutating_tool_warns_without_blocking():
    controller = ToolCallGuardrailController()
    args = {"command": "ls -la /tmp"}

    decisions = []
    for _ in range(5):
        assert controller.before_call("terminal", args).action == "allow"
        decisions.append(
            controller.after_call("terminal", args, '{"exit_code":0}', failed=False)
        )

    assert [d.action for d in decisions[:3]] == ["allow", "allow", "allow"]
    assert [d.action for d in decisions[3:]] == ["warn", "warn"]
    assert {d.code for d in decisions[3:]} == {"repeated_success_warning"}
    assert decisions[3].count == 4
    assert decisions[4].count == 5
    assert "redundant" in decisions[3].message
    # Warn-only: execution is never prevented and no halt is recorded.
    assert controller.before_call("terminal", args).action == "allow"
    assert controller.halt_decision is None


def test_success_streak_broken_by_different_args_or_failure():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(repeated_success_warn_after=3)
    )

    # Two identical successes, then different args: streak restarts.
    controller.after_call("terminal", {"command": "a"}, '{"exit_code":0}', failed=False)
    controller.after_call("terminal", {"command": "a"}, '{"exit_code":0}', failed=False)
    controller.after_call("terminal", {"command": "b"}, '{"exit_code":0}', failed=False)
    third = controller.after_call("terminal", {"command": "a"}, '{"exit_code":0}', failed=False)
    assert third.action == "allow"

    # Two identical successes, then a failure of the same call: streak resets.
    controller.reset_for_turn()
    controller.after_call("terminal", {"command": "a"}, '{"exit_code":0}', failed=False)
    controller.after_call("terminal", {"command": "a"}, '{"exit_code":0}', failed=False)
    controller.after_call("terminal", {"command": "a"}, '{"exit_code":1}', failed=True)
    after_failure = controller.after_call(
        "terminal", {"command": "a"}, '{"exit_code":0}', failed=False
    )
    assert after_failure.action == "allow"


def test_idempotent_tools_use_no_progress_code_not_repeated_success():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(repeated_success_warn_after=2, no_progress_warn_after=2)
    )
    args = {"path": "/tmp/same.txt"}

    first = controller.after_call("read_file", args, "same contents", failed=False)
    second = controller.after_call("read_file", args, "same contents", failed=False)

    assert first.action == "allow"
    assert second.action == "warn"
    assert second.code == "idempotent_no_progress_warning"

    # Identical args but progressing results (e.g. polling) must not warn as
    # a redundant-success loop: no-progress covers the idempotent family.
    controller.reset_for_turn()
    a = controller.after_call("read_file", args, "contents v1", failed=False)
    b = controller.after_call("read_file", args, "contents v2", failed=False)
    c = controller.after_call("read_file", args, "contents v3", failed=False)
    assert [a.action, b.action, c.action] == ["allow", "allow", "allow"]


def test_turn_volume_warns_once_per_turn_and_resets_at_turn_boundary():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(turn_volume_warn_after=3)
    )

    first = controller.after_call("web_search", {"query": "q1"}, "r1", failed=False)
    second = controller.after_call("terminal", {"command": "c1"}, '{"exit_code":0}', failed=False)
    third = controller.after_call("write_file", {"path": "/tmp/a", "content": "x"}, "ok", failed=False)
    fourth = controller.after_call("web_search", {"query": "q2"}, "r2", failed=False)

    assert [first.action, second.action] == ["allow", "allow"]
    assert third.action == "warn"
    assert third.code == "turn_volume_warning"
    assert third.count == 3
    # Only once per turn — subsequent calls stay allow.
    assert fourth.action == "allow"
    assert controller.halt_decision is None

    controller.reset_for_turn()
    after_reset = controller.after_call("web_search", {"query": "q3"}, "r3", failed=False)
    assert after_reset.action == "allow"


def test_turn_volume_counts_failed_calls_too():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            turn_volume_warn_after=2,
            exact_failure_warn_after=99,
            same_tool_failure_warn_after=99,
        )
    )

    first = controller.after_call("terminal", {"command": "a"}, '{"exit_code":1}', failed=True)
    second = controller.after_call("terminal", {"command": "b"}, '{"exit_code":1}', failed=True)

    assert first.action == "allow"
    assert second.action == "warn"
    assert second.code == "turn_volume_warning"


def test_warnings_disabled_suppresses_success_and_volume_warnings():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            warnings_enabled=False,
            repeated_success_warn_after=2,
            turn_volume_warn_after=2,
        )
    )
    args = {"command": "same"}

    for _ in range(4):
        decision = controller.after_call("terminal", args, '{"exit_code":0}', failed=False)
        assert decision.action == "allow"


def test_reset_for_turn_clears_success_streak_state():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(repeated_success_warn_after=3)
    )
    args = {"command": "same"}
    controller.after_call("terminal", args, '{"exit_code":0}', failed=False)
    controller.after_call("terminal", args, '{"exit_code":0}', failed=False)

    controller.reset_for_turn()

    # Streak restarted: the third identical success is call #1 of the new turn.
    resumed = controller.after_call("terminal", args, '{"exit_code":0}', failed=False)
    assert resumed.action == "allow"
