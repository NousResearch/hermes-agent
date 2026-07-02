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
    assert cfg.block_enabled is True
    assert cfg.hard_stop_enabled is False
    assert cfg.exact_failure_warn_after == 2
    assert cfg.same_tool_failure_warn_after == 3
    assert cfg.no_progress_warn_after == 2
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


def test_default_repeated_identical_failed_call_warns_then_soft_blocks():
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

    # 6th identical attempt: soft-blocked without executing, turn keeps going.
    blocked = controller.before_call("web_search", args)
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"
    assert blocked.soft is True
    assert blocked.allows_execution is False
    assert blocked.should_halt is False
    assert controller.halt_decision is None


def test_warn_only_legacy_behavior_when_block_disabled():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(block_enabled=False)
    )
    args = {"query": "same"}

    for _ in range(8):
        assert controller.before_call("web_search", args).action == "allow"
        controller.after_call("web_search", args, '{"error":"boom"}', failed=True)

    assert controller.before_call("web_search", args).action == "allow"
    assert controller.halt_decision is None


def test_consecutive_soft_blocks_escalate_count_and_vary_message():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(exact_failure_block_after=2)
    )
    args = {"command": "npm run dev"}

    controller.after_call("terminal", args, '{"exit_code":-1,"error":"boom"}', failed=True)
    controller.after_call("terminal", args, '{"exit_code":-1,"error":"boom"}', failed=True)

    first = controller.before_call("terminal", args)
    second = controller.before_call("terminal", args)
    third = controller.before_call("terminal", args)

    assert [d.action for d in (first, second, third)] == ["block", "block", "block"]
    assert [d.soft for d in (first, second, third)] == [True, True, True]
    # Blocked attempts count as failed attempts so the streak keeps escalating.
    assert [d.count for d in (first, second, third)] == [3, 4, 5]
    # Wording must differ between consecutive blocks — uniform guidance gets
    # absorbed into the loop pattern instead of breaking it.
    assert len({first.message, second.message, third.message}) == 3
    assert controller.halt_decision is None


def test_soft_block_success_after_changed_args_resets_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(exact_failure_block_after=2)
    )
    bad = {"command": "npm run dev"}
    fixed = {"command": "npm run dev", "background": True}

    controller.after_call("terminal", bad, '{"exit_code":-1,"error":"boom"}', failed=True)
    controller.after_call("terminal", bad, '{"exit_code":-1,"error":"boom"}', failed=True)
    assert controller.before_call("terminal", bad).action == "block"

    # A different call is never blocked.
    assert controller.before_call("terminal", fixed).action == "allow"
    controller.after_call("terminal", fixed, '{"exit_code":0}', failed=False)

    # The old signature stays blocked until its streak is cleared by a success.
    assert controller.before_call("terminal", bad).action == "block"
    controller.after_call("terminal", bad, '{"exit_code":0}', failed=False)
    assert controller.before_call("terminal", bad).action == "allow"


def test_soft_block_metadata_marks_decision_as_soft():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(exact_failure_block_after=1)
    )
    controller.after_call("web_search", {"q": "x"}, '{"error":"boom"}', failed=True)
    blocked = controller.before_call("web_search", {"q": "x"})

    metadata = blocked.to_metadata()
    assert metadata["soft"] is True
    assert metadata["action"] == "block"


def test_config_parses_block_enabled_flag():
    cfg = ToolCallGuardrailConfig.from_mapping({"block_enabled": False})
    assert cfg.block_enabled is False
    assert ToolCallGuardrailConfig.from_mapping({}).block_enabled is True


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


def test_idempotent_no_progress_repeated_result_warns_then_soft_blocks_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )
    args = {"path": "/tmp/same.txt"}
    result = "same file contents"

    assert controller.before_call("read_file", args).action == "allow"
    assert controller.after_call("read_file", args, result, failed=False).action == "allow"
    assert controller.before_call("read_file", args).action == "allow"
    warn = controller.after_call("read_file", args, result, failed=False)
    assert warn.action == "warn"
    assert warn.code == "idempotent_no_progress_warning"

    # Threshold reached: identical read is soft-blocked, turn keeps going.
    first = controller.before_call("read_file", args)
    second = controller.before_call("read_file", args)
    assert [d.action for d in (first, second)] == ["block", "block"]
    assert [d.soft for d in (first, second)] == [True, True]
    assert first.code == "idempotent_no_progress_block"
    assert first.message != second.message
    assert controller.halt_decision is None

    # A changed result clears the streak.
    controller.after_call("read_file", args, "new contents", failed=False)
    assert controller.before_call("read_file", args).action == "allow"


def test_no_progress_soft_block_disabled_with_block_enabled_false():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(block_enabled=False, no_progress_warn_after=2, no_progress_block_after=2)
    )
    args = {"path": "/tmp/same.txt"}

    for _ in range(5):
        assert controller.before_call("read_file", args).action == "allow"
        controller.after_call("read_file", args, "same file contents", failed=False)

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
