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


# ── tool_repetition guardrail tests ──────────────────────────────────────────


def test_default_tool_repetition_config_values():
    cfg = ToolCallGuardrailConfig()

    assert cfg.tool_repetition_warn_after == 5
    assert cfg.tool_repetition_block_after == 8


def test_config_parses_tool_repetition_thresholds_from_nested_sections():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warn_after": {"tool_repetition": 3},
            "hard_stop_after": {"tool_repetition": 6},
        }
    )

    assert cfg.tool_repetition_warn_after == 3
    assert cfg.tool_repetition_block_after == 6


def test_config_parses_tool_repetition_thresholds_from_flat_keys():
    """top-level flat keys (legacy compat) still work."""
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "tool_repetition_warn_after": 4,
            "tool_repetition_block_after": 7,
        }
    )

    assert cfg.tool_repetition_warn_after == 4
    assert cfg.tool_repetition_block_after == 7


def test_warns_on_repeated_identical_mutating_tool_call_without_hard_stop():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(tool_repetition_warn_after=2, tool_repetition_block_after=8)
    )
    args = {"command": "ls"}

    decision = None
    # 4 identical calls; warning threshold at 2, no hard stop
    for i in range(4):
        assert controller.before_call("terminal", args).action == "allow"
        decision = controller.after_call("terminal", args, '{"exit_code":0}', failed=False)

    assert decision is not None
    assert decision.action == "warn"
    assert decision.code == "tool_repetition_warning"
    assert decision.count == 4
    assert controller.halt_decision is None
    assert controller.before_call("terminal", args).action == "allow"


def test_hard_stop_blocks_repeated_identical_mutating_tool_call_before_next():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            tool_repetition_warn_after=2,
            tool_repetition_block_after=3,
        )
    )
    args = {"command": "false"}

    decision = None
    # 2 identical "successful" calls — count=2 triggers warn (not block yet at 3)
    for i in range(2):
        assert controller.before_call("terminal", args).action == "allow"
        decision = controller.after_call("terminal", args, '{"exit_code":0}', failed=False)

    assert decision is not None
    assert decision.action == "warn"
    assert decision.code == "tool_repetition_warning"

    # Third attempt: after_call count=3 triggers block
    assert controller.before_call("terminal", args).action == "allow"
    blocked = controller.after_call("terminal", args, '{"exit_code":0}', failed=False)
    assert blocked.action == "block"
    assert blocked.code == "tool_repetition_block"
    assert blocked.count == 3

    # Fourth attempt blocked BEFORE execution
    fourth = controller.before_call("terminal", args)
    assert fourth.action == "block"


def test_different_args_create_different_signatures_so_repetition_independent():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            tool_repetition_warn_after=2,
        )
    )

    # Call A with args {command: "cmd-a"} 3× → should warn
    for _ in range(2):
        controller.before_call("terminal", {"command": "cmd-a"})
        controller.after_call("terminal", {"command": "cmd-a"}, "ok", failed=False)
    decision_a = controller.after_call("terminal", {"command": "cmd-a"}, "ok", failed=False)
    assert decision_a.code == "tool_repetition_warning"

    # Call B with different args — fresh counter
    assert controller.before_call("terminal", {"command": "cmd-b"}).action == "allow"
    decision_b = controller.after_call("terminal", {"command": "cmd-b"}, "ok", failed=False)
    assert decision_b.action == "allow"


def test_single_success_between_repeated_calls_resets_repetition_counter():
    """A different tool call breaks the repetition streak for THAT signature."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(tool_repetition_warn_after=2)
    )

    # 2 identical calls → should warn
    controller.after_call("terminal", {"command": "ls"}, "ok", failed=False)
    warn = controller.after_call("terminal", {"command": "ls"}, "ok", failed=False)
    assert warn.code == "tool_repetition_warning"

    # A completely different tool call does NOT reset — repetition is per-signature
    controller.after_call("read_file", {"path": "/tmp/x"}, "contents", failed=False)

    # Same terminal("ls") signature already at count 2 — still triggers
    decision = controller.after_call("terminal", {"command": "ls"}, "ok", failed=False)
    assert decision.code == "tool_repetition_warning"
    assert decision.count == 3


def test_tool_repetition_warns_for_mutating_tools_including_browser_navigate():
    """browser_navigate is a mutating tool; repeated identical calls should warn."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(tool_repetition_warn_after=2)
    )

    decision = None
    for _ in range(3):
        controller.before_call("browser_navigate", {"url": "https://example.com/404"})
        decision = controller.after_call(
            "browser_navigate",
            {"url": "https://example.com/404"},
            "Page loaded: 404 Not Found",
            failed=False,
        )

    assert decision is not None
    assert decision.code == "tool_repetition_warning"
    assert "browser_navigate" in decision.message


def test_reset_for_turn_clears_tool_repetition_state():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            tool_repetition_block_after=2,
        )
    )

    controller.after_call("terminal", {"command": "ls"}, "ok", failed=False)
    controller.after_call("terminal", {"command": "ls"}, "ok", failed=False)

    assert controller.before_call("terminal", {"command": "ls"}).action == "block"

    controller.reset_for_turn()

    assert controller.before_call("terminal", {"command": "ls"}).action == "allow"


def test_failed_call_does_not_count_toward_tool_repetition():
    """Failed calls should be handled by failure guardrails, not repetition."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            tool_repetition_warn_after=2,
            exact_failure_warn_after=2,
        )
    )

    # Two failed calls — triggers failure warning, NOT repetition
    controller.after_call("terminal", {"command": "bad"}, '{"exit_code":1}', failed=True)
    decision = controller.after_call("terminal", {"command": "bad"}, '{"exit_code":1}', failed=True)
    assert decision.code == "repeated_exact_failure_warning"

    # Repetition counter for this signature should be ZERO (failures don't count)
    assert controller._tool_repetition_counts.get(
        ToolCallSignature.from_call("terminal", {"command": "bad"}), 0
    ) == 0
