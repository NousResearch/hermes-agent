"""Pure tool-call guardrail primitive tests."""

import json

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    _terminal_probe_family,
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


def test_default_config_warns_and_redirects_without_hard_stop():
    cfg = ToolCallGuardrailConfig()

    assert cfg.warnings_enabled is True
    assert cfg.hard_stop_enabled is False
    assert cfg.exact_failure_warn_after == 2
    assert cfg.same_tool_failure_warn_after == 3
    assert cfg.no_progress_warn_after == 2
    assert cfg.low_information_warn_after == 3
    assert cfg.exact_failure_block_after == 5
    assert cfg.same_tool_failure_halt_after == 8
    assert cfg.no_progress_block_after == 5
    assert cfg.low_information_redirect_after == 4
    assert cfg.low_information_halt_after == 6


def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
                "low_information": 6,
            },
            "hard_stop_after": {
                "exact_failure": 6,
                "same_tool_failure": 7,
                "idempotent_no_progress": 8,
                "low_information_redirect": 9,
                "low_information": 10,
            },
        }
    )

    assert cfg.warnings_enabled is False
    assert cfg.hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 3
    assert cfg.same_tool_failure_warn_after == 4
    assert cfg.no_progress_warn_after == 5
    assert cfg.low_information_warn_after == 6
    assert cfg.exact_failure_block_after == 6
    assert cfg.same_tool_failure_halt_after == 7
    assert cfg.no_progress_block_after == 8
    assert cfg.low_information_redirect_after == 9
    assert cfg.low_information_halt_after == 10


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


def test_tool_reported_loop_block_halts_even_when_hard_stop_disabled():
    controller = ToolCallGuardrailController()
    args = {"pattern": "def.*drain", "path": "/repo", "target": "content"}
    result = json.dumps(
        {
            "error": (
                "BLOCKED: You have run this exact search 4 times in a row. "
                "The results have NOT changed."
            ),
            "already_searched": 4,
        }
    )

    decision = controller.after_call(
        "search_files",
        args,
        result,
        failed=True,
    )

    assert decision.action == "halt"
    assert decision.code == "tool_reported_loop_block"
    assert decision.count == 4
    assert controller.halt_decision == decision


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
    assert "verify the commit/PR state" in second.message
    assert "python3.11" in second.message
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


def test_low_information_search_variants_redirect_before_exact_repeat_block():
    controller = ToolCallGuardrailController()
    empty_search = json.dumps({"total_count": 0})

    decisions = [
        controller.after_call(
            "search_files",
            {"pattern": pattern, "path": "/repo", "target": "content"},
            empty_search,
            failed=False,
        )
        for pattern in ("def.*drain", "drain_command", "dispatch_command", "select.*task")
    ]

    assert [decision.action for decision in decisions] == ["allow", "allow", "warn", "warn"]
    assert decisions[2].code == "low_information_strategy_warning"
    assert decisions[3].code == "low_information_strategy_warning"

    redirected = controller.before_call(
        "search_files",
        {"pattern": "another variation", "path": "/repo", "target": "content"},
    )

    assert redirected.action == "redirect"
    assert redirected.code == "low_information_tool_redirect"
    assert redirected.should_halt is False
    assert controller.halt_decision is None

    controller.after_call(
        "terminal",
        {"command": "pwd && rg --files | head"},
        json.dumps({"exit_code": 0, "output": "/repo\nrun_agent.py"}),
        failed=False,
    )

    assert controller.before_call(
        "search_files",
        {"pattern": "run_conversation", "path": "/repo", "target": "content"},
    ).action == "allow"


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


def test_terminal_filtered_empty_probe_streak_redirects_more_filter_churn():
    controller = ToolCallGuardrailController()
    empty_success = json.dumps({"exit_code": 0, "stdout": "", "stderr": ""})
    commands = [
        'cd ~/Workspaces/Projects/meshboard && git diff main --name-only 2>/dev/null | head -30',
        'cd ~/Workspaces/Projects/meshboard && git diff main --name-only 2>/dev/null | grep -v "^tests/" | head -20',
        'cd ~/Workspaces/Projects/meshboard && grep -r "dispatch.*history" tools/ --include="*.py" -l 2>/dev/null | head -10',
        'cd ~/Workspaces/Projects/meshboard && rg -n "dispatch history" tools | head -20',
    ]

    decisions = []
    for command in commands:
        assert controller.before_call("terminal", {"command": command}).action == "allow"
        decisions.append(
            controller.after_call(
                "terminal",
                {"command": command},
                empty_success,
                failed=False,
            )
        )

    assert decisions[2].code == "low_information_strategy_warning"
    assert decisions[3].code == "low_information_strategy_warning"

    redirected = controller.before_call(
        "terminal",
        {
            "command": (
                'cd ~/Workspaces/Projects/meshboard && git diff main --name-only '
                '2>/dev/null | grep -v "^docs/" | head -20'
            )
        },
    )

    assert redirected.action == "redirect"
    assert redirected.code == "low_information_tool_redirect"
    assert "filtered shell probes" in redirected.message


def test_terminal_filter_redirect_allows_broad_diagnostic_to_reset():
    controller = ToolCallGuardrailController()
    empty_success = json.dumps({"exit_code": 0, "output": ""})
    for i in range(4):
        controller.after_call(
            "terminal",
            {"command": f'git diff main --name-only | grep -v "path-{i}" | head -20'},
            empty_success,
            failed=False,
        )

    broad = controller.before_call("terminal", {"command": "pwd && ls -la"})

    assert broad.action == "allow"
    assert controller.before_call(
        "terminal",
        {"command": 'git diff main --name-only | grep -v "docs" | head -20'},
    ).action == "allow"


def test_terminal_probe_family_detects_find_xargs_grep_but_not_xargs_mutation():
    assert _terminal_probe_family(
        {"command": 'find tools -type f -name "*.py" | xargs grep -n "dispatch"'}
    ) == "filter_probe"
    assert _terminal_probe_family(
        {"command": 'find tools -type f -print0 | xargs -0 rg -n "dispatch"'}
    ) == "filter_probe"
    assert _terminal_probe_family(
        {"command": 'find tools -type f -name "*.tmp" | xargs rm'}
    ) is None


def test_terminal_filtered_empty_probe_streak_includes_find_xargs_grep_chains():
    controller = ToolCallGuardrailController()
    empty_success = json.dumps({"exit_code": 0, "stdout": "", "stderr": ""})
    commands = [
        'find tools -type f -name "*.py" | xargs grep -n "dispatch"',
        'find tools -type f -name "*.py" | xargs grep -n "dispatch_history"',
        'find tools -type f -print0 | xargs -0 grep -n "route history"',
        'find tools -type f -print0 | xargs -0 rg -n "dispatch history"',
    ]

    for command in commands:
        assert controller.before_call("terminal", {"command": command}).action == "allow"
        controller.after_call("terminal", {"command": command}, empty_success, failed=False)

    redirected = controller.before_call(
        "terminal",
        {"command": 'find tools -type f -name "*.py" | xargs grep -n "again"'},
    )

    assert redirected.action == "redirect"
    assert redirected.code == "low_information_tool_redirect"


def test_terminal_filter_redirect_does_not_halt_by_default():
    controller = ToolCallGuardrailController()
    empty_success = json.dumps({"exit_code": 0, "stdout": "", "stderr": ""})
    for i in range(4):
        controller.after_call(
            "terminal",
            {"command": f'git diff main --name-only | grep "path-{i}" | head -20'},
            empty_success,
            failed=False,
        )

    last = None
    for i in range(10):
        last = controller.before_call(
            "terminal",
            {"command": f'git diff main --name-only | grep "again-{i}" | head -20'},
        )

    assert last is not None
    assert last.action == "redirect"
    assert last.code == "low_information_tool_redirect"
    assert not last.should_halt
    assert controller.halt_decision is None


def test_terminal_filter_redirect_can_halt_when_hard_stop_is_enabled():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True)
    )
    empty_success = json.dumps({"exit_code": 0, "stdout": "", "stderr": ""})
    for i in range(4):
        controller.after_call(
            "terminal",
            {"command": f'git diff main --name-only | grep "path-{i}" | head -20'},
            empty_success,
            failed=False,
        )

    last = None
    for i in range(2):
        last = controller.before_call(
            "terminal",
            {"command": f'git diff main --name-only | grep "again-{i}" | head -20'},
        )

    assert last is not None
    assert last.action == "halt"
    assert last.code == "low_information_tool_halt"
    assert controller.halt_decision == last


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
