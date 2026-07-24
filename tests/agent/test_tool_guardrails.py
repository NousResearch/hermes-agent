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
    assert cfg.cycle_warn_after == 3
    assert cfg.cycle_block_after == 5


def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
                "cycle": 6,
            },
            "hard_stop_after": {
                "exact_failure": 6,
                "same_tool_failure": 7,
                "idempotent_no_progress": 8,
                "cycle": 9,
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
    assert cfg.cycle_warn_after == 6
    assert cfg.cycle_block_after == 9


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
    # cycle thresholds raised: alternating two identical calls IS a legitimate
    # length-2 cycle; this test isolates the no_progress behavior only.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            no_progress_warn_after=2,
            no_progress_block_after=2,
            cycle_warn_after=99,
            cycle_block_after=99,
        )
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


def test_after_call_survives_lone_surrogates_in_result_and_args():
    # Scraped web/social text can contain unpaired UTF-16 surrogates (e.g. the
    # first half of a mathematical-bold pair, '\ud835'). str.encode('utf-8')
    # rejects them, and the result hasher crashed the whole conversation loop
    # (live outage: "Outer loop error in API call #34 ... surrogates not
    # allowed"). Weird text must never take down the loop.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, exact_failure_block_after=2, no_progress_block_after=2)
    )
    dirty = "price \ud835 update"

    decision = controller.after_call("web_search", {"query": dirty}, dirty, failed=False)
    assert decision.action in {"allow", "warn"}

    # hashing stays deterministic: the same dirty failure twice still trips
    # the exact-failure guard, proving the hash is stable across calls
    controller.after_call("web_search", {"query": dirty}, '{"error":"\ud835 boom"}', failed=True)
    controller.after_call("web_search", {"query": dirty}, '{"error":"\ud835 boom"}', failed=True)
    assert controller.before_call("web_search", {"query": dirty}).action == "block"


# ── Cyclic tool-call loop detection (ported from google-gemini/gemini-cli#28429) ──


def _cycle(controller, calls):
    """Feed (tool_name, args) pairs through after_call; return the decisions.

    Results are unique per call so the idempotent no-progress guard stays
    quiet and only cycle behavior is exercised.
    """
    return [
        controller.after_call(name, args, f"ok-{i}", failed=False)
        for i, (name, args) in enumerate(calls)
    ]


def test_alternating_two_call_cycle_warns_by_default():
    controller = ToolCallGuardrailController()
    a = ("read_file", {"path": "/tmp/loop_a.txt"})
    b = ("browser_click", {"ref": "@e5"})

    decisions = _cycle(controller, [a, b] * 3)

    assert [d.action for d in decisions[:-1]] == ["allow"] * 5
    last = decisions[-1]
    assert last.action == "warn"
    assert last.code == "tool_call_cycle_warning"
    assert last.count == 3
    assert "read_file -> browser_click" in last.message
    assert controller.halt_decision is None


def test_three_call_cycle_warns_by_default():
    controller = ToolCallGuardrailController()
    seq = [
        ("read_file", {"path": "/tmp/a"}),
        ("read_file", {"path": "/tmp/b"}),
        ("terminal", {"command": "ls /tmp"}),
    ]

    decisions = _cycle(controller, seq * 3)

    assert decisions[-1].action == "warn"
    assert decisions[-1].code == "tool_call_cycle_warning"
    assert decisions[-1].count == 3


def test_cycle_halts_with_hard_stop_enabled():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, cycle_warn_after=2, cycle_block_after=3)
    )
    a = ("web_search", {"query": "alpha"})
    b = ("web_search", {"query": "beta"})

    decisions = _cycle(controller, [a, b] * 3)

    assert decisions[-1].action == "halt"
    assert decisions[-1].code == "tool_call_cycle_halt"
    assert decisions[-1].count == 3
    assert controller.halt_decision is decisions[-1]


def test_broken_pattern_does_not_trip_cycle_detector():
    controller = ToolCallGuardrailController()
    a = ("read_file", {"path": "/tmp/loop_a.txt"})
    b = ("read_file", {"path": "/tmp/loop_b.txt"})
    c = ("read_file", {"path": "/tmp/loop_c.txt"})

    # A B A B, then break the pattern with C, then A B A B again
    decisions = _cycle(controller, [a, b, a, b, c, a, b, a, b])

    assert all(d.action == "allow" for d in decisions)


def test_uniform_self_repeats_do_not_trip_cycle_detector():
    # Back-to-back repeats of one call (e.g. polling a background process) are
    # deliberately out of scope for the cycle detector.
    controller = ToolCallGuardrailController()
    args = {"action": "poll", "session_id": "watch_1"}

    decisions = _cycle(controller, [("process", args)] * 12)

    assert all(d.action == "allow" for d in decisions)


def test_more_specific_failure_warning_wins_over_cycle_warning():
    controller = ToolCallGuardrailController()
    a = ("web_search", {"query": "same"})
    b = ("read_file", {"path": "/tmp/x"})

    for _ in range(3):
        controller.after_call(a[0], a[1], '{"error":"boom"}', failed=True)
        controller.after_call(b[0], b[1], "ok", failed=False)
    decision = controller.after_call(a[0], a[1], '{"error":"boom"}', failed=True)

    # cycle reps hit 3 here, but the exact-failure warning is more specific
    assert decision.action == "warn"
    assert decision.code == "repeated_exact_failure_warning"


def test_reset_for_turn_clears_cycle_state():
    controller = ToolCallGuardrailController()
    a = ("read_file", {"path": "/tmp/loop_a.txt"})
    b = ("browser_click", {"ref": "@e5"})

    _cycle(controller, [a, b, a, b])
    controller.reset_for_turn()
    decisions = _cycle(controller, [a, b])

    assert all(d.action == "allow" for d in decisions)
