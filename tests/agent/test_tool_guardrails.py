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














def test_mutating_or_unknown_tools_are_not_blocked_for_repeated_identical_success_output_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )

    for _ in range(3):
        assert controller.before_call("write_file", {"path": "/tmp/x", "content": "x"}).action == "allow"
        assert controller.after_call("write_file", {"path": "/tmp/x", "content": "x"}, "ok", failed=False).action == "allow"
        assert controller.before_call("custom_tool", {"x": 1}).action == "allow"
        assert controller.after_call("custom_tool", {"x": 1}, "ok", failed=False).action == "allow"






# ── Per-turn runaway-loop caps (Claude Code v2.1.212, Week 29) ──────────────

from agent.tool_guardrails import LoopCapConfig  # noqa: E402






def test_loop_cap_zero_disables_and_junk_falls_back():
    # 0 is a legitimate "unlimited" value; negatives / junk fall back to default.
    assert LoopCapConfig.from_mapping({"max_web_searches": 0}).max_web_searches == 0
    assert LoopCapConfig.from_mapping({"max_web_searches": -5}).max_web_searches == 50
    assert LoopCapConfig.from_mapping({"max_subagents": "nope"}).max_subagents == 50


def test_web_search_cap_blocks_after_limit_regardless_of_hard_stop():
    # Loop caps fire even with hard_stop_enabled=False (the per-turn loop
    # detector's flag). Each distinct query avoids the loop detector so we know
    # the block came from the loop cap, not exact-failure repetition.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=False,
            loop_caps=LoopCapConfig(max_web_searches=3),
        )
    )
    for i in range(3):
        assert controller.before_call("web_search", {"query": f"q{i}"}).action == "allow"
    decision = controller.before_call("web_search", {"query": "q4"})
    assert decision.action == "block"
    assert decision.code == "loop_web_search_cap"
    assert decision.should_halt is True


# ── Read-shaped calls of mutating tools (#92475) ─────────────────────────────

from agent.tool_guardrails import is_read_shaped_call  # noqa: E402


def test_todo_no_arg_read_gets_no_progress_warn_and_block():
    # `todo` is classified MUTATING (worst-case call), but the no-todos call
    # is a pure read (tools/todo_tool.py: `store.read()` whenever todos is
    # None). It cannot fail and returns identical bytes, so neither the
    # failure counters nor — before the fix — the no-progress detector ever
    # saw it. With hard_stop_enabled=True the read must warn at 2 identical
    # results and block the NEXT execution at 5, like any idempotent read.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True)
    )
    args = {"merge": False}

    actions = []
    for _ in range(5):
        assert controller.before_call("todo", args).action == "allow"
        actions.append(controller.after_call("todo", args, '{"todos":[]}').action)

    assert actions == ["allow", "warn", "warn", "warn", "warn"]

    blocked = controller.before_call("todo", args)
    assert blocked.action == "block"
    assert blocked.code == "idempotent_no_progress_block"


def test_todo_explicit_null_todos_follows_the_tools_real_read_branch():
    # todo_tool branches on `todos is not None`, so an EXPLICIT null is also
    # a pure read and must be detected identically to the absent-argument form.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True)
    )

    for i in range(2):
        assert controller.before_call("todo", {"todos": None}).action == "allow"
        controller.after_call("todo", {"todos": None}, '{"todos":[]}')

    assert controller.before_call("todo", {"todos": None}).action == "allow"
    third = controller.after_call("todo", {"todos": None}, '{"todos":[]}')
    assert third.action == "warn"
    assert third.code == "idempotent_no_progress_warning"


def test_todo_write_shaped_calls_keep_today_behaviour_exactly():
    # Any call that carries a todos payload goes through store.write() —
    # including an EMPTY list, which clears the list. Those calls must remain
    # outside no-progress tracking (and must reset any read streak recorded
    # under their own signature, which they always did by classification).
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True)
    )
    args = {"todos": [], "merge": False}

    for _ in range(7):
        assert controller.before_call("todo", args).action == "allow"
        assert controller.after_call("todo", args, '{"ok":true}').action == "allow"


def test_process_polling_stays_exempt_from_no_progress_detection():
    # Guard rail against over-generalizing: `process` repeats deliberately
    # (polling) and is already notice-exempt via STALL_GUARD_REPEATABLE_TOOLS.
    # It must never enter the no-progress tracker.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True)
    )
    args = {"action": "poll", "session_id": "s1"}

    for _ in range(7):
        assert controller.before_call("process", args).action == "allow"
        assert controller.after_call("process", args, '{"status":"running"}').action == "allow"


def test_read_shaped_call_table_matches_todo_branch_semantics():
    # The helper mirrors todo_tool's own branch (`todos is not None` means
    # write): only absent or explicitly-null payloads count as reads.
    assert is_read_shaped_call("todo", {}) is True
    assert is_read_shaped_call("todo", {"merge": False}) is True
    assert is_read_shaped_call("todo", {"todos": None}) is True
    assert is_read_shaped_call("todo", {"todos": []}) is False
    assert is_read_shaped_call("todo", {"todos": [{"id": "x"}]}) is False
    # Unlisted tools have no read shape — name-set behaviour is untouched.
    assert is_read_shaped_call("write_file", {}) is False
    assert is_read_shaped_call("web_search", {}) is False










