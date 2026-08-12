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











# ── guardrail lifecycle hook emission ───────────────────────────────────────

import pytest  # noqa: E402

from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager  # noqa: E402


@pytest.fixture()
def captured_guardrail_hooks():
    """Capture guardrail_block/guardrail_halt firings; restore the manager."""
    manager = get_plugin_manager()
    fired = []

    def _capture(event):
        def _cb(**kwargs):
            fired.append((event, kwargs))
        return _cb

    for event in ("guardrail_block", "guardrail_halt"):
        manager._hooks.setdefault(event, []).append(_capture(event))
    try:
        yield fired
    finally:
        for event in ("guardrail_block", "guardrail_halt"):
            manager._hooks[event] = [
                cb for cb in manager._hooks.get(event, [])
                if not getattr(cb, "__name__", "") == "_cb"
            ]


def _hard_stop_controller(**overrides):
    return ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, **overrides)
    )


def test_guardrail_events_registered_as_valid_hooks():
    assert "guardrail_block" in VALID_HOOKS
    assert "guardrail_halt" in VALID_HOOKS


def test_exact_failure_block_fires_guardrail_block_hook(captured_guardrail_hooks):
    controller = _hard_stop_controller()
    args = {"command": "false"}
    for _ in range(5):
        controller.before_call("terminal", args)
        controller.after_call("terminal", args, '{"exit_code": 1}', failed=True)

    assert captured_guardrail_hooks == []  # nothing below the block threshold
    decision = controller.before_call("terminal", args)
    assert decision.action == "block"
    assert decision.code == "repeated_exact_failure_block"
    assert len(captured_guardrail_hooks) == 1
    event, kwargs = captured_guardrail_hooks[0]
    assert event == "guardrail_block"
    assert kwargs["tool_name"] == "terminal"
    assert kwargs["code"] == "repeated_exact_failure_block"
    assert kwargs["count"] == 5
    assert kwargs["action"] == "block"
    assert kwargs["message"] == decision.message


def test_no_progress_block_fires_guardrail_block_hook(captured_guardrail_hooks):
    controller = _hard_stop_controller()
    args = {"query": "same"}
    for _ in range(5):
        controller.before_call("web_search", args)
        controller.after_call("web_search", args, "same result", failed=False)

    decision = controller.before_call("web_search", args)
    assert decision.action == "block"
    assert decision.code == "idempotent_no_progress_block"
    assert [event for event, _ in captured_guardrail_hooks] == ["guardrail_block"]
    assert captured_guardrail_hooks[0][1]["count"] == 5


def test_same_tool_failure_halt_fires_guardrail_halt_hook(captured_guardrail_hooks):
    controller = _hard_stop_controller()
    decision = None
    for i in range(8):  # distinct args: streak counts the tool, not the args
        decision = controller.after_call(
            "terminal", {"command": f"cmd-{i}"}, "Error: boom", failed=True
        )
    assert decision is not None
    assert decision.action == "halt"
    assert decision.code == "same_tool_failure_halt"
    assert [event for event, _ in captured_guardrail_hooks] == ["guardrail_halt"]
    kwargs = captured_guardrail_hooks[0][1]
    assert kwargs["tool_name"] == "terminal"
    assert kwargs["code"] == "same_tool_failure_halt"
    assert kwargs["count"] == 8
    assert kwargs["action"] == "halt"


def test_allow_and_warn_decisions_do_not_fire_hooks(captured_guardrail_hooks):
    controller = _hard_stop_controller()
    args = {"query": "same"}
    for _ in range(4):  # warns from the 2nd on; below every block/halt threshold
        assert controller.before_call("web_search", args).action == "allow"
        decision = controller.after_call("web_search", args, "x", failed=True)
    assert decision.action == "warn"
    assert controller.halt_decision is None
    assert captured_guardrail_hooks == []


def test_loop_cap_block_fires_guardrail_block_hook(captured_guardrail_hooks):
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=False,
            loop_caps=LoopCapConfig(max_web_searches=3),
        )
    )
    for i in range(3):
        controller.before_call("web_search", {"query": f"q{i}"})
    decision = controller.before_call("web_search", {"query": "q4"})
    assert decision.code == "loop_web_search_cap"
    assert [event for event, _ in captured_guardrail_hooks] == ["guardrail_block"]


def test_raising_hook_does_not_break_decisions():
    manager = get_plugin_manager()

    def _boom(**kwargs):
        raise RuntimeError("hook exploded")

    manager._hooks.setdefault("guardrail_block", []).append(_boom)
    try:
        controller = _hard_stop_controller()
        args = {"command": "false"}
        for _ in range(5):
            controller.before_call("terminal", args)
            controller.after_call("terminal", args, "x", failed=True)
        # The block decision must still be made and returned.
        decision = controller.before_call("terminal", args)
        assert decision.action == "block"
        assert controller.halt_decision is decision
    finally:
        manager._hooks["guardrail_block"] = [
            cb for cb in manager._hooks["guardrail_block"] if cb is not _boom
        ]
