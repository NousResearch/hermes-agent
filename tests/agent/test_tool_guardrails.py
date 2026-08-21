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














def test_memory_terminal_degradation_is_not_a_tool_failure():
    # The memory tool's terminal graceful-degradation result (#42405) carries
    # done=True and already tells the model to stop retrying. Counting it as a
    # failure feeds the same-tool halt counter, which aborts the turn and eats
    # the user-facing reply - the exact outcome #42405 exists to prevent.
    # The verdict is shared with agent/display.py:_detect_tool_failure through
    # agent.tool_result_classification.classify_memory_result.
    terminal = json.dumps({
        "success": False,
        "done": True,
        "error": (
            "Memory consolidation failed 4 times this turn. Stop retrying "
            "memory calls - leave memory unchanged for now and continue with "
            "your reply to the user."
        ),
    })
    full = json.dumps({
        "success": False,
        "error": "Memory at 3,990/4,000 chars. Adding this entry would exceed the limit.",
    })

    assert classify_tool_failure("memory", terminal) == (False, "")
    # A genuine at-capacity error is still a failure.
    assert classify_tool_failure("memory", full) == (True, " [full]")
    # And a memory error that is neither settled nor at-capacity still falls
    # through to the generic rules — the shared classifier returns "no verdict"
    # for it rather than swallowing it as a success.
    other = json.dumps({"success": False, "error": "target 'USER' not found"})
    assert classify_tool_failure("memory", other) == (True, " [error]")


def test_repeated_terminal_degradation_never_halts_the_turn():
    """The consequence, not just the classification.

    The #42405 symptom is the halt controller aborting the turn after enough
    same-tool failures. Feeding the terminal payload straight through the
    controller — past the halt threshold — is what proves the reply survives;
    a classifier assertion alone would not notice a consumer that stopped
    honouring it.
    """
    terminal = json.dumps({
        "success": False,
        "done": True,
        "error": "Memory consolidation failed 4 times this turn. Stop retrying memory calls.",
    })
    args = {"action": "add", "content": "Adrien prefers HT amounts"}

    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, same_tool_failure_halt_after=3)
    )
    for _ in range(6):  # twice the halt threshold
        decision = controller.after_call("memory", args, terminal)
        assert decision.code != "same_tool_failure_halt"
        assert decision.should_halt is False


def test_repeated_real_memory_failure_still_halts_the_turn():
    """Control for the test above: the halt itself is not what got broken."""
    failing = json.dumps({"success": False, "error": "target 'USER' not found"})
    args = {"action": "add", "content": "Adrien prefers HT amounts"}

    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, same_tool_failure_halt_after=3)
    )
    decisions = [controller.after_call("memory", args, failing) for _ in range(3)]
    assert decisions[-1].code == "same_tool_failure_halt"
    assert decisions[-1].should_halt is True


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










