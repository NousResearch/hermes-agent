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


# ── Read-only shell commands must be loop-detected (#loop-517) ──────────────
# A session repeated `gh api .../reviews` 517 times, each returning an identical
# empty `[]` with exit 0. `terminal` is in MUTATING_TOOL_NAMES, so the
# no-progress detector never looked at it, and exit 0 meant the failure
# detector never fired either. Nothing stopped it until max_iterations.


def test_read_only_shell_command_repeating_identical_output_is_blocked():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=3)
    )
    args = {"command": "cd ~/repo && gh api repos/o/r/pulls/1/reviews 2>&1", "timeout": 15}
    result = '{"output": "[]", "exit_code": 0, "error": null}'

    for _ in range(3):
        assert controller.before_call("terminal", args).allows_execution
        controller.after_call("terminal", args, result, failed=False)

    decision = controller.before_call("terminal", args)
    assert decision.action == "block"
    assert decision.code == "idempotent_no_progress_block"


def test_mutating_shell_command_repeating_identical_output_is_not_blocked():
    # `git push` succeeding identically twice is not evidence of a loop, and
    # blocking a write would be worse than letting it through.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=2)
    )
    args = {"command": "git push origin main", "timeout": 30}
    result = '{"output": "Everything up-to-date", "exit_code": 0, "error": null}'

    for _ in range(4):
        assert controller.before_call("terminal", args).allows_execution
        controller.after_call("terminal", args, result, failed=False)


def test_read_only_classification_covers_common_inspection_commands():
    from agent.tool_guardrails import shell_command_is_read_only

    assert shell_command_is_read_only("gh api repos/o/r/pulls/1/reviews 2>&1")
    assert shell_command_is_read_only("cd ~/repo && git status")
    assert shell_command_is_read_only("ls -la /tmp")
    assert shell_command_is_read_only("  GH_PAGER=cat gh pr view 1 ")
    assert shell_command_is_read_only("git ls-files | wc -l")
    assert shell_command_is_read_only("ls -la /tmp 2>/dev/null")

    assert not shell_command_is_read_only("gh pr merge 1 --merge")
    assert not shell_command_is_read_only("cd ~/repo && git push")
    assert not shell_command_is_read_only("rm -rf /tmp/x")
    assert not shell_command_is_read_only("git status && git commit -m x")
    assert not shell_command_is_read_only("git diff > patch.txt")
    assert not shell_command_is_read_only("ls $(rm -rf /tmp/x)")
    assert not shell_command_is_read_only("")
    assert not shell_command_is_read_only(None)


def test_write_flags_are_scoped_per_command_not_global():
    # A global mutating-flag list conflates unrelated meanings: `-f` is
    # `--field` to gh but `--file` to grep, `-x` is `--method` to gh but
    # `--exclude-type` to df. Scoping per command keeps readers readable.
    from agent.tool_guardrails import shell_command_is_read_only

    assert shell_command_is_read_only("grep -f patterns.txt file")
    assert shell_command_is_read_only("df -x tmpfs")
    assert shell_command_is_read_only("test -f /tmp/x")
    assert shell_command_is_read_only("ps -f")

    assert not shell_command_is_read_only("gh api -f key=value repos/o/r")
    assert not shell_command_is_read_only("gh api -X POST repos/o/r/issues")


def test_read_only_subcommands_do_not_leak_their_write_siblings():
    # An allowlisted noun is not enough: `git config --get` reads while
    # `git config k v` writes, `gh label list` reads while `gh label create`
    # does not, and `find` writes when told to `-delete` or `-exec`.
    from agent.tool_guardrails import shell_command_is_read_only

    assert shell_command_is_read_only("git config --get user.name")
    assert shell_command_is_read_only("git config --list")
    assert shell_command_is_read_only("git branch")
    assert shell_command_is_read_only("git branch --contains HEAD")
    assert shell_command_is_read_only("git tag --list 'v1*'")
    assert shell_command_is_read_only("gh label list")
    assert shell_command_is_read_only("sort in.txt")

    assert not shell_command_is_read_only("git config user.name someone")
    assert not shell_command_is_read_only("git config --unset user.name")
    assert not shell_command_is_read_only("git branch newbranch")
    assert not shell_command_is_read_only("git branch -D feature")
    assert not shell_command_is_read_only("git tag v1.0")
    assert not shell_command_is_read_only("gh label create bug")
    assert not shell_command_is_read_only("find . -delete")
    assert not shell_command_is_read_only("find . -exec rm {} ;")
    assert not shell_command_is_read_only("sort -o out.txt in.txt")
