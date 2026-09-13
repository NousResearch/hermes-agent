"""Pure tool-call guardrail primitive tests."""

import json

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    canonical_tool_args,
    classify_tool_failure,
    is_no_progress_marker_result,
    normalize_tool_args_for_guardrail,
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
    assert cfg.non_interactive_hard_stop_enabled is True
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


def test_gateway_platform_defaults_to_hard_stop_without_changing_interactive_defaults():
    interactive_configs = [
        ToolCallGuardrailConfig.from_mapping({}, platform=platform)
        for platform in ("cli", "tui", "desktop", "acp")
    ]
    telegram_cfg = ToolCallGuardrailConfig.from_mapping({}, platform="telegram")
    cron_cfg = ToolCallGuardrailConfig.from_mapping({}, platform="cron")

    assert all(cfg.hard_stop_enabled is False for cfg in interactive_configs)
    assert telegram_cfg.hard_stop_enabled is True
    assert cron_cfg.hard_stop_enabled is True


def test_non_interactive_hard_stop_can_be_disabled_explicitly():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {"non_interactive_hard_stop_enabled": False},
        platform="telegram",
    )

    assert cfg.hard_stop_enabled is False
    assert cfg.non_interactive_hard_stop_enabled is False


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














def test_skill_read_tools_are_idempotent_and_block_repeated_identical_success_output():
    cases = [
        (
            "skill_view",
            {"name": "gui-agent-ml-operations"},
            '{"success":true,"name":"gui-agent-ml-operations","content":"same"}',
        ),
        (
            "skills_list",
            {"category": "mlops"},
            '{"success":true,"skills":[{"name":"gui-agent-ml-operations"}]}',
        ),
    ]

    for tool_name, args, result in cases:
        controller = ToolCallGuardrailController(
            ToolCallGuardrailConfig(
                hard_stop_enabled=True,
                no_progress_warn_after=2,
                no_progress_block_after=2,
            )
        )

        assert controller.before_call(tool_name, args).action == "allow"
        assert controller.after_call(tool_name, args, result, failed=False).action == "allow"
        assert controller.before_call(tool_name, args).action == "allow"
        warn = controller.after_call(tool_name, args, result, failed=False)
        assert warn.action == "warn"
        assert warn.code == "idempotent_no_progress_warning"

        blocked = controller.before_call(tool_name, args)
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


def test_explicit_dedup_results_continue_no_progress_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=3,
        )
    )
    args = {"path": "README.md"}

    assert controller.before_call("read_file", args).action == "allow"
    assert controller.after_call(
        "read_file",
        args,
        '{"path":"README.md","content":"same"}',
        failed=False,
    ).action == "allow"

    assert controller.before_call("read_file", args).action == "allow"
    second = controller.after_call(
        "read_file",
        args,
        "[Duplicate tool output — same content as a more recent call]",
        failed=False,
    )
    assert second.action == "warn"
    assert second.code == "idempotent_no_progress_warning"
    assert second.count == 2

    assert controller.before_call("read_file", args).action == "allow"
    third = controller.after_call(
        "read_file",
        args,
        '{"status":"unchanged","dedup":true,"content_returned":false}',
        failed=False,
    )
    assert third.action == "warn"
    assert third.count == 3

    blocked = controller.before_call("read_file", args)
    assert blocked.action == "block"
    assert blocked.code == "idempotent_no_progress_block"
    assert blocked.count == 3


def test_identical_call_streak_halts_any_tool_when_hard_stop_enabled():
    # #89069 / #100849 bundle: a model replaying the same SUCCESSFUL
    # terminal/skill_view call with a byte-identical result is not covered by
    # the idempotent_tools no-progress block. The consecutive-identical
    # streak (observe_call) is tool-agnostic; under hard_stop it must halt.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=5)
    )
    args = {"command": "hermes config get memory.provider"}
    for i in range(1, 5):
        controller.after_call("terminal", args, "local\n", failed=False)
        controller.observe_call("terminal", args, "local\n", failed=False)
        assert controller.halt_decision is None, f"halted early at {i}"

    controller.after_call("terminal", args, "local\n", failed=False)
    controller.observe_call("terminal", args, "local\n", failed=False)
    halt = controller.halt_decision
    assert halt is not None and halt.should_halt
    assert halt.code == "identical_call_streak_halt"
    assert halt.tool_name == "terminal" and halt.count == 5


def test_identical_call_streak_never_halts_when_hard_stop_disabled_or_for_pollers():
    soft = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=False, no_progress_block_after=2)
    )
    for _ in range(6):
        soft.observe_call("terminal", {"command": "ls"}, "a\nb\n", failed=False)
    assert soft.halt_decision is None  # notice-only in interactive sessions

    hard = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=2)
    )
    for _ in range(6):
        hard.observe_call("process_manage", {"action": "poll", "session_id": "p1"}, "running", failed=False)
    assert hard.halt_decision is None  # an unchanged poll is legitimate progress

    # A changed result resets the streak.
    for i in range(6):
        hard.observe_call("terminal", {"command": "date"}, f"t{i}", failed=False)
    assert hard.halt_decision is None






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












def test_new_user_turn_clears_no_progress_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=3,
        )
    )
    args = {"todos": [{"id": "a", "content": "same", "status": "in_progress"}]}

    for _ in range(3):
        assert controller.before_call("todo", args).action == "allow"
        controller.after_call("todo", args, "same-list", failed=False)

    blocked = controller.before_call("todo", args)
    assert blocked.action == "block"
    assert blocked.code == "idempotent_no_progress_block"

    controller.reset_for_turn()
    assert controller.before_call("todo", args).action == "allow"


def test_changed_read_result_restarts_no_progress_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=3,
        )
    )
    args = {"query": "latest state"}

    for result in ("one", "two", "three", "four"):
        assert controller.before_call("web_search", args).action == "allow"
        decision = controller.after_call("web_search", args, result, failed=False)
        assert decision.action == "allow"
        assert decision.count == 1


def test_guardrail_signature_normalizes_housekeeping_arg_jitter():
    todo_a = ToolCallSignature.from_call(
        "todo",
        {
            "merge": True,
            "todos": [
                {"id": "b", "content": "same", "status": "pending"},
                {"id": "a", "content": "same", "status": "in_progress"},
            ],
        },
    )
    todo_b = ToolCallSignature.from_call(
        "todo",
        {
            "merge": False,
            "todos": [
                {"id": "a", "content": "same", "status": "in_progress"},
                {"id": "b", "content": "same", "status": "pending"},
            ],
        },
    )
    # Todo list order is priority and merge changes write semantics, so this
    # jitter must remain visible to the guardrail.
    assert todo_a != todo_b

    assert ToolCallSignature.from_call("skill_view", {"name": "hermes-agent"}) == ToolCallSignature.from_call(
        "skill_view",
        {"name": "hermes-agent", "file_path": None},
    )
    assert ToolCallSignature.from_call("read_file", {"path": "x"}) == ToolCallSignature.from_call(
        "read_file",
        {"path": "x", "offset": 1, "limit": 2000},
    )

    # Non-housekeeping tools keep raw args; shell-string differences may be semantic.
    assert ToolCallSignature.from_call("terminal", {"command": "pwd"}) != ToolCallSignature.from_call(
        "terminal",
        {"command": "pwd "},
    )


def test_no_progress_blocks_repeated_identical_todo_state():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=3,
        )
    )
    args = {"todos": [{"id": "a", "content": "same", "status": "in_progress"}]}

    for _ in range(3):
        assert controller.before_call("todo", args).action == "allow"
        controller.after_call("todo", args, "same-list", failed=False)

    blocked = controller.before_call("todo", args)
    assert blocked.action == "block"
    assert blocked.count == 3


def test_arbitrary_mutating_tool_is_not_blocked_from_identical_stdout():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=3,
        )
    )
    args = {"command": "make step"}

    for _ in range(5):
        assert controller.before_call("terminal", args).action == "allow"
        decision = controller.after_call("terminal", args, "ok", failed=False)
        assert decision.action == "allow"


def test_skill_pruned_reload_loop_is_blocked_across_turn_boundaries():
    """Regression: the 2026-08-19 twenty-turn loop.

    A post-compression banner listed 18 pruned skills. Each ``skill_view``
    returned a ``[SKILL_PRUNED: ...]`` marker whose text instructs the agent to
    reissue the very same call. Every lap was a SEPARATE turn, so any counter
    cleared by ``reset_for_turn`` would never reach its threshold.
    """
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=3,
        )
    )
    args = {"name": "kibana-evals"}
    pruned = "[skill_view] name=kibana-evals (7,719 chars) [SKILL_PRUNED: content lost in compression]"

    for _ in range(3):
        # Each lap is a compaction-triggered turn restart, NOT new user input.
        controller.reset_for_turn(new_user_input=False)
        assert controller.before_call("skill_view", args).action == "allow"
        controller.after_call("skill_view", args, pruned, failed=False)

    controller.reset_for_turn(new_user_input=False)
    blocked = controller.before_call("skill_view", args)
    assert blocked.action == "block"
    assert blocked.code == "idempotent_no_progress_block"

    # A genuine new user request is allowed to re-read the same skill.
    controller.reset_for_turn(new_user_input=True)
    assert controller.before_call("skill_view", args).action == "allow"


def test_dedup_stub_counts_as_no_progress_despite_different_hash():
    """The compressor replaces a repeat with a stub that hashes differently.

    Without explicit handling the streak resets to 1 forever, which is exactly
    how the observed todo loop stayed invisible to the hash comparison.
    """
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=3,
        )
    )
    args = {"todos": [{"id": "a", "content": "same", "status": "in_progress"}]}

    controller.after_call("todo", args, "the real list", failed=False)
    for _ in range(2):
        stub = "[Duplicate tool output — same content as a more recent call]"
        controller.after_call("todo", args, stub, failed=False)

    blocked = controller.before_call("todo", args)
    assert blocked.action == "block"
    assert blocked.code == "idempotent_no_progress_block"


def test_real_file_write_clears_streak_so_reread_is_allowed():
    """A read repeated AFTER a landed write is progress, not a loop."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=3,
        )
    )
    read_args = {"path": "/tmp/x.py"}

    for _ in range(3):
        controller.after_call("read_file", read_args, "same content", failed=False)
    assert controller.before_call("read_file", read_args).action == "block"

    controller.after_call(
        "write_file",
        {"path": "/tmp/x.py", "content": "new"},
        json.dumps({"success": True, "verified": True, "path": "/tmp/x.py"}),
        failed=False,
    )

    assert controller.before_call("read_file", read_args).action == "allow"


def test_cross_turn_carry_never_blocks_edit_then_reread_iteration():
    """The mitigation the cross-turn carry needs to be safe.

    Carrying a no-progress streak across turn boundaries is what makes a
    compaction-spanning loop reachable at all. The same carry would be a
    regression if it also survived REAL work, so a landed mutation must clear
    it even when the streak was inherited from an earlier turn.

    Without ``note_progress()`` on a landed write, the fourth read below is
    blocked and a normal edit -> re-read cycle dies at the guardrail.
    """
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=3,
        )
    )
    read_args = {"path": "/tmp/x.py"}

    # Three unchanged reads spread across compaction-driven turn restarts:
    # the streak is carried, exactly as the cross-turn half intends.
    for _ in range(3):
        controller.reset_for_turn(new_user_input=False)
        controller.after_call("read_file", read_args, "same content", failed=False)

    controller.reset_for_turn(new_user_input=False)
    assert controller.before_call("read_file", read_args).action == "block"

    # A real edit lands. The world moved, so the carried streak is stale even
    # though no new user message arrived.
    controller.after_call(
        "write_file",
        {"path": "/tmp/x.py", "content": "new"},
        json.dumps({"success": True, "verified": True, "path": "/tmp/x.py"}),
        failed=False,
    )

    controller.reset_for_turn(new_user_input=False)
    assert controller.before_call("read_file", read_args).action == "allow"


def test_bookkeeping_success_does_not_clear_a_carried_streak():
    """The other half of the mitigation: it must not defeat the guardrail.

    ``todo`` is in ``PROGRESS_RESET_TOOL_NAMES`` but changes nothing
    observable, so if a successful todo call cleared no-progress streaks, the
    2026-08-19 loop — skill_view reload interleaved with todo bookkeeping —
    would restart its streak on every lap and never reach the threshold, which
    is the exact bug the cross-turn carry exists to fix.
    """
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=3,
        )
    )
    view_args = {"name": "kibana-evals"}
    pruned = "[skill_view] name=kibana-evals [SKILL_PRUNED: content lost in compression]"

    for _ in range(3):
        controller.reset_for_turn(new_user_input=False)
        controller.after_call("skill_view", view_args, pruned, failed=False)
        # Interleaved bookkeeping that succeeds but changes nothing.
        controller.after_call(
            "todo", {"todos": [{"id": "a", "status": "in_progress"}]}, "ok", failed=False
        )

    controller.reset_for_turn(new_user_input=False)
    blocked = controller.before_call("skill_view", view_args)
    assert blocked.action == "block"
    assert blocked.code == "idempotent_no_progress_block"


def test_marker_normalization_never_blocks_a_changed_read():
    """The marker half must not swallow real change.

    A dedup marker continues the streak, but the moment the tool returns a
    genuinely different payload the streak restarts — otherwise a file that is
    actually being edited would stay blocked.
    """
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=3,
        )
    )
    args = {"path": "README.md"}

    controller.after_call("read_file", args, '{"content":"v1"}', failed=False)
    controller.after_call(
        "read_file",
        args,
        '{"status":"unchanged","dedup":true,"content_returned":false}',
        failed=False,
    )
    # Real new content arrives: this is progress and must restart the streak.
    controller.after_call("read_file", args, '{"content":"v2"}', failed=False)
    assert controller.before_call("read_file", args).action == "allow"


def test_partial_dedup_envelope_is_not_treated_as_unchanged():
    """Only the full envelope both real emitters write counts as a marker.

    A tool that happens to return ``dedup: true`` without the ``unchanged``
    status is not making the identity claim this predicate relies on, so it
    must not silently extend a streak toward a hard stop.
    """
    assert is_no_progress_marker_result(
        '{"status":"unchanged","dedup":true,"content_returned":false}'
    )
    assert not is_no_progress_marker_result('{"dedup":true,"content_returned":false}')
    assert not is_no_progress_marker_result('{"status":"unchanged","dedup":true}')
    assert not is_no_progress_marker_result('{"content":"a real payload"}')
    assert not is_no_progress_marker_result(None)


# ── Legitimate flows must survive hard stops (Teknium, Sep 2026) ────────────
# Hard stops default ON for unattended platforms. These pin the flows that
# must NEVER be cut off there: edit -> re-run loops, diagnostic sweeps of
# distinct red commands, and browser retry-after-action — while the pure
# replay (same call, nothing changed between attempts) is still stopped.

_HARD = lambda: ToolCallGuardrailController(  # noqa: E731
    ToolCallGuardrailConfig(hard_stop_enabled=True)
)
_PYTEST = {"command": "pytest tests/test_x.py -q"}
_RED = '{"output": "1 failed", "exit_code": 1}'


def _run_red(c, args=_PYTEST):
    assert c.before_call("terminal", args).allows_execution
    return c.after_call("terminal", args, _RED, failed=True)


def test_fix_retest_loop_is_never_hard_stopped():
    c = _HARD()
    for i in range(12):
        d = _run_red(c)
        assert not d.should_halt, f"halted on red run {i + 1}"
        # the model edits between runs — a landed mutation is progress
        c.after_call("patch", {"path": "x.py", "old_string": "a", "new_string": f"b{i}"},
                     '{"success": true, "diff": "..."}', failed=False)
    assert c.halt_decision is None
    assert c.before_call("terminal", _PYTEST).allows_execution


def test_pure_replay_with_no_intervening_change_is_still_blocked():
    c = _HARD()
    for _ in range(5):
        _run_red(c)
    d = c.before_call("terminal", _PYTEST)
    assert d.action == "block" and d.code == "repeated_exact_failure_block"


def test_intervening_mutation_resets_the_replay_streak_only_once():
    # 4 reds, one edit, then 4 reds with NO edit: the second run of 4 is a
    # fresh streak, and the 5th unchanged retry after it is blocked.
    c = _HARD()
    for _ in range(4):
        _run_red(c)
    c.after_call("write_file", {"path": "x.py", "content": "y"}, '{"bytes_written": 1}', failed=False)
    for _ in range(5):
        assert c.before_call("terminal", _PYTEST).allows_execution
        c.after_call("terminal", _PYTEST, _RED, failed=True)
    assert c.before_call("terminal", _PYTEST).action == "block"


def test_distinct_failing_terminal_commands_warn_but_never_halt():
    # A diagnostic sweep: grep with no matches, missing binaries, red builds.
    c = _HARD()
    for i in range(12):
        args = {"command": f"grep -q needle{i} haystack.txt"}
        d = c.after_call("terminal", args, _RED, failed=True)
        assert not d.should_halt, f"same_tool halt on distinct command #{i + 1}"
    assert c.halt_decision is None
    # ...while a non-tolerant tool failing 8 distinct ways still halts.
    c2 = _HARD()
    last = None
    for i in range(8):
        last = c2.after_call("send_message", {"to": f"u{i}"}, '{"error": "no route"}', failed=True)
    assert last.should_halt and last.code == "same_tool_failure_halt"


def test_browser_retry_after_action_is_not_a_replay():
    c = _HARD()
    nav = {"url": "https://example.test/app"}
    for _ in range(8):
        assert c.before_call("browser_navigate", nav).allows_execution
        c.after_call("browser_navigate", nav, '{"error": "timeout"}', failed=True)
        c.after_call("browser_click", {"selector": "#retry"}, '{"ok": true}', failed=False)
    assert c.halt_decision is None


def test_supervised_task_platforms_keep_warning_only_default():
    for platform in ("subagent", "api_server", "cli"):
        cfg = ToolCallGuardrailConfig.from_mapping({}, platform=platform)
        assert cfg.hard_stop_enabled is False, platform
    for platform in ("telegram", "discord", "cron", "kanban"):
        cfg = ToolCallGuardrailConfig.from_mapping({}, platform=platform)
        assert cfg.hard_stop_enabled is True, platform


# ── The carry must be reachable from production, not just from tests ────────


def _SKILL_VIEW_DEDUP(name: str = "hermes-development") -> str:
    """The exact envelope tools/skills_tool.py emits for a repeat view."""
    return json.dumps(
        {
            "success": True,
            "status": "unchanged",
            "name": name,
            "file": "SKILL.md",
            "dedup": True,
            "content_returned": False,
            "message": "already served this turn",
        }
    )


def test_internal_continuation_carries_the_streak_into_the_next_turn():
    """An agent-authored continuation must not launder a no-progress streak.

    build_turn_context() calls reset_for_turn(new_user_input=not
    internal_continuation). Before that argument was threaded through, every
    turn passed the default and the streak restarted at 1 forever -- the
    mechanism existed but no production path could reach it.
    """
    c = _HARD()
    args = {"name": "hermes-development"}
    blocked_at = None
    for turn in range(1, 8):
        c.reset_for_turn(new_user_input=False)  # what an internal retry does
        if not c.before_call("skill_view", args).allows_execution:
            blocked_at = turn
            break
        c.after_call("skill_view", args, _SKILL_VIEW_DEDUP())
    assert blocked_at is not None, "carried streak never reached the block"


def test_a_genuine_new_user_request_still_clears_the_streak():
    """The other half of the same argument: real user turns must never block."""
    c = _HARD()
    args = {"name": "hermes-development"}
    for _ in range(8):
        c.reset_for_turn(new_user_input=True)
        assert c.before_call("skill_view", args).allows_execution
        c.after_call("skill_view", args, _SKILL_VIEW_DEDUP())
    assert c.halt_decision is None
