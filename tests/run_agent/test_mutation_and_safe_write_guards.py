"""Tests for the repeated-mutation halt and the destructive-overwrite guard."""

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
)

OK_RESULT = '{"status": "ok"}'


def _controller(**overrides):
    data = {
        "warnings_enabled": True,
        "hard_stop_enabled": True,
        "no_progress_warn_after": 3,
        "no_progress_block_after": 5,
    }
    data.update(overrides)
    return ToolCallGuardrailController(ToolCallGuardrailConfig.from_mapping(data))


# --------------------------------------------------------- repeated-mutation halt


def test_identical_successful_mutation_warns_then_halts():
    controller = _controller()
    args = {"path": "/tmp/does-not-exist-loop.md", "content": "same content"}
    decisions = [
        controller.after_call("write_file", args, OK_RESULT, failed=False)
        for _ in range(5)
    ]
    assert [d.action for d in decisions[:2]] == ["allow", "allow"]
    assert decisions[2].action == "warn"
    assert decisions[2].code == "repeated_mutation_warning"
    assert decisions[4].action == "halt"
    assert decisions[4].code == "repeated_mutation_halt"
    assert controller.halt_decision is not None


def test_iterative_edits_with_different_content_never_flag():
    controller = _controller()
    for i in range(8):
        decision = controller.after_call(
            "write_file",
            {"path": "/tmp/iterative-edit.md", "content": f"version {i}"},
            OK_RESULT,
            failed=False,
        )
        assert decision.action == "allow", (i, decision.action, decision.code)


def test_mutation_counter_resets_each_turn():
    controller = _controller()
    args = {"path": "/tmp/x.md", "content": "y"}
    for _ in range(4):
        controller.after_call("write_file", args, OK_RESULT, failed=False)
    controller.reset_for_turn()
    decision = controller.after_call("write_file", args, OK_RESULT, failed=False)
    assert decision.action == "allow"


def test_mutation_halt_requires_hard_stop_enabled():
    controller = _controller(hard_stop_enabled=False)
    args = {"path": "/tmp/x.md", "content": "y"}
    last = None
    for _ in range(7):
        last = controller.after_call("write_file", args, OK_RESULT, failed=False)
    assert last.action in ("allow", "warn")
    assert controller.halt_decision is None


# --------------------------------------------------------- destructive-overwrite


def _victim(tmp_path, size=1000):
    victim = tmp_path / "victim.md"
    victim.write_text("x" * size)
    return victim


def test_blocks_drastic_shrink_without_halting_turn(tmp_path):
    controller = _controller()
    victim = _victim(tmp_path)
    decision = controller.before_call(
        "write_file", {"path": str(victim), "content": "stub"}
    )
    assert decision.action == "block"
    assert decision.code == "destructive_overwrite_block"
    # block, not halt: the model can recover in-turn with the full content
    assert controller.halt_decision is None


def test_identical_reissue_confirms_intent(tmp_path):
    controller = _controller()
    victim = _victim(tmp_path)
    args = {"path": str(victim), "content": "intentional shrink"}
    assert controller.before_call("write_file", args).action == "block"
    assert controller.before_call("write_file", args).action == "allow"


def test_allows_small_files_new_files_and_growth(tmp_path):
    controller = _controller()
    small = tmp_path / "small.md"
    small.write_text("x" * 50)  # below safe_write_min_bytes
    assert (
        controller.before_call("write_file", {"path": str(small), "content": ""}).action
        == "allow"
    )
    assert (
        controller.before_call(
            "write_file", {"path": str(tmp_path / "new.md"), "content": "hello"}
        ).action
        == "allow"
    )
    victim = _victim(tmp_path)
    assert (
        controller.before_call(
            "write_file", {"path": str(victim), "content": "x" * 900}
        ).action
        == "allow"
    )


def test_safe_write_independent_of_hard_stop(tmp_path):
    controller = _controller(hard_stop_enabled=False)
    victim = _victim(tmp_path)
    decision = controller.before_call(
        "write_file", {"path": str(victim), "content": "stub"}
    )
    assert decision.action == "block"


def test_safe_write_flag_off_allows(tmp_path):
    controller = _controller(safe_write_enabled=False)
    victim = _victim(tmp_path)
    decision = controller.before_call(
        "write_file", {"path": str(victim), "content": ""}
    )
    assert decision.action == "allow"
