"""Bounded gateway continuation after the Hermes tool-call ceiling."""

from gateway.run import (
    _max_iteration_recovery_prompt,
    _strip_auto_continue_noise,
)


def _ceiling_result(**overrides):
    result = {
        "turn_exit_reason": "max_iterations_reached(90/90)",
        "completed": False,
        "partial": True,
        "final_response": "Reached maximum iterations (90).",
    }
    result.update(overrides)
    return result


def test_max_iteration_builds_one_recovery_turn():
    prompt = _max_iteration_recovery_prompt(_ceiling_result())

    assert prompt is not None
    assert prompt.startswith("[System note: Your previous turn")
    assert "durable progress" in prompt
    assert "Do not repeat non-idempotent external actions" in prompt
    assert "at most 30 recovery iterations" in prompt


def test_recovery_budget_is_configurable_and_clamped():
    prompt = _max_iteration_recovery_prompt(
        _ceiling_result(), recovery_max_turns=12
    )
    assert prompt is not None
    assert "at most 12 recovery iterations" in prompt

    clamped = _max_iteration_recovery_prompt(
        _ceiling_result(), recovery_max_turns=900
    )
    assert clamped is not None
    assert "at most 90 recovery iterations" in clamped


def test_recovery_is_bounded_by_depth():
    assert _max_iteration_recovery_prompt(
        _ceiling_result(), recovery_depth=1, max_attempts=1
    ) is None


def test_human_pending_message_wins_over_recovery():
    assert _max_iteration_recovery_prompt(
        _ceiling_result(), pending="new user instruction"
    ) is None
    assert _max_iteration_recovery_prompt(
        _ceiling_result(), pending_event=object()
    ) is None


def test_normal_completion_does_not_recover():
    assert _max_iteration_recovery_prompt(
        {"turn_exit_reason": "completed", "completed": True, "final_response": "Done."}
    ) is None


def test_completed_response_discussing_ceiling_does_not_recover():
    assert _max_iteration_recovery_prompt(
        {
            "turn_exit_reason": "completed_no_tools",
            "completed": True,
            "final_response": "I fixed the tool-call ceiling handling.",
        }
    ) is None


def test_completed_response_with_stale_ceiling_reason_does_not_recover():
    assert _max_iteration_recovery_prompt(
        {
            "turn_exit_reason": "max_iterations_reached(90/90)",
            "completed": True,
            "final_response": "Verified complete.",
        }
    ) is None


def test_legacy_ceiling_text_is_classified():
    prompt = _max_iteration_recovery_prompt(
        {"completed": False, "error": "You've reached the maximum number of tool-calling iterations allowed"}
    )
    assert prompt is not None


def test_synthetic_recovery_note_is_not_replayed_later():
    prompt = _max_iteration_recovery_prompt(_ceiling_result())
    assert prompt is not None
    assert _strip_auto_continue_noise(prompt) == ""
