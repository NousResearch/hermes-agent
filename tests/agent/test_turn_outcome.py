import json

import pytest

from agent.turn_outcome import TURN_OUTCOMES, classify_turn_outcome


def _classify(**overrides):
    facts = {
        "final_response": "answer",
        "failed": False,
        "interrupted": False,
        "_turn_exit_reason": "text_response(finish_reason=stop)",
        "verification_status": None,
        "timeout": False,
        "unresolved": False,
    }
    facts.update(overrides)
    return classify_turn_outcome(**facts)


def test_normal_verified_response_has_canonical_serializable_outcome():
    result = _classify(verification_status="passed")

    assert result == {"outcome": "verified", "reason": "verification passed"}
    assert json.loads(json.dumps(result)) == result


def test_iteration_budget_fallback_is_partial_even_with_response_text():
    result = _classify(
        _turn_exit_reason="max_iterations_reached(60/60)",
    )

    assert result["outcome"] == "partial"
    assert "iteration budget" in result["reason"]


def test_provider_failure_with_response_text_is_failed():
    result = _classify(
        failed=True,
        _turn_exit_reason="provider_failure",
    )

    assert result["outcome"] == "failed"


@pytest.mark.parametrize(
    ("exit_reason", "expected_outcome"),
    [
        ("guardrail_halt", "blocked"),
        ("error_near_max_iterations(repeated failure)", "failed"),
        ("all_retries_exhausted_no_response", "failed"),
        ("ollama_runtime_context_too_small", "failed"),
        ("partial_stream_recovery", "partial"),
        ("fallback_prior_turn_content", "partial"),
        ("empty_response_exhausted", "partial"),
        ("pending_tool_result", "partial"),
    ],
)
def test_known_non_success_exit_reason_overrides_response_text(
    exit_reason, expected_outcome
):
    result = _classify(_turn_exit_reason=exit_reason)

    assert result["outcome"] == expected_outcome


def test_interrupt_with_response_text_is_interrupted():
    result = _classify(
        interrupted=True,
        _turn_exit_reason="interrupted_by_user",
    )

    assert result["outcome"] == "interrupted"


def test_blocked_approval_with_response_text_is_blocked():
    result = _classify(
        _turn_exit_reason="approval_blocked",
    )

    assert result["outcome"] == "blocked"


def test_unresolved_timeout_with_response_text_is_unresolved():
    result = _classify(
        _turn_exit_reason="tool_timeout",
        timeout=True,
        unresolved=True,
    )

    assert result["outcome"] == "unresolved"


def test_completed_but_unverified_response_is_not_verified():
    result = _classify(verification_status="unverified")

    assert result["outcome"] == "completed_unverified"


def test_outcome_vocabulary_is_finite():
    assert TURN_OUTCOMES == (
        "verified",
        "completed_unverified",
        "partial",
        "blocked",
        "failed",
        "interrupted",
        "unresolved",
        "cancelled",
    )
