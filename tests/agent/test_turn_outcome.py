import json

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
