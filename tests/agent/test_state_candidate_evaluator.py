from agent.state_candidate_evaluator import (
    CandidateStatus,
    DeltaType,
    Evidence,
    evaluate_state_candidate,
)


def _state(value="old", scope="project-a"):
    return {
        "key": "policy",
        "value": value,
        "scope": scope,
        "source_ref": "state-1",
    }


def test_matching_evidence_is_corroboration():
    result = evaluate_state_candidate(
        active_context="project-a",
        current_state=_state(),
        evidence=Evidence("old", "user_statement", "msg-1", "project-a"),
    )
    assert result.status is CandidateStatus.CORROBORATION
    assert result.delta_type is DeltaType.NONE


def test_changed_user_evidence_is_pending_without_side_effects():
    current = _state()
    result = evaluate_state_candidate(
        active_context="project-a",
        current_state=current,
        evidence=Evidence("new", "user_statement", "msg-2", "project-a"),
    )
    assert result.status is CandidateStatus.PENDING
    assert result.delta_type is DeltaType.REPLACE
    assert result.scope == "project-a"
    assert result.evidence_ref == "msg-2"
    assert current["value"] == "old"


def test_scope_mismatch_is_unverified():
    result = evaluate_state_candidate(
        active_context="project-a",
        current_state=_state(),
        evidence=Evidence("new", "user_statement", "msg-3", "project-b"),
    )
    assert result.status is CandidateStatus.UNVERIFIED
    assert result.delta_type is DeltaType.SCOPE_CHANGE


def test_inference_only_is_unverified():
    result = evaluate_state_candidate(
        active_context="project-a",
        current_state=_state(),
        evidence=Evidence("new", "assistant_inference", "inf-1", "project-a"),
    )
    assert result.status is CandidateStatus.UNVERIFIED


def test_missing_source_and_malformed_state_are_rejected():
    missing_source = evaluate_state_candidate(
        active_context="project-a",
        current_state=_state(),
        evidence=Evidence("new", "user_statement", "", "project-a"),
    )
    malformed = evaluate_state_candidate(
        active_context="project-a",
        current_state={"key": "policy"},
        evidence=Evidence("new", "user_statement", "msg-4", "project-a"),
    )
    assert missing_source.status is CandidateStatus.REJECTED
    assert malformed.status is CandidateStatus.REJECTED
