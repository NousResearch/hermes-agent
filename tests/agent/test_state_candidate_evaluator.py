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


def test_current_state_scope_mismatch_is_unverified():
    result = evaluate_state_candidate(
        active_context="project-a",
        current_state=_state(scope="project-b"),
        evidence=Evidence("new", "user_statement", "msg-5", "project-a"),
    )
    assert result.status is CandidateStatus.UNVERIFIED
    assert result.delta_type is DeltaType.SCOPE_CHANGE


def test_malformed_evidence_and_prior_candidate_are_rejected():
    malformed_evidence = evaluate_state_candidate(
        active_context="project-a", current_state=_state(), evidence=None,
    )
    malformed_prior = evaluate_state_candidate(
        active_context="project-a", current_state=_state(),
        evidence=Evidence("new", "user_statement", "msg-6", "project-a"),
        existing_candidates=[None],  # type: ignore[list-item]
    )
    malformed_field = evaluate_state_candidate(
        active_context="project-a", current_state=_state(),
        evidence=Evidence("new", "user_statement", 123, "project-a"),  # type: ignore[arg-type]
    )
    malformed_priors = evaluate_state_candidate(
        active_context="project-a", current_state=_state(),
        evidence=Evidence("new", "user_statement", "msg-7", "project-a"),
        existing_candidates=None,  # type: ignore[arg-type]
    )
    assert malformed_evidence.status is CandidateStatus.REJECTED
    assert malformed_prior.status is CandidateStatus.REJECTED
    assert malformed_field.status is CandidateStatus.REJECTED
    assert malformed_priors.status is CandidateStatus.REJECTED


def test_blank_evidence_value_is_rejected():
    result = evaluate_state_candidate(
        active_context="project-a", current_state=_state(),
        evidence=Evidence("", "user_statement", "msg-8", "project-a"),
    )
    assert result.status is CandidateStatus.REJECTED
