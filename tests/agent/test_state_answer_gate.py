from agent.state_answer_gate import evaluate_answer_gate
from agent.state_candidate_evaluator import CandidateStatus, DeltaType, StateCandidateResult
from agent.state_response_policy import PersistencePolicy, ResponsePolicy


def candidate(status):
    return StateCandidateResult(
        candidate_id="id", delta_type=DeltaType.REPLACE, status=status,
        state_key="budget", scope="project-a",
    )


def test_pending_dependency_blocks_model_and_preserves_pending_only():
    result = evaluate_answer_gate(
        candidate(CandidateStatus.PENDING),
        requested_state_keys=["budget"], answer_scope="project-a",
    )
    assert result.depends_on_candidate is True
    assert result.model_call_allowed is False
    assert result.decision.response_policy is ResponsePolicy.ASK_CONFIRMATION
    assert result.decision.persistence_policy is PersistencePolicy.PENDING_ONLY


def test_conflict_dependency_blocks_model():
    result = evaluate_answer_gate(
        candidate(CandidateStatus.CONFLICT),
        requested_state_keys=["budget"], answer_scope="project-a",
    )
    assert result.model_call_allowed is False
    assert result.decision.response_policy is ResponsePolicy.CONFLICT_STOP


def test_unrelated_pending_candidate_allows_model():
    result = evaluate_answer_gate(
        candidate(CandidateStatus.PENDING),
        requested_state_keys=["timeline"], answer_scope="project-a",
    )
    assert result.depends_on_candidate is False
    assert result.model_call_allowed is True
    assert result.decision.response_policy is ResponsePolicy.CONTINUE


def test_no_candidate_allows_model():
    result = evaluate_answer_gate(
        None, requested_state_keys=["budget"], answer_scope="project-a",
    )
    assert result.model_call_allowed is True
    assert result.decision.persistence_policy is PersistencePolicy.NO_WRITE
