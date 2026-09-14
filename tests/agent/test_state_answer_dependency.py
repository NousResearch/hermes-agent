from agent.state_answer_dependency import answer_depends_on_candidate
from agent.state_candidate_evaluator import CandidateStatus, DeltaType, StateCandidateResult
from agent.state_response_policy import PersistencePolicy, ResponsePolicy, route_candidate_result


def candidate(*, state_key="budget", scope="project-a"):
    return StateCandidateResult(
        candidate_id="candidate-1",
        delta_type=DeltaType.REPLACE,
        status=CandidateStatus.PENDING,
        state_key=state_key,
        scope=scope,
    )


def test_matching_key_and_scope_is_dependent():
    assert answer_depends_on_candidate(
        candidate(), requested_state_keys=["budget"], answer_scope="project-a"
    )


def test_unrelated_key_is_not_dependent():
    assert not answer_depends_on_candidate(
        candidate(), requested_state_keys=["timeline"], answer_scope="project-a"
    )


def test_different_scope_is_not_dependent():
    assert not answer_depends_on_candidate(
        candidate(), requested_state_keys=["budget"], answer_scope="project-b"
    )


def test_candidate_presence_alone_is_not_dependency():
    assert not answer_depends_on_candidate(
        candidate(), requested_state_keys=[], answer_scope="project-a"
    )


def test_missing_candidate_is_not_dependent():
    assert not answer_depends_on_candidate(
        None, requested_state_keys=["budget"], answer_scope="project-a"
    )


def test_empty_candidate_identity_is_not_dependent():
    assert not answer_depends_on_candidate(
        candidate(state_key=""), requested_state_keys=["budget"], answer_scope="project-a"
    )


def test_pending_matching_dependency_requires_confirmation():
    result = candidate()
    depends = answer_depends_on_candidate(
        result, requested_state_keys=["budget"], answer_scope="project-a"
    )
    decision = route_candidate_result(result, answer_depends_on_candidate=depends)
    assert depends is True
    assert decision.response_policy is ResponsePolicy.ASK_CONFIRMATION
    assert decision.persistence_policy is PersistencePolicy.PENDING_ONLY


def test_conflict_unrelated_dependency_does_not_stop_answer():
    result = candidate()
    result = StateCandidateResult(
        result.candidate_id, result.delta_type, CandidateStatus.CONFLICT,
        result.state_key, result.old_value, result.new_value, result.scope,
        result.evidence_ref, result.reason,
    )
    depends = answer_depends_on_candidate(
        result, requested_state_keys=["timeline"], answer_scope="project-a"
    )
    decision = route_candidate_result(result, answer_depends_on_candidate=depends)
    assert depends is False
    assert decision.response_policy is ResponsePolicy.CONTINUE

