from agent.state_candidate_evaluator import CandidateStatus, DeltaType, StateCandidateResult
from agent.state_response_policy import (
    PersistencePolicy,
    ResponsePolicy,
    route_candidate_result,
)


def result(status):
    return StateCandidateResult("id", DeltaType.REPLACE, status)


def test_no_candidate_continues_without_write():
    decision = route_candidate_result(None, answer_depends_on_candidate=True)
    assert decision.response_policy is ResponsePolicy.CONTINUE
    assert decision.persistence_policy is PersistencePolicy.NO_WRITE


def test_pending_unrelated_answer_continues_but_only_pending_may_be_held():
    decision = route_candidate_result(result(CandidateStatus.PENDING), answer_depends_on_candidate=False)
    assert decision.response_policy is ResponsePolicy.CONTINUE
    assert decision.persistence_policy is PersistencePolicy.PENDING_ONLY


def test_pending_dependent_answer_requires_confirmation():
    decision = route_candidate_result(result(CandidateStatus.PENDING), answer_depends_on_candidate=True)
    assert decision.response_policy is ResponsePolicy.ASK_CONFIRMATION
    assert decision.persistence_policy is PersistencePolicy.PENDING_ONLY


def test_conflict_dependent_answer_stops():
    decision = route_candidate_result(result(CandidateStatus.CONFLICT), answer_depends_on_candidate=True)
    assert decision.response_policy is ResponsePolicy.CONFLICT_STOP
    assert decision.persistence_policy is PersistencePolicy.NO_WRITE


def test_unverified_unrelated_answer_continues_without_write():
    decision = route_candidate_result(result(CandidateStatus.UNVERIFIED), answer_depends_on_candidate=False)
    assert decision.response_policy is ResponsePolicy.CONTINUE
    assert decision.persistence_policy is PersistencePolicy.NO_WRITE


def test_rejected_dependent_answer_is_held():
    decision = route_candidate_result(result(CandidateStatus.REJECTED), answer_depends_on_candidate=True)
    assert decision.response_policy is ResponsePolicy.HOLD
    assert decision.persistence_policy is PersistencePolicy.NO_WRITE


def test_corroboration_and_duplicate_continue():
    for status in (CandidateStatus.CORROBORATION, CandidateStatus.DUPLICATE):
        decision = route_candidate_result(result(status), answer_depends_on_candidate=True)
        assert decision.response_policy is ResponsePolicy.CONTINUE
        assert decision.persistence_policy is PersistencePolicy.NO_WRITE
