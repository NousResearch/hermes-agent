from agent.conversation_loop import _CTX_FIELDS, _LoopState
from agent.state_candidate_evaluator import CandidateStatus, DeltaType, StateCandidateResult


def test_state_candidate_result_is_threaded_into_real_loop_state_fields():
    assert "state_candidate_result" in _CTX_FIELDS
    assert "state_candidate_result" in _LoopState.__annotations__


def test_loop_state_mapping_preserves_candidate_result_identity():
    expected = StateCandidateResult(
        candidate_id="candidate-1",
        delta_type=DeltaType.REPLACE,
        status=CandidateStatus.PENDING,
        state_key="budget",
        scope="project-a",
    )
    turn_context = type("Context", (), {"state_candidate_result": expected})()
    mapped = {
        field_name: getattr(turn_context, field_name)
        for field_name in ("state_candidate_result",)
    }
    assert mapped["state_candidate_result"] is expected
