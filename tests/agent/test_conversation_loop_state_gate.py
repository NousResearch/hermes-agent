from types import SimpleNamespace

from agent.conversation_loop import _apply_state_answer_gate
from agent.state_candidate_evaluator import CandidateStatus, DeltaType, StateCandidateResult
from agent.state_response_policy import ResponsePolicy


def candidate(status):
    return StateCandidateResult(
        candidate_id="id", delta_type=DeltaType.REPLACE, status=status,
        state_key="budget", scope="project-a",
    )


def test_pending_dependent_turn_is_blocked_before_model_call():
    agent = SimpleNamespace(
        _state_answer_keys_for_turn=["budget"], _state_answer_scope="project-a"
    )
    state = SimpleNamespace(state_candidate_result=candidate(CandidateStatus.PENDING))

    assert _apply_state_answer_gate(agent, state) is True
    assert state._state_gate_blocked is True
    assert state.blocked is True
    assert state.final_response
    assert state._turn_exit_reason == "state_gate_ask_confirmation"
    assert state.state_answer_gate_result.decision.response_policy is ResponsePolicy.ASK_CONFIRMATION


def test_conflict_dependent_turn_is_blocked_before_model_call():
    agent = SimpleNamespace(
        _state_answer_keys_for_turn=["budget"], _state_answer_scope="project-a"
    )
    state = SimpleNamespace(state_candidate_result=candidate(CandidateStatus.CONFLICT))

    assert _apply_state_answer_gate(agent, state) is True
    assert state._state_gate_blocked is True
    assert state.blocked is True
    assert state._turn_exit_reason == "state_gate_conflict_stop"


def test_without_explicit_answer_metadata_gate_is_noop():
    agent = SimpleNamespace()
    state = SimpleNamespace(state_candidate_result=candidate(CandidateStatus.CONFLICT))

    assert _apply_state_answer_gate(agent, state) is False
    assert not hasattr(state, "_state_gate_blocked")
