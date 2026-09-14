from unittest.mock import MagicMock

from agent.turn_context import TurnContext
from agent.state_candidate_evaluator import CandidateStatus, DeltaType, StateCandidateResult
from tests.agent.test_turn_context import _build, _FakeAgent


def test_build_turn_context_attaches_explicit_candidate_result_after_prefetch():
    agent = _FakeAgent()
    events = []
    memory = MagicMock()
    memory.prefetch_all.side_effect = lambda _query: events.append("prefetch") or ""
    agent._memory_manager = memory

    expected = StateCandidateResult(
        candidate_id="candidate-1",
        delta_type=DeltaType.REPLACE,
        status=CandidateStatus.PENDING,
        state_key="policy",
        old_value="old",
        new_value="new",
        scope="project-a",
        evidence_ref="msg-1",
    )

    def evaluator(_agent, original_message):
        events.append(("evaluate", original_message))
        return expected

    agent._state_candidate_evaluator = evaluator
    context = _build(agent, user_message="new")

    assert isinstance(context, TurnContext)
    assert context.state_candidate_result is expected
    assert events == ["prefetch", ("evaluate", "new")]


def test_build_turn_context_without_explicit_evaluator_is_unchanged():
    agent = _FakeAgent()
    context = _build(agent, user_message="hello")
    assert isinstance(context, TurnContext)
    assert context.state_candidate_result is None


def test_evaluator_failure_is_fail_open():
    agent = _FakeAgent()

    def failing_evaluator(_agent, _message):
        raise RuntimeError("probe failure")

    agent._state_candidate_evaluator = failing_evaluator
    context = _build(agent, user_message="hello")
    assert isinstance(context, TurnContext)
    assert context.state_candidate_result is None
    assert context.messages[-1]["content"] == "hello"
