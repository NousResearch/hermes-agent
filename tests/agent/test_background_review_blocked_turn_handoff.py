"""A ``pre_llm_call`` veto hands no review candidate to any owner; a delivered turn's candidate
still reaches the gateway owner.

Upstream moved the post-turn review spawn into ``agent/turn_finalizer._mirror_turn_to_memory``
behind the veto gate (a blocked prompt never left the host, so there is nothing to learn from
it). The review hand-off (``_hand_off_background_review``: gateway admission, durable lease
holder, or a bare spawn) lives behind the same gate, so a blocked turn hands off nothing and its
gateway admission releases the live-turn token candidate-free, while a normal turn's candidate
is captured on the admission, never spawned from inside the live turn. Real
``AIAgent.run_conversation`` against a local provider, like ``test_pre_llm_block_hook``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.turn_context import PRE_LLM_BLOCK_EXIT_REASON
from hermes_state import SessionDB
from tests.agent.test_pre_llm_block_hook import (  # provider_url is a fixture
    BLOCK_MSG,
    SECRET,
    _Provider,
    _agent,
    provider_url,
)


def _reviewing_agent(url, db):
    """An agent whose turn trips the skill-review trigger, bound to a gateway review owner."""
    agent = _agent(url, db)
    agent._skill_nudge_interval = 1
    agent._iters_since_skill = 1
    agent.valid_tool_names = set(agent.valid_tool_names) | {"skill_manage"}
    agent._gateway_review_admission = MagicMock(name="gateway_review_admission")
    agent._spawn_background_review = MagicMock(name="_spawn_background_review")
    return agent


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield db
    finally:
        db.close()


def test_a_vetoed_turn_hands_off_no_review_candidate(
    provider_url,  # health: allow F811 -- pytest injects the imported fixture by parameter name
    db,
):
    agent = _reviewing_agent(provider_url, db)
    verdicts = [{"action": "block", "message": BLOCK_MSG}]
    with patch(
        "hermes_cli.plugins.invoke_hook",
        side_effect=lambda hook, **kw: list(verdicts) if hook == "pre_llm_call" else [],
    ):
        result = agent.run_conversation(
            f"look up {SECRET}", conversation_history=[], task_id="t"
        )

    assert result["turn_exit_reason"] == PRE_LLM_BLOCK_EXIT_REASON
    assert result["final_response"] == BLOCK_MSG and _Provider.chat_requests == []
    agent._gateway_review_admission.capture_candidate.assert_not_called()
    agent._spawn_background_review.assert_not_called()
    assert getattr(agent, "_post_turn_background_review_candidate", None) is None


def test_a_delivered_turn_hands_its_candidate_to_the_gateway_owner(
    provider_url,  # health: allow F811 -- pytest injects the imported fixture by parameter name
    db,
):
    agent = _reviewing_agent(provider_url, db)

    result = agent.run_conversation(
        "plain question", conversation_history=[], task_id="t"
    )

    assert result["final_response"] == "MODEL-ANSWER"
    agent._gateway_review_admission.capture_candidate.assert_called_once()
    captured = agent._gateway_review_admission.capture_candidate.call_args
    assert captured.args[0] is agent and captured.kwargs["review_skills"] is True
    assert [message["role"] for message in captured.args[1]][-2:] == [
        "user",
        "assistant",
    ]
    agent._spawn_background_review.assert_not_called()
