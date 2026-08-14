"""The agent facade forwards ``turn_author`` into the conversation loop.

Every dispatcher gates the keyword on the callee's signature, so a facade that does not declare it
would drop the author for every real agent without an error.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.interrupt_compat import _accepts_keyword
from agent.turn_facade import TurnFacadeMixin

AUTHOR = {"id": "bot:coder", "name": "coder", "is_bot": True}


def test_real_agent_signature_accepts_turn_author():
    from run_agent import AIAgent

    assert _accepts_keyword(AIAgent.run_conversation, "turn_author")


@pytest.mark.parametrize("author", [AUTHOR, None])
def test_facade_forwards_turn_author_to_the_conversation_loop(monkeypatch, author):
    recorded = {}

    def fake_loop(agent, user_message, *args, **kwargs):
        recorded.update(kwargs, user_message=user_message)
        return {"final_response": "ok"}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_loop)
    monkeypatch.setattr("agent.background_review.cancel_background_review_for_live_turn", lambda agent: None)
    monkeypatch.setattr("agent.turn_facade_lease.admit_durable_turn_lease",
                        lambda agent, **kw: SimpleNamespace(early_result=None, lease=None, conversation_history=[]))
    monkeypatch.setattr("agent.relay_runtime.SESSION_COORDINATOR", MagicMock())
    monkeypatch.setattr("hermes_cli.observability.relay_shared_metrics.start_task_run", lambda **kw: None)
    monkeypatch.setattr("hermes_cli.observability.relay_shared_metrics.finish_task_run", lambda **kw: None)
    monkeypatch.setattr("agent.subagent_lifecycle.bind_subagent_parent", lambda agent: nullcontext())
    monkeypatch.setattr("agent.auxiliary_client.scoped_runtime_main", lambda runtime: nullcontext())
    agent = SimpleNamespace(
        session_id="s1", platform="cli", model="m", _session_db=None,
        _conversation_root_id=lambda: "root", _reset_activity_labels_after_turn=lambda: None,
    )

    result = TurnFacadeMixin.run_conversation(agent, "hello", **({"turn_author": author} if author else {}))

    assert result == {"final_response": "ok"}
    assert recorded["user_message"] == "hello"
    assert recorded["turn_author"] == author
