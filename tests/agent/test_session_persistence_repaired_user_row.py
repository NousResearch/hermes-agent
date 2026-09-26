"""A user row repaired in place after it was written must not reach state.db a second time.

The pre-request alternation repair merges a following user row into an already-written one and
pops its persist marker; the flush only appends, so the merged row used to land again, and with the
length-continuation nudge Hermes's internal instruction was saved as the user's own words and replayed
on every later turn. Both flows drive the real turn loop against a real SessionDB (LLM faked).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import FINISH_REASON_LENGTH
from hermes_state import SessionDB


def _response(content, finish_reason="stop", reasoning_content=None):
    message = SimpleNamespace(content=content, tool_calls=None)
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    return SimpleNamespace(
        id="chatcmpl-test", model="test/model", usage=None,
        choices=[SimpleNamespace(index=0, message=message, finish_reason=finish_reason)],
    )


@pytest.fixture()
def session_agent(tmp_path, monkeypatch):
    # No request may leave the machine: any real HTTP client dies on a dead local proxy.
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setattr("agent.turn_context._maybe_title_session_at_turn_start", lambda *a, **k: None)
    monkeypatch.setattr("agent.turn_context.start_deferred_title_upgrade", lambda *a, **k: None)

    from run_agent import AIAgent

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "repaired-user-row"
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            session_db=db, session_id=session_id,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    try:
        yield agent, db, session_id
    finally:
        db.close()


def _answer_with(agent, *responses):
    agent.client.chat.completions.create.side_effect = list(responses)


def _saved_user_texts(db, session_id):
    return [m["content"] for m in db.get_messages_as_conversation(session_id) if m["role"] == "user"]


def test_thinking_only_length_retry_saves_the_prompt_once_and_never_the_nudge(session_agent):
    agent, db, session_id = session_agent
    _answer_with(agent, _response("Answer one."))
    first = agent.run_conversation("first question")

    # A reasoning model spends the whole output cap on thinking, then answers on the retry.
    _answer_with(
        agent,
        _response("", finish_reason=FINISH_REASON_LENGTH, reasoning_content="thinking " * 50),
        _response("Here is the full report."),
    )
    second = agent.run_conversation("write me a long report", conversation_history=first["messages"])

    assert second["completed"] is True
    prompts = ["first question", "write me a long report"]
    assert _saved_user_texts(db, session_id) == prompts
    # A CLI host replays result["messages"] as the next turn's history.
    assert [m["content"] for m in second["messages"] if m["role"] == "user"] == prompts


def test_prompt_carried_past_an_interrupted_lease_wait_is_saved_once(session_agent):
    agent, db, session_id = session_agent
    _answer_with(agent, _response("Answer one."))
    first = agent.run_conversation("first question")

    # Another Hermes process holds the session; the user sends B while A waits for it.
    other_holder = "pid=0:turn=other:platform=desktop"
    assert db.acquire_session_turn_lease(session_id, other_holder, ttl_seconds=300.0, wait_seconds=0.0)
    waiting = threading.Event()
    agent.status_callback = lambda *_: waiting.set()
    outcome = {}
    waiter = threading.Thread(target=lambda: outcome.setdefault(
        "result", agent.run_conversation("message A", conversation_history=first["messages"])))
    waiter.start()
    assert waiting.wait(30)
    agent.interrupt("message B")
    waiter.join(30)
    assert outcome["result"]["interrupted"] is True
    agent.status_callback = None
    db.release_session_turn_lease(session_id, other_holder)

    _answer_with(agent, _response("Answer to A and B."))
    follow_up = agent.run_conversation("message B", conversation_history=outcome["result"]["messages"])

    assert follow_up["completed"] is True
    assert _saved_user_texts(db, session_id) == ["first question", "message A", "message B"]
