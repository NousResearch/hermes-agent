"""A Codex app-server turn's review candidate follows the same ownership hand-off as every
other delivered turn.

``agent/codex_runtime.py::_finish_codex_turn`` is the only post-turn review site for
``api_mode == "codex_app_server"``: ``run_conversation`` returns the Codex result before
``finalize_turn``. Spawning the automatic review there, inside the live turn, is refused on
every surface that owns the foreground: under the gateway the turn's ``_GatewayReviewAdmission``
token is live beside the facade token (``live_turn_active``), and a turn holding the durable
session row would refuse its own fork (``durable_foreground_active``). The hand-off
(``agent/turn_finalizer.py::_hand_off_background_review``) keeps the review: the gateway owner
spawns it once after the confirmed delivery, a durable-lease holder stages it for the facade,
and a bare turn spawns it directly, exactly as the chat-completions finalizer does.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import codex_runtime, review_admission
from agent.turn_facade_review import ForegroundReviewOwnership
from gateway.run_turn_review import _GatewayReviewAdmission


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": "done"},
    ]


def _turn():
    return SimpleNamespace(
        final_text="done", interrupted=False, error=None, tool_iterations=0
    )


def _agent(session_id: str, **overrides):
    """The slice of AIAgent state ``_finish_codex_turn`` reads; the spawn is a recorder."""
    attrs = dict(
        session_id=session_id,
        skip_background_review=False,
        _iters_since_skill=0,
        _skill_nudge_interval=0,
        valid_tool_names=set(),
        _active_turn_token=None,
        _active_turn_profile_key=None,
        _active_session_turn_lease_holder=None,
        _spawn_background_review=MagicMock(),
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _finish(agent, messages, *, should_review_memory: bool = True):
    return codex_runtime._finish_codex_turn(
        agent,
        _turn(),
        messages,
        original_user_message="ask",
        should_review_memory=should_review_memory,
        final_response="done",
    )


@pytest.fixture(autouse=True)
def _bookkeeping_stubs(monkeypatch):
    """Compaction and usage bookkeeping read live Codex session state the review path never needs."""
    monkeypatch.setattr(
        codex_runtime, "_record_codex_app_server_compaction", lambda a, t: None
    )
    monkeypatch.setattr(
        codex_runtime, "_record_codex_app_server_usage", lambda a, t, messages=None: {}
    )


def test_gateway_turn_captures_on_its_admission_and_spawns_once_after_delivery():
    """Both foreground tokens live (the gateway admission's and the facade's): nothing spawns
    inside the turn; the candidate sits on the admission under the admission's token and starts
    exactly once when the delivery is confirmed."""
    session_id, profile_key = "codex-gateway-turn", "/profiles/codex-gateway"
    gateway_token = review_admission.note_turn_started(session_id, profile_key)
    facade_token = review_admission.note_turn_started(session_id, profile_key)
    admission = _GatewayReviewAdmission(session_id, profile_key, gateway_token)
    agent = _agent(
        session_id,
        _active_turn_token=facade_token,
        _active_turn_profile_key=profile_key,
    )
    admission.bind_agent(agent)
    messages = _messages()
    try:
        _finish(agent, messages)

        agent._spawn_background_review.assert_not_called()
        candidate = admission._candidate
        assert isinstance(candidate, dict)
        assert candidate["_spawning_turn_token"] == admission.token
        assert candidate["_review_session_id"] == session_id
        assert candidate["messages_snapshot"] == messages
        assert candidate["review_memory"] is True
        assert candidate["review_skills"] is False
        # The facade releases its token when run_conversation returns, before the adapter delivers.
        review_admission.note_turn_finished(session_id, facade_token, profile_key)
        facade_token = None

        spawner = admission.finish(delivery_succeeded=True)
        assert spawner is not None
        spawner.join(timeout=10)
        assert not spawner.is_alive()
    finally:
        if facade_token is not None:
            review_admission.note_turn_finished(session_id, facade_token, profile_key)
        admission.finish(delivery_succeeded=False)

    agent._spawn_background_review.assert_called_once()
    spawned = agent._spawn_background_review.call_args.kwargs
    assert spawned["_spawning_turn_token"] == gateway_token
    assert spawned["messages_snapshot"] == messages
    assert not review_admission.other_live_turn(session_id, None, profile_key)


def test_durable_lease_holder_stages_for_the_facade_which_spawns_on_release():
    """A CLI/TUI turn holding the durable session row cannot fork beside itself: the candidate
    waits on the agent and ``ForegroundReviewOwnership.finish`` starts it after the release."""
    session_id = "codex-durable-turn"
    agent = _agent(
        session_id,
        _active_session_turn_lease_holder="durable-holder",
        _active_turn_token=42,
        _active_turn_profile_key="/profiles/codex-durable",
    )
    messages = _messages()

    _finish(agent, messages)

    agent._spawn_background_review.assert_not_called()
    staged = {
        "messages_snapshot": messages,
        "review_memory": True,
        "review_skills": False,
        "_spawning_turn_token": 42,
        "_review_profile_key": "/profiles/codex-durable",
        "_review_session_id": session_id,
    }
    assert agent._post_turn_background_review_candidate == staged

    ownership = ForegroundReviewOwnership(agent, session_id)
    ownership.finish()

    agent._spawn_background_review.assert_called_once_with(**staged)
    assert not hasattr(agent, "_post_turn_background_review_candidate")


def test_bare_turn_spawns_directly_with_the_finalizer_signature():
    """No gateway owner and no durable lease: the review starts from the turn as before, on a
    snapshot that does not alias the live transcript."""
    agent = _agent("codex-bare-turn")
    messages = _messages()

    _finish(agent, messages)

    agent._spawn_background_review.assert_called_once()
    spawned = agent._spawn_background_review.call_args.kwargs
    assert spawned == {
        "messages_snapshot": messages,
        "review_memory": True,
        "review_skills": False,
    }
    assert spawned["messages_snapshot"] is not messages
    assert not hasattr(agent, "_post_turn_background_review_candidate")


def test_skip_background_review_suppresses_every_hand_off_sink():
    """``skip_background_review`` (cron, managed workers, safe-worker policy) is the finalizer's
    gate for the fork; the Codex site honours the same switch before any owner sees a candidate."""
    admission = MagicMock()
    agent = _agent(
        "codex-skipped-turn",
        skip_background_review=True,
        _gateway_review_admission=admission,
        _active_session_turn_lease_holder="durable-holder",
    )

    _finish(agent, _messages())

    admission.capture_candidate.assert_not_called()
    agent._spawn_background_review.assert_not_called()
    assert not hasattr(agent, "_post_turn_background_review_candidate")
