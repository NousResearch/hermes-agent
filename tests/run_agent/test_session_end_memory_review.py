"""Regression tests for last-chance memory review at session boundaries.

Issue #58669: sessions that end below ``memory.nudge_interval`` never reach
the periodic background review, so durable facts can disappear at reset.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import agent.background_review as background_review
from run_agent import AIAgent


def _messages():
    return [
        {"role": "user", "content": "I prefer concise status updates."},
        {"role": "assistant", "content": "Understood."},
    ]


def _reviewable_agent(*, pending_turns: int = 1):
    thread = MagicMock()
    agent = SimpleNamespace(
        _memory_store=object(),
        _memory_enabled=True,
        _user_profile_enabled=False,
        _memory_nudge_interval=10,
        _turns_since_memory=pending_turns,
        _background_review_state_lock=threading.Lock(),
        _last_memory_review_thread=None,
        _session_messages=_messages(),
        session_id="short-session",
    )

    def _spawn(**kwargs):
        agent._last_memory_review_thread = thread
        return thread

    agent._spawn_background_review = MagicMock(side_effect=_spawn)
    return agent, thread


def test_session_end_schedules_review_for_unreviewed_short_session():
    agent, thread = _reviewable_agent(pending_turns=3)

    result = background_review.schedule_session_end_memory_review(agent, _messages())

    assert result is thread
    assert agent._turns_since_memory == 0
    agent._spawn_background_review.assert_called_once_with(
        messages_snapshot=_messages(),
        review_memory=True,
        review_skills=False,
    )


def test_session_end_review_uses_agent_transcript_when_messages_omitted():
    agent, _thread = _reviewable_agent()

    background_review.schedule_session_end_memory_review(agent)

    assert agent._spawn_background_review.call_args.kwargs["messages_snapshot"] == _messages()


def test_session_end_review_is_claimed_only_once():
    agent, _thread = _reviewable_agent()

    background_review.schedule_session_end_memory_review(agent, _messages())
    background_review.schedule_session_end_memory_review(agent, _messages())

    agent._spawn_background_review.assert_called_once()


def test_session_end_review_restores_pending_counter_if_spawn_fails():
    agent, _thread = _reviewable_agent(pending_turns=2)
    agent._spawn_background_review.side_effect = RuntimeError("thread unavailable")

    try:
        background_review.schedule_session_end_memory_review(agent, _messages())
    except RuntimeError as exc:
        assert str(exc) == "thread unavailable"
    else:  # pragma: no cover - assertion guard
        raise AssertionError("spawn failure should be propagated")

    assert agent._turns_since_memory == 2


def test_session_end_review_skips_when_periodic_review_is_disabled():
    agent, _thread = _reviewable_agent()
    agent._memory_nudge_interval = 0

    assert background_review.schedule_session_end_memory_review(agent, _messages()) is None
    agent._spawn_background_review.assert_not_called()


def test_session_end_review_skips_when_no_turns_are_pending():
    agent, _thread = _reviewable_agent(pending_turns=0)

    assert background_review.schedule_session_end_memory_review(agent, _messages()) is None
    agent._spawn_background_review.assert_not_called()


def test_session_end_review_supports_user_profile_only_mode():
    agent, _thread = _reviewable_agent()
    agent._memory_enabled = False
    agent._user_profile_enabled = True

    background_review.schedule_session_end_memory_review(agent, _messages())

    agent._spawn_background_review.assert_called_once()


def test_session_end_review_keeps_user_fact_from_interrupted_exchange():
    agent, _thread = _reviewable_agent()

    result = background_review.schedule_session_end_memory_review(
        agent,
        [{"role": "user", "content": "cancelled before an answer"}],
    )

    assert result is not None
    assert agent._turns_since_memory == 0
    agent._spawn_background_review.assert_called_once()


def test_session_end_review_skips_empty_transcript():
    agent, _thread = _reviewable_agent()

    result = background_review.schedule_session_end_memory_review(agent, [])

    assert result is None
    assert agent._turns_since_memory == 1
    agent._spawn_background_review.assert_not_called()


def test_session_end_review_skips_when_builtin_memory_is_unavailable():
    agent, _thread = _reviewable_agent()
    agent._memory_store = None

    assert background_review.schedule_session_end_memory_review(agent, _messages()) is None
    agent._spawn_background_review.assert_not_called()


def test_wait_for_pending_memory_review_joins_latest_thread():
    agent, thread = _reviewable_agent()
    agent._last_memory_review_thread = thread
    thread.is_alive.return_value = False

    assert background_review.wait_for_pending_memory_review(agent, timeout=4.5) is True
    thread.join.assert_called_once_with(timeout=4.5)


def test_wait_for_pending_memory_review_is_bounded():
    agent, thread = _reviewable_agent()
    agent._last_memory_review_thread = thread
    thread.is_alive.return_value = True

    assert background_review.wait_for_pending_memory_review(agent, timeout=0) is False
    thread.join.assert_called_once_with(timeout=0.0)


def test_commit_memory_session_schedules_last_chance_review():
    agent, _thread = _reviewable_agent(pending_turns=1)
    agent._memory_manager = MagicMock()
    agent.context_compressor = MagicMock()
    agent.commit_memory_session = AIAgent.commit_memory_session.__get__(agent)

    agent.commit_memory_session(_messages())

    agent._memory_manager.on_session_end.assert_called_once_with(_messages())
    agent.context_compressor.on_session_end.assert_called_once_with(
        "short-session", _messages()
    )
    agent._spawn_background_review.assert_called_once()


def test_commit_memory_session_can_wait_at_true_teardown_boundary(monkeypatch):
    agent, _thread = _reviewable_agent(pending_turns=1)
    agent._memory_manager = None
    agent.context_compressor = None
    agent.commit_memory_session = AIAgent.commit_memory_session.__get__(agent)
    wait = MagicMock(return_value=True)
    monkeypatch.setattr(background_review, "wait_for_pending_memory_review", wait)

    agent.commit_memory_session(_messages(), wait_for_review=True)

    agent._spawn_background_review.assert_called_once()
    wait.assert_called_once_with(agent)


def test_shutdown_waits_for_last_chance_review_after_provider_flush():
    events = []

    class MemoryManager:
        def on_session_end(self, messages):
            events.append("provider-flush")

        def shutdown_all(self):
            events.append("provider-shutdown")

    class ContextEngine:
        def on_session_end(self, session_id, messages):
            events.append("context-end")

    class ReviewThread:
        def join(self, timeout):
            events.append("review-join")

        def is_alive(self):
            return False

    agent, _thread = _reviewable_agent(pending_turns=1)
    review_thread = ReviewThread()

    def _spawn(**kwargs):
        events.append("review-spawn")
        agent._last_memory_review_thread = review_thread
        return review_thread

    agent._spawn_background_review = _spawn
    agent._memory_manager = MemoryManager()
    agent.context_compressor = ContextEngine()
    agent.shutdown_memory_provider = AIAgent.shutdown_memory_provider.__get__(agent)

    agent.shutdown_memory_provider(_messages())

    assert events == [
        "provider-flush",
        "provider-shutdown",
        "review-spawn",
        "context-end",
        "review-join",
    ]


def test_background_memory_reviews_execute_serially(monkeypatch):
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def _target():
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1

    monkeypatch.setattr(
        background_review,
        "spawn_background_review_thread",
        lambda *args, **kwargs: (_target, "review"),
    )

    agent = object.__new__(AIAgent)
    agent._background_review_execution_lock = threading.Lock()
    agent._background_review_state_lock = threading.Lock()
    agent._last_memory_review_thread = None

    first = AIAgent._spawn_background_review(
        agent, _messages(), review_memory=True
    )
    second = AIAgent._spawn_background_review(
        agent, _messages(), review_memory=True
    )
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert max_active == 1
    assert agent._last_memory_review_thread is second


def test_background_review_captures_parent_state_before_session_rotation(monkeypatch):
    captured = {}
    agent = SimpleNamespace(
        model="old-model",
        provider="old-provider",
        platform="telegram",
        session_id="old-session",
        session_start="old-start",
        enabled_toolsets=["memory"],
        disabled_toolsets=["browser"],
        _credential_pool=object(),
        _memory_store=object(),
        _memory_enabled=True,
        _user_profile_enabled=False,
        _cached_system_prompt="old-prompt",
        memory_notifications="on",
        _safe_print=MagicMock(),
        background_review_callback=MagicMock(),
        _emit_auxiliary_failure=MagicMock(),
    )
    monkeypatch.setattr(
        background_review,
        "_resolve_review_runtime",
        lambda _agent: {
            "model": "old-model",
            "provider": "old-provider",
            "api_key": "captured-key",
        },
    )

    def fake_worker(_agent, _messages, _prompt, review_context=None):
        captured.update(review_context or {})

    monkeypatch.setattr(background_review, "_run_review_in_thread", fake_worker)

    target, _prompt = background_review.spawn_background_review_thread(
        agent,
        _messages(),
        review_memory=True,
    )
    agent.session_id = "new-session"
    agent.model = "new-model"
    agent._cached_system_prompt = "new-prompt"
    target()

    assert captured["session_id"] == "old-session"
    assert captured["model"] == "old-model"
    assert captured["cached_system_prompt"] == "old-prompt"
    assert captured["runtime"]["api_key"] == "captured-key"


def test_cli_new_schedules_builtin_review_without_external_memory_manager():
    from cli import HermesCLI

    agent, _thread = _reviewable_agent(pending_turns=2)
    agent._memory_manager = None
    agent.context_compressor = MagicMock()
    console = object.__new__(HermesCLI)
    console.agent = agent

    queued = console._launch_session_boundary_memory_flush(
        _messages(),
        session_id="short-session",
    )

    assert queued is None
    agent.context_compressor.on_session_end.assert_called_once_with(
        "short-session", _messages()
    )
    agent._spawn_background_review.assert_called_once()
    assert agent._turns_since_memory == 0
