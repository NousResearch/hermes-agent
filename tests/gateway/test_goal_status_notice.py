from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE


class FakeAdapter:
    def __init__(self):
        self.calls = []
        self.callbacks = {}
        self._active_sessions = {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SimpleNamespace(success=True)

    def register_post_delivery_callback(self, session_key, callback, *, generation=None):
        self.callbacks[session_key] = (generation, callback)


def _goal_continuation_event(source, goal="finish the task"):
    return MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal=goal),
        message_type=MessageType.TEXT,
        source=source,
    )


@pytest.mark.asyncio
async def test_goal_status_notice_defers_until_post_delivery_callback():
    """Regression: goal status must appear after the agent's visible reply.

    _post_turn_goal_continuation runs before BasePlatformAdapter sends the
    returned final response. It should therefore register a post-delivery
    callback, not send the judge status immediately.
    """
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False)

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
        user_id="user-1",
    )

    await runner._defer_goal_status_notice_after_delivery(source, "✓ Goal achieved: done")

    assert adapter.calls == []
    assert len(adapter.callbacks) == 1

    _, callback = next(iter(adapter.callbacks.values()))
    result = callback()
    if hasattr(result, "__await__"):
        await result

    assert adapter.calls == [
        {
            "chat_id": "parent-channel",
            "content": "✓ Goal achieved: done",
            "reply_to": None,
            "metadata": {"thread_id": "thread-123"},
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_notice"),
    [
        (
            "missing",
            "Goal status unavailable: session goal storage is unavailable",
        ),
        (
            "read",
            "Goal status unavailable: persisted goal read failed",
        ),
        (
            "invalid",
            "Goal status unavailable: persisted goal state is invalid",
        ),
    ],
)
async def test_post_turn_goal_read_failure_is_reported_without_judge_or_enqueue(
    monkeypatch, caplog, failure, expected_notice
):
    """An unverified read failure must not look like verified goal absence."""
    from hermes_cli import goals

    class GoalDB:
        def get_meta(self, _key):
            if failure == "read":
                raise OSError("simulated goal read failure")
            return "{invalid goal json"

    monkeypatch.setattr(
        goals,
        "_get_session_db",
        (lambda: None) if failure == "missing" else (lambda: GoalDB()),
    )
    judge = MagicMock(side_effect=AssertionError("goal judge must not run"))
    monkeypatch.setattr(goals, "judge_goal", judge)

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = SimpleNamespace(
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner._warm_goals_session_db = AsyncMock()
    runner._goal_max_turns_from_config = MagicMock(return_value=20)
    runner._enqueue_fifo = MagicMock()
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
        user_id="user-1",
    )
    session_entry = SimpleNamespace(session_id="goal-read-failure-session")

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=source,
            final_response="partial progress",
        )

    judge.assert_not_called()
    runner._enqueue_fifo.assert_not_called()
    assert adapter.calls == []
    assert len(adapter.callbacks) == 1
    assert expected_notice in caplog.text

    _, callback = next(iter(adapter.callbacks.values()))
    await callback()
    assert [call["content"] for call in adapter.calls] == [expected_notice]


@pytest.mark.asyncio
async def test_post_turn_verified_goal_absence_remains_a_silent_noop(monkeypatch):
    """An authoritative empty row is still a normal no-goal control case."""
    from hermes_cli import goals

    class EmptyGoalDB:
        def get_meta(self, _key):
            return None

    monkeypatch.setattr(goals, "_get_session_db", lambda: EmptyGoalDB())
    judge = MagicMock(side_effect=AssertionError("goal judge must not run"))
    monkeypatch.setattr(goals, "judge_goal", judge)

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = SimpleNamespace(
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner._warm_goals_session_db = AsyncMock()
    runner._goal_max_turns_from_config = MagicMock(return_value=20)
    runner._enqueue_fifo = MagicMock()
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
        user_id="user-1",
    )

    await runner._post_turn_goal_continuation(
        session_entry=SimpleNamespace(session_id="verified-empty-goal-session"),
        source=source,
        final_response="ordinary response",
    )

    judge.assert_not_called()
    runner._enqueue_fifo.assert_not_called()
    assert adapter.calls == []
    assert adapter.callbacks == {}


@pytest.mark.asyncio
async def test_late_goal_read_failure_from_real_hook_is_reported_without_judge_or_enqueue(
    monkeypatch, caplog
):
    """A read failing after gateway precheck must not end continuation silently."""
    from hermes_cli import goals

    active_raw = goals.GoalState(goal="finish the task").to_json()
    db_calls = 0

    class ReadableGoalDB:
        def get_meta(self, _key):
            return active_raw

    class FailingGoalDB:
        def get_meta(self, _key):
            raise OSError("sensitive staged-store details")

    def staged_goal_db():
        nonlocal db_calls
        db_calls += 1
        if db_calls < 3:
            return ReadableGoalDB()
        return FailingGoalDB()

    monkeypatch.setattr(goals, "_get_session_db", staged_goal_db)
    judge = MagicMock(side_effect=AssertionError("goal judge must not run"))
    monkeypatch.setattr(goals, "judge_goal", judge)

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = SimpleNamespace(
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner._warm_goals_session_db = AsyncMock()
    runner._goal_max_turns_from_config = MagicMock(return_value=20)
    runner._enqueue_fifo = MagicMock()
    runner._post_turn_loop_completion = AsyncMock()

    async def run_inline(func, *args):
        return func(*args)

    runner._run_in_executor_with_context = run_inline
    session_entry = SimpleNamespace(session_id="late-goal-read-failure-session")
    runner.session_store = MagicMock()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=session_entry),
    )
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
        user_id="user-1",
    )
    expected_notice = "Goal status unavailable: persisted goal read failed"

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await runner._run_post_turn_hooks(
            agent_result={"final_response": "partial progress"},
            source=source,
            is_internal=False,
        )

    assert db_calls == 3
    judge.assert_not_called()
    runner._enqueue_fifo.assert_not_called()
    runner._post_turn_loop_completion.assert_awaited_once()
    assert adapter.calls == []
    assert len(adapter.callbacks) == 1
    assert expected_notice in caplog.text
    assert "sensitive staged-store details" not in caplog.text

    _, callback = next(iter(adapter.callbacks.values()))
    await callback()
    assert [call["content"] for call in adapter.calls] == [expected_notice]
    assert "sensitive staged-store details" not in adapter.calls[0]["content"]


