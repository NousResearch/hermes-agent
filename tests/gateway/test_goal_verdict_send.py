"""Tests for gateway /goal verdict-message delivery.

The judge verdict message ("✓ Goal achieved", "⏸ budget exhausted", etc.)
must reach the user after each turn. Before this fix the code checked
``hasattr(adapter, "send_message")`` — but adapters expose ``send()``,
never ``send_message``, so the check always evaluated False and users
never saw verdicts. This test locks in the fix.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


class _RecordingAdapter:
    """Minimal adapter that records send() invocations."""

    def __init__(self) -> None:
        self._pending_messages: dict = {}
        self.sends: list[dict] = []

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None):
        self.sends.append({"chat_id": chat_id, "content": content, "metadata": metadata})

        class _R:
            success = True
            message_id = "mock-msg"

        return _R()


def _make_runner_with_adapter(session_id: str = None):
    from gateway.run import GatewayRunner
    import uuid

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._queued_events = {}
    runner._background_tasks = set()
    runner._goal_background_missions = {}

    src = _make_source()
    # Default to a unique session_id so xdist parallel runs on the same worker
    # don't see each other's GoalManager state (DEFAULT_DB_PATH gets frozen at
    # module-import time, defeating per-test HERMES_HOME monkeypatches).
    session_entry = SessionEntry(
        session_key=build_session_key(src),
        session_id=session_id or f"goal-sess-{uuid.uuid4().hex[:8]}",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._generate_session_key.return_value = build_session_key(src)

    adapter = _RecordingAdapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    return runner, adapter, session_entry, src


@pytest.mark.asyncio
async def test_goal_verdict_done_sent_via_adapter_send(hermes_home):
    """When the judge says done, the '✓ Goal achieved' message must reach
    the user through the adapter's ``send()`` method."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="I shipped the feature.",
        )
        # fire-and-forget create_task — give the loop a tick
        await asyncio.sleep(0.05)

    assert len(adapter.sends) == 1, f"expected 1 send, got {len(adapter.sends)}: {adapter.sends}"
    msg = adapter.sends[0]
    assert msg["chat_id"] == "c1"
    assert "Goal achieved" in msg["content"]
    assert "the feature shipped" in msg["content"]


@pytest.mark.asyncio
async def test_goal_verdict_continue_enqueues_continuation(hermes_home):
    """When the judge says continue, both the 'continuing' status and the
    continuation-prompt event must be delivered. The continuation prompt is
    routed through the adapter's pending-messages FIFO so the goal loop
    proceeds on the next turn."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("polish the docs")

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "still needs work", False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="here's a partial edit",
        )
        await asyncio.sleep(0.05)

    # Status line sent back
    assert len(adapter.sends) == 1
    assert "Continuing toward goal" in adapter.sends[0]["content"]
    # Continuation prompt enqueued for next turn
    assert adapter._pending_messages, "continuation prompt must be enqueued in pending_messages"


@pytest.mark.asyncio
async def test_goal_verdict_budget_exhausted_sends_pause(hermes_home):
    """When the budget is exhausted, a '⏸ Goal paused' message must be sent
    and no further continuation enqueued."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager, save_goal

    mgr = GoalManager(session_entry.session_id, default_max_turns=2)
    state = mgr.set("tiny goal", max_turns=2)
    state.turns_used = 2
    save_goal(session_entry.session_id, state)

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "keep going", False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="still partial",
        )
        await asyncio.sleep(0.05)

    assert len(adapter.sends) == 1
    content = adapter.sends[0]["content"]
    assert "paused" in content.lower()
    assert "turns used" in content.lower()
    # No continuation enqueued when budget is exhausted
    assert not adapter._pending_messages


@pytest.mark.asyncio
async def test_goal_verdict_skipped_when_no_active_goal(hermes_home):
    """No goal set → the hook is a no-op. Nothing is sent, nothing enqueued."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    await runner._post_turn_goal_continuation(
        session_entry=session_entry,
        source=src,
        final_response="anything",
    )
    await asyncio.sleep(0.05)

    assert adapter.sends == []
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_goal_verdict_survives_adapter_without_send(hermes_home):
    """Bad adapter (no ``send`` attribute) must not crash the judge hook."""
    runner, _adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("survive missing send")

    class _NoSendAdapter:
        def __init__(self):
            self._pending_messages: dict = {}

    runner.adapters[Platform.TELEGRAM] = _NoSendAdapter()

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "ok", False)):
        # must not raise
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="whatever",
        )
        await asyncio.sleep(0.05)


def _make_goal_event(text: str = "/goal ship the feature") -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


@pytest.mark.asyncio
async def test_goal_set_starts_background_mission_without_replaying_goal_as_plain_text(hermes_home):
    """/goal <text> should not enqueue <text> as a normal foreground turn."""
    runner, _adapter, _session_entry, _src = _make_runner_with_adapter()
    runner._enqueue_fifo = MagicMock()
    runner._run_goal_background_mission = AsyncMock()

    response = await runner._handle_goal_command(_make_goal_event("/goal ship the feature"))
    await asyncio.sleep(0)

    assert "Goal set" in response
    runner._enqueue_fifo.assert_not_called()
    runner._run_goal_background_mission.assert_awaited_once()
    await_args = runner._run_goal_background_mission.await_args
    assert await_args is not None
    state_arg, source_arg = await_args.args
    assert state_arg.goal == "ship the feature"
    assert source_arg.chat_id == "c1"


@pytest.mark.asyncio
async def test_goal_resume_restarts_background_mission(hermes_home):
    """/goal resume must actually restart the autonomous background mission."""
    runner, _adapter, session_entry, _src = _make_runner_with_adapter()
    runner._run_goal_background_mission = AsyncMock()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    state = mgr.set("ship the feature")
    mgr.pause(reason="test pause")

    response = await runner._handle_goal_command(_make_goal_event("/goal resume"))
    await asyncio.sleep(0)

    assert "Goal resumed" in response
    assert "background mission" in response
    runner._run_goal_background_mission.assert_awaited_once()
    await_args = runner._run_goal_background_mission.await_args
    assert await_args is not None
    state_arg, source_arg = await_args.args
    assert state_arg.goal == state.goal
    assert source_arg.chat_id == "c1"


@pytest.mark.asyncio
async def test_goal_background_mission_judges_response_and_marks_goal_done(hermes_home):
    """The dedicated /goal mission must still feed output into GoalManager."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    state = GoalManager(session_entry.session_id).set("ship the feature")
    runner._run_background_task = AsyncMock(return_value="I shipped the feature.")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False)):
        await runner._run_goal_background_mission(state, src)

    runner._run_background_task.assert_awaited_once()
    assert any("Goal achieved" in send["content"] for send in adapter.sends)
    assert not GoalManager(session_entry.session_id).is_active()


@pytest.mark.asyncio
async def test_goal_background_mission_does_not_resurrect_cleared_goal(hermes_home):
    """A stale background mission must not overwrite /goal clear."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    state = mgr.set("ship the feature")

    async def _clear_during_background(_prompt, _source, _task_id):
        GoalManager(session_entry.session_id).clear()
        return "I shipped the old feature."

    runner._run_background_task = AsyncMock(side_effect=_clear_during_background)

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "old goal done", False)):
        await runner._run_goal_background_mission(state, src)

    current = GoalManager(session_entry.session_id)._state
    assert current is not None
    assert current.status == "cleared"
    assert adapter.sends == []


@pytest.mark.asyncio
async def test_goal_background_mission_does_not_overwrite_replaced_goal(hermes_home):
    """A stale mission for one goal must not mutate a newer replacement goal."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    state = mgr.set("old goal")

    async def _replace_during_background(_prompt, _source, _task_id):
        GoalManager(session_entry.session_id).set("new goal")
        return "I completed the old goal."

    runner._run_background_task = AsyncMock(side_effect=_replace_during_background)

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "old goal done", False)):
        await runner._run_goal_background_mission(state, src)

    current = GoalManager(session_entry.session_id)._state
    assert current is not None
    assert current.goal == "new goal"
    assert current.status == "active"
    assert adapter.sends == []


@pytest.mark.asyncio
async def test_post_turn_goal_continuation_skips_when_background_mission_owns_goal(hermes_home):
    """Foreground turns must not enqueue goal continuations during background /goal work."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    state = GoalManager(session_entry.session_id).set("ship the feature")
    runner._goal_background_missions[session_entry.session_id] = (state.goal, state.created_at)
    runner._enqueue_fifo = MagicMock()

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "keep going", False)) as judge:
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="foreground chat reply",
        )

    judge.assert_not_called()
    runner._enqueue_fifo.assert_not_called()
    assert adapter.sends == []
