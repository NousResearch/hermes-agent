"""Regression tests for /goal state across compression session splits."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import uuid

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_state import SessionDB


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


class _RecordingAdapter:
    def __init__(self) -> None:
        self._pending_messages: dict = {}
        self.sends: list[dict] = []

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None):
        self.sends.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SimpleNamespace(success=True, message_id="mock-msg")


def _source(*, thread_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-1",
        chat_id="chat-1",
        user_name="tester",
        chat_type="dm",
        thread_id=thread_id,
    )


def _make_runner(session_entry: SessionEntry, source: SessionSource) -> tuple[GatewayRunner, _RecordingAdapter]:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    runner.adapters = {}
    runner._queued_events = {}
    runner._session_db = SessionDB()
    runner._session_model_overrides = {}

    session_key = session_entry.session_key
    runner.session_store = MagicMock()
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store._generate_session_key.return_value = session_key
    runner.session_store.get_or_create_session.return_value = session_entry

    adapter = _RecordingAdapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    return runner, adapter


def _session_entry(source: SessionSource, session_id: str) -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=source.platform,
        chat_type=source.chat_type,
    )


def _create_compression_child(db: SessionDB, parent_id: str, child_id: str) -> None:
    db.create_session(parent_id, "telegram")
    db.end_session(parent_id, "compression")
    db.create_session(child_id, "telegram", parent_session_id=parent_id)
    assert db.get_compression_tip(parent_id) == child_id


def _create_compression_chain(db: SessionDB, root_id: str, middle_id: str, tip_id: str) -> None:
    _create_compression_child(db, root_id, middle_id)
    db.end_session(middle_id, "compression")
    db.create_session(tip_id, "telegram", parent_session_id=middle_id)
    assert db.get_compression_tip(root_id) == tip_id
    assert db.get_compression_tip(middle_id) == tip_id


def _create_non_compression_child(db: SessionDB, parent_id: str, child_id: str) -> None:
    db.create_session(parent_id, "telegram")
    db.create_session(child_id, "telegram", parent_session_id=parent_id)
    assert db.get_compression_tip(parent_id) == parent_id


def _sid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_goal_continuation_migrates_and_judges_after_compression_split(hermes_home):
    """A compressed child must inherit the parent /goal before post-turn judging."""
    db = SessionDB()
    parent_id = _sid("goal-compress-parent")
    child_id = _sid("goal-compress-child")
    _create_compression_child(db, parent_id, child_id)

    from hermes_cli.goals import GoalManager, load_goal

    parent_goal = GoalManager(parent_id, default_max_turns=5).set("finish the migration")
    parent_goal.turns_used = 2
    parent_goal.last_verdict = "continue"
    parent_goal.last_reason = "needs another turn"
    parent_goal.subgoals = ["keep it compression-only"]
    from hermes_cli.goals import save_goal

    save_goal(parent_id, parent_goal)

    source = _source()
    session_entry = _session_entry(source, parent_id)
    runner, adapter = _make_runner(session_entry, source)

    runner._handle_compression_session_switch(
        session_key=session_entry.session_key,
        session_entry=session_entry,
        old_session_id=parent_id,
        new_session_id=child_id,
        source=source,
        reason="agent-result-compression",
    )

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "not done", False)) as judge:
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=source,
            final_response="partial progress",
        )

    child_goal = load_goal(child_id)
    old_goal = load_goal(parent_id)
    assert child_goal is not None
    assert child_goal.goal == "finish the migration"
    assert child_goal.subgoals == ["keep it compression-only"]
    assert child_goal.turns_used == 3
    assert child_goal.last_verdict == "continue"
    assert child_goal.last_reason == "not done"
    assert old_goal is not None
    assert old_goal.status == "cleared"
    assert f"migrated to {child_id}" in (old_goal.paused_reason or "")
    assert session_entry.session_id == child_id
    judge.assert_called_once()
    assert judge.call_args.args[0] == "finish the migration"
    assert len(adapter.sends) == 1
    assert "Continuing toward goal" in adapter.sends[0]["content"]
    assert adapter._pending_messages


@pytest.mark.asyncio
async def test_non_compression_child_does_not_inherit_parent_goal(hermes_home):
    db = SessionDB()
    parent_id = _sid("goal-non-compression-parent")
    child_id = _sid("goal-non-compression-child")
    _create_non_compression_child(db, parent_id, child_id)

    from hermes_cli.goals import GoalManager, load_goal

    GoalManager(parent_id).set("do not leak")
    source = _source()
    session_entry = _session_entry(source, parent_id)
    runner, adapter = _make_runner(session_entry, source)

    runner._handle_compression_session_switch(
        session_key=session_entry.session_key,
        session_entry=session_entry,
        old_session_id=parent_id,
        new_session_id=child_id,
        source=source,
        reason="agent-result-compression",
    )

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "should not run", False)) as judge:
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=source,
            final_response="partial progress",
        )

    assert load_goal(child_id) is None
    judge.assert_not_called()
    assert adapter.sends == []
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_goal_continuation_migrates_when_compression_tip_walk_routes_to_latest_child(hermes_home):
    db = SessionDB()
    root_id = _sid("goal-tip-root")
    middle_id = _sid("goal-tip-middle")
    tip_id = _sid("goal-tip-final")
    _create_compression_chain(db, root_id, middle_id, tip_id)

    from hermes_cli.goals import GoalManager, load_goal

    GoalManager(root_id).set("survive tip walk")
    source = _source(thread_id="topic-1")
    session_entry = _session_entry(source, root_id)
    runner, adapter = _make_runner(session_entry, source)

    runner._handle_compression_session_switch(
        session_key=session_entry.session_key,
        session_entry=session_entry,
        old_session_id=root_id,
        new_session_id=tip_id,
        source=source,
        reason="compression-tip-walk",
    )

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "not done", False)) as judge:
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=source,
            final_response="still working",
        )

    assert load_goal(tip_id) is not None
    assert load_goal(tip_id).goal == "survive tip walk"
    assert load_goal(root_id).status == "cleared"
    assert load_goal(middle_id) is None
    judge.assert_called_once()
    assert adapter._pending_messages


@pytest.mark.asyncio
async def test_compression_tip_walk_migrates_active_intermediate_goal_to_latest_child(hermes_home):
    db = SessionDB()
    root_id = _sid("goal-tip-intermediate-root")
    middle_id = _sid("goal-tip-intermediate-middle")
    tip_id = _sid("goal-tip-intermediate-final")
    _create_compression_chain(db, root_id, middle_id, tip_id)

    from hermes_cli.goals import GoalManager, GoalState, load_goal, save_goal

    save_goal(root_id, GoalState(goal="already migrated", status="cleared", paused_reason=f"migrated to {middle_id}"))
    GoalManager(middle_id).set("live on intermediate")
    source = _source(thread_id="topic-1")
    session_entry = _session_entry(source, root_id)
    runner, adapter = _make_runner(session_entry, source)

    runner._handle_compression_session_switch(
        session_key=session_entry.session_key,
        session_entry=session_entry,
        old_session_id=root_id,
        new_session_id=tip_id,
        source=source,
        reason="compression-tip-walk",
    )

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "not done", False)) as judge:
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=source,
            final_response="still working",
        )

    assert load_goal(tip_id) is not None
    assert load_goal(tip_id).goal == "live on intermediate"
    assert load_goal(middle_id).status == "cleared"
    judge.assert_called_once()
    assert adapter._pending_messages


@pytest.mark.asyncio
async def test_compression_tip_walk_does_not_cross_non_compression_edge(hermes_home):
    db = SessionDB()
    root_id = _sid("goal-tip-unsafe-root")
    middle_id = _sid("goal-tip-unsafe-middle")
    unsafe_child_id = _sid("goal-tip-unsafe-child")
    _create_compression_child(db, root_id, middle_id)
    db.create_session(unsafe_child_id, "telegram", parent_session_id=middle_id)
    assert db.get_compression_tip(root_id) == middle_id

    from hermes_cli.goals import GoalManager, load_goal

    GoalManager(root_id).set("do not cross unsafe edge")
    source = _source(thread_id="topic-1")
    session_entry = _session_entry(source, root_id)
    runner, adapter = _make_runner(session_entry, source)

    runner._handle_compression_session_switch(
        session_key=session_entry.session_key,
        session_entry=session_entry,
        old_session_id=root_id,
        new_session_id=unsafe_child_id,
        source=source,
        reason="compression-tip-walk",
    )

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "should not run", False)) as judge:
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=source,
            final_response="partial progress",
        )

    assert load_goal(unsafe_child_id) is None
    assert load_goal(root_id).status == "active"
    judge.assert_not_called()
    assert adapter.sends == []
    assert adapter._pending_messages == {}
