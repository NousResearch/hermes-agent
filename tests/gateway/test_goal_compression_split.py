"""Regression tests for /goal state across compression session splits."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
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

    def _switch_session(_session_key: str, target_session_id: str) -> SessionEntry:
        return SessionEntry(
            session_key=_session_key,
            session_id=target_session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            origin=source,
            platform=source.platform,
            chat_type=source.chat_type,
        )

    runner.session_store.switch_session = MagicMock(side_effect=_switch_session)

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


def _event(text: str, source: SessionSource) -> MessageEvent:
    return MessageEvent(text=text, source=source, message_id="mock-message")


def _history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def _fake_compression_agent(child_id: str, compressed: list[dict[str, str]]) -> MagicMock:
    agent = MagicMock()
    agent.session_id = child_id
    agent._cached_system_prompt = ""
    agent.tools = []
    agent._compress_context.return_value = (compressed, "")
    agent.context_compressor = SimpleNamespace(
        has_content_to_compress=lambda _messages: True,
        _last_compress_aborted=False,
        _last_summary_error=None,
        _last_aux_model_failure_model=None,
        _last_aux_model_failure_error=None,
    )
    return agent


def _prepare_runner_for_manual_compress(
    runner: GatewayRunner,
    source: SessionSource,
    session_entry: SessionEntry,
    history: list[dict[str, str]],
) -> None:
    runner.session_store.load_transcript.return_value = history
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._session_key_for_source = lambda _source: session_entry.session_key
    runner._resolve_session_agent_runtime = lambda **_kwargs: (
        "test-model",
        {"api_key": "***"},
    )
    runner._evict_cached_agent = MagicMock()
    runner._cleanup_agent_resources = MagicMock()


def _prepare_runner_for_hygiene_message(
    runner: GatewayRunner,
    source: SessionSource,
    session_entry: SessionEntry,
    history: list[dict[str, str]],
) -> None:
    _prepare_runner_for_manual_compress(runner, source, session_entry, history)
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.clear_resume_pending = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = MagicMock()
    runner._is_telegram_topic_lane = lambda _source: False
    runner._is_telegram_topic_root_lobby = lambda _source: False
    runner._set_session_env = lambda _context: None
    runner._clear_session_env = MagicMock()
    runner._set_session_reasoning_override = MagicMock()
    runner._format_session_info = MagicMock(return_value="")
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner._reply_anchor_for_event = lambda _event: None
    runner._deliver_platform_notice = AsyncMock()
    runner._prepare_inbound_message_text = AsyncMock(return_value="after hygiene")
    runner._bind_adapter_run_generation = MagicMock()
    runner._run_agent = AsyncMock(
        return_value={
            "success": True,
            "final_response": "hygiene turn complete",
            "session_id": session_entry.session_id,
            "messages": [],
        }
    )
    runner._is_session_run_current = MagicMock(return_value=True)
    runner._clear_restart_failure_count = MagicMock()
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = MagicMock()
    runner._emit_gateway_run_progress = MagicMock()
    runner._read_user_config = lambda: {}
    runner._show_reasoning = False
    runner.hooks = SimpleNamespace(emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]))


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


@pytest.mark.asyncio
async def test_manual_compress_migrates_goal_to_compressed_session(hermes_home):
    db = SessionDB()
    parent_id = _sid("goal-manual-compress-parent")
    child_id = _sid("goal-manual-compress-child")
    _create_compression_child(db, parent_id, child_id)

    from hermes_cli.goals import GoalManager, load_goal

    GoalManager(parent_id).set("survive manual compress")
    source = _source()
    session_entry = _session_entry(source, parent_id)
    runner, _adapter = _make_runner(session_entry, source)
    runner._session_db = db
    history = _history()
    compressed = [history[0], history[-1]]
    _prepare_runner_for_manual_compress(runner, source, session_entry, history)
    fake_agent = _fake_compression_agent(child_id, compressed)

    with (
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=fake_agent),
        patch("agent.manual_compression_feedback.summarize_manual_compression", return_value={
            "headline": "compressed",
            "token_line": "tokens reduced",
            "note": "",
        }),
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        result = await runner._handle_compress_command(_event("/compress", source))

    child_goal = load_goal(child_id)
    old_goal = load_goal(parent_id)
    assert "compressed" in result
    assert child_goal is not None
    assert child_goal.goal == "survive manual compress"
    assert old_goal.status == "cleared"
    assert f"migrated to {child_id}" in (old_goal.paused_reason or "")
    assert session_entry.session_id == child_id
    runner.session_store.rewrite_transcript.assert_called_once_with(child_id, compressed)


@pytest.mark.asyncio
async def test_hygiene_compress_migrates_goal_before_agent_turn(hermes_home):
    db = SessionDB()
    parent_id = _sid("goal-hygiene-compress-parent")
    child_id = _sid("goal-hygiene-compress-child")
    _create_compression_child(db, parent_id, child_id)

    from hermes_cli.goals import GoalManager, load_goal

    GoalManager(parent_id).set("survive hygiene compress")
    source = _source()
    session_entry = _session_entry(source, parent_id)
    session_entry.last_prompt_tokens = 900
    runner, _adapter = _make_runner(session_entry, source)
    runner._session_db = db
    history = _history()
    compressed = [history[0], history[-1]]
    _prepare_runner_for_hygiene_message(runner, source, session_entry, history)
    fake_agent = _fake_compression_agent(child_id, compressed)

    with (
        patch("gateway.run._load_gateway_config", return_value={
            "model": {"default": "test-model", "context_length": 1000},
            "compression": {"enabled": True, "hygiene_hard_message_limit": 4},
        }),
        patch("agent.model_metadata.get_model_context_length", return_value=1000),
        patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=900),
        patch("run_agent.AIAgent", return_value=fake_agent),
    ):
        result = await runner._handle_message_with_agent(
            _event("continue after hygiene", source),
            source,
            session_entry.session_key,
            1,
        )

    child_goal = load_goal(child_id)
    old_goal = load_goal(parent_id)
    assert result == "hygiene turn complete"
    assert child_goal is not None
    assert child_goal.goal == "survive hygiene compress"
    assert old_goal.status == "cleared"
    assert f"migrated to {child_id}" in (old_goal.paused_reason or "")
    assert session_entry.session_id == child_id
    runner._run_agent.assert_awaited_once()
    assert runner._run_agent.call_args.kwargs["session_id"] == child_id
    runner.session_store.rewrite_transcript.assert_called_once_with(child_id, compressed)
