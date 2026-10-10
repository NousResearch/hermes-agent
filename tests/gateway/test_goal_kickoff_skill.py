"""Regression tests for Issue #130316:
A session opened by /goal never loads the channel's bound skill
(kickoff drops auto_skill; a non-generating command spends first-turn loading).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_busy import GatewayBusySessionMixin
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionEntry, SessionSource, SessionStore
from gateway.slash_commands_goals import GatewayGoalCommandsMixin


class _GoalTestRunner(GatewayBusySessionMixin, GatewayGoalCommandsMixin, GatewayTurnMixin):
    """Test runner combining goal commands and turn dispatch mixins."""

    def __init__(self) -> None:
        self.config = GatewayConfig()
        self.hooks = SimpleNamespace(emit=AsyncMock())
        self.enqueued: list[MessageEvent] = []

    def _adapter_and_key_for(self, event):
        return object(), "slack:C1:U1"

    def _enqueue_fifo(self, quick_key, turn, adapter):
        self.enqueued.append(turn)


def test_enqueue_goal_turn_kickoff_preserves_auto_skill():
    """A /goal kickoff turn must carry event.auto_skill into the enqueued event."""
    runner = _GoalTestRunner()
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    event = MessageEvent(
        text="/goal optimize database",
        source=source,
        message_id="msg_123",
        channel_prompt="sys prompt",
        auto_skill=["db-tuning", "metrics"],
    )

    runner._enqueue_goal_turn(event, "start prompt", label="test kickoff", kickoff=True)

    assert len(runner.enqueued) == 1
    turn = runner.enqueued[0]
    assert turn.text == "start prompt"
    assert turn.message_id == "msg_123"
    assert turn.channel_prompt == "sys prompt"
    assert turn.auto_skill == ["db-tuning", "metrics"]


def test_enqueue_goal_turn_continuation_drops_auto_skill():
    """A /goal resume or follow-up continuation must not re-inject auto_skill."""
    runner = _GoalTestRunner()
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    event = MessageEvent(
        text="/goal resume",
        source=source,
        message_id="msg_456",
        channel_prompt="sys prompt",
        auto_skill=["db-tuning"],
    )

    runner._enqueue_goal_turn(event, "continue prompt", label="test continuation", kickoff=False)

    assert len(runner.enqueued) == 1
    turn = runner.enqueued[0]
    assert turn.text == "continue prompt"
    assert turn.message_id is None
    assert turn.channel_prompt is None
    assert turn.auto_skill is None


@pytest.mark.asyncio
async def test_hmwa_open_session_new_when_touched_by_command_without_prior_turns():
    """A non-generating command (/goal, /retry, /goal status) touches the session first,
    moving updated_at. _hmwa_open_session must still treat the session as new because no agent
    turns have run (empty transcript)."""
    runner = _GoalTestRunner()
    now = datetime(2026, 10, 1, 12, 0, 0)
    later = now + timedelta(seconds=5)

    entry = SessionEntry(
        session_key="slack:C1:U1",
        session_id="sess_123",
        created_at=now,
        updated_at=later,  # updated_at advanced by /goal or /retry command
    )
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")

    # Transcript is empty — no agent turn has persisted history yet
    runner.async_session_store = SimpleNamespace(
        load_transcript=AsyncMock(return_value=[])
    )

    was_auto_reset, is_new_session = await runner._hmwa_open_session(
        entry, entry.session_key, source
    )

    assert was_auto_reset is False
    assert is_new_session is True
    runner.hooks.emit.assert_awaited_once_with("session:start", {
        "platform": "slack",
        "user_id": "U1",
        "session_id": "sess_123",
        "session_key": "slack:C1:U1",
    })


@pytest.mark.asyncio
async def test_hmwa_open_session_not_new_when_prior_turns_exist():
    """When an agent turn has already run in the session, it is no longer new."""
    runner = _GoalTestRunner()
    now = datetime(2026, 10, 1, 12, 0, 0)
    later = now + timedelta(seconds=10)

    entry = SessionEntry(
        session_key="slack:C1:U1",
        session_id="sess_123",
        created_at=now,
        updated_at=later,
    )
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")

    runner.async_session_store = SimpleNamespace(
        load_transcript=AsyncMock(return_value=[
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
        ])
    )

    was_auto_reset, is_new_session = await runner._hmwa_open_session(
        entry, entry.session_key, source
    )

    assert was_auto_reset is False
    assert is_new_session is False
    runner.hooks.emit.assert_not_called()


@pytest.mark.asyncio
async def test_hmwa_open_session_fresh_reset_forces_new():
    """is_fresh_reset (e.g. from /reset or /new) forces is_new_session=True even with history."""
    runner = _GoalTestRunner()
    now = datetime(2026, 10, 1, 12, 0, 0)
    later = now + timedelta(seconds=10)

    entry = SessionEntry(
        session_key="slack:C1:U1",
        session_id="sess_reset",
        created_at=now,
        updated_at=later,
        is_fresh_reset=True,
    )
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")

    runner.async_session_store = SimpleNamespace(
        load_transcript=AsyncMock(return_value=[{"role": "user", "content": "old"}])
    )

    was_auto_reset, is_new_session = await runner._hmwa_open_session(
        entry, entry.session_key, source
    )

    assert is_new_session is True
    assert entry.is_fresh_reset is False  # consumed
    runner.hooks.emit.assert_awaited_once()


@pytest.mark.asyncio
async def test_goal_kickoff_loads_bound_skill_on_fresh_session():
    """Verify that auto_skill is loaded on kickoff turn when session was touched by /goal."""
    runner = _GoalTestRunner()
    now = datetime(2026, 10, 1, 12, 0, 0)
    later = now + timedelta(seconds=2)

    entry = SessionEntry(
        session_key="slack:C1:U1",
        session_id="sess_goal",
        created_at=now,
        updated_at=later,  # moved by /goal command dispatch
    )
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")

    # Mock _hmwa_auto_load_skills to verify it gets called
    auto_load_mock = MagicMock()
    runner._hmwa_auto_load_skills = auto_load_mock

    # Setup runner dependencies for _hmwa_prepare_turn
    runner._set_session_env = lambda context: {}
    runner._clear_session_env = lambda tokens: None
    runner._pinned_session_context_prompt = lambda *args, **kwargs: ""
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock()
    runner._hmwa_run_session_hygiene = AsyncMock(return_value=[])
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *args: None
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="kickoff goal text")
    runner._build_inbound_turn_context = lambda *args, **kwargs: SimpleNamespace()
    runner._bind_adapter_run_generation = lambda *args: None
    runner._delivery_adapter_for = lambda source: None
    runner._set_pending_turn_sidecar_notes = lambda *args: None
    runner._hmwa_apply_message_timestamp = lambda event, text: (text, text, 1234567.8)
    runner.async_session_store = SimpleNamespace(
        load_transcript=AsyncMock(return_value=[])
    )

    kickoff_event = MessageEvent(
        text="kickoff goal text",
        source=source,
        message_id="msg_1",
        auto_skill=["incident-investigator"],
    )

    prepared, _ = await runner._hmwa_prepare_turn(
        kickoff_event, source, entry, entry.session_key, "slack:C1:U1", run_generation=1
    )

    # auto_load_skills must have been called with the bound skill
    auto_load_mock.assert_called_once_with(
        kickoff_event, ["incident-investigator"], "slack:C1:U1", "slack:C1:U1"
    )


@pytest.mark.asyncio
async def test_goal_continuation_does_not_reload_skill():
    """Verify that a subsequent continuation turn does not re-invoke auto_load_skills."""
    runner = _GoalTestRunner()
    now = datetime(2026, 10, 1, 12, 0, 0)
    later = now + timedelta(seconds=10)

    entry = SessionEntry(
        session_key="slack:C1:U1",
        session_id="sess_goal",
        created_at=now,
        updated_at=later,
    )
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")

    auto_load_mock = MagicMock()
    runner._hmwa_auto_load_skills = auto_load_mock

    runner._set_session_env = lambda context: {}
    runner._clear_session_env = lambda tokens: None
    runner._pinned_session_context_prompt = lambda *args, **kwargs: ""
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock()
    runner._hmwa_run_session_hygiene = AsyncMock(return_value=[])
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *args: None
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="continuation text")
    runner._build_inbound_turn_context = lambda *args, **kwargs: SimpleNamespace()
    runner._bind_adapter_run_generation = lambda *args: None
    runner._delivery_adapter_for = lambda source: None
    runner._set_pending_turn_sidecar_notes = lambda *args: None
    runner._hmwa_apply_message_timestamp = lambda event, text: (text, text, 1234567.8)
    runner.async_session_store = SimpleNamespace(
        load_transcript=AsyncMock(return_value=[
            {"role": "user", "content": "turn 1 user"},
            {"role": "assistant", "content": "turn 1 assistant"},
        ])
    )

    continuation_event = MessageEvent(
        text="continuation prompt",
        source=source,
        message_id=None,
        auto_skill=None,
    )

    prepared, _ = await runner._hmwa_prepare_turn(
        continuation_event, source, entry, entry.session_key, "slack:C1:U1", run_generation=2
    )

    auto_load_mock.assert_not_called()


@pytest.mark.asyncio
async def test_hmwa_open_session_not_new_when_emptied_by_rewind():
    """/undo or /retry rewinding the oldest turn leaves an empty transcript, but the
    session already ran turns (token counters survive truncation): it must NOT take
    the new-session branch again — no second session:start, no skill re-prepend (#130381)."""
    runner = _GoalTestRunner()
    now = datetime(2026, 10, 1, 12, 0, 0)
    later = now + timedelta(minutes=30)

    entry = SessionEntry(
        session_key="slack:C1:U1",
        session_id="sess_rewound",
        created_at=now,
        updated_at=later,  # touched by earlier user activity
        input_tokens=1200,  # turns ran before the rewind emptied the transcript
        output_tokens=340,
    )
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")

    runner.async_session_store = SimpleNamespace(
        load_transcript=AsyncMock(return_value=[])
    )

    was_auto_reset, is_new_session = await runner._hmwa_open_session(
        entry, entry.session_key, source
    )

    assert was_auto_reset is False
    assert is_new_session is False
    runner.hooks.emit.assert_not_called()


@pytest.mark.asyncio
async def test_retry_replay_preserves_auto_skill(tmp_path, monkeypatch):
    """/retry replays the last user message as a new turn: it must forward
    event.auto_skill like /queue and /goal kickoff do (#130381 P2)."""
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    session_id = "retry_skill_session"
    store._db.create_session(session_id=session_id, source="test")
    for msg in [
        {"role": "user", "content": "orig question"},
        {"role": "assistant", "content": "orig answer"},
    ]:
        store.append_to_transcript(session_id, msg)

    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = store
    session_entry = MagicMock(session_id=session_id)
    session_entry.last_prompt_tokens = 0
    gw.session_store.get_or_create_session = MagicMock(return_value=session_entry)

    seen = {}

    async def fake_handle_message(event):
        seen["auto_skill"] = event.auto_skill
        return "ok"

    gw._handle_message = AsyncMock(side_effect=fake_handle_message)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    event = MessageEvent(
        text="/retry", message_type=MessageType.TEXT, source=source,
        auto_skill=["incident-investigator"],
    )

    assert await gw._handle_retry_command(event) == "ok"
    assert seen["auto_skill"] == ["incident-investigator"]


@pytest.mark.asyncio
async def test_steer_fallback_preserves_auto_skill():
    """/steer queued as a follow-up turn (no running agent) must forward
    event.auto_skill like /queue does (#130381 P2)."""
    runner = _GoalTestRunner()
    runner._peek_session_state = lambda quick_key: None
    runner._delivery_adapter_for = lambda source: object()
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    event = MessageEvent(
        text="/steer focus on latency", message_type=MessageType.TEXT, source=source,
        auto_skill=["incident-investigator"],
    )

    await runner._busy_steer_command(event, "slack:C1:U1", source)

    assert len(runner.enqueued) == 1
    assert runner.enqueued[0].auto_skill == ["incident-investigator"]
