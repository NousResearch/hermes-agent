"""Matrix Project A / Project B context-isolation regressions."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from plugins.platforms.matrix.adapter import MatrixRelation, _parse_relates_to
from hermes_state import AsyncSessionDB
from gateway.session import (
    SessionContext,
    SessionEntry,
    SessionSource,
    build_session_context_prompt,
    build_session_key,
)

PROJECT_A_ROOM_ID = "!projectA:example.org"
PROJECT_B_ROOM_ID = "!projectB:example.org"
PROJECT_A_NAME = "Project - Project A"
PROJECT_B_NAME = "Project - Project B"
PROJECT_A_TOPIC = "Architecture and deploy plan for Project A"
PROJECT_B_TOPIC = "Migration and branch plan for Project B"
PROJECT_A_ALIAS = "#project-a:example.org"
PROJECT_B_ALIAS = "#project-b:example.org"
SENDER = "@alice:example.org"


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"},
        )
    )
    adapter._user_id = "@bot:example.org"
    adapter._require_mention = False
    adapter._auto_thread = False
    adapter._matrix_session_scope = "room"
    adapter._text_batch_delay_seconds = 0
    adapter._background_read_receipt = MagicMock()
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._client = _FakeMatrixClient()
    return adapter


class _FakeMatrixClient:
    def __init__(self):
        self.state_store = MagicMock()
        self.state_store.get_members = AsyncMock(return_value=["@bot:example.org", SENDER])

    async def get_state_event(self, room_id, event_type):
        rid = str(room_id)
        state = {
            PROJECT_A_ROOM_ID: {
                "m.room.name": {"content": {"name": PROJECT_A_NAME}},
                "m.room.topic": {"content": {"topic": PROJECT_A_TOPIC}},
                "m.room.canonical_alias": {"content": {"alias": PROJECT_A_ALIAS}},
            },
            PROJECT_B_ROOM_ID: {
                "m.room.name": {"content": {"name": PROJECT_B_NAME}},
                "m.room.topic": {"content": {"topic": PROJECT_B_TOPIC}},
                "m.room.canonical_alias": {"content": {"alias": PROJECT_B_ALIAS}},
            },
        }
        value = state.get(rid, {}).get(str(event_type))
        if value is None:
            raise KeyError((rid, event_type))
        return value


async def _source_for(adapter, room_id: str, event_id: str = "$event"):
    ctx = await adapter._resolve_message_context(
        room_id=room_id,
        sender=SENDER,
        event_id=event_id,
        body="What is next?",
        source_content={"body": "What is next?"},
        relation=MatrixRelation(),
    )
    assert ctx is not None
    return ctx[-1]


def _matrix_event(room_id: str, event_id: str, body: str = "What is next?"):
    event = MagicMock()
    event.room_id = room_id
    event.sender = SENDER
    event.event_id = event_id
    event.timestamp = int(time.time() * 1000)
    event.server_timestamp = event.timestamp
    event.content = {"msgtype": "m.text", "body": body}
    return event


def _context_for(source: SessionSource) -> SessionContext:
    return SessionContext(
        source=source,
        connected_platforms=[Platform.MATRIX],
        home_channels={},
        session_key=build_session_key(source),
        session_id="session-test",
    )


@pytest.mark.asyncio
async def test_matrix_source_includes_room_name_topic_and_message_id():
    adapter = _make_adapter()
    source = await _source_for(adapter, PROJECT_B_ROOM_ID, "$project-b-msg")

    assert source.chat_id == PROJECT_B_ROOM_ID
    assert source.chat_name == PROJECT_B_NAME
    assert source.chat_topic == PROJECT_B_TOPIC
    assert source.guild_id == "example.org"
    assert source.message_id == "$project-b-msg"
    assert source.parent_chat_id is None


@pytest.mark.asyncio
async def test_matrix_project_a_and_project_b_have_distinct_session_keys():
    adapter = _make_adapter()
    source_a = await _source_for(adapter, PROJECT_A_ROOM_ID, "$a")
    source_b = await _source_for(adapter, PROJECT_B_ROOM_ID, "$b")

    assert source_a.chat_id != source_b.chat_id
    assert source_a.chat_name == PROJECT_A_NAME
    assert source_b.chat_name == PROJECT_B_NAME
    assert build_session_key(source_a) != build_session_key(source_b)


@pytest.mark.asyncio
async def test_matrix_project_b_prompt_contains_project_b_not_project_a():
    adapter = _make_adapter()
    source_b = await _source_for(adapter, PROJECT_B_ROOM_ID, "$b")

    prompt = build_session_context_prompt(_context_for(source_b))

    assert PROJECT_B_NAME in prompt
    assert PROJECT_B_TOPIC in prompt
    assert PROJECT_B_ROOM_ID in prompt
    assert "Matrix room boundary" in prompt
    assert PROJECT_A_NAME not in prompt
    assert PROJECT_A_TOPIC not in prompt


@pytest.mark.asyncio
async def test_matrix_project_context_survives_sequential_messages():
    adapter = _make_adapter()
    adapter._matrix_session_scope = "room"
    first = await _source_for(adapter, PROJECT_B_ROOM_ID, "$b1")
    second = await _source_for(adapter, PROJECT_B_ROOM_ID, "$b2")

    assert first.thread_id is None
    assert second.thread_id is None
    assert first.chat_name == PROJECT_B_NAME
    assert second.chat_name == PROJECT_B_NAME
    assert build_session_key(first) == build_session_key(second)


@pytest.mark.asyncio
async def test_matrix_session_scope_auto_and_thread_preserve_synthetic_threads():
    adapter = _make_adapter()
    # Override member_count to 3 so the named project room is NOT classified as
    # a DM (the DM fix uses member_count <= 2 as the primary DM signal).
    adapter._get_room_member_count = AsyncMock(return_value=3)
    adapter._auto_thread = True
    adapter._matrix_session_scope = "auto"
    auto_source = await _source_for(adapter, PROJECT_B_ROOM_ID, "$auto")
    assert auto_source.thread_id == "$auto"

    adapter._matrix_session_scope = "thread"
    thread_source = await _source_for(adapter, PROJECT_B_ROOM_ID, "$thread")
    assert thread_source.thread_id == "$thread"

    real_thread = await adapter._resolve_message_context(
        room_id=PROJECT_B_ROOM_ID,
        sender=SENDER,
        event_id="$reply",
        body="thread reply",
        source_content={"body": "thread reply"},
        relation=_parse_relates_to(
            {"rel_type": "m.thread", "event_id": "$root"}
        ),
    )
    assert real_thread is not None
    assert real_thread[-1].thread_id == "$root"


@pytest.mark.asyncio
async def test_matrix_project_context_survives_concurrent_messages():
    from gateway.run import GatewayRunner
    from gateway.session_context import get_session_env

    async def observe(room_id: str):
        adapter = _make_adapter()
        source = await _source_for(adapter, room_id, f"${room_id}")
        context = _context_for(source)
        runner = object.__new__(GatewayRunner)
        tokens = runner._set_session_env(context)
        try:
            await asyncio.sleep(0)
            return SimpleNamespace(
                chat_id=get_session_env("HERMES_SESSION_CHAT_ID"),
                chat_name=get_session_env("HERMES_SESSION_CHAT_NAME"),
                session_key=get_session_env("HERMES_SESSION_KEY"),
            )
        finally:
            runner._clear_session_env(tokens)

    observed_a, observed_b = await asyncio.gather(
        observe(PROJECT_A_ROOM_ID),
        observe(PROJECT_B_ROOM_ID),
    )

    assert observed_a.chat_id == PROJECT_A_ROOM_ID
    assert observed_b.chat_id == PROJECT_B_ROOM_ID
    assert observed_a.chat_name == PROJECT_A_NAME
    assert observed_b.chat_name == PROJECT_B_NAME
    assert observed_a.session_key != observed_b.session_key


@pytest.mark.asyncio
async def test_matrix_inbound_handler_emits_project_b_metadata_not_project_a():
    adapter = _make_adapter()
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture

    await adapter._on_room_message(_matrix_event(PROJECT_B_ROOM_ID, "$project-b"))

    assert len(captured) == 1
    source = captured[0].source
    assert source.chat_id == PROJECT_B_ROOM_ID
    assert source.chat_name == PROJECT_B_NAME
    assert source.chat_topic == PROJECT_B_TOPIC
    assert source.message_id == "$project-b"
    assert PROJECT_A_NAME not in repr(source.to_dict())


@pytest.mark.asyncio
async def test_matrix_inbound_handler_keeps_project_a_and_b_distinct():
    adapter = _make_adapter()
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture

    await adapter._on_room_message(_matrix_event(PROJECT_A_ROOM_ID, "$project-a", "A"))
    await adapter._on_room_message(_matrix_event(PROJECT_B_ROOM_ID, "$project-b", "B"))

    assert [event.source.chat_id for event in captured] == [
        PROJECT_A_ROOM_ID,
        PROJECT_B_ROOM_ID,
    ]
    assert [event.source.chat_name for event in captured] == [
        PROJECT_A_NAME,
        PROJECT_B_NAME,
    ]
    assert build_session_key(captured[0].source) != build_session_key(captured[1].source)


def test_matrix_room_scope_group_sessions_per_user_true_separates_users():
    alice = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    bob = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    bob.user_id = "@bob:example.org"
    alice.thread_id = None
    bob.thread_id = None

    assert build_session_key(alice, group_sessions_per_user=True) != build_session_key(
        bob,
        group_sessions_per_user=True,
    )


def test_matrix_room_scope_group_sessions_per_user_false_shares_room():
    alice = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    bob = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    bob.user_id = "@bob:example.org"
    alice.thread_id = None
    bob.thread_id = None

    assert build_session_key(alice, group_sessions_per_user=False) == build_session_key(
        bob,
        group_sessions_per_user=False,
    )


def _make_matrix_source(room_id: str, room_name: str, topic: str) -> SessionSource:
    return SessionSource(
        platform=Platform.MATRIX,
        chat_id=room_id,
        chat_name=room_name,
        chat_type="group",
        user_id=SENDER,
        user_name="Alice",
        chat_topic=topic,
    )


def _entry(source: SessionSource, session_id: str, title: str | None = None) -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        display_name=title or source.chat_name,
        platform=Platform.MATRIX,
        chat_type="group",
    )


def _make_runner(current_source: SessionSource, entries: list[SessionEntry]):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.MATRIX: PlatformConfig(enabled=True)})
    adapter = MagicMock()
    adapter._matrix_session_scope = "room"
    runner.adapters = {Platform.MATRIX: adapter}
    runner.session_store = MagicMock()
    runner.session_store._entries = {entry.session_key: entry for entry in entries}
    current = next((e for e in entries if e.origin and e.origin.chat_id == current_source.chat_id), entries[0])
    runner.session_store.get_or_create_session.return_value = current
    runner.session_store.switch_session.return_value = current
    runner.session_store.load_transcript.return_value = [{"role": "user", "content": "hello"}]
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._release_running_agent_state = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._queue_depth = MagicMock(return_value=0)
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.list_sessions_rich.return_value = [
        {"id": entry.session_id, "title": entry.display_name, "preview": ""}
        for entry in entries
    ]
    runner._session_db._db.resolve_resume_session_id.side_effect = lambda sid: sid
    runner._session_db._db.get_session_title.side_effect = lambda sid: {
        entry.session_id: entry.display_name for entry in entries
    }.get(sid)
    runner._session_db._db.get_session.return_value = None
    return runner


def _event(text: str, source: SessionSource) -> MessageEvent:
    return MessageEvent(text=text, source=source, message_id="$cmd")


@pytest.mark.asyncio
async def test_matrix_status_reports_current_matrix_room_scope():
    source_a = _make_matrix_source(PROJECT_A_ROOM_ID, PROJECT_A_NAME, PROJECT_A_TOPIC)
    source_b = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    entry_b = _entry(source_b, "session-b", "Project B Plan")
    runner = _make_runner(source_b, [_entry(source_a, "session-a", "Project A Plan"), entry_b])

    result = await runner._handle_status_command(_event("/status", source_b))

    assert "Matrix scope:" in result
    assert PROJECT_B_NAME in result
    assert PROJECT_B_ROOM_ID in result
    assert "session_scope: room" in result
    session_key = build_session_key(source_b)
    assert session_key not in result
    assert session_key[:8] not in result
    assert "session_key: sha256:" in result
    assert PROJECT_A_NAME not in result
    assert PROJECT_A_ROOM_ID not in result


@pytest.mark.asyncio
async def test_matrix_resume_does_not_cross_rooms_by_default():
    source_a = _make_matrix_source(PROJECT_A_ROOM_ID, PROJECT_A_NAME, PROJECT_A_TOPIC)
    source_b = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    entry_a = _entry(source_a, "session-a", "Project A Plan")
    entry_b = _entry(source_b, "session-b", "Project B Plan")
    runner = _make_runner(source_b, [entry_a, entry_b])
    runner._session_db._db.resolve_session_by_title.return_value = "session-a"

    result = await runner._handle_resume_command(_event("/resume Project A Plan", source_b))

    assert "blocked" in result
    assert PROJECT_A_NAME in result
    runner.session_store.switch_session.assert_not_called()


@pytest.mark.asyncio
async def test_matrix_resume_allows_same_room_session():
    source_b = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    entry_b = _entry(source_b, "session-b-old", "Project B Plan")
    runner = _make_runner(source_b, [entry_b])
    runner.session_store.get_or_create_session.return_value = _entry(
        source_b, "session-b-current", "Current Project B"
    )
    runner.session_store.switch_session.return_value = entry_b
    runner._session_db._db.resolve_session_by_title.return_value = "session-b-old"

    result = await runner._handle_resume_command(_event("/resume Project B Plan", source_b))

    assert "Resumed session" in result
    runner.session_store.switch_session.assert_called_once()


@pytest.mark.asyncio
async def test_matrix_resume_quoted_title_same_room():
    source_b = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    entry_b = _entry(source_b, "session-b-old", "Project B Plan")
    runner = _make_runner(source_b, [entry_b])
    runner.session_store.get_or_create_session.return_value = _entry(
        source_b, "session-b-current", "Current Project B"
    )
    runner.session_store.switch_session.return_value = entry_b
    runner._session_db._db.resolve_session_by_title.return_value = "session-b-old"

    result = await runner._handle_resume_command(
        _event('/resume "Project B Plan"', source_b)
    )

    assert "Resumed session" in result
    runner._session_db._db.resolve_session_by_title.assert_called_once_with("Project B Plan")


@pytest.mark.asyncio
async def test_matrix_resume_quoted_title_cross_room_blocked():
    source_a = _make_matrix_source(PROJECT_A_ROOM_ID, PROJECT_A_NAME, PROJECT_A_TOPIC)
    source_b = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    entry_a = _entry(source_a, "session-a", "Project A Plan")
    entry_b = _entry(source_b, "session-b", "Project B Plan")
    runner = _make_runner(source_b, [entry_a, entry_b])
    runner._session_db._db.resolve_session_by_title.return_value = "session-a"

    result = await runner._handle_resume_command(
        _event('/resume "Project A Plan"', source_b)
    )

    assert "blocked" in result
    runner.session_store.switch_session.assert_not_called()


@pytest.mark.asyncio
async def test_matrix_resume_malformed_quote_returns_helpful_error():
    source_b = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    runner = _make_runner(source_b, [_entry(source_b, "session-b", "Project B Plan")])

    result = await runner._handle_resume_command(
        _event('/resume "Project B Plan', source_b)
    )

    assert "Could not parse" in result
    assert "quotes" in result


@pytest.mark.asyncio
async def test_matrix_resume_cross_room_requires_explicit_flag_and_warns():
    source_a = _make_matrix_source(PROJECT_A_ROOM_ID, PROJECT_A_NAME, PROJECT_A_TOPIC)
    source_b = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    entry_a = _entry(source_a, "session-a", "Project A Plan")
    entry_b = _entry(source_b, "session-b", "Project B Plan")
    runner = _make_runner(source_b, [entry_a, entry_b])
    runner.session_store.switch_session.return_value = entry_a
    runner._session_db._db.resolve_session_by_title.return_value = "session-a"

    result = await runner._handle_resume_command(
        _event("/resume --cross-room Project A Plan", source_b)
    )

    assert "Cross-room resume" in result
    assert PROJECT_B_NAME in result
    runner.session_store.switch_session.assert_called_once()


@pytest.mark.asyncio
async def test_matrix_resume_lists_only_current_room_by_default():
    source_a = _make_matrix_source(PROJECT_A_ROOM_ID, PROJECT_A_NAME, PROJECT_A_TOPIC)
    source_b = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    runner = _make_runner(
        source_b,
        [_entry(source_a, "session-a", "Project A Plan"), _entry(source_b, "session-b", "Project B Plan")],
    )

    result = await runner._handle_resume_command(_event("/resume", source_b))

    assert "Project B Plan" in result
    assert "Project A Plan" not in result


@pytest.mark.asyncio
async def test_matrix_resume_all_lists_room_names():
    source_a = _make_matrix_source(PROJECT_A_ROOM_ID, PROJECT_A_NAME, PROJECT_A_TOPIC)
    source_b = _make_matrix_source(PROJECT_B_ROOM_ID, PROJECT_B_NAME, PROJECT_B_TOPIC)
    runner = _make_runner(
        source_b,
        [_entry(source_a, "session-a", "Project A Plan"), _entry(source_b, "session-b", "Project B Plan")],
    )
    # Cross-room `/resume --all` listing is admin-gated (IDOR scoping), so this
    # cross-room listing test must run as a configured admin.
    runner._resume_caller_is_admin = lambda _src: True

    result = await runner._handle_resume_command(_event("/resume --all", source_b))

    assert "Project A Plan" in result
    assert PROJECT_A_NAME in result
    assert "Project B Plan" in result
