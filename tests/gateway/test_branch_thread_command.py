"""Tests for /branch defaulting to a new thread on Discord/Telegram/Slack."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.slash_commands import (
    _branch_dest_source,
    _branch_thread_parent_id,
    _parse_branch_command_args,
)
from hermes_state import AsyncSessionDB


def test_parse_branch_command_args_plain_name():
    assert _parse_branch_command_args("refactor approach") == (False, "refactor approach")
    assert _parse_branch_command_args("") == (False, "")


def test_parse_branch_command_args_here_flags():
    assert _parse_branch_command_args("--here") == (True, "")
    assert _parse_branch_command_args("--here alt path") == (True, "alt path")
    assert _parse_branch_command_args("here alt path") == (True, "alt path")
    assert _parse_branch_command_args("HERE name") == (True, "name")


def test_branch_thread_parent_id_resolution():
    channel = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan-1",
        chat_type="group",
        user_id="u1",
    )
    assert _branch_thread_parent_id(channel) == "chan-1"

    in_thread = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thr-9",
        chat_type="thread",
        user_id="u1",
        thread_id="thr-9",
        parent_chat_id="chan-1",
    )
    assert _branch_thread_parent_id(in_thread) == "chan-1"

    orphan = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thr-9",
        chat_type="thread",
        user_id="u1",
        thread_id="thr-9",
    )
    assert _branch_thread_parent_id(orphan) is None


def test_branch_dest_source_matches_discord_inbound_shape():
    origin = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan-1",
        chat_type="group",
        user_id="u1",
        user_name="tester",
        scope_id="guild-1",
    )
    dest = _branch_dest_source(
        origin,
        parent_chat_id="chan-1",
        new_thread_id="thr-new",
        title="branch title",
    )
    assert dest.chat_id == "thr-new"
    assert dest.thread_id == "thr-new"
    assert dest.chat_type == "thread"
    assert dest.parent_chat_id == "chan-1"
    assert build_session_key(dest) == "agent:main:discord:thread:thr-new:thr-new"


def test_branch_dest_source_slack_keeps_parent_chat():
    origin = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="group",
        user_id="U1",
    )
    dest = _branch_dest_source(
        origin,
        parent_chat_id="C123",
        new_thread_id="1710000.0001",
        title="t",
    )
    assert dest.chat_id == "C123"
    assert dest.thread_id == "1710000.0001"
    assert dest.chat_type == "group"


def _make_discord_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="chan-1",
        user_name="tester",
        chat_type="group",
        scope_id="guild-1",
    )


def _make_entry(session_id: str, source: SessionSource) -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=source.platform,
        chat_type=source.chat_type,
    )


def _make_branch_thread_runner():
    from gateway.run import GatewayRunner

    source = _make_discord_source()
    current_entry = _make_entry("current-session", source)
    dest_source = _branch_dest_source(
        source,
        parent_chat_id="chan-1",
        new_thread_id="thr-new",
        title="alt path",
    )
    dest_entry = _make_entry("branched-session", dest_source)

    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value="thr-new")
    adapter._threads = MagicMock()

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = current_entry
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    runner.session_store.switch_session.return_value = dest_entry
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.get_session_title.return_value = "Current Work"
    runner._session_db._db.get_next_title_in_lineage.return_value = "Current Work #2"
    return runner, source, adapter


@pytest.mark.asyncio
async def test_branch_default_on_discord_opens_new_thread():
    runner, source, adapter = _make_branch_thread_runner()
    origin_key = build_session_key(source)

    event = MessageEvent(text="/branch alt path", source=source, message_id="m1")
    result = await runner._handle_branch_command(event)

    assert "new thread" in result
    assert "alt path" in result
    assert "<#thr-new>" in result

    adapter.create_handoff_thread.assert_awaited_once_with("chan-1", "alt path")
    adapter._threads.mark.assert_called_once_with("thr-new")

    switch_calls = runner.session_store.switch_session.call_args_list
    assert len(switch_calls) == 1
    dest_key, new_sid = switch_calls[0].args
    assert dest_key != origin_key
    assert dest_key == "agent:main:discord:thread:thr-new:thr-new"
    assert new_sid != "current-session"
    assert "_" in new_sid


@pytest.mark.asyncio
async def test_branch_here_stays_on_origin_surface():
    runner, source, adapter = _make_branch_thread_runner()
    origin_key = build_session_key(source)
    branched = _make_entry("branched-session", source)
    runner.session_store.switch_session.return_value = branched

    event = MessageEvent(text="/branch --here alt path", source=source, message_id="m1")
    result = await runner._handle_branch_command(event)

    assert "Branched to" in result
    assert "new thread" not in result
    assert "alt path" in result
    adapter.create_handoff_thread.assert_not_awaited()
    switch_calls = runner.session_store.switch_session.call_args_list
    assert len(switch_calls) == 1
    assert switch_calls[0].args[0] == origin_key


@pytest.mark.asyncio
async def test_branch_on_whatsapp_stays_in_place():
    """Non-thread platforms always branch on the current surface."""
    from gateway.run import GatewayRunner

    source = SessionSource(
        platform=Platform.WHATSAPP,
        user_id="u1",
        chat_id="15551234567",
        chat_type="dm",
    )
    current_entry = _make_entry("current-session", source)
    branched = _make_entry("branched-session", source)
    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value=None)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.config = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = current_entry
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "hello"},
    ]
    runner.session_store.switch_session.return_value = branched
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.get_session_title.return_value = None
    runner._session_db._db.get_next_title_in_lineage.return_value = "branch #2"

    event = MessageEvent(text="/branch", source=source, message_id="m1")
    result = await runner._handle_branch_command(event)

    assert "Branched to" in result
    assert "new thread" not in result
    adapter.create_handoff_thread.assert_not_awaited()
    runner.session_store.switch_session.assert_called_once()
    assert runner.session_store.switch_session.call_args.args[0] == build_session_key(source)


@pytest.mark.asyncio
async def test_branch_missing_adapter_falls_back_in_place():
    runner, source, adapter = _make_branch_thread_runner()
    runner.adapters = {}
    origin_key = build_session_key(source)
    branched = _make_entry("branched-session", source)
    runner.session_store.switch_session.return_value = branched

    event = MessageEvent(text="/branch", source=source, message_id="m1")
    result = await runner._handle_branch_command(event)

    assert "Branched to" in result
    assert "new thread" not in result
    assert runner.session_store.switch_session.call_args.args[0] == origin_key
