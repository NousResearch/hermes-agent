"""Gateway visible-thread fork behavior and performance contracts."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import AsyncSessionStore, SessionEntry, SessionSource, build_session_key
from hermes_state import AsyncSessionDB, SessionDB


class FakeSessionStore:
    def __init__(self, source: SessionSource, session_id: str, history: list[dict]):
        key = build_session_key(source)
        self.entries = {
            key: SessionEntry(
                session_key=key,
                session_id=session_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                origin=source,
                display_name=source.chat_name,
                platform=source.platform,
                chat_type=source.chat_type,
            )
        }
        self.histories = {session_id: list(history)}
        self.bound: list[tuple[str, str]] = []

    def get_or_create_session(self, source: SessionSource) -> SessionEntry:
        return self.entries[build_session_key(source)]

    def load_transcript(self, session_id: str) -> list[dict]:
        return list(self.histories.get(session_id, []))

    def bind_session(
        self,
        source: SessionSource,
        target_session_id: str,
        display_name: str | None = None,
    ) -> SessionEntry:
        key = build_session_key(source)
        entry = SessionEntry(
            session_key=key,
            session_id=target_session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            origin=source,
            display_name=display_name or source.chat_name,
            platform=source.platform,
            chat_type=source.chat_type,
        )
        self.entries[key] = entry
        self.bound.append((key, target_session_id))
        return entry


@pytest.fixture
def current_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="source-thread",
        chat_name="Home Server / work / Source Lane",
        chat_type="thread",
        user_id="user-1",
        user_name="User One",
        thread_id="source-thread",
        scope_id="guild-1",
        parent_chat_id="parent-forum",
        role_authorized=True,
        profile="work",
        chat_topic="Durable work surface.",
    )


@pytest.fixture
def visible_fork_runner(tmp_path, monkeypatch, current_source):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    history = [
        {"role": "user", "content": "Remember VF-CONTINUITY."},
        {
            "role": "assistant",
            "content": "VF-CONTINUITY stored.",
            "finish_reason": "stop",
            "reasoning": "kept",
            "reasoning_details": [{"type": "summary", "text": "kept"}],
        },
    ]
    db.create_session(
        "source-session",
        source="discord",
        user_id=current_source.user_id,
        chat_id=current_source.chat_id,
        chat_type=current_source.chat_type,
        thread_id=current_source.thread_id,
    )
    db.replace_messages("source-session", history)
    db.set_session_title("source-session", "Source Lane")
    db.replace_messages = MagicMock(wraps=db.replace_messages)
    db.append_message = MagicMock(wraps=db.append_message)

    from gateway.run import GatewayRunner

    runner: Any = object.__new__(GatewayRunner)
    runner.config = {}
    adapter = MagicMock()
    adapter.create_visible_fork_thread = AsyncMock(return_value="fork-thread")
    adapter.delete_visible_fork_thread = AsyncMock(return_value=True)
    adapter.send = AsyncMock()
    runner.adapters = {Platform.DISCORD: adapter}
    store = FakeSessionStore(current_source, "source-session", history)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(cast(Any, store))
    runner._session_db = AsyncSessionDB(db)
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()
    yield runner, adapter, store, db
    db.close()


@pytest.mark.asyncio
async def test_visible_fork_preserves_parent_and_binds_child(
    visible_fork_runner, current_source
):
    runner, adapter, store, db = visible_fork_runner

    response = await runner._handle_visible_fork_command(
        MessageEvent(text="/fork Fork Lane", source=current_source, message_id="m1")
    )

    adapter.create_visible_fork_thread.assert_awaited_once_with("parent-forum", "Fork Lane")
    assert "Fork Lane" in response
    assert "fork-thread" in response

    children = [
        row
        for row in db.list_sessions_rich(limit=20)
        if row["parent_session_id"] == "source-session"
    ]
    assert len(children) == 1
    child_id = children[0]["id"]
    assert db.get_session_title(child_id) == "Fork Lane"
    copied = db.get_messages_as_conversation(child_id)
    for message in copied:
        message.pop("timestamp", None)
    assert copied == [
        {"role": "user", "content": "Remember VF-CONTINUITY."},
        {
            "role": "assistant",
            "content": "VF-CONTINUITY stored.",
            "finish_reason": "stop",
            "reasoning": "kept",
            "reasoning_details": [{"type": "summary", "text": "kept"}],
        },
    ]

    child_key = build_session_key(
        SessionSource(
            platform=Platform.DISCORD,
            chat_id="fork-thread",
            chat_name="Fork Lane",
            chat_type="thread",
            user_id="user-1",
            thread_id="fork-thread",
            scope_id="guild-1",
            parent_chat_id="parent-forum",
            role_authorized=True,
            profile="work",
        )
    )
    assert (child_key, child_id) in store.bound
    bound_source = store.entries[child_key].origin
    assert bound_source.scope_id == "guild-1"
    assert bound_source.profile == "work"
    assert bound_source.role_authorized is True
    assert store.entries[child_key].session_id == child_id
    assert store.entries[build_session_key(current_source)].session_id == "source-session"
    assert db.get_session("source-session")["ended_at"] is None

    runner.hooks.emit.assert_awaited_once()
    event_type, context = runner.hooks.emit.await_args.args
    assert event_type == "session:fork"
    assert context == {
        "platform": "discord",
        "user_id": "user-1",
        "parent_session_id": "source-session",
        "session_id": child_id,
        "session_key": child_key,
        "source_chat_id": "source-thread",
        "source_thread_id": "source-thread",
        "thread_id": "fork-thread",
        "parent_chat_id": "parent-forum",
        "title": "Fork Lane",
    }


@pytest.mark.asyncio
async def test_visible_fork_copies_history_in_one_batch_offload(
    visible_fork_runner, current_source
):
    runner, _adapter, store, db = visible_fork_runner
    store.histories["source-session"] = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message-{i}"}
        for i in range(200)
    ]

    await runner._handle_visible_fork_command(
        MessageEvent(text="/fork Fast Fork", source=current_source, message_id="m1")
    )

    # Structural performance contract: message count does not multiply DB
    # thread offloads/transactions. One replace_messages call copies all rows.
    db.replace_messages.assert_called_once()
    db.append_message.assert_not_called()
    copied_messages = db.replace_messages.call_args.args[1]
    assert len(copied_messages) == 200


@pytest.mark.asyncio
async def test_visible_fork_rejects_non_discord_platforms(
    visible_fork_runner, current_source
):
    runner, adapter, _store, _db = visible_fork_runner
    current_source.platform = Platform.TELEGRAM

    response = await runner._handle_visible_fork_command(
        MessageEvent(text="/fork Other", source=current_source, message_id="m1")
    )

    assert "only from a Discord forum thread" in response
    adapter.create_visible_fork_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_visible_fork_prepares_db_before_creating_external_thread(
    visible_fork_runner, current_source
):
    runner, adapter, _store, db = visible_fork_runner
    db.replace_messages.side_effect = RuntimeError("copy failed")

    response = await runner._handle_visible_fork_command(
        MessageEvent(text="/fork Broken Copy", source=current_source, message_id="m1")
    )

    assert "could not create child session" in response
    adapter.create_visible_fork_thread.assert_not_awaited()
    children = [
        row
        for row in db.list_sessions_rich(limit=20)
        if row["parent_session_id"] == "source-session"
    ]
    assert len(children) == 1
    assert children[0]["ended_at"] is not None


@pytest.mark.asyncio
async def test_visible_fork_ends_prepared_child_when_forum_creation_fails(
    visible_fork_runner, current_source
):
    runner, adapter, _store, db = visible_fork_runner
    adapter.create_visible_fork_thread.side_effect = RuntimeError(
        "Discord /fork currently requires a forum thread"
    )

    response = await runner._handle_visible_fork_command(
        MessageEvent(text="/fork Not Forum", source=current_source, message_id="m1")
    )

    assert "requires a forum thread" in response
    children = [
        row
        for row in db.list_sessions_rich(limit=20)
        if row["parent_session_id"] == "source-session"
    ]
    assert len(children) == 1
    assert children[0]["ended_at"] is not None
    adapter.delete_visible_fork_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_visible_fork_cleans_external_thread_when_bind_fails(
    visible_fork_runner, current_source
):
    runner, adapter, store, db = visible_fork_runner
    store.bind_session = MagicMock(side_effect=RuntimeError("bind failed"))

    response = await runner._handle_visible_fork_command(
        MessageEvent(text="/fork Broken Bind", source=current_source, message_id="m1")
    )

    adapter.delete_visible_fork_thread.assert_awaited_once_with("fork-thread")
    assert "unbound forum thread was removed" in response
    children = [
        row
        for row in db.list_sessions_rich(limit=20)
        if row["parent_session_id"] == "source-session"
    ]
    assert len(children) == 1
    assert children[0]["ended_at"] is not None


@pytest.mark.asyncio
async def test_visible_fork_default_title_and_parent_fallback(
    visible_fork_runner, current_source
):
    runner, adapter, _store, _db = visible_fork_runner
    current_source.parent_chat_id = None
    raw_channel = SimpleNamespace(parent_id="parent-forum")
    event = MessageEvent(
        text="/fork",
        source=current_source,
        message_id="m1",
        raw_message=SimpleNamespace(channel=raw_channel),
    )

    response = await runner._handle_visible_fork_command(event)

    adapter.create_visible_fork_thread.assert_awaited_once_with(
        "parent-forum", "Source Lane Fork"
    )
    assert "Source Lane Fork" in response


@pytest.mark.asyncio
async def test_visible_fork_requires_existing_thread(visible_fork_runner, current_source):
    runner, adapter, _store, db = visible_fork_runner
    current_source.thread_id = None

    response = await runner._handle_visible_fork_command(
        MessageEvent(text="/fork Fork Lane", source=current_source, message_id="m1")
    )

    assert "existing Discord forum thread" in response
    adapter.create_visible_fork_thread.assert_not_awaited()
    assert not [
        row
        for row in db.list_sessions_rich(limit=20)
        if row["parent_session_id"] == "source-session"
    ]


def test_visible_fork_thread_refs_are_platform_aware(visible_fork_runner):
    runner, _adapter, _store, _db = visible_fork_runner
    assert runner._visible_fork_thread_ref(Platform.DISCORD, "123") == "<#123>"
    assert runner._visible_fork_thread_ref(Platform.TELEGRAM, "456") == "`456`"


def test_fork_is_distinct_canonical_command():
    from hermes_cli.commands import resolve_command

    branch = resolve_command("branch")
    fork = resolve_command("fork")
    assert branch is not None and branch.name == "branch"
    assert fork is not None and fork.name == "fork"
