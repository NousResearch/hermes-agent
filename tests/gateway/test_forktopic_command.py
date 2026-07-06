"""Tests for /forktopic gateway slash command."""

import dataclasses
import json
import re
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore, build_session_key


def _source(
    thread_id="555",
    *,
    chat_id="-1001234567890",
    chat_type="forum",
) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id="u1",
        user_name="tester",
        thread_id=thread_id,
    )


def _runner(tmp_path, monkeypatch):
    import hermes_state
    from gateway.run import GatewayRunner

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
    assert store._db is not None

    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = store
    runner._session_db = store._db
    clear_security = MagicMock()
    evict_agent = MagicMock()
    release_running = MagicMock()
    setattr(runner, "_clear_session_boundary_security_state", clear_security)
    setattr(runner, "_evict_cached_agent", evict_agent)
    setattr(runner, "_release_running_agent_state", release_running)

    adapter = SimpleNamespace(
        create_forum_topic=AsyncMock(return_value="888"),
        send=AsyncMock(return_value=SimpleNamespace(success=True)),
        delete_forum_topic=AsyncMock(return_value=True),
    )
    setattr(runner, "adapters", {Platform.TELEGRAM: adapter})
    return runner, store, adapter, clear_security, evict_agent, release_running


def _seed_current_session(store: SessionStore, source: SessionSource) -> str:
    entry = store.get_or_create_session(source)
    store.append_to_transcript(entry.session_id, {"role": "user", "content": "original ask"})
    store.append_to_transcript(entry.session_id, {"role": "assistant", "content": "original answer"})
    db = store._db
    assert db is not None
    db.set_session_title(entry.session_id, "Original Topic")
    return entry.session_id


def test_forktopic_registered_for_gateway_and_telegram_menu():
    from hermes_cli.commands import resolve_command, telegram_bot_commands

    cmd = resolve_command("forktopic")
    assert cmd is not None
    assert cmd.name == "forktopic"
    assert cmd.gateway_only is True
    assert ("forktopic", "Fork this Telegram topic into a new seeded topic") in telegram_bot_commands()


@pytest.mark.asyncio
async def test_forktopic_creates_seeded_topic_copies_session_and_binds_routing(tmp_path, monkeypatch):
    runner, store, adapter, clear_security, evict_agent, release_running = _runner(tmp_path, monkeypatch)
    source = _source(chat_id="208214988", chat_type="dm")
    db = store._db
    assert db is not None
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="u1")
    parent_session_id = _seed_current_session(store, source)

    event = MessageEvent(text="/forktopic Forked Lane", source=source, message_id="m1")
    result = await runner._handle_forktopic_command(event)

    assert "Forked into Telegram topic" in result
    assert "Forked Lane" in result
    adapter.create_forum_topic.assert_awaited_once_with("208214988", "Forked Lane")
    adapter.delete_forum_topic.assert_not_awaited()

    assert adapter.send.await_count == 3
    send_kwargs = adapter.send.await_args_list[0].kwargs
    assert send_kwargs["chat_id"] == "208214988"
    assert send_kwargs["metadata"]["thread_id"] == "888"
    assert "Forked conversation: Forked Lane" in send_kwargs["content"]
    assert "original ask" not in send_kwargs["content"]
    assert "original answer" not in send_kwargs["content"]
    assert "**You**" in adapter.send.await_args_list[1].kwargs["content"]
    assert "original ask" in adapter.send.await_args_list[1].kwargs["content"]
    assert "**Hermes**" in adapter.send.await_args_list[2].kwargs["content"]
    assert "original answer" in adapter.send.await_args_list[2].kwargs["content"]

    match = re.search(r"Fork: `([^`]+)`", result)
    assert match, result
    fork_session_id = match.group(1)
    assert fork_session_id != parent_session_id

    original_key = build_session_key(source)
    fork_source = dataclasses.replace(source, chat_type="dm", thread_id="888")
    fork_key = build_session_key(fork_source)
    assert store._entries[original_key].session_id == parent_session_id
    assert store._entries[fork_key].session_id == fork_session_id
    binding = db.get_telegram_topic_binding(chat_id="208214988", thread_id="888")
    assert binding is not None
    assert binding["session_key"] == fork_key
    assert binding["session_id"] == fork_session_id

    fork_messages = db.get_messages_as_conversation(fork_session_id)
    assert [m["role"] for m in fork_messages] == ["user", "assistant"]
    assert fork_messages[0]["content"] == "original ask"
    assert fork_messages[1]["content"] == "original answer"

    conn = db._conn
    assert conn is not None
    with db._lock:
        session_rows = conn.execute("SELECT id FROM sessions").fetchall()
    assert {row["id"] for row in session_rows} == {parent_session_id, fork_session_id}

    row = db.get_session(fork_session_id)
    assert row is not None
    meta = json.loads(row["model_config"])
    assert meta["_branched_from"] == parent_session_id
    assert meta["_forktopic_from"] == parent_session_id
    assert meta["_forktopic_chat_id"] == "208214988"
    assert meta["_forktopic_thread_id"] == "888"
    assert db.get_session_title(fork_session_id) == "Forked Lane"
    clear_security.assert_called_once_with(fork_key)
    evict_agent.assert_called_once_with(fork_key)
    release_running.assert_called_once_with(fork_key)


@pytest.mark.asyncio
async def test_forktopic_sends_recent_context_as_separate_timestamped_messages(tmp_path, monkeypatch):
    runner, store, adapter, clear_security, evict_agent, release_running = _runner(tmp_path, monkeypatch)
    source = _source(thread_id="777")
    entry = store.get_or_create_session(source)
    base_ts = datetime(2026, 6, 30, 12, 0).timestamp()
    long_recent = "recent user five " + ("kept in full " * 80).strip()
    messages = [
        ("user", "oldest user message should not be visible", base_ts),
        ("assistant", "older assistant message should not be visible", base_ts + 60),
        ("user", "recent user one", base_ts + 120),
        ("assistant", "recent assistant two", base_ts + 180),
        ("user", "recent user three", base_ts + 240),
        ("assistant", "recent assistant four", base_ts + 300),
        ("user", long_recent, base_ts + 360),
    ]
    for role, content, timestamp in messages:
        store.append_to_transcript(
            entry.session_id,
            {"role": role, "content": content, "timestamp": timestamp},
        )

    event = MessageEvent(text="/forktopic Context Fork", source=source, message_id="m1")
    result = await runner._handle_forktopic_command(event)

    assert "Forked into Telegram topic" in result
    assert adapter.send.await_count == 6
    sent_contents = [call.kwargs["content"] for call in adapter.send.await_args_list]
    seed_content = sent_contents[0]
    context_messages = sent_contents[1:]
    assert "Forked conversation: Context Fork" in seed_content
    assert "Recent context" not in seed_content
    assert "oldest user message should not be visible" not in "\n".join(sent_contents)
    assert "older assistant message should not be visible" not in "\n".join(sent_contents)
    assert "**You** · " in context_messages[0]
    assert "recent user one" in context_messages[0]
    assert "**Hermes** · " in context_messages[1]
    assert "recent assistant two" in context_messages[1]
    assert "**You** · " in context_messages[2]
    assert "recent user three" in context_messages[2]
    assert "**Hermes** · " in context_messages[3]
    assert "recent assistant four" in context_messages[3]
    assert "**You** · " in context_messages[4]
    assert long_recent in context_messages[4]
    assert datetime.fromtimestamp(base_ts + 120).strftime("%Y-%m-%d %H:%M") in context_messages[0]
    assert all("Forked conversation" not in message for message in context_messages)

    match = re.search(r"Fork: `([^`]+)`", result)
    assert match, result
    db = store._db
    assert db is not None
    fork_messages = db.get_messages_as_conversation(match.group(1))
    assert [m["content"] for m in fork_messages] == [content for _, content, _ in messages]


@pytest.mark.asyncio
async def test_forktopic_recent_context_skips_contentless_assistant_tool_turns(tmp_path, monkeypatch):
    runner, store, adapter, clear_security, evict_agent, release_running = _runner(tmp_path, monkeypatch)
    source = _source(thread_id="778")
    entry = store.get_or_create_session(source)
    base_ts = datetime(2026, 7, 1, 12, 0).timestamp()
    visible = [
        ("user", "actual user one", base_ts),
        ("assistant", "actual assistant two", base_ts + 60),
        ("user", "actual user three", base_ts + 120),
        ("assistant", "actual assistant four", base_ts + 180),
        ("user", "actual user five", base_ts + 240),
    ]
    for role, content, timestamp in visible:
        store.append_to_transcript(
            entry.session_id,
            {"role": role, "content": content, "timestamp": timestamp},
        )
    for index in range(4):
        store.append_to_transcript(
            entry.session_id,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
                "timestamp": base_ts + 300 + index,
            },
        )

    event = MessageEvent(text="/forktopic Context Fork", source=source, message_id="m1")
    result = await runner._handle_forktopic_command(event)

    assert "Forked into Telegram topic" in result
    assert adapter.send.await_count == 6
    sent_contents = [call.kwargs["content"] for call in adapter.send.await_args_list]
    context_text = "\n".join(sent_contents[1:])
    assert "(empty)" not in context_text
    assert "actual user one" in context_text
    assert "actual assistant two" in context_text
    assert "actual user three" in context_text
    assert "actual assistant four" in context_text
    assert "actual user five" in context_text


@pytest.mark.asyncio
async def test_forktopic_rolls_back_created_topic_when_binding_fails(tmp_path, monkeypatch):
    runner, store, adapter, clear_security, evict_agent, release_running = _runner(tmp_path, monkeypatch)
    source = _source()
    parent_session_id = _seed_current_session(store, source)
    store.switch_session = MagicMock(return_value=None)

    event = MessageEvent(text="/forktopic Bad Fork", source=source, message_id="m1")
    result = await runner._handle_forktopic_command(event)

    assert "Topic was not forked" in result
    adapter.create_forum_topic.assert_awaited_once_with("-1001234567890", "Bad Fork")
    adapter.delete_forum_topic.assert_awaited_once_with("-1001234567890", "888")
    adapter.send.assert_not_awaited()
    db = store._db
    assert db is not None
    conn = db._conn
    assert conn is not None
    with db._lock:
        fork_rows = conn.execute(
            "SELECT id FROM sessions WHERE parent_session_id = ?",
            (parent_session_id,),
        ).fetchall()
    assert fork_rows == []
