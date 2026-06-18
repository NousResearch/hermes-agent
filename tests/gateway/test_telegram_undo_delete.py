from __future__ import annotations

from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_state import SessionDB


@pytest.fixture()
def session_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    store._db = db
    return store


class _DeleteAdapter:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls: list[tuple[str, str]] = []

    async def delete_message(self, chat_id, message_id) -> bool:
        self.calls.append((str(chat_id), str(message_id)))
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return bool(result)
        return True


class _UndoRunner(GatewaySlashCommandsMixin):
    def __init__(self, store, adapter=None):
        self.session_store = store
        self._async_session_store = AsyncSessionStore(store)
        self.adapters = {}
        if adapter is not None:
            self.adapters[Platform.TELEGRAM] = adapter
        self.evicted: list[str] = []

    def _evict_cached_agent(self, session_key):
        self.evicted.append(session_key)

    @property
    def async_session_store(self):
        return self._async_session_store


def _telegram_source(chat_id="-1003554245341", chat_type="supergroup"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id="42",
        user_name="tester",
    )


def _event(source, text="/undo"):
    return MessageEvent(text=text, source=source, message_id="9001")


def _seed_turns(store, source):
    entry = store.get_or_create_session(source)
    sid = entry.session_id
    store._db.append_message(sid, "user", "q1", platform_message_id="101")
    store._db.append_message(sid, "assistant", "a1", platform_message_id="102")
    store._db.append_message(sid, "user", "q2", platform_message_id="201")
    store._db.append_message(
        sid,
        "assistant",
        "a2",
        platform_message_id='["202", "203"]',
    )
    return entry


@pytest.mark.asyncio
async def test_telegram_undo_deletes_rewound_visible_messages(session_store):
    source = _telegram_source()
    _seed_turns(session_store, source)
    adapter = _DeleteAdapter()
    runner = _UndoRunner(session_store, adapter)

    response = await runner._handle_undo_command(_event(source))

    assert "q2" in response
    assert adapter.calls == [
        ("-1003554245341", "201"),
        ("-1003554245341", "202"),
        ("-1003554245341", "203"),
    ]
    assert len(session_store.load_transcript(runner.session_store.get_or_create_session(source).session_id)) == 2


@pytest.mark.asyncio
async def test_telegram_undo_reports_partial_cleanup_failure(session_store):
    source = _telegram_source(chat_id="42", chat_type="dm")
    _seed_turns(session_store, source)
    adapter = _DeleteAdapter(results=[True, False, RuntimeError("no rights")])
    runner = _UndoRunner(session_store, adapter)

    response = await runner._handle_undo_command(_event(source))

    assert adapter.calls == [("42", "201"), ("42", "202"), ("42", "203")]
    assert "deleted 1/3" in response or "удалил 1/3" in response


@pytest.mark.asyncio
async def test_non_telegram_undo_does_not_delete_platform_messages(session_store):
    source = SessionSource(platform=Platform.DISCORD, chat_id="discord-chat", user_id="42")
    _seed_turns(session_store, source)
    adapter = _DeleteAdapter()
    runner = _UndoRunner(session_store, adapter)

    response = await runner._handle_undo_command(_event(source))

    assert "q2" in response
    assert adapter.calls == []

@pytest.mark.asyncio
async def test_telegram_undo_deletes_split_assistant_message_ids(session_store):
    """Assistant replies split across multiple Telegram messages must all clean up."""
    source = _telegram_source(chat_id="77", chat_type="dm")
    entry = session_store.get_or_create_session(source)
    sid = entry.session_id
    session_store._db.append_message(sid, "user", "split-q", platform_message_id="301")
    session_store._db.append_message(
        sid,
        "assistant",
        "split-a",
        platform_message_id='["401", "402", "403"]',
    )
    adapter = _DeleteAdapter()
    runner = _UndoRunner(session_store, adapter)

    response = await runner._handle_undo_command(_event(source))

    assert "split-q" in response
    assert adapter.calls == [
        ("77", "301"),
        ("77", "401"),
        ("77", "402"),
        ("77", "403"),
    ]

