"""Fake persistence tests for Discord thread session restart behavior.

These tests use temporary ``sessions.json`` and ``state.db`` files only. They
document that the durable session_key -> session_id index is required to route
the same Discord thread back to its prior transcript after restart.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import sqlite3

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, build_session_key


THREAD_ID = "222222222222222222"
PARENT_CHANNEL_ID = "111111111111111111"
GUILD_ID = "999999999999999999"
USER_ID = "333333333333333333"
EXPECTED_THREAD_KEY = f"agent:main:discord:thread:{THREAD_ID}:{THREAD_ID}"


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    """Build SessionStore instances pinned to temp sessions and temp SQLite."""
    import hermes_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    stores: list[SessionStore] = []

    def _make_store() -> SessionStore:
        store = SessionStore(
            sessions_dir=tmp_path / "sessions",
            config=GatewayConfig(),
        )
        stores.append(store)
        return store

    yield _make_store

    for store in stores:
        if store._db:
            store._db.close()


def _discord_thread_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=THREAD_ID,
        chat_type="thread",
        user_id=USER_ID,
        thread_id=THREAD_ID,
        guild_id=GUILD_ID,
        parent_chat_id=PARENT_CHANNEL_ID,
    )


def _append_fake_transcript(store: SessionStore, session_id: str) -> None:
    store.append_to_transcript(
        session_id,
        {
            "role": "user",
            "content": "remember the Jenny project thread context",
            "message_id": "444444444444444444",
        },
    )
    store.append_to_transcript(
        session_id,
        {
            "role": "assistant",
            "content": "I will keep using this thread context.",
        },
    )


def test_discord_thread_session_key_stays_stable():
    source = _discord_thread_source()

    assert build_session_key(source) == EXPECTED_THREAD_KEY


def test_discord_thread_session_persists_routing_metadata(store_factory):
    source = _discord_thread_source()
    store = store_factory()

    entry = store.get_or_create_session(source)

    with sqlite3.connect(store._db.db_path) as conn:
        row = conn.execute(
            """
            SELECT session_key, platform, chat_type, chat_id, thread_id,
                   parent_chat_id, guild_id, user_id
            FROM session_routing_metadata
            WHERE session_id = ?
            """,
            (entry.session_id,),
        ).fetchone()

    assert row == (
        EXPECTED_THREAD_KEY,
        "discord",
        "thread",
        THREAD_ID,
        THREAD_ID,
        PARENT_CHANNEL_ID,
        GUILD_ID,
        USER_ID,
    )


def test_discord_thread_session_and_transcript_survive_restart(store_factory):
    source = _discord_thread_source()
    first_store = store_factory()

    first_entry = first_store.get_or_create_session(source)
    _append_fake_transcript(first_store, first_entry.session_id)

    restarted_store = store_factory()
    restarted_entry = restarted_store.get_or_create_session(source)
    transcript = restarted_store.load_transcript(restarted_entry.session_id)

    assert restarted_entry.session_key == EXPECTED_THREAD_KEY
    assert restarted_entry.session_id == first_entry.session_id
    assert [message["content"] for message in transcript] == [
        "remember the Jenny project thread context",
        "I will keep using this thread context.",
    ]


def test_lost_sessions_json_mapping_starts_new_discord_thread_session(
    store_factory,
    tmp_path,
):
    """state.db alone cannot reconstruct the Discord thread routing mapping.

    If ``sessions.json`` is lost while the old transcript rows remain in
    state.db, the same Discord thread gets a new active session_id and loads an
    empty transcript by default.
    """
    source = _discord_thread_source()
    first_store = store_factory()

    first_entry = first_store.get_or_create_session(source)
    _append_fake_transcript(first_store, first_entry.session_id)

    (tmp_path / "sessions" / "sessions.json").unlink()

    restarted_store = store_factory()
    restarted_entry = restarted_store.get_or_create_session(source)
    new_transcript = restarted_store.load_transcript(restarted_entry.session_id)
    old_transcript = restarted_store.load_transcript(first_entry.session_id)

    assert restarted_entry.session_key == EXPECTED_THREAD_KEY
    assert restarted_entry.session_id != first_entry.session_id
    assert new_transcript == []
    assert [message["content"] for message in old_transcript] == [
        "remember the Jenny project thread context",
        "I will keep using this thread context.",
    ]


def test_pruned_discord_thread_mapping_starts_new_session_but_keeps_old_db_transcript(
    store_factory,
):
    """Pruning removes the active routing index, not historical DB rows."""
    source = _discord_thread_source()
    store = store_factory()

    first_entry = store.get_or_create_session(source)
    _append_fake_transcript(store, first_entry.session_id)
    with store._lock:
        store._entries[first_entry.session_key].updated_at = (
            datetime.now() - timedelta(days=2)
        )
        store._save()

    assert store.prune_old_entries(max_age_days=1) == 1

    next_entry = store.get_or_create_session(source)
    next_transcript = store.load_transcript(next_entry.session_id)
    old_transcript = store.load_transcript(first_entry.session_id)

    assert next_entry.session_key == EXPECTED_THREAD_KEY
    assert next_entry.session_id != first_entry.session_id
    assert next_transcript == []
    assert [message["content"] for message in old_transcript] == [
        "remember the Jenny project thread context",
        "I will keep using this thread context.",
    ]
