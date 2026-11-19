"""Tests for SessionStore.rewind_session — the gateway /undo [N] primitive.

The gateway /undo backs up N user turns by soft-deleting the truncated rows
in state.db (active=0, kept for audit, hidden from re-prompts/search) via
the store's messages.rewind_user_turn, rather than the old hard rewrite_transcript.
load_transcript returns only the active view. See issue #21910.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder, rewind_store_operation
from tests.fixtures.storage_seed import seed_messages, seed_session


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    s = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    yield s
    s.close_all_db_handles()


def _seed(store, sid, source="telegram", turns=3):
    seed_session(store.persistence, sid, source)
    for i in range(1, turns + 1):
        seed_messages(store.persistence, sid, {"role": "user", "content": f"q{i}"},
                      {"role": "assistant", "content": f"a{i}"})
    return sid


def test_rewind_default_one_turn(store):
    sid = _seed(store, "gw-1")
    res = store.rewind_session(sid)
    assert res["turns_undone"] == 1
    assert res["target_text"] == "q3"
    assert res["rewound_count"] == 2  # q3 + a3
    active = store.load_transcript(sid)
    assert [m["role"] for m in active] == ["user", "assistant", "user", "assistant"]


def test_rewind_n_turns(store):
    sid = _seed(store, "gw-2")
    res = store.rewind_session(sid, 2)
    assert res["turns_undone"] == 2
    assert res["target_text"] == "q2"
    assert res["rewound_count"] == 4  # q2,a2,q3,a3
    assert len(store.load_transcript(sid)) == 2  # q1,a1


def test_rewind_pins_raw_active_ids_when_projection_hides_review_harness(store):
    sid = _seed(store, "gw-review-harness", turns=2)
    seed_messages(
        store.persistence,
        sid,
        {"role": "user", "content": "Review the conversation above and update the skill library safely"},
        {"role": "assistant", "content": "curator-only reply"},
    )

    # Legacy background-review rows are intentionally absent from replay, but
    # they remain physical active rows that the rewind CAS must pin.
    assert [message["content"] for message in store.load_transcript(sid)] == [
        "q1",
        "a1",
        "q2",
        "a2",
    ]

    result = store.rewind_session(sid)

    assert result is not None
    assert result["target_text"] == "q2"
    assert result["rewound_count"] == 4
    assert [message["content"] for message in store.load_transcript(sid)] == [
        "q1",
        "a1",
    ]


def test_rewind_fails_closed_when_transcript_changes_after_snapshot(
    store, monkeypatch
):
    sid = _seed(store, "gw-cas", turns=2)
    sibling = open_persistence(profile_scope(home=store.persistence.scope.profile_dir))
    original_rewind = rewind_store_operation("rewind_to_message")  # the store's write behind messages.rewind_user_turn

    def _append_then_rewind(*args, **kwargs):
        seed_messages(sibling, sid, {"role": "assistant", "content": "concurrent tail"})
        return original_rewind(*args, **kwargs)

    monkeypatch.setattr("persistence.sqlite.conversation._store.rewind.rewind_to_message", _append_then_rewind)

    assert store.rewind_session(sid) is None

    with raw_holder(store.persistence)._read_ctx() as conn:
        rows = conn.execute(
            "SELECT content, active FROM messages "
            "WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("q1", 1),
        ("a1", 1),
        ("q2", 1),
        ("a2", 1),
        ("concurrent tail", 1),
    ]
    sibling.close()


def test_rewind_fails_closed_when_new_turn_lands_after_id_snapshot(
    store, monkeypatch
):
    sid = _seed(store, "gw-snapshot-order", turns=2)
    sibling = open_persistence(profile_scope(home=store.persistence.scope.profile_dir))
    original_load = rewind_store_operation("get_messages_as_conversation")  # the reads behind messages.rewind_user_turn

    calls = []

    def _load_then_append(*args, **kwargs):
        snapshot = original_load(*args, **kwargs)
        if not calls:  # one new turn lands between the id snapshot and the write, however often the DB is read
            seed_messages(sibling, sid, {"role": "user", "content": "q3-from-other-process"},
                          {"role": "assistant", "content": "a3-from-other-process"})
        calls.append(1)
        return snapshot

    monkeypatch.setattr("persistence.sqlite.conversation._store.rewind.get_messages_as_conversation", _load_then_append)

    assert store.rewind_session(sid) is None

    with raw_holder(store.persistence)._read_ctx() as conn:
        rows = conn.execute(
            "SELECT content, active FROM messages "
            "WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("q1", 1),
        ("a1", 1),
        ("q2", 1),
        ("a2", 1),
        ("q3-from-other-process", 1),
        ("a3-from-other-process", 1),
    ]
    sibling.close()


@pytest.mark.asyncio
async def test_undo_evicts_cached_agent_under_profile_namespaced_key():
    """/undo must evict under the key the cache is keyed by. On a multiplexed gateway that is
    ``agent:<profile>:…``; a bare ``build_session_key(source)`` yields ``agent:main:…``, the
    eviction misses and the next turn reuses an agent still holding the undone turns."""
    from gateway.platforms.base import Platform, SessionSource
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run import GatewayRunner
    from gateway.session import build_session_key

    class _Entry:
        session_id = "sid"
        last_prompt_tokens = 0

    class _Store:
        async def get_or_create_session(self, source):
            return _Entry()

        async def rewind_session(self, sid, n):
            return {"target_text": "q3", "turns_undone": 1, "rewound_count": 2}

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", user_id="u1")
    source.profile = "work"
    evicted = []
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.session_store = _Store()
    runner._async_session_store = _Store()
    runner._async_session_store._store = runner.session_store
    runner._evict_cached_agent = evicted.append
    runner._session_key_for_source = lambda s: build_session_key(s, profile=s.profile)

    await runner._handle_undo_command(MessageEvent(text="/undo", message_type=MessageType.TEXT, source=source))

    assert evicted == ["agent:work:telegram:dm:123"]
