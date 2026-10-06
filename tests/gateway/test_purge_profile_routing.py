"""In-memory routing purge for `hermes profile delete`.

The routing index lives in ``SessionStore._entries`` and is written back periodically, so a durable
DB delete made anywhere else is undone by this process's next save — the store has to drop its own
copy, and the drop has to be persisted, or a deleted profile's chats keep resolving to it. This is
the delete-side sibling of the rename rekey in #111926.
"""

from __future__ import annotations


def _make_store(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    store = SessionStore(
        sessions_dir,
        GatewayConfig(sessions_dir=sessions_dir, write_sessions_json=False,
                      multiplex_profiles=True),
    )
    store._ensure_loaded()
    return store


def _entry(session_key, chat_id, profile):
    from gateway.session import SessionEntry, SessionSource, Platform
    from gateway.session_lifecycle import _now
    now = _now()
    return SessionEntry(
        session_key=session_key, session_id=f"sid-{chat_id}",
        platform=Platform.FEISHU, chat_type="dm", created_at=now, updated_at=now,
        origin=SessionSource(platform=Platform.FEISHU, chat_id=chat_id, profile=profile),
    )


def test_purges_deleted_namespace_and_persists_the_drop(tmp_path):
    store = _make_store(tmp_path)
    with store._lock:
        store._entries["agent:gone:feishu:dm:chatA"] = _entry(
            "agent:gone:feishu:dm:chatA", "chatA", "gone")
        store._entries["agent:keepme:feishu:dm:chatB"] = _entry(
            "agent:keepme:feishu:dm:chatB", "chatB", "keepme")
    store._save()

    dropped = store.purge_profile_routing("gone")

    assert dropped == 1
    assert "agent:gone:feishu:dm:chatA" not in store._entries
    assert "agent:keepme:feishu:dm:chatB" in store._entries
    # The drop is durable: a store reloading the same home must not resurrect the deleted profile.
    reloaded = _make_store(tmp_path)
    assert "agent:gone:feishu:dm:chatA" not in reloaded._entries
    assert "agent:keepme:feishu:dm:chatB" in reloaded._entries


def test_namespace_scoped_and_idempotent(tmp_path):
    store = _make_store(tmp_path)
    with store._lock:
        store._entries["agent:foo_bar:feishu:dm:chatA"] = _entry(
            "agent:foo_bar:feishu:dm:chatA", "chatA", "foo_bar")
        store._entries["agent:fooXbar:feishu:dm:chatB"] = _entry(
            "agent:fooXbar:feishu:dm:chatB", "chatB", "fooXbar")

    assert store.purge_profile_routing("foo_bar") == 1
    assert store.purge_profile_routing("foo_bar") == 0

    assert "agent:foo_bar:feishu:dm:chatA" not in store._entries
    assert "agent:fooXbar:feishu:dm:chatB" in store._entries
