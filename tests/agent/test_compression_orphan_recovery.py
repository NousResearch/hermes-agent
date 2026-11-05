"""Recovery for legacy compression parents with no continuation child."""

from types import SimpleNamespace

from agent.conversation_compression import recover_rotated_compression_session
from persistence import CompressionClosed
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


def test_recover_rotated_compression_session_reopens_legacy_orphan(tmp_path):
    store = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(store, "orphan", "cli")
        seed_messages(store, "orphan", {"role": "user", "content": "before compression"})
        store.sessions.end("orphan", "compression")
        agent = SimpleNamespace(_session_db=store, persistence=store, session_id="orphan")

        assert recover_rotated_compression_session(agent) is None
        seed_messages(store, "orphan", {"role": "user", "content": "after recovery"})
    finally:
        store.close()


def test_recover_rotated_compression_session_keeps_parent_closed_with_child(
    tmp_path,
):
    store = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(store, "parent", "cli")
        seed_messages(store, "parent", {"role": "user", "content": "before compression"})
        store.sessions.end("parent", "compression")
        seed_session(store, "child", "cli", parent_session_id="parent")
        agent = SimpleNamespace(_session_db=store, persistence=store, session_id="parent")

        assert recover_rotated_compression_session(agent) is None
        try:
            seed_messages(store, "parent", {"role": "user", "content": "must stay closed"})
        except CompressionClosed:
            pass
        else:
            raise AssertionError("compression parent with child was reopened")
    finally:
        store.close()
