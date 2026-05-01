"""Regression tests for Honcho duplicate flush guards."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


class RecordingPeer:
    def __init__(self, peer_id: str):
        self.peer_id = peer_id

    def message(self, content: str):
        return SimpleNamespace(peer_id=self.peer_id, content=content)


def test_honcho_flush_skips_recently_synced_duplicate_batch():
    mgr = HonchoSessionManager()
    session = HonchoSession(
        key="cli:test",
        user_peer_id="eri",
        assistant_peer_id="hermes",
        honcho_session_id="cli-test",
    )
    session.add_message("user", "same prompt")
    session.add_message("assistant", "same answer")

    user_peer = RecordingPeer("eri")
    assistant_peer = RecordingPeer("hermes")
    honcho_session = MagicMock()
    mgr._peers_cache["eri"] = user_peer
    mgr._peers_cache["hermes"] = assistant_peer
    mgr._sessions_cache["cli-test"] = honcho_session

    assert mgr._flush_session(session) is True
    assert honcho_session.add_messages.call_count == 1

    duplicate = HonchoSession(
        key="cli:test",
        user_peer_id="eri",
        assistant_peer_id="hermes",
        honcho_session_id="cli-test",
    )
    duplicate.add_message("user", "same prompt")
    duplicate.add_message("assistant", "same answer")

    assert mgr._flush_session(duplicate) is True
    assert honcho_session.add_messages.call_count == 1
    assert all(msg.get("_synced") for msg in duplicate.messages)
