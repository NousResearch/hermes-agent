"""Gateway /undo N clamps to the oldest user turn, so its out-of-range case is a transcript with no user turn at all:
``SessionStore.rewind_session`` refuses with ``None`` and no stored row changes."""

from __future__ import annotations

from pathlib import Path

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from persistence.conversation.messages import TranscriptRead, TranscriptView
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session


def test_undo_with_no_user_turn_returns_none_and_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    every_row = TranscriptRead(view=TranscriptView.ALL)
    try:
        seed_session(store.persistence, "oob-gateway", "cli")
        seed_messages(store.persistence, "oob-gateway", {"role": "user", "content": "q1"},
                      {"role": "assistant", "content": "a1"})
        before = transcript_rows(store.persistence, "oob-gateway", every_row)
        seed_session(store.persistence, "oob-gateway-empty", "cli")
        assert store.rewind_session("oob-gateway-empty", 1) is None
        assert transcript_rows(store.persistence, "oob-gateway-empty", every_row) == []
        assert transcript_rows(store.persistence, "oob-gateway", every_row) == before
    finally:
        store.close_all_db_handles()
