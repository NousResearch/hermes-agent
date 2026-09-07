"""A failed transcript read must not masquerade as an empty history (#100788).

The gateway restore path (``_handle_message``) already fails closed on
current main (#100910). This file covers the surviving half of PR #100887:
the slash-command handlers, which used to let ``TranscriptReadError``
propagate into the dispatch wrapper and reply with nothing at all.

Incident shape: a malformed ``state.db`` made every
``SessionStore.load_transcript`` raise; the except-block swallowed it and
returned ``[]``.  Restore then rebuilt the turn from "no history", so a
long-running chat silently restarted as a brand-new conversation and the
model happily answered as if nothing had ever been discussed.

Two guarantees under test:
  A. ``load_transcript`` raises ``TranscriptReadError`` on a read failure,
     while a genuinely empty session still returns ``[]``.
  B. Slash-command handlers that read the transcript reply with
     ``HISTORY_UNREADABLE`` instead of raising into the dispatch wrapper
     (which logs and sends nothing).

Offline: SQLite on tmp_path only, no network.
"""

import sqlite3

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore, TranscriptReadError


@pytest.fixture
def store(tmp_path):
    return SessionStore(sessions_dir=tmp_path / "gw", config=GatewayConfig())


# --------------------------------------------------------------------------
# A. read failure != empty transcript (landed on main via #100910; kept as
#    the contract the slash-command handlers below rely on)
# --------------------------------------------------------------------------


class TestLoadTranscriptReadFailure:
    def test_read_failure_raises_instead_of_returning_empty(self, store, monkeypatch):
        db = store._db
        assert db is not None
        db.create_session("s1", "telegram", session_key="telegram:1")
        db.append_message("s1", "user", "the conversation we must not forget")

        boom = sqlite3.DatabaseError("database disk image is malformed")

        def _raise(*_args, **_kwargs):
            raise boom

        monkeypatch.setattr(db, "get_messages_as_conversation", _raise)

        with pytest.raises(TranscriptReadError) as excinfo:
            store.load_transcript("s1")

        assert excinfo.value.session_id == "s1"
        assert excinfo.value.__cause__ is boom

    def test_genuinely_empty_session_still_returns_empty_list(self, store):
        db = store._db
        assert db is not None
        db.create_session("s2", "telegram", session_key="telegram:2")

        assert store.load_transcript("s2") == []

    def test_no_db_still_returns_empty_list(self, store):
        # "No DB for this session" really is an empty transcript, not a
        # failure — that path must keep its [] contract.
        store._db = None
        assert store.load_transcript("nope") == []


# --------------------------------------------------------------------------
# B. slash-command handlers surface the failure instead of dying silently.
#    Before: the handler raised, base.py's dispatch wrapper logged
#    "Command '/x' dispatch failed" and the user got NO reply at all.
# --------------------------------------------------------------------------


class TestSlashCommandsOnUnreadableTranscript:
    def test_history_unreadable_text_is_explicit(self):
        from gateway.slash_commands import HISTORY_UNREADABLE

        assert "unreadable" in HISTORY_UNREADABLE
        assert "not a new conversation" in HISTORY_UNREADABLE

    def test_every_transcript_reading_handler_catches_the_error(self):
        """No `await ...load_transcript(` in the mixin may be left uncaught."""
        import inspect
        import re

        from gateway import slash_commands as sc

        src = inspect.getsource(sc)
        # Each awaited load_transcript must sit inside a try: whose handlers
        # include TranscriptReadError within the following ~6 lines.
        for m in re.finditer(r"await self\.async_session_store\.load_transcript\(", src):
            window = src[m.end() : m.end() + 400]
            assert "except TranscriptReadError" in window, src[m.start() - 200 : m.end() + 100]
