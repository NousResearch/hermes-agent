"""A session row deleted under a live agent must heal on the next flush (#123583).

Before the fix, ``_flush_messages_to_session_db`` trusts the cached
``_session_db_created`` flag: after ``hermes sessions delete`` (or Desktop delete /
bulk prune / profile-repair move / in-place store rebuild) removes the row, every
later turn's append fails the FK and is dropped with one WARNING per turn — the
durable transcript silently stops growing and leaves no trace in the store.

Maintainer triage direction (issue #123583, maintainer pass): classify the FK
failure as ``session_row_missing``, recreate the row, and replay the FULL
in-memory transcript — not just the current tail.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def _make_agent(session_db, session_id):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )


def test_flush_recreates_row_deleted_under_live_agent():
    """Real turn shape (messages = history + tail, conversation_history=history): every
    deletion round replays the whole in-memory transcript onto the recreated row."""
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "test.db")
        agent = _make_agent(db, "sess-live")

        history = []
        for n in ("one", "two", "three"):
            if history:
                # Row removed under the idle live agent (sessions delete / Desktop / prune).
                assert db.delete_session("sess-live") is True
            tail = [{"role": "user", "content": f"turn {n}"}, {"role": "assistant", "content": f"answer {n}"}]
            messages = list(history) + tail
            assert agent._flush_messages_to_session_db(messages, history) is True
            history = messages
            assert [r["content"] for r in db.get_messages("sess-live")] == [m["content"] for m in history]
        assert agent._last_persistence_error_cause == "session_row_missing"
        db.close()


def test_flush_fails_open_when_row_cannot_be_recreated(monkeypatch):
    """Scenario B: if row creation fails too, the flush returns False instead of
    appending into a guaranteed rollback — fail-open, batch stays unmarked."""
    import sqlite3 as _sqlite3

    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "test.db")
        agent = _make_agent(db, "sess-gone")

        agent._flush_messages_to_session_db([{"role": "user", "content": "a"}], [])
        assert len(db.get_messages("sess-gone")) == 1
        assert db.delete_session("sess-gone") is True

        # Row creation inside the heal now raises (transient store trouble).
        def _broken_create(*a, **kw):
            raise _sqlite3.OperationalError("unable to open database file")

        monkeypatch.setattr(db, "create_session", _broken_create)

        healed = agent._flush_messages_to_session_db(
            [{"role": "user", "content": "b"}], []
        )
        assert healed is False
        assert agent._session_db_created is False
