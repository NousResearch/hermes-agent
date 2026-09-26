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
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "test.db")
        agent = _make_agent(db, "sess-live")

        transcript = [
            {"role": "user", "content": "turn one"},
            {"role": "assistant", "content": "answer one"},
        ]
        agent._flush_messages_to_session_db(transcript, [])
        assert len(db.get_messages("sess-live")) == 2
        assert agent._session_db_created is True

        # Row removed under the live agent: the same store API behind
        # `hermes sessions delete`, the Desktop/web delete, and bulk prune.
        assert db.delete_session("sess-live") is True

        # The live transcript keeps growing: two more turns join the same list
        # (marked dicts from the earlier flush + fresh tail), as in a real session.
        transcript += [
            {"role": "user", "content": "turn two"},
            {"role": "assistant", "content": "answer two"},
        ]
        healed = agent._flush_messages_to_session_db(transcript, [])

        assert healed is True
        assert agent._last_persistence_error_cause == "session_row_missing"
        rows = db.get_messages("sess-live")
        assert len(rows) == 4, (
            "Heal must replay the FULL in-memory transcript onto the recreated "
            "row (4 messages), not just the current tail; a silent drop here is "
            "the #123583 transcript-loss bug."
        )
        assert agent._session_db_created is True
        db.close()


def test_flush_recovers_when_row_deleted_between_turns_twice():
    """The healed state is stable: a second deletion keeps healing, not just once."""
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "test.db")
        agent = _make_agent(db, "sess-live-2")

        agent._flush_messages_to_session_db([{"role": "user", "content": "a"}], [])
        assert len(db.get_messages("sess-live-2")) == 1

        for round_no in ("x", "y"):
            assert db.delete_session("sess-live-2") is True
            healed = agent._flush_messages_to_session_db(
                [{"role": "user", "content": round_no}], []
            )
            assert healed is True
            rows = db.get_messages("sess-live-2")
            assert len(rows) == 1 and rows[-1]["role"] == "user"
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
