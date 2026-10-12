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

        # A heal during a muted notification turn hides only that turn's new rows; the replayed
        # history rows (and their in-memory dicts) keep their visibility.
        assert db.delete_session("sess-live") is True
        agent._mute_notification_reply = True
        tail = [{"role": "user", "content": "notif"}, {"role": "assistant", "content": "muted"}]
        assert agent._flush_messages_to_session_db(list(history) + tail, history) is True
        rows = db.get_messages("sess-live")
        assert [(r["content"], r.get("display_kind")) for r in rows] == (
            [(m["content"], None) for m in history] + [("notif", "hidden"), ("muted", "hidden")]
        )
        assert [m.get("display_kind") for m in history] == [None] * len(history)
        db.close()


def test_flush_fails_closed_when_row_cannot_be_recreated(monkeypatch):
    """Scenario B: if row creation fails too, the flush fails closed — returns False
    instead of appending into a guaranteed rollback, and the batch stays unmarked. A FK
    failure is only healed when the row is confirmed gone, and a failed lookup is no proof."""
    import sqlite3 as _sqlite3

    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "test.db")
        agent = _make_agent(db, "sess-gone")

        turn1 = [{"role": "user", "content": "a"}]
        agent._flush_messages_to_session_db(turn1, [])
        assert len(db.get_messages("sess-gone")) == 1
        assert db.delete_session("sess-gone") is True

        # Row creation inside the heal now raises (transient store trouble).
        def _broken_create(*a, **kw):
            raise _sqlite3.OperationalError("unable to open database file")

        monkeypatch.setattr(db, "create_session", _broken_create)

        turn2 = turn1 + [{"role": "user", "content": "b"}]
        healed = agent._flush_messages_to_session_db(turn2, turn1)
        assert healed is False
        assert agent._session_db_created is False
        monkeypatch.undo()

        # The next flush recreates the row before any FK error (so no heal runs) and must still
        # replay the history prefix the failed heal left unwritten.
        turn3 = turn2 + [{"role": "assistant", "content": "c"}]
        assert agent._flush_messages_to_session_db(turn3, turn2) is True
        assert [r["content"] for r in db.get_messages("sess-gone")] == ["a", "b", "c"]
        assert agent._session_row_replay_pending is None

        # The heal recreates the row but its single retry write fails (lock/lease/disk): the next
        # flush finds a live row (no FK, no heal) and must still replay the history prefix.
        retry = _make_agent(db, "sess-retry")
        t1 = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "a1"}]
        assert retry._flush_messages_to_session_db(t1, []) is True
        assert db.delete_session("sess-retry") is True
        real_append, calls = db.append_messages_batch, []

        def _fk_then_locked(*a, **kw):
            calls.append(1)
            if len(calls) == 2:
                raise _sqlite3.OperationalError("database is locked")
            return real_append(*a, **kw)  # call 1 hits the real FK error -> heal

        monkeypatch.setattr(db, "append_messages_batch", _fk_then_locked)
        t2 = t1 + [{"role": "user", "content": "two"}]
        assert retry._flush_messages_to_session_db(t2, t1) is False
        t3 = t2 + [{"role": "assistant", "content": "a2"}]
        assert retry._flush_messages_to_session_db(t3, t2) is True
        assert [r["content"] for r in db.get_messages("sess-retry")] == ["one", "a1", "two", "a2"]
        monkeypatch.undo()

        # FK failure while the session row still exists (e.g. a sessions-table FK): no heal, and
        # no replay of the history prefix onto the live transcript.
        live = _make_agent(db, "sess-fk-live")
        history = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "a1"}]
        assert live._flush_messages_to_session_db(history, []) is True
        real_append, calls = db.append_messages_batch, []

        def _fk_once(*a, **kw):
            calls.append(1)
            if len(calls) == 1:
                raise _sqlite3.IntegrityError("FOREIGN KEY constraint failed")
            return real_append(*a, **kw)

        monkeypatch.setattr(db, "append_messages_batch", _fk_once)
        assert live._flush_messages_to_session_db(list(history) + [{"role": "user", "content": "two"}], history) is False
        assert [r["content"] for r in db.get_messages("sess-fk-live")] == ["one", "a1"]
        monkeypatch.undo()

        # Delegate child whose row is gone: a raising parent-row lookup fails the flush closed
        # instead of letting the OperationalError escape.
        db.create_session("sess-parent", source="cli")
        child = _make_agent(db, "sess-child")
        child._parent_session_id = "sess-parent"
        assert child._flush_messages_to_session_db([{"role": "user", "content": "x"}], []) is True
        db.delete_session("sess-parent")
        db.delete_session("sess-child")
        real_get = db.get_session

        def _parent_lookup_raises(sid):
            if sid == "sess-parent":
                raise _sqlite3.OperationalError("disk I/O error")
            return real_get(sid)

        monkeypatch.setattr(db, "get_session", _parent_lookup_raises)
        assert child._flush_messages_to_session_db([{"role": "user", "content": "y"}], []) is False
        db.close()
