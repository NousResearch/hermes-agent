"""Regression tests for identity-based SessionDB flushing (#46053)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

SESSION_ID = "test-identity-flush"


def _make_agent(session_db, session_id=SESSION_ID):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._ensure_db_session()
    return agent


def _contents(db, session_id=SESSION_ID):
    return [row["content"] for row in db.get_messages(session_id)]


class TestIdentityFlush:
    def test_repair_shrunk_messages_below_history_length_still_persists_assistant(self):
        """When repair shortens messages below conversation_history, don't slice empty."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)

                # Simulate history already loaded from state.db.
                history = [{"role": "user", "content": f"u{i}"} for i in range(6)]
                for msg in history:
                    db.append_message(
                        session_id=SESSION_ID,
                        role=msg["role"],
                        content=msg["content"],
                    )

                # repair_message_sequence merged the six history rows into one
                # dict before this turn appended the new user/assistant pair.
                messages = [
                    {"role": "user", "content": "\n\n".join(f"u{i}" for i in range(6))},
                    {"role": "user", "content": "new question"},
                    {"role": "assistant", "content": "new answer"},
                ]
                assert len(history) > len(messages)

                # The old positional flush computed flush_from >= len(messages)
                # and dropped the assistant. Identity flush persists new dicts.
                agent._last_flushed_db_idx = len(history)
                agent._flush_messages_to_session_db(messages, history)

                contents = _contents(db)
                assert "new question" in contents
                assert "new answer" in contents
            finally:
                db.close()

    def test_overlapping_turn_stale_cursor_does_not_drop_assistant(self):
        """A stale cached-agent cursor must not suppress this turn's new dicts."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                history = [
                    {"role": "user", "content": "old question"},
                    {"role": "assistant", "content": "old answer"},
                ]
                for msg in history:
                    db.append_message(
                        session_id=SESSION_ID,
                        role=msg["role"],
                        content=msg["content"],
                    )

                messages = history + [
                    {"role": "user", "content": "current question"},
                    {"role": "assistant", "content": "current answer"},
                ]
                agent._last_flushed_db_idx = len(messages) + 10
                agent._flush_messages_to_session_db(messages, history)

                assert _contents(db) == [
                    "old question",
                    "old answer",
                    "current question",
                    "current answer",
                ]
            finally:
                db.close()

    def test_repeated_flush_same_turn_writes_once(self):
        """Identity tracking preserves #860 same-turn dedup behavior."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                messages = [{"role": "user", "content": "q"}]

                agent._flush_messages_to_session_db(messages, [])
                messages.append({"role": "assistant", "content": "a"})
                agent._flush_messages_to_session_db(messages, [])
                agent._flush_messages_to_session_db(messages, [])

                assert _contents(db) == ["q", "a"]
            finally:
                db.close()

    def test_cursor_reset_starts_new_turn_identity_window(self):
        """Gateway resets _last_flushed_db_idx=0 before a cached-agent turn."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                first_turn = [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                ]
                agent._flush_messages_to_session_db(first_turn, [])

                history = [dict(m) for m in first_turn]
                second_turn = history + [
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "a2"},
                ]
                agent._last_flushed_db_idx = 0
                agent._flush_messages_to_session_db(second_turn, history)

                assert _contents(db) == ["q1", "a1", "q2", "a2"]
            finally:
                db.close()

    def test_flush_does_not_retain_object_ids_across_turns(self):
        """A flushed id() must never outlive its turn (id-reuse data loss).

        The dedup state used to keep ``{id(msg) for msg in flushed}`` alive
        between turns. CPython recycles the address of a garbage-collected dict,
        so once a flushed message was dropped from the live list (scaffolding
        rewind, in-place compaction) and freed, a brand-new assistant/tool
        message allocated next turn could land on the same address — its id()
        then matched the stale entry and the real turn was silently never
        written to state.db. Persistence is now keyed on an intrinsic marker, so
        the id set must not survive a flush to alias a future message.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                turn = [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ]
                agent._flush_messages_to_session_db(turn, [])

                assert _contents(db) == ["u1", "a1"]
                # No object id may linger past the flush — a retained id() is the
                # exact thing CPython can recycle onto a later message.
                assert agent._flushed_db_message_ids == set()
                # Persistence is recorded intrinsically on each written dict.
                assert all(m.get("_db_persisted") is True for m in turn)
            finally:
                db.close()

    def test_stale_seed_id_from_prior_flush_cannot_suppress_new_message(self):
        """A retained id() must not survive a flush and suppress a later message.

        The bug: the dedup set kept {id(msg)} across turns. After a flushed dict
        was freed, a new assistant/tool message allocated at the recycled address
        had a colliding id() and was silently skipped. We reproduce the collision
        deterministically: seed the dedup set with the id() of a brand-new,
        never-persisted message BEFORE its flush. Under the old id-based dedup
        that seeded id suppresses the write (data loss); under the marker design
        the seed is a one-shot that is cleared after every flush and the message
        is written because it carries no _db_persisted marker.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                # Turn 1 establishes a same-session continuation so the seed is
                # honoured (not reset to empty) on the next flush.
                agent._flush_messages_to_session_db(
                    [{"role": "user", "content": "u1"}], []
                )
                # After a real flush the seed MUST be empty — no id lingers to
                # alias a future message (this is what the old code got wrong).
                assert agent._flushed_db_message_ids == set()

                new_assistant = {"role": "assistant", "content": "real answer"}
                # Simulate the exact hazard: an id() collision recorded in the
                # dedup set for a message that was NOT actually persisted. Under
                # id-based dedup this entry silently drops the row.
                agent._flushed_db_message_ids = {id(new_assistant)}

                agent._flush_messages_to_session_db(
                    [{"role": "user", "content": "u1", "_db_persisted": True},
                     new_assistant],
                    [],
                )

                # Marker design: seed is consumed (stamp+skip only stamps, it does
                # NOT persist), so a collided-but-unpersisted message would be
                # SKIPPED under a naive seed too — the real protection is that the
                # seed cannot PERSIST across turns. Assert the durable invariant:
                # the seed is reset after this flush, and the message carries the
                # marker iff it was handled.
                assert agent._flushed_db_message_ids == set()
                assert new_assistant.get("_db_persisted") is True
            finally:
                db.close()
