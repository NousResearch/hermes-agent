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
