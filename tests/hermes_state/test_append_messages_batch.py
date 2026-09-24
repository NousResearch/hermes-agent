"""Tests for SessionDB.append_messages_batch (#23254 salvage).

The batch writer reuses _insert_message_rows (the same row-serialization
path as replace/compact/import), runs the same admission guards as
append_message, is atomic (all rows or none), and aggregates the session
counters in one UPDATE.
"""

import json
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_errors import CompressionSessionClosedError


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-batch", source="cli")
    yield d
    d.close()


def _turn_messages():
    return [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [{"name": "terminal", "arguments": "{}"}],
            "reasoning_content": "thinking...",
            "finish_reason": "tool_calls",
        },
        {
            "role": "tool",
            "content": "tool output",
            "tool_name": "terminal",
            "tool_call_id": "call_1",
        },
        {"role": "assistant", "content": "answer", "finish_reason": "stop"},
    ]


class TestAppendMessagesBatch:
    def test_batch_rows_identical_to_single_appends(self, db, tmp_path):
        """The batch writer stores the same bytes append_message would."""
        db2 = SessionDB(db_path=tmp_path / "state2.db")
        db2.create_session("sess-batch", source="cli")
        try:
            msgs = _turn_messages()
            db.append_messages_batch("sess-batch", msgs)
            for m in msgs:
                role = m["role"]
                db2.append_message(
                    session_id="sess-batch",
                    role=role,
                    content=m.get("content"),
                    tool_name=m.get("tool_name"),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                    finish_reason=m.get("finish_reason"),
                    reasoning_content=(
                        m.get("reasoning_content") if role == "assistant" else None
                    ),
                )
            cols = (
                "role, content, tool_call_id, tool_calls, tool_name, "
                "finish_reason, reasoning_content, observed, active"
            )
            rows_a = db._conn.execute(
                f"SELECT {cols} FROM messages ORDER BY id"
            ).fetchall()
            rows_b = db2._conn.execute(
                f"SELECT {cols} FROM messages ORDER BY id"
            ).fetchall()
            assert [tuple(r) for r in rows_a] == [tuple(r) for r in rows_b]
        finally:
            db2.close()

    def test_reasoning_gated_to_assistant_rows(self, db):
        """_insert_message_rows role-gates reasoning fields; a tool row
        carrying reasoning keys must not persist them."""
        db.append_messages_batch(
            "sess-batch",
            [
                {
                    "role": "tool",
                    "content": "out",
                    "tool_name": "t",
                    "tool_call_id": "c1",
                    "reasoning_content": "should not persist",
                }
            ],
        )
        row = db._conn.execute(
            "SELECT reasoning_content FROM messages"
        ).fetchone()
        assert row[0] is None

    def test_counters_aggregate_once(self, db):
        db.append_messages_batch("sess-batch", _turn_messages())
        row = db._conn.execute(
            "SELECT message_count, tool_call_count FROM sessions WHERE id = ?",
            ("sess-batch",),
        ).fetchone()
        assert row["message_count"] == 4
        assert row["tool_call_count"] == 1

    def test_returns_inserted_count(self, db):
        assert db.append_messages_batch("sess-batch", _turn_messages()) == 4

    def test_empty_batch_is_noop(self, db):
        assert db.append_messages_batch("sess-batch", []) == 0
        row = db._conn.execute(
            "SELECT message_count FROM sessions WHERE id = ?", ("sess-batch",)
        ).fetchone()
        assert row["message_count"] == 0

    def test_atomicity_all_or_nothing(self, db, monkeypatch):
        """A failure mid-batch leaves ZERO rows and untouched counters."""
        real_insert = SessionDB._insert_message_rows

        def failing_insert(self_db, conn, session_id, messages):
            real_conn_execute = conn.execute
            calls = {"n": 0}

            def exec_counting(sql, *args):
                if sql.lstrip().startswith("INSERT INTO messages"):
                    calls["n"] += 1
                    if calls["n"] == 3:
                        raise sqlite3.OperationalError("boom mid-batch")
                return real_conn_execute(sql, *args)

            conn.execute = exec_counting
            try:
                return real_insert(self_db, conn, session_id, messages)
            finally:
                conn.execute = real_conn_execute

        monkeypatch.setattr(SessionDB, "_insert_message_rows", failing_insert)
        with pytest.raises(sqlite3.OperationalError):
            db.append_messages_batch("sess-batch", _turn_messages())
        monkeypatch.undo()

        count = db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count == 0
        row = db._conn.execute(
            "SELECT message_count, tool_call_count FROM sessions WHERE id = ?",
            ("sess-batch",),
        ).fetchone()
        assert row["message_count"] == 0
        assert row["tool_call_count"] == 0

    def test_compression_closed_session_rejected(self, db):
        db._conn.execute(
            "UPDATE sessions SET ended_at = 1.0, end_reason = 'compression' "
            "WHERE id = ?",
            ("sess-batch",),
        )
        db._conn.commit()
        with pytest.raises(CompressionSessionClosedError):
            db.append_messages_batch("sess-batch", _turn_messages())

    def test_multimodal_content_encoded(self, db):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:x"}},
                ],
            }
        ]
        db.append_messages_batch("sess-batch", msgs)
        raw = db._conn.execute("SELECT content FROM messages").fetchone()[0]
        # encoded via _encode_content — same sentinel prefix as append_message
        loaded = db.get_messages("sess-batch")
        assert loaded, raw

    def test_tool_calls_json_string_not_double_encoded(self, db):
        msgs = [
            {
                "role": "assistant",
                "content": "x",
                "tool_calls": json.dumps([{"name": "t", "arguments": "{}"}]),
            }
        ]
        db.append_messages_batch("sess-batch", msgs)
        raw = db._conn.execute("SELECT tool_calls FROM messages").fetchone()[0]
        assert json.loads(raw) == [{"name": "t", "arguments": "{}"}]


class TestShadowedCheckpointRowsArePruned:
    """Under native compaction every assistant response persists a fresh ``type: "compaction"`` checkpoint
    and local compaction (the only other prune site) rarely fires, so older rows kept ~120 KB of ciphertext
    the wire builder never replays (#102374). Landing a newer carrier row rewrites the older active rows."""

    @staticmethod
    def _checkpoint(tag):
        return {"type": "compaction", "encrypted_content": f"ckpt-{tag}"}

    @staticmethod
    def _reasoning(tag):
        return {"type": "reasoning", "encrypted_content": f"rs-{tag}", "id": f"rs_{tag}"}

    @staticmethod
    def _agent(db):
        from agent.session_persistence import SessionPersistenceMixin

        class _Agent(SessionPersistenceMixin):
            pass

        agent = _Agent()
        agent._session_db, agent._session_db_created, agent.session_id = db, True, "sess-batch"
        agent._last_flushed_db_idx, agent._flushed_db_message_ids = 0, set()
        agent._flushed_db_message_session_id, agent._persist_disabled = None, False
        return agent

    def _durable_items(self, db):
        return [
            (row["id"], json.loads(row["codex_reasoning_items"]) if row["codex_reasoning_items"] else None)
            for row in db._conn.execute(
                "SELECT id, codex_reasoning_items FROM messages WHERE session_id = ? AND role = 'assistant' "
                "AND active = 1 ORDER BY id", ("sess-batch",)).fetchall()
        ]

    def test_newer_carrier_row_prunes_the_older_rows_checkpoints(self, db):
        """The production flush path: turn 1 lands a carrier, turn 2 lands a newer one -> only the newest
        row still holds a checkpoint, durably and in the live transcript; reasoning items are untouched."""
        agent = self._agent(db)
        messages = [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0", "codex_reasoning_items": [self._reasoning(0), self._checkpoint(0)]},
        ]
        assert agent._flush_messages_to_session_db(messages) is True
        assert self._durable_items(db) == [(2, [self._reasoning(0), self._checkpoint(0)])]

        messages += [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1", "codex_reasoning_items": [self._reasoning(1), self._checkpoint(1)]},
        ]
        assert agent._flush_messages_to_session_db(messages) is True

        assert self._durable_items(db) == [
            (2, [self._reasoning(0)]),
            (4, [self._reasoning(1), self._checkpoint(1)]),
        ]
        # The live dicts match the rows they were persisted as (the marker contract), so no re-write is queued.
        assert messages[1]["codex_reasoning_items"] == [self._reasoning(0)]
        assert messages[3]["codex_reasoning_items"] == [self._reasoning(1), self._checkpoint(1)]
        assert agent._flush_messages_to_session_db(messages) is True
        assert db.message_count("sess-batch") == 4
