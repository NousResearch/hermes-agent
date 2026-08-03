"""Tests for SessionDB.append_messages_batch (#23254 salvage).

The batch writer must be row-shape-identical to append_message (shared
_prepare_message_row + _MESSAGE_INSERT_SQL), atomic (all rows or none),
and must aggregate the session counters in one UPDATE.
"""

import json
import sqlite3

import pytest

from hermes_state import (
    CompressionSessionClosedError,
    SessionDB,
)


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

    def test_counters_aggregate_once(self, db):
        db.append_messages_batch("sess-batch", _turn_messages())
        row = db._conn.execute(
            "SELECT message_count, tool_call_count FROM sessions WHERE id = ?",
            ("sess-batch",),
        ).fetchone()
        assert row["message_count"] == 4
        assert row["tool_call_count"] == 1

    def test_returns_row_ids_in_input_order(self, db):
        ids = db.append_messages_batch("sess-batch", _turn_messages())
        assert ids == sorted(ids)
        assert len(ids) == 4

    def test_empty_batch_is_noop(self, db):
        assert db.append_messages_batch("sess-batch", []) == []
        row = db._conn.execute(
            "SELECT message_count FROM sessions WHERE id = ?", ("sess-batch",)
        ).fetchone()
        assert row["message_count"] == 0

    def test_atomicity_all_or_nothing(self, db, monkeypatch):
        """A failure mid-batch leaves ZERO rows and untouched counters."""
        real_execute_write = db._execute_write
        original_insert = SessionDB._MESSAGE_INSERT_SQL

        calls = {"n": 0}

        def _do_wrapper(fn, **kwargs):
            def failing(conn):
                real_conn_execute = conn.execute

                def exec_counting(sql, *args):
                    if sql == original_insert:
                        calls["n"] += 1
                        if calls["n"] == 3:
                            raise sqlite3.OperationalError("boom mid-batch")
                    return real_conn_execute(sql, *args)

                conn.execute = exec_counting
                try:
                    return fn(conn)
                finally:
                    conn.execute = real_conn_execute

            return real_execute_write(failing, **kwargs)

        monkeypatch.setattr(db, "_execute_write", _do_wrapper)
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
