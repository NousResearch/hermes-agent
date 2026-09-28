"""Lossless storage deduplication of plaintext reasoning (#125273)."""
import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("writer", ["single", "batch", "replace"])
def test_shared_reasoning_stored_once_and_restored(tmp_path, writer):
    path = tmp_path / "state.db"
    text = "A sufficiently long reasoning trace. " * 100
    cases = [(text, text), (text, None), (None, text), (text, "different"),
             ("", ""), (text, ""), (None, None)]
    messages = [dict(role="assistant", content="answer", reasoning=a, reasoning_content=b)
                for a, b in cases]
    with SessionDB(db_path=path) as db:
        db.create_session(session_id="s", source="cli")
        if writer == "single":
            for msg in messages:
                db.append_message("s", **msg)
        elif writer == "batch":
            db.append_messages_batch("s", messages)
        else:
            db.replace_messages("s", messages)
        with sqlite3.connect(path) as conn:
            stored = conn.execute("SELECT reasoning, reasoning_content FROM messages ORDER BY id").fetchall()
        assert sum(len(v) for v in stored[0] if v is not None) == len(text)
    with SessionDB(db_path=path) as db:
        rows = db.get_messages("s")
        assert [(r["reasoning"], r["reasoning_content"]) for r in rows] == cases
        replay = db.get_messages_as_conversation("s")
        for row, (a, b) in zip(replay, cases):
            assert row.get("reasoning") == (a or None)
            assert ("reasoning_content" in row) == (b is not None)
            assert row.get("reasoning_content") == b
        db.replace_messages("s", rows)
        assert [(r["reasoning"], r["reasoning_content"]) for r in db.get_messages("s")] == cases
        assert all("reasoning_shared" not in r for r in rows)
        exported = db.export_all()
    with SessionDB(db_path=tmp_path / "imported.db") as restored:
        restored.import_sessions(exported)
        assert [(r["reasoning"], r["reasoning_content"]) for r in restored.get_messages("s")] == cases


def test_legacy_schema_and_concurrent_compaction_tail(tmp_path):
    path = tmp_path / "legacy.db"
    with SessionDB(db_path=path) as db:
        db.create_session(session_id="s", source="cli")
        db.append_message("s", role="user", content="question")
        # A pre-dedup store has ordinary independent columns and no flag.
        with sqlite3.connect(path) as conn:
            conn.execute("ALTER TABLE messages DROP COLUMN reasoning_shared")
            conn.execute("INSERT INTO messages (session_id, role, content, reasoning, reasoning_content, timestamp) "
                         "VALUES ('s', 'assistant', 'legacy', 'old', 'old', 1)")
    with SessionDB(db_path=path) as db:
        assert db.get_messages_as_conversation("s")[-1]["reasoning"] == "old"
        watermark = db.get_active_message_watermark("s")
        db.append_message("s", role="assistant", content="concurrent", reasoning="new", reasoning_content="new")
        db.archive_and_compact("s", [{"role": "user", "content": "summary"}], watermark=watermark)
        row = db.get_messages("s")[-1]
        assert (row["reasoning"], row["reasoning_content"]) == ("new", "new")
        assert db.get_messages_as_conversation("s")[-1]["reasoning"] == "new"
        with sqlite3.connect(path) as conn:
            raw = conn.execute("SELECT reasoning, reasoning_content FROM messages WHERE active = 1 AND role = 'assistant'").fetchone()
        assert raw == (None, "new")
