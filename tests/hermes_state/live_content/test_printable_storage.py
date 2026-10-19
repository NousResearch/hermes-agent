"""Actual file content survives SQLite writes, FTS mirrors, and a database reopen."""

import gzip
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("operation", ["append", "replace"])
def test_real_text_and_compressed_bytes_are_printable_in_both_fts_indexes(tmp_path, operation):
    root = Path(__file__).resolve().parents[3]
    text = (root / "README.md").read_text(encoding="utf-8")[:512]
    search_term = re.findall(r"[A-Za-z]{4,}", text)[0]
    structured = [{"type": "text", "text": text}]
    compressed = gzip.compress(text.encode("utf-8"))
    rows = [{"role": "user", "content": structured}, {"role": "assistant", "content": compressed}]
    path = tmp_path / "state.db"
    sid = str(uuid4())
    db = SessionDB(db_path=path)
    try:
        db.create_session(sid, source="cli")
        if operation == "replace":
            db.append_message(sid, "user", text)
            db.replace_messages(sid, rows)
        else:
            for row in rows:
                db.append_message(sid, **row)
        persisted = db.get_messages(sid)
        assert [row["content"] for row in persisted] == [structured, compressed]
        assert persisted[0]["id"] in {row["id"] for row in db.search_messages(search_term)}
    finally:
        db.close()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        available = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"messages_fts", "messages_fts_trigram"}.issubset(available)
        for record in persisted:
            raw = conn.execute("SELECT content FROM messages WHERE id=?", (record["id"],)).fetchone()[0]
            assert isinstance(raw, str) and "\x00" not in raw
            for table in ("messages_fts", "messages_fts_trigram"):
                mirrored = conn.execute("SELECT content FROM " + table + " WHERE rowid=?", (record["id"],)).fetchone()
                assert mirrored is not None and mirrored[0] == raw
    reopened = SessionDB(db_path=path)
    try:
        replay = reopened.get_messages_as_conversation(sid)
    finally:
        reopened.close()
    assert [row["content"] for row in replay] == [structured, compressed]
    assert gzip.decompress(replay[1]["content"]).decode("utf-8") == text
