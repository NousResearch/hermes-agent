"""Storage-refactor boundaries for #125273: import, raw clones, dump and blank/non-assistant pads."""

import sqlite3
from types import SimpleNamespace

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    append_message, archive_and_compact, get_active_message_watermark, get_messages,
)
from persistence.sqlite.conversation._store.sessions import create_session
from persistence.sqlite.housekeeping._portability import SessionPortability

TEXT = "consider both branches"


@pytest.mark.parametrize("role,reasoning,content,stored,restored", [
    ("assistant", TEXT, TEXT, None, TEXT),
    ("assistant", None, TEXT, None, TEXT),
    ("assistant", TEXT, "different", TEXT, TEXT),
    ("assistant", " ", " ", " ", " "),
    ("assistant", "\n\n", "\n\n", "\n\n", "\n\n"),
    ("assistant", "", "", "", ""),
    ("assistant", None, " ", None, None),
    ("user", TEXT, TEXT, TEXT, TEXT),
])
def test_only_nonblank_assistant_pairs_fold(tmp_path, role, reasoning, content, stored, restored):
    with StoreHolder(tmp_path / "state.db") as db:
        create_session(db, "s", source="cli")
        append_message(db, "s", role=role, content="reply", reasoning=reasoning, reasoning_content=content)
        with db._read_ctx() as conn:
            row = conn.execute("SELECT reasoning, reasoning_content FROM messages").fetchone()
        assert tuple(row) == (stored, content)
        message = get_messages(db, "s")[0]
        assert (message["reasoning"], message["reasoning_content"]) == (restored, content)


def test_portability_import_and_bulk_export_use_the_codec(tmp_path):
    with StoreHolder(tmp_path / "state.db") as db:
        portability = SessionPortability()
        portability._store = SimpleNamespace(holder=lambda: db)
        result = portability.import_sessions([{
            "id": "s", "source": "cli",
            "messages": [{"role": "assistant", "content": "answer", "reasoning": TEXT, "reasoning_content": TEXT}],
        }])
        assert result.ok and result.imported == 1
        with db._read_ctx() as conn:
            row = conn.execute("SELECT reasoning, reasoning_content FROM messages").fetchone()
        assert tuple(row) == (None, TEXT)
        exported = portability.export_all()[0].messages[0]
        assert exported.reasoning == exported.reasoning_content == TEXT


def test_concurrent_sql_clone_preserves_legacy_payload_and_remains_readable(tmp_path):
    """Raw SQL clones are deliberately NOT a backfill: preserve opaque and future columns verbatim."""
    with StoreHolder(tmp_path / "state.db") as db:
        create_session(db, "s", source="cli")
        append_message(db, "s", role="user", content="ask")
        watermark = get_active_message_watermark(db, "s")
        append_message(db, "s", role="assistant", content="concurrent", reasoning=TEXT, reasoning_content=TEXT)
        db._execute_write(lambda conn: conn.execute("UPDATE messages SET reasoning = reasoning_content WHERE role = 'assistant'"))
        archive_and_compact(db, "s", [{"role": "user", "content": "summary"}], watermark=watermark)
        with db._read_ctx() as conn:
            copies = conn.execute("SELECT reasoning, reasoning_content FROM messages WHERE role = 'assistant' ORDER BY id").fetchall()
        assert [tuple(row) for row in copies] == [(TEXT, TEXT), (TEXT, TEXT)]
        assert get_messages(db, "s")[-1]["reasoning"] == TEXT


def test_sqlite_dump_preserves_null_without_a_marker(tmp_path):
    with StoreHolder(tmp_path / "state.db") as db:
        create_session(db, "s", source="cli")
        append_message(db, "s", role="assistant", content="answer", reasoning=TEXT, reasoning_content=TEXT)
        with db._read_ctx() as conn:
            # Dump the actual message pair through a plain SQLite table. The store's external-content
            # FTS tables need their own rebuild on whole-store restoration; that is unrelated to the codec.
            pair = conn.execute("SELECT reasoning, reasoning_content FROM messages").fetchone()
        with sqlite3.connect(":memory:") as donor:
            donor.execute("CREATE TABLE pair(reasoning TEXT, reasoning_content TEXT)")
            donor.execute("INSERT INTO pair VALUES(?, ?)", tuple(pair))
            dump = "\n".join(donor.iterdump())
        with sqlite3.connect(":memory:") as restored:
            restored.executescript(dump)
            pair = restored.execute("SELECT reasoning, reasoning_content FROM pair").fetchone()
        assert pair == (None, TEXT)
