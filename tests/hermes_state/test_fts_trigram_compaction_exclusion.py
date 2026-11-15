"""Compacted/inactive messages (active = 0) stay out of the trigram FTS index.

When context compaction runs or turns are rewound, historical messages are
marked with active = 0. These rows remain in `messages` and the standard
`messages_fts` index, but are excluded from `messages_fts_trigram_src` so the
trigram inverted index (messages_fts_trigram_data) does not bloat.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    if not session_db._trigram_available:
        session_db.close()
        pytest.skip("trigram tokenizer unavailable in this SQLite build")
    yield session_db
    session_db.close()


def _trigram_rowids(db: SessionDB) -> set[int]:
    return {
        row[0]
        for row in db._conn.execute("SELECT id FROM messages_fts_trigram_docsize").fetchall()
    }


def test_inactive_compaction_messages_excluded_from_trigram_src(db: SessionDB):
    db.create_session("sess1", source="desktop")
    m1 = db.append_message("sess1", role="user", content="active message content 一二三")
    m2 = db.append_message("sess1", role="assistant", content="compacted dead message content 四五六")

    # Mark m2 as inactive (as done by compaction or rewind)
    db._conn.execute("UPDATE messages SET active = 0, compacted = 1 WHERE id = ?", (m2,))
    db._conn.commit()

    # Query the view directly
    rows = db._conn.execute("SELECT id FROM messages_fts_trigram_src WHERE id IN (?, ?)", (m1, m2)).fetchall()
    view_ids = {r[0] for r in rows}

    assert m1 in view_ids, "Active message should be present in messages_fts_trigram_src"
    assert m2 not in view_ids, "Inactive compaction message must be excluded from messages_fts_trigram_src"
