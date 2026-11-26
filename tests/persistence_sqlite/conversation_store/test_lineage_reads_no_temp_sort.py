"""Compression-lineage reads return rows in id order without a temp b-tree.

``session_id IN (a, b, ...) ORDER BY id`` has no index in that order, so SQLite copied every selected
column of the whole lineage into a sorter that spills to TMPDIR once it outgrows the page cache. On a
15-segment lineage one 120-row Desktop transcript page (``GET /api/sessions/<id>/messages``) wrote about
170 MB, and the gateway wrote 137 GB in seven hours. The reads now select the ids first and fetch rows by
rowid (``_lineage_rows_sql``), which returns the same rows in the same order.
"""

import re
import sqlite3

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    _resume_lineage_ids,
    append_messages_batch,
    get_ancestor_display_prefix,
    get_messages,
    get_messages_as_conversation,
    get_resume_conversations,
)
from persistence.sqlite.conversation._store.sessions import create_session, end_session
from persistence.sqlite.conversation._store.stored_reads import display_generations

_TIP = "seg2"

_READERS = {
    "get_messages active": lambda db: get_messages(db, _TIP, include_ancestors=True),
    "get_messages display": lambda db: get_messages(db, _TIP, include_ancestors=True, include_compacted=True),
    "get_messages audit": lambda db: get_messages(db, _TIP, include_ancestors=True, include_inactive=True),
    "conversation": lambda db: get_messages_as_conversation(db, _TIP, include_ancestors=True),
    "conversation display": lambda db: get_messages_as_conversation(
        db, _TIP, include_ancestors=True, include_compacted=True),
    "resume": lambda db: get_resume_conversations(db, _TIP),
    "ancestor prefix": lambda db: get_ancestor_display_prefix(db, _TIP),
    "transcript generations": lambda db: display_generations(db, _TIP, _resume_lineage_ids(db, _TIP)),
}


def _lineage_db(path, *, analyze=False):
    db = StoreHolder(path)
    parent = None
    for seg in range(3):
        sid = f"seg{seg}"
        create_session(db, sid, "tui", parent_session_id=parent)
        append_messages_batch(db, sid, [
            {"role": "user" if j % 2 == 0 else "assistant", "content": f"{sid} message {j}", "timestamp": 1_000 + j}
            for j in range(12)
        ])
        if seg < 2:
            end_session(db, sid, "compression")
        parent = sid
    db.close()
    if analyze:  # planner statistics must not bring the sort back
        raw = sqlite3.connect(path)
        raw.execute("ANALYZE")
        raw.commit()
        raw.close()


def _traced_reads(path, monkeypatch, reader):
    statements = []
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    db = StoreHolder(path)
    statements.clear()
    result = reader(db)
    db.close()
    monkeypatch.undo()
    return result, statements


@pytest.mark.parametrize("analyze", [False, True], ids=["no-stats", "analyzed"])
@pytest.mark.parametrize("reader", sorted(_READERS))
def test_lineage_reads_need_no_temp_btree(tmp_path, monkeypatch, reader, analyze):
    path = tmp_path / "state.db"
    _lineage_db(path, analyze=analyze)
    result, statements = _traced_reads(path, monkeypatch, _READERS[reader])
    assert result

    lineage_reads = [s for s in statements
                     if re.match(r"\s*SELECT\b", s, re.IGNORECASE) and re.search(r"session_id IN \('seg0', ?'seg1', ?'seg2'\)", s)]
    assert lineage_reads, statements
    plan_conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    plans = {sql: [row[3] for row in plan_conn.execute("EXPLAIN QUERY PLAN " + sql)] for sql in lineage_reads}
    plan_conn.close()
    sorted_reads = {sql: plan for sql, plan in plans.items() if any("TEMP B-TREE" in step for step in plan)}
    assert not sorted_reads, sorted_reads


def test_lineage_read_returns_every_segment_in_id_order(tmp_path):
    path = tmp_path / "state.db"
    _lineage_db(path)
    db = StoreHolder(path)
    rows = get_messages(db, _TIP, include_ancestors=True)
    db.close()
    assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)
    assert [row["content"] for row in rows] == [f"seg{s} message {j}" for s in range(3) for j in range(12)]
