"""Search storage on SQLite next to main's ``SessionDB`` on its own file, both written with the same rows: the FTS
layout and its storage version 3 after the conversation verbs write, the indexes the appended rows land in, the
route each query takes (FTS5, the trigram index for 3+ character substrings, the LIKE scan for short CJK
tokens), the fallback when an index is missing or marked stale, and the rebuild.
"""

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]

T0 = 1_700_000_000.0
# (session, source, role, content, tool_name): tool rows and cron sessions stay out of the substring indexes.
ROWS = (
    ("chat-1", "cli", "user", "payment question about invoices", None),
    ("chat-1", "cli", "assistant", "the concatenate helper", None),
    ("chat-1", "cli", "tool", "payment tool output", "terminal"),
    ("chat-2", "cli", "user", "迁移内容 完成了", None),
    ("chat-2", "cli", "assistant", "修改youer服务端", None),
    ("chat-2", "cli", "user", "广 字", None),
    ("cron-1", "cron", "user", "定时迁移内容 payment report", None),
)
# (query, the path main names for it without the CJK bigram index, whether hits carry FTS snippet markers).
ROUTES = (
    ("payment", "fts5", True),
    ("ayment", "fts5", True),  # a unicode61 miss, answered by the trigram retry
    ("concat", "fts5", True),
    ("invoices OR helper", "fts5", True),
    ("迁移内容", "trigram", True),
    ("迁移", "like_scan", False),
    ("广", "like_scan", False),
    ("zzz", "fts5", True),
)


@pytest.fixture
def stores(backend, scope_a, legacy_store):
    """Main's SessionDB and the backend's handle, each on its own file, written with ROWS in order."""
    from hermes_state import SessionDB

    m1_path = Path(os.environ["HERMES_HOME"]) / "m1" / "state.db"
    m1_path.parent.mkdir(parents=True, exist_ok=True)
    m1 = SessionDB(db_path=m1_path)
    store = backend.open(scope_a)
    try:
        for index, (session_id, source, role, content, tool_name) in enumerate(ROWS):
            if m1.get_session(session_id) is None:
                m1.create_session(session_id, source, model="model-a")
                store.sessions.create(NewSession(session_id=session_id, source=source, model="model-a"))
            m1.append_message(session_id, role, content=content, tool_name=tool_name, timestamp=T0 + index)
            store.messages.append(session_id, NewMessage(role=role, content=content, tool_name=tool_name,
                                                         timestamp=T0 + index))
        yield m1, store, m1_path, store_for(scope_a)
    finally:
        m1.close()


def _layout(path: Path):
    """The FTS objects of a store with their stored SQL, and its storage-version stamp."""
    from hermes_state_fts import fts_layout_current

    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        objects = sorted(
            (kind, name, " ".join((sql or "").split()))
            for kind, name, sql in conn.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE name LIKE 'messages_fts%'"))
        stamp = conn.execute("SELECT value FROM state_meta WHERE key = 'fts_storage_version'").fetchone()
        current = fts_layout_current(conn.cursor())
    return objects, stamp, current


def _indexed(path: Path):
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL TABLE%'")]
        return {table: sorted(row[0] for row in conn.execute(f'SELECT id FROM "{table}_docsize"')) for table in tables}


def _raw(path: Path, sql: str, params=()):
    with closing(sqlite3.connect(str(path), isolation_level=None)) as conn:
        conn.execute(sql, params)


def _m1_hits(m1, query):
    return [(row["id"], row["session_id"], row["role"], row["snippet"]) for row in m1.search_messages(query)]


def _backend_hits(store, query):
    return [(hit.message_id, hit.session_id, hit.role, hit.snippet) for hit in store.search.search_messages(query)]


def test_conversation_writes_keep_storage_version_3(stores):
    from hermes_state_common import FTS_STORAGE_VERSION

    m1, store, m1_path, path = stores
    store.sessions.set_title("chat-1", "Payments")
    store.sessions.end("chat-2", "user_exit")
    m1.set_session_title("chat-1", "Payments")
    m1.end_session("chat-2", "user_exit")
    objects, stamp, current = _layout(path)
    assert (objects, stamp, current) == _layout(m1_path)
    assert stamp == (str(FTS_STORAGE_VERSION),) and FTS_STORAGE_VERSION == 3 and current
    names = {name for _, name, _ in objects}
    assert {"messages_fts", "messages_fts_trigram", "messages_fts_insert", "messages_fts_trigram_insert"} <= names


def test_appended_rows_land_in_the_indexes_m1_fills(stores):
    _, store, m1_path, path = stores
    indexed = _indexed(path)
    assert indexed == _indexed(m1_path)
    ids = {(row.role, row.content): row.id for session_id in ("chat-1", "chat-2", "cron-1")
           for row in store.messages.read_transcript(session_id)}
    dialogue = {ids[(role, content)] for session_id, source, role, content, _ in ROWS
                if role != "tool" and source != "cron"}
    assert set(ids.values()) <= set(indexed["messages_fts"])
    assert dialogue <= set(indexed["messages_fts_trigram"])


@pytest.mark.parametrize(("query", "path_name", "marked"), ROUTES, ids=[query for query, _, _ in ROUTES])
def test_search_takes_m1s_route(stores, query, path_name, marked):
    m1, store, _, _ = stores
    if m1._fts_cjk_available and path_name in ("trigram", "like_scan") and query != "广":
        path_name, marked = "fts_cjk", True
    assert m1._describe_search_path(query) == path_name
    hits = _backend_hits(store, query)
    assert hits == _m1_hits(m1, query)
    assert bool(hits) == (query != "zzz")
    assert all((">>>" in snippet) == marked for _, _, _, snippet in hits)


def test_a_stale_mark_takes_the_like_fallback(stores):
    from hermes_state_common import FTS_STALE_KEY

    m1, store, m1_path, path = stores
    for db in (m1_path, path):
        _raw(db, "INSERT INTO state_meta (key, value) VALUES (?, '1')", (FTS_STALE_KEY,))
    for query in ("payment", "迁移内容", "ayment"):
        hits = _backend_hits(store, query)
        assert hits == _m1_hits(m1, query)
        assert hits and not any(">>>" in snippet for _, _, _, snippet in hits)
    assert m1._describe_search_path("payment") == "like_scan_fts_stale"


def test_a_missing_index_takes_m1s_fallback(stores):
    m1, store, m1_path, path = stores
    for db in (m1_path, path):
        _raw(db, "DROP TABLE messages_fts_trigram")
    expected = {query: _m1_hits(m1, query) for query in ("迁移内容", "ayment", "payment")}
    assert {query: _backend_hits(store, query) for query in expected} == expected
    assert expected["迁移内容"] and not any(">>>" in snippet for _, _, _, snippet in expected["迁移内容"])
    assert expected["ayment"] == [] and expected["payment"]

    for db in (m1_path, path):
        _raw(db, "DROP TABLE messages_fts")
    assert _backend_hits(store, "payment") == _m1_hits(m1, "payment") == []


def test_rebuild_returns_m1s_count_and_keeps_layout_3(stores):
    from hermes_state_common import FTS_STORAGE_VERSION

    m1, store, m1_path, path = stores
    before = {query: _backend_hits(store, query) for query, _, _ in ROUTES}
    assert store.search.rebuild_search_index() == m1.rebuild_fts() > 0
    assert _layout(path) == _layout(m1_path)
    assert _layout(path)[1:] == ((str(FTS_STORAGE_VERSION),), True)
    assert _indexed(path) == _indexed(m1_path)
    assert {query: _backend_hits(store, query) for query, _, _ in ROUTES} == before
    assert store.search.search_index_rebuild_status() is None and m1.fts_rebuild_status() is None
