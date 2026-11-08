"""``descendants`` and ``list_gateway_sessions`` on SQLite return the rows the legacy statements return, in the only
order those statements define: none for the descendants walk, last_active descending (ties unordered) for the gateway
listing."""

from __future__ import annotations

import sqlite3
from collections import Counter
from itertools import groupby

from persistence.conversation.sessions import NewSession
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]


def _descendants_sql() -> str:
    """Main's descendants statement (hermes_cli/web_server_sessions.py at main), the reference the walk is compared
    with, built from main's legacy store."""
    from hermes_state_common import _RESET_CHILD_SQL, _sql_json_extract

    return f"""
            WITH RECURSIVE descendants(id, parent_session_id, started_at) AS (
                SELECT id, parent_session_id, started_at FROM sessions WHERE id = ?
                UNION
                SELECT s.id, s.parent_session_id, s.started_at
                FROM sessions s
                JOIN descendants d ON s.parent_session_id = d.id
                -- Continuation edges only (same predicate as the session list's chain CTE): a subagent run,
                -- a /branch fork, a /new reset child or a tool-owned row is its own conversation, and resuming
                -- INTO one parks the user's chat in a row the sidebar never lists (#115092).
                WHERE {_sql_json_extract('s.model_config', '$._delegate_from')} IS NULL
                  AND {_sql_json_extract('s.model_config', '$._branched_from')} IS NULL
                  AND NOT ({_RESET_CHILD_SQL.format(a='s')})
                  AND COALESCE(s.source, '') != 'tool'
            )
            SELECT id, parent_session_id, started_at FROM descendants
            """


# (id, parent, routing key, started_at), in creation order: ties on started_at, created out of id order.
TREE = (
    ("root", None, None, 1.0),
    ("b", "root", None, 10.0),
    ("a", "root", None, 10.0),
    ("b1", "b", None, 20.0),
    ("a1", "a", None, 20.0),
    ("other", None, None, 5.0),
)
KEYED = (
    ("k1-old", None, "agent:main:k1", 5.0),
    ("k3", None, "agent:main:k3", 20.0),
    ("k1", None, "agent:main:k1", 30.0),
    ("k2", None, "agent:main:k2", 20.0),
    ("k5", None, "agent:main:k5", 20.0),
    ("k4", None, "agent:main:k4", 10.0),
)


def _seed(backend, scope, tmp_path, rows):
    from hermes_state import SessionDB

    sessions = backend.open(scope).sessions
    db = SessionDB(db_path=tmp_path / "legacy.db")
    for sid, parent, key, _ in rows:
        sessions.create(NewSession(sid, "telegram", session_key=key, parent_session_id=parent))
        db.create_session(sid, "telegram", session_key=key, parent_session_id=parent)
    started = [(at, sid) for sid, _, _, at in rows]
    with sqlite3.connect(store_for(scope)) as connection:
        connection.executemany("UPDATE sessions SET started_at=? WHERE id=?", started)
    db._execute_write(lambda conn: conn.executemany("UPDATE sessions SET started_at=? WHERE id=?", started))
    return sessions, db


def test_descendants_are_the_legacy_walk_as_a_multiset(backend, scope_a, tmp_path, legacy_store):
    sessions, db = _seed(backend, scope_a, tmp_path, TREE)
    try:
        for root in ("root", "a", "b1", "other"):
            ours = Counter((e.session_id, e.parent_session_id, e.started_at) for e in sessions.descendants(root))
            legacy = Counter(tuple(row) for row in db._conn.execute(_descendants_sql(), (root,)).fetchall())
            assert ours == legacy
        assert {e.session_id for e in sessions.descendants("root")} == {"root", "a", "b", "a1", "b1"}
    finally:
        db.close()


def test_gateway_listing_orders_by_last_active_only(backend, scope_a, tmp_path, legacy_store):
    sessions, db = _seed(backend, scope_a, tmp_path, KEYED)
    try:
        ours = [(e.last_active, e.session.session_id) for e in sessions.list_gateway_sessions()]
        legacy = [(row["last_active"], row["id"]) for row in db.list_gateway_sessions()]
    finally:
        db.close()
    assert [at for at, _ in ours] == sorted((at for at, _ in ours), reverse=True) == [at for at, _ in legacy]
    groups = [(at, {sid for _, sid in rows}) for at, rows in groupby(ours, key=lambda pair: pair[0])]
    assert groups == [(at, {sid for _, sid in rows}) for at, rows in groupby(legacy, key=lambda pair: pair[0])]
    assert groups == [(30.0, {"k1"}), (20.0, {"k2", "k3", "k5"}), (10.0, {"k4"})]
