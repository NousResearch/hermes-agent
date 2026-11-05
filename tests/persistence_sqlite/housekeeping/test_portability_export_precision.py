"""Exports keep the exact epochs stored in the session row and its messages."""

import sqlite3

from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.persistence_sqlite.conftest import store_for


def test_export_keeps_exact_stored_epochs(backend, scope_a):
    stamp = 1700000000.1234567
    path = store_for(scope_a)
    store = backend.open(scope_a)
    seed_session(store, "s", "cli")
    seed_messages(store, "s", {"role": "user", "content": "exact", "timestamp": stamp})
    raw_holder(store)._write_sql("UPDATE sessions SET started_at=?, ended_at=?, last_activity_at=? WHERE id=?",
                                 (stamp, stamp, stamp, "s"))
    export = store.maintenance.export_session("s")
    with sqlite3.connect(path) as conn:
        stored = conn.execute("SELECT s.started_at, s.ended_at, s.last_activity_at, m.timestamp FROM sessions s "
                              "JOIN messages m ON m.session_id=s.id WHERE s.id='s'").fetchone()
    assert stored == (stamp, stamp, stamp, stamp)
    assert export.session.started_at == stamp
    assert export.session.ended_at == stamp
    assert export.session.last_activity_at == stamp
    assert export.messages[0].timestamp == stamp
    assert export.session.message_count == 1
    assert export.session.tool_call_count == 0
