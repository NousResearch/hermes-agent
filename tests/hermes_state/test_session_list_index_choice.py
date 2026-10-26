"""Session-list reads stay on the (session_id, timestamp) index whatever the planner stats say (#119403).

A ``sqlite_stat1`` written before ``idx_messages_session_id`` existed has no row for that index, so the
planner guesses it is selective and routes the per-session last-active and preview subqueries through
it: every listed session then reads all of its message rows (≈0.5 GB per sidebar list on a 125k-message
store before this fix).
"""

import re
import sqlite3

import pytest

from hermes_state import SessionDB

_READERS = {
    "sidebar": lambda db: db.list_sessions_rich(limit=2, order_by_last_active=True, include_pinned=True),
    "session_search": lambda db: db.list_recent_sessions_bounded(limit=2),
    "telegram_unlinked": lambda db: db.list_unlinked_telegram_sessions_for_user(chat_id="c", user_id="u", limit=2),
}


def _stale_stats_db(path):
    db = SessionDB(db_path=path)
    for i in range(4):
        sid = f"s{i}"
        db.create_session(sid, "telegram", user_id="u")
        db.append_messages_batch(sid, [
            {"role": "user" if j % 2 == 0 else "assistant", "content": f"{sid} message {j}", "timestamp": 1_000 + j}
            for j in range(40)
        ])
    db.set_session_pinned("s0", True)
    db.close()
    raw = sqlite3.connect(path)
    raw.execute("ANALYZE")
    raw.execute("DELETE FROM sqlite_stat1 WHERE idx = 'idx_messages_session_id'")
    raw.commit()
    raw.close()


@pytest.mark.parametrize("reader", sorted(_READERS))
def test_session_listings_search_messages_only_through_the_timestamp_index(tmp_path, monkeypatch, reader):
    path = tmp_path / "state.db"
    _stale_stats_db(path)
    statements = []
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    db = SessionDB(db_path=path)
    statements.clear()
    rows = _READERS[reader](db)
    db.close()
    monkeypatch.undo()

    assert rows and all(r["preview"] for r in rows)
    reads = [s for s in statements if s.lstrip().split(None, 1)[0].upper() in ("SELECT", "WITH")]
    plan_conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    plans = [row[3] for sql in reads for row in plan_conn.execute("EXPLAIN QUERY PLAN " + sql)]
    plan_conn.close()
    message_searches = [p for p in plans if re.match(r"(SEARCH|SCAN) (m|_act_m)\b", p)]
    # Preview (m) and last-active (_act_m) both run, and both only through idx_messages_session.
    assert {p.split()[1] for p in message_searches} == {"m", "_act_m"}, plans
    assert all(re.search(r"INDEX idx_messages_session \(", p) for p in message_searches), message_searches
