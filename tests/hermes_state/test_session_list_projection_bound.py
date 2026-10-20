"""Page projection regression for #119408, contributed by diamondpearson-netizen.

Adapted from gist 8807976cf3e7e36cf16bace2114ca9fa for the current preview helper.
"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hermes_state_sessions
from hermes_state import SessionDB


class TestSessionListProjectionBound(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        env = patch.dict(os.environ, {"HERMES_HOME": str(self.home), "HERMES_TEST_ISOLATION": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.db = SessionDB(self.home / "state.db")
        self.addCleanup(self.db.close)
        for i in range(30):
            sid = f"s{i:02d}"
            self.db.create_session(sid, "cli", model_config={"test": i})
        with self.db._lock:
            for i in range(30):
                ts = 1_700_000_000 + i
                self.db._conn.execute(
                    "INSERT INTO messages(session_id,role,content,timestamp) VALUES(?,?,?,?)",
                    (f"s{i:02d}", "user", f"preview {i}", ts),
                )
                self.db._conn.execute(
                    "UPDATE sessions SET started_at=?,last_activity_at=? WHERE id=?",
                    (ts, ts, f"s{i:02d}"),
                )
                self.db._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (ts, f"s{i:02d}"))
            self.db._conn.commit()

    def test_preview_work_is_bounded_by_requested_page(self):
        calls = []
        def preview(sid):
            calls.append(sid)
            return sid
        def execute(sql, params=()):
            with self.db._read_ctx() as conn:
                conn.create_function("count_preview", 1, preview)
                return conn.execute(sql, params).fetchall()
        with patch.object(hermes_state_sessions, "_sql_preview_raw", lambda session_id="s.id": f"count_preview({session_id}) AS _preview_raw"), patch.object(self.db, "_read_all", execute):
            for compact in (False, True):
                for id_query, search_query in ((None, None), ("s", None), (None, "s")):
                    for limit, offset in ((3, 2), (3, 20), (0, 0)):
                        with self.subTest(compact=compact, id_query=id_query, search_query=search_query, limit=limit, offset=offset):
                            calls.clear()
                            rows = self.db.list_sessions_rich(
                                limit=limit, offset=offset, compact_rows=compact,
                                order_by_last_active=True, id_query=id_query, search_query=search_query,
                            )
                            expected = [f"s{i:02d}" for i in range(29, -1, -1)][offset:offset + limit]
                            self.assertEqual([r["id"] for r in rows], expected)
                            self.assertEqual(calls, expected, "off-page previews must not be evaluated")

    def test_recent_pages_preserve_row_projection_and_pinned_backfill(self):
        with self.db._lock:
            self.db._conn.execute("UPDATE sessions SET pinned=1 WHERE id='s00'")
            self.db._conn.commit()
        for compact in (False, True):
            all_rows = self.db.list_sessions_rich(limit=50, compact_rows=compact, order_by_last_active=True)
            for limit, offset in ((0, 0), (1, 0), (3, 2), (3, 40)):
                with self.subTest(compact=compact, limit=limit, offset=offset):
                    actual = self.db.list_sessions_rich(limit=limit, offset=offset, compact_rows=compact, order_by_last_active=True, include_pinned=True)
                    expected = all_rows[offset:offset + limit]
                    if not any(r["id"] == "s00" for r in expected):
                        expected = expected + [r for r in all_rows if r["id"] == "s00"]
                    self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()