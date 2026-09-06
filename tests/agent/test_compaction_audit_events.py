"""Durable compaction audit events (#104099): start/end brackets + orphan detection."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="compaction-audit-test-"))

from agent.compaction_audit import find_orphaned_compaction_starts, record_compaction_event
from hermes_state import SessionDB


def _store() -> SessionDB:
    fd, path = tempfile.mkstemp(prefix="audit-", suffix=".db")
    os.close(fd)
    os.unlink(path)
    db = SessionDB(Path(path))
    db.create_session("sess-a", source="cli")
    db.create_session("sess-b", source="cli")
    return db


class TestCompactionEventsStore(unittest.TestCase):
    def setUp(self):
        self.db = _store()

    def tearDown(self):
        self.db.close()

    def test_start_end_pair_is_not_orphaned(self):
        self.assertTrue(record_compaction_event(self.db, "sess-a", "att-1", "start", {"message_count": 40}))
        self.assertTrue(record_compaction_event(self.db, "sess-a", "att-1", "end", {"commit_status": "committed"}))
        self.assertEqual(find_orphaned_compaction_starts(self.db), [])

    def test_orphaned_start_detected_per_attempt(self):
        record_compaction_event(self.db, "sess-a", "att-1", "start", {})
        record_compaction_event(self.db, "sess-a", "att-1", "end", {"commit_status": "aborted"})
        record_compaction_event(self.db, "sess-b", "att-2", "start", {"approx_tokens": 190000})
        orphans = find_orphaned_compaction_starts(self.db)
        self.assertEqual(len(orphans), 1)
        self.assertEqual(orphans[0]["attempt_id"], "att-2")
        self.assertEqual(orphans[0]["session_id"], "sess-b")

    def test_orphan_query_scopes_to_session(self):
        record_compaction_event(self.db, "sess-a", "att-1", "start", {})
        record_compaction_event(self.db, "sess-b", "att-2", "start", {})
        self.assertEqual(len(find_orphaned_compaction_starts(self.db, "sess-a")), 1)
        self.assertEqual(find_orphaned_compaction_starts(self.db, "sess-a")[0]["session_id"], "sess-a")

    def test_end_event_survives_without_start(self):
        # Abort paths before lease acquisition emit telemetry (an ``end``) with no start; the
        # orphan query must stay silent — it only flags start-without-end.
        record_compaction_event(self.db, "sess-a", "att-pre", "end", {"failure_class": "cooldown"})
        self.assertEqual(find_orphaned_compaction_starts(self.db), [])

    def test_payload_is_content_free_and_bounded(self):
        record_compaction_event(self.db, "sess-a", "att-1", "start", {"blob": "x" * 50_000})
        rows = self.db._read_all("SELECT payload_json FROM compaction_events WHERE session_id = 'sess-a'")
        self.assertLessEqual(len(rows[0][0]), 8_192)
        self.assertIn("truncated", json.loads(rows[0][0]))

    def test_failed_write_never_raises(self):
        class Exploding:
            def append_compaction_event(self, *a, **k):
                raise RuntimeError("store is gone")

        self.assertFalse(record_compaction_event(Exploding(), "sess-a", "att-1", "start", {}))

    def test_missing_session_or_attempt_is_noop(self):
        self.assertFalse(record_compaction_event(self.db, "", "att-1", "start"))
        self.assertFalse(record_compaction_event(self.db, "sess-a", "", "start"))

    def test_store_without_append_method_is_noop(self):
        self.assertFalse(record_compaction_event(SimpleNamespace(), "sess-a", "att-1", "start"))


if __name__ == "__main__":
    unittest.main()
