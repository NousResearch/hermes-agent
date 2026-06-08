"""Tests for memory locations (markers)."""

import os
import tempfile
import unittest
from pathlib import Path

from hermes_state import SessionDB
from agent.memory_locations import MemoryLocationStore


class TestMemoryLocationStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "state.db"
        self.db = SessionDB(db_path=self.db_path)
        
        # Create test sessions for foreign key constraints
        with self.db._lock:
            self.db._conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                ("session-1", "cli", 1000.0)
            )
            self.db._conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                ("session-2", "cli", 1000.0)
            )
            self.db._conn.commit()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_and_get(self):
        store = MemoryLocationStore(session_db=self.db)
        loc_id = store.create(
            session_id="session-1",
            label="Test Marker",
            time_type="point",
            tags=["bug", "frontend"],
            is_persistent=False,
        )
        self.assertIsNotNone(loc_id)
        self.assertGreater(loc_id, 0)

        loc = store.get(loc_id)
        self.assertIsNotNone(loc)
        self.assertEqual(loc["label"], "Test Marker")
        self.assertEqual(loc["session_id"], "session-1")
        self.assertEqual(loc["tags"], ["bug", "frontend"])
        self.assertEqual(loc["is_persistent"], 0)

    def test_list_locations(self):
        store = MemoryLocationStore(session_db=self.db)
        store.create(
            session_id="session-1",
            label="Marker 1",
            time_type="point",
            is_persistent=False,
        )
        store.create(
            session_id="session-2",
            label="Marker 2",
            time_type="point",
            is_persistent=True,
        )

        all_locs = store.list()
        self.assertEqual(len(all_locs), 2)

        session_1_locs = store.list(session_id="session-1")
        self.assertEqual(len(session_1_locs), 1)
        self.assertEqual(session_1_locs[0]["label"], "Marker 1")

        persistent_locs = store.list(persistent_only=True)
        self.assertEqual(len(persistent_locs), 1)
        self.assertEqual(persistent_locs[0]["label"], "Marker 2")

    def test_delete_location(self):
        store = MemoryLocationStore(session_db=self.db)
        loc_id = store.create(
            session_id="session-1",
            label="To Delete",
            time_type="point",
        )
        
        self.assertTrue(store.delete(loc_id))
        self.assertIsNone(store.get(loc_id))

    def test_update_location(self):
        store = MemoryLocationStore(session_db=self.db)
        loc_id = store.create(
            session_id="session-1",
            label="Original",
            time_type="point",
        )
        
        self.assertTrue(store.update(loc_id, label="Updated", is_persistent=True))
        loc = store.get(loc_id)
        self.assertEqual(loc["label"], "Updated")
        self.assertEqual(loc["is_persistent"], 1)

    def test_fts_search(self):
        store = MemoryLocationStore(session_db=self.db)
        store.create(
            session_id="session-1",
            label="Bug Fix Frontend",
            time_type="point",
            tags=["bug", "frontend"],
        )
        store.create(
            session_id="session-1",
            label="Database Migration",
            time_type="point",
            tags=["db", "backend"],
        )

        results = store.search_fts("bug")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["label"], "Bug Fix Frontend")

    def test_resolve_anchor(self):
        store = MemoryLocationStore(session_db=self.db)
        
        # Create a session
        with self.db._lock:
            self.db._conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                ("parent-session", "cli", 1000.0)
            )
            self.db._conn.execute(
                "INSERT INTO sessions (id, source, parent_session_id, started_at) VALUES (?, ?, ?, ?)",
                ("child-session", "cli", "parent-session", 1100.0)
            )
            self.db._conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, message_guid) VALUES (?, ?, ?, ?, ?)",
                ("parent-session", "user", "Hello", 1001.0, "guid-123")
            )
            self.db._conn.commit()

        # Resolve in parent session (directly)
        msg_id = store.resolve_anchor("guid-123", "parent-session")
        self.assertIsNotNone(msg_id)

        # Resolve in child session (lineage-aware walk)
        msg_id_child = store.resolve_anchor("guid-123", "child-session")
        self.assertIsNotNone(msg_id_child)
        self.assertEqual(msg_id, msg_id_child)


if __name__ == "__main__":
    unittest.main()
