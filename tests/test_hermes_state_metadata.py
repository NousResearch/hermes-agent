"""Tests for session metadata CRUD methods on SessionDB.

Tests the metadata column, set/get/search operations added
for issue #27013: Agents lose project context across session restarts.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from hermes_state import SessionDB


class TestSessionMetadataCRUD(unittest.TestCase):
    """Test get/set/search session metadata operations."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = SessionDB(db_path=Path(self.tmp.name))
        self.session_id = self.db.create_session(
            session_id="test_meta_1", source="test"
        )

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_set_and_get_metadata(self):
        self.db.set_session_metadata(self.session_id, "project", "hermes")
        meta = self.db.get_session_metadata(self.session_id)
        self.assertEqual(meta, {"project": "hermes"})

    def test_get_metadata_empty_by_default(self):
        meta = self.db.get_session_metadata(self.session_id)
        self.assertEqual(meta, {})

    def test_overwrite_metadata_key(self):
        self.db.set_session_metadata(self.session_id, "project", "old")
        self.db.set_session_metadata(self.session_id, "project", "new")
        meta = self.db.get_session_metadata(self.session_id)
        self.assertEqual(meta, {"project": "new"})

    def test_remove_metadata_key(self):
        self.db.set_session_metadata(self.session_id, "project", "hermes")
        self.db.set_session_metadata(self.session_id, "project", None)
        meta = self.db.get_session_metadata(self.session_id)
        self.assertEqual(meta, {})

    def test_multiple_keys(self):
        self.db.set_session_metadata(self.session_id, "project", "hermes")
        self.db.set_session_metadata(self.session_id, "topic", "gateway")
        meta = self.db.get_session_metadata(self.session_id)
        self.assertEqual(meta, {"project": "hermes", "topic": "gateway"})

    def test_search_by_metadata(self):
        sid2 = self.db.create_session(session_id="test_meta_2", source="test")
        sid3 = self.db.create_session(session_id="test_meta_3", source="cli")
        self.db.set_session_metadata(self.session_id, "project", "A")
        self.db.set_session_metadata(sid2, "project", "A")
        self.db.set_session_metadata(sid3, "project", "B")

        results = self.db.search_sessions_by_metadata("project", "A")
        self.assertEqual(len(results), 2)

        results = self.db.search_sessions_by_metadata("project", "B")
        self.assertEqual(len(results), 1)

    def test_search_with_source_filter(self):
        sid2 = self.db.create_session(session_id="test_meta_4", source="cli")
        self.db.set_session_metadata(self.session_id, "project", "A")
        self.db.set_session_metadata(sid2, "project", "A")

        results = self.db.search_sessions_by_metadata(
            "project", "A", source="test"
        )
        self.assertEqual(len(results), 1)

    def test_search_nonexistent_key_returns_empty(self):
        results = self.db.search_sessions_by_metadata("nonexistent", "value")
        self.assertEqual(results, [])

    def test_metadata_on_existing_session(self):
        """Verify metadata can be added to existing non-empty sessions."""
        self.db.set_session_metadata(self.session_id, "project", "hermes")
        session = self.db.get_session(self.session_id)
        self.assertIsNotNone(session)
        meta = json.loads(session["metadata"]) if session["metadata"] else {}
        self.assertEqual(meta.get("project"), "hermes")
