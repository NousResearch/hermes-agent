#!/usr/bin/env python3
"""Unit tests for qdrant_snapshot.py."""

import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path.home() / ".hermes" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import qdrant_snapshot as qs


# ---------------------------------------------------------------------------
# _load_collections
# ---------------------------------------------------------------------------

class TestLoadCollections(unittest.TestCase):

    def _patch_read_text(self, content):
        """Context manager that patches Path.read_text to return `content`."""
        return patch("pathlib.Path.read_text", return_value=content)

    def test_returns_name_from_valid_mem0_json(self):
        cfg = {
            "oss": {
                "vector_store": {
                    "config": {
                        "collection_name": "my_custom_collection"
                    }
                }
            }
        }
        with self._patch_read_text(json.dumps(cfg)):
            result = qs._load_collections()
        self.assertEqual(result, ["my_custom_collection"])

    def test_fallback_when_file_missing(self):
        with patch("pathlib.Path.read_text", side_effect=FileNotFoundError("no file")):
            result = qs._load_collections()
        self.assertEqual(result, ["hermes_memories"])

    def test_fallback_when_collection_name_key_absent(self):
        cfg = {
            "oss": {
                "vector_store": {
                    "config": {}  # collection_name not present
                }
            }
        }
        with self._patch_read_text(json.dumps(cfg)):
            result = qs._load_collections()
        self.assertEqual(result, ["hermes_memories"])

    def test_fallback_when_oss_key_absent(self):
        cfg = {"user_id": "clark"}
        with self._patch_read_text(json.dumps(cfg)):
            result = qs._load_collections()
        self.assertEqual(result, ["hermes_memories"])

    def test_fallback_on_malformed_json(self):
        with self._patch_read_text("not valid json{{"):
            result = qs._load_collections()
        self.assertEqual(result, ["hermes_memories"])

    def test_fallback_when_collection_name_is_empty_string(self):
        """Empty string is falsy — should fall back to default."""
        cfg = {
            "oss": {
                "vector_store": {
                    "config": {
                        "collection_name": ""
                    }
                }
            }
        }
        with self._patch_read_text(json.dumps(cfg)):
            result = qs._load_collections()
        self.assertEqual(result, ["hermes_memories"])

    def test_fallback_on_permission_error(self):
        with patch("pathlib.Path.read_text", side_effect=PermissionError("denied")):
            result = qs._load_collections()
        self.assertEqual(result, ["hermes_memories"])


# ---------------------------------------------------------------------------
# COLLECTIONS module-level constant
# ---------------------------------------------------------------------------

class TestCollectionsConstant(unittest.TestCase):

    def test_collections_is_a_list(self):
        self.assertIsInstance(qs.COLLECTIONS, list)

    def test_collections_is_not_a_string(self):
        self.assertNotIsInstance(qs.COLLECTIONS, str)

    def test_collections_is_non_empty(self):
        self.assertTrue(len(qs.COLLECTIONS) > 0)

    def test_collections_contains_strings(self):
        for item in qs.COLLECTIONS:
            self.assertIsInstance(item, str)


if __name__ == "__main__":
    unittest.main()
