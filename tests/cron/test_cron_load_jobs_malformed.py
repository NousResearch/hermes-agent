"""Regression tests for #22569 — guard load_jobs against malformed jobs.json shapes."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# cron/jobs.py uses a module-level JOBS_FILE path.
# We patch it to a temp file for isolation.


class TestLoadJobsMalformedShapes(unittest.TestCase):
    """load_jobs must not crash when jobs.json is not a dict."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.jobs_file = Path(self.tmp_dir.name) / "jobs.json"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _write_json(self, obj):
        self.jobs_file.write_text(json.dumps(obj), encoding="utf-8")

    def _load(self):
        # Import inside test so JOBS_FILE can be patched
        import importlib
        import cron.jobs as jobs_mod
        with patch.object(jobs_mod, "JOBS_FILE", self.jobs_file):
            # ensure_dirs writes to the parent, already exists via tmp_dir
            return jobs_mod.load_jobs()

    def test_bare_list(self):
        """Legacy bare-list format should be auto-migrated."""
        self._write_json([{"id": "j1", "name": "test"}])
        result = self._load()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "j1")
        # After migration, file should be rewritten as dict shape
        data = json.loads(self.jobs_file.read_text(encoding="utf-8"))
        self.assertIn("jobs", data)

    def test_null(self):
        """null root should return empty list, not crash."""
        self._write_json(None)
        result = self._load()
        self.assertEqual(result, [])

    def test_scalar_string(self):
        """String root should return empty list, not crash."""
        self._write_json("corrupted")
        result = self._load()
        self.assertEqual(result, [])

    def test_scalar_number(self):
        """Number root should return empty list, not crash."""
        self._write_json(42)
        result = self._load()
        self.assertEqual(result, [])

    def test_normal_dict(self):
        """Normal dict shape should work as before."""
        self._write_json({"jobs": [{"id": "j1"}], "updated_at": "2026-01-01T00:00:00"})
        result = self._load()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "j1")

    def test_empty_dict(self):
        """Empty dict should return empty list."""
        self._write_json({})
        result = self._load()
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
