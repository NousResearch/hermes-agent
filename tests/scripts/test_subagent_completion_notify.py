#!/usr/bin/env python3
"""Unit tests for subagent-completion-notify.py."""

import importlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import the module under test.
# The filename contains hyphens, so we must use importlib.util directly.
# ---------------------------------------------------------------------------
HOOK_PATH = Path.home() / ".hermes" / "agent-hooks" / "subagent-completion-notify.py"

spec = importlib.util.spec_from_file_location("subagent_completion_notify", HOOK_PATH)
scn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scn)


# ---------------------------------------------------------------------------
# duration_ms parsing
# ---------------------------------------------------------------------------

class TestDurationMsParsing(unittest.TestCase):
    """
    The expression under test (reproduced from the module):
        duration_ms: int = int(float(extra.get("duration_ms") or 0))

    We test each branch by exercising the expression directly.
    """

    def _parse(self, extra):
        return int(float(extra.get("duration_ms") or 0))

    def test_integer_string(self):
        self.assertEqual(self._parse({"duration_ms": "4200"}), 4200)

    def test_float_string_was_broken_case(self):
        """Previously raised ValueError; now handled via int(float(...))."""
        self.assertEqual(self._parse({"duration_ms": "4200.0"}), 4200)

    def test_actual_int(self):
        self.assertEqual(self._parse({"duration_ms": 4200}), 4200)

    def test_missing_key_gives_zero(self):
        self.assertEqual(self._parse({}), 0)

    def test_none_value_gives_zero(self):
        self.assertEqual(self._parse({"duration_ms": None}), 0)

    def test_zero_string(self):
        self.assertEqual(self._parse({"duration_ms": "0"}), 0)

    def test_float_value(self):
        self.assertEqual(self._parse({"duration_ms": 4200.7}), 4200)


# ---------------------------------------------------------------------------
# main() integration test
# ---------------------------------------------------------------------------

class TestMainIntegration(unittest.TestCase):

    def _build_payload(self, **extra_overrides):
        extra = {
            "parent_session_id": "sess-abc123",
            "child_role": "leaf",
            "child_summary": "Ran the tests",
            "child_status": "completed",
            "duration_ms": "4200.0",  # float string — the fixed case
        }
        extra.update(extra_overrides)
        return {
            "hook_event_name": "subagent_stop",
            "session_id": "sess-abc123",
            "tool_name": None,
            "tool_input": None,
            "cwd": "/tmp",
            "extra": extra,
        }

    # The module catches (urllib.error.URLError, OSError) around the POST.
    # We patch the urlopen that the module's urllib.request sees.
    _URL_ERROR = urllib.error.URLError("no server")

    def test_writes_jsonl_record_with_correct_fields(self):
        payload = self._build_payload()
        stdin_data = json.dumps(payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            completions_dir = Path(tmpdir) / "completions"

            with (
                patch("sys.stdin", io.StringIO(stdin_data)),
                patch.object(scn, "COMPLETIONS_DIR", completions_dir),
                patch.object(scn.urllib.request, "urlopen",
                             side_effect=self._URL_ERROR),
                patch("builtins.print"),  # suppress stdout {}
            ):
                scn.main()

            # Verify the .jsonl file was written
            expected_file = completions_dir / "sess-abc123.jsonl"
            self.assertTrue(expected_file.exists(), "JSONL file was not created")

            lines = expected_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)

            record = json.loads(lines[0])

            self.assertEqual(record["session_id"], "sess-abc123")
            self.assertEqual(record["child_role"], "leaf")
            self.assertEqual(record["child_status"], "completed")
            self.assertEqual(record["child_summary"], "Ran the tests")
            self.assertEqual(record["duration_ms"], 4200)
            self.assertIn("timestamp", record)
            self.assertIn("raw_extra", record)

    def test_multiple_calls_append_to_same_file(self):
        """Two subagent completions for the same session → two lines."""
        payload1 = self._build_payload(child_summary="First task")
        payload2 = self._build_payload(child_summary="Second task")

        with tempfile.TemporaryDirectory() as tmpdir:
            completions_dir = Path(tmpdir) / "completions"

            for p in (payload1, payload2):
                with (
                    patch("sys.stdin", io.StringIO(json.dumps(p))),
                    patch.object(scn, "COMPLETIONS_DIR", completions_dir),
                    patch.object(scn.urllib.request, "urlopen",
                                 side_effect=self._URL_ERROR),
                    patch("builtins.print"),
                ):
                    scn.main()

            expected_file = completions_dir / "sess-abc123.jsonl"
            lines = expected_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

    def test_empty_stdin_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            completions_dir = Path(tmpdir) / "completions"
            with (
                patch("sys.stdin", io.StringIO("")),
                patch.object(scn, "COMPLETIONS_DIR", completions_dir),
                patch.object(scn.urllib.request, "urlopen",
                             side_effect=self._URL_ERROR),
                patch("builtins.print"),
            ):
                scn.main()  # must not raise

    def test_session_id_falls_back_to_top_level(self):
        """When extra.parent_session_id is absent, use payload.session_id."""
        payload = {
            "hook_event_name": "subagent_stop",
            "session_id": "top-level-sess",
            "extra": {
                "child_status": "completed",
                "duration_ms": 1000,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            completions_dir = Path(tmpdir) / "completions"
            with (
                patch("sys.stdin", io.StringIO(json.dumps(payload))),
                patch.object(scn, "COMPLETIONS_DIR", completions_dir),
                patch.object(scn.urllib.request, "urlopen",
                             side_effect=self._URL_ERROR),
                patch("builtins.print"),
            ):
                scn.main()

            expected_file = completions_dir / "top-level-sess.jsonl"
            self.assertTrue(expected_file.exists())
            record = json.loads(expected_file.read_text().strip())
            self.assertEqual(record["session_id"], "top-level-sess")

    def test_http_post_is_attempted(self):
        """Verify urllib.request.urlopen is called (POST attempt is made)."""
        payload = self._build_payload()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_urlopen = MagicMock(return_value=mock_ctx)

        with tempfile.TemporaryDirectory() as tmpdir:
            completions_dir = Path(tmpdir) / "completions"
            with (
                patch("sys.stdin", io.StringIO(json.dumps(payload))),
                patch.object(scn, "COMPLETIONS_DIR", completions_dir),
                patch.object(scn.urllib.request, "urlopen", mock_urlopen),
                patch("builtins.print"),
            ):
                scn.main()

        mock_urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
