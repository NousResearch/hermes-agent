#!/usr/bin/env python3
"""
Unit tests for Agathos
Agent Guardian & Health Oversight System
Tests entropy detection, decision logic, and database operations.
"""

import os
import sys
import json
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Import agathos package (conftest.py sets up the paths)
import agathos
from agathos.agathos import Agathos
from agathos import entropy as _entropy


class TestAgathos(unittest.TestCase):
    """Test cases for Agathos class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Create Agathos with temporary database
        self.agathos = Agathos()
        self.agathos.db_path = self.temp_db.name
        self.agathos._init_database()
        self.agathos._load_schema()

    def tearDown(self):
        """Clean up test fixtures."""
        if self.agathos.conn:
            self.agathos.conn.close()
        os.unlink(self.temp_db.name)

    def test_database_initialization(self):
        """Test database initialization."""
        # Check that tables exist
        self.agathos.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in self.agathos.cursor.fetchall()]

        expected_tables = [
            "sessions",
            "tool_calls",
            "file_changes",
            "terminal_commands",
            "quality_metrics",
            "entropy_detections",
            "watcher_actions",
            "notifications",
            "directive_checks",
        ]

        for table in expected_tables:
            self.assertIn(table, tables)

    def test_session_registration(self):
        """Test session registration."""
        session = {
            "session_id": "test_session_1",
            "session_type": "cron",
            "job_id": "test_job_1",
            "task_description": "Test session",
            "model": "test_model",
            "provider": "test_provider",
            "metadata": json.dumps({"test": "data"}),
        }

        self.agathos.register_session(session)

        # Verify session was registered
        self.agathos.cursor.execute(
            "SELECT * FROM sessions WHERE session_id = ?", ("test_session_1",)
        )
        result = self.agathos.cursor.fetchone()

        self.assertIsNotNone(result)
        self.assertEqual(result["session_type"], "cron")
        self.assertEqual(result["job_id"], "test_job_1")

    def test_repeat_tool_calls_detection(self):
        """Test repeat tool calls detection."""
        session_id = "test_session_2"

        # Register session
        self.agathos.register_session(
            {
                "session_id": session_id,
                "session_type": "manual",
                "task_description": "Test session",
            }
        )

        # Insert repeated tool calls (use UTC to match SQLite datetime('now'))
        from datetime import timezone

        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(5):
            self.agathos.cursor.execute(
                """
                INSERT INTO tool_calls (session_id, tool_name, tool_args, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (session_id, "read_file", '{"path": "test.py"}', utc_now),
            )

        self.agathos.conn.commit()

        # Detect entropy
        detections = _entropy.detect_repeat_tool_calls(self.agathos.cursor, session_id)

        # Verify detection
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["entropy_type"], "repeat_tool_calls")
        self.assertEqual(detections[0]["severity"], "critical")

    def test_repeat_commands_detection(self):
        """Test repeat commands detection."""
        session_id = "test_session_3"

        # Register session
        self.agathos.register_session(
            {
                "session_id": session_id,
                "session_type": "cron",
                "task_description": "Test session",
            }
        )

        # Insert repeated commands (use UTC to match SQLite datetime('now'))
        from datetime import timezone

        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(4):
            self.agathos.cursor.execute(
                """
                INSERT INTO terminal_commands (session_id, command, timestamp)
                VALUES (?, ?, ?)
            """,
                (session_id, "ls -la", utc_now),
            )

        self.agathos.conn.commit()

        # Detect entropy
        detections = _entropy.detect_repeat_commands(self.agathos.cursor, session_id)

        # Verify detection
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["entropy_type"], "repeat_commands")
        self.assertEqual(detections[0]["severity"], "warning")

    def test_stuck_loop_detection(self):
        """Test stuck loop detection."""
        session_id = "test_session_4"

        # Register session
        self.agathos.register_session(
            {
                "session_id": session_id,
                "session_type": "delegate_task",
                "task_description": "Test session",
            }
        )

        # Insert repeating pattern of tool calls
        pattern = [
            ("read_file", '{"path": "a.py"}'),
            ("write_file", '{"path": "a.py"}'),
            ("read_file", '{"path": "b.py"}'),
        ]

        # Repeat pattern twice
        for _ in range(2):
            for tool_name, tool_args in pattern:
                self.agathos.cursor.execute(
                    """
                    INSERT INTO tool_calls (session_id, tool_name, tool_args, timestamp)
                    VALUES (?, ?, ?, ?)
                """,
                    (session_id, tool_name, tool_args, datetime.now().isoformat()),
                )

        self.agathos.conn.commit()

        # Detect entropy
        detections = _entropy.detect_stuck_loops(self.agathos.cursor, session_id)

        # Verify detection
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["entropy_type"], "stuck_loop")
        self.assertEqual(detections[0]["severity"], "critical")

    def test_no_file_changes_detection(self):
        """Test no file changes detection."""
        session_id = "test_session_5"

        # Register session
        self.agathos.register_session(
            {
                "session_id": session_id,
                "session_type": "cron",
                "task_description": "Test session",
            }
        )

        # Insert write operations without file changes (use UTC)
        from datetime import timezone

        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(3):
            self.agathos.cursor.execute(
                """
                INSERT INTO tool_calls (session_id, tool_name, file_path, file_changed, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (session_id, "write_file", "test.py", False, utc_now),
            )

        self.agathos.conn.commit()

        # Detect entropy
        detections = _entropy.detect_no_file_changes(self.agathos.cursor, session_id)

        # Verify detection
        self.assertEqual(len(detections), 3)
        for detection in detections:
            self.assertEqual(detection["entropy_type"], "no_file_changes")
            self.assertEqual(detection["severity"], "critical")

    def test_error_cascade_detection(self):
        """Test error cascade detection (3+ consecutive tool failures)."""
        session_id = "test_session_cascade"

        # Register session
        self.agathos.register_session(
            {
                "session_id": session_id,
                "session_type": "cron",
                "task_description": "Test session",
            }
        )

        # Insert tool calls: 2 success, then 3 errors (cascade), then 1 success
        from datetime import timezone

        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        tool_sequence = [
            ("read_file", True, None),
            ("terminal", True, None),
            ("write_file", False, "FileNotFoundError: /missing.txt"),
            ("patch", False, "old_string not found"),
            ("terminal", False, "exit 1"),
            ("read_file", True, None),
        ]
        for i, (tool, success, error) in enumerate(tool_sequence):
            self.agathos.cursor.execute(
                """
                INSERT INTO tool_calls (session_id, tool_name, success, error_message, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (session_id, tool, success, error, utc_now),
            )

        self.agathos.conn.commit()

        # Detect error cascade
        detections = _entropy.detect_error_cascade(self.agathos.cursor, session_id)

        # Should detect the 3 consecutive errors (write_file, patch, terminal)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["entropy_type"], "error_cascade")
        self.assertEqual(detections[0]["severity"], "warning")
        details = json.loads(detections[0]["details"])
        self.assertEqual(details["consecutive_errors"], 3)
        self.assertEqual(details["tools"], ["write_file", "patch", "terminal"])

    def test_error_cascade_critical_at_five(self):
        """Test error cascade severity escalates to critical at 5+ errors."""
        session_id = "test_session_cascade_critical"

        self.agathos.register_session(
            {
                "session_id": session_id,
                "session_type": "delegate_task",
                "task_description": "Test session",
            }
        )

        from datetime import timezone

        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # 5 consecutive errors
        for i in range(5):
            self.agathos.cursor.execute(
                """
                INSERT INTO tool_calls (session_id, tool_name, success, error_message, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (session_id, "terminal", False, "exit 1", utc_now),
            )

        self.agathos.conn.commit()

        detections = _entropy.detect_error_cascade(self.agathos.cursor, session_id)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["severity"], "critical")

    def test_error_cascade_no_false_positive(self):
        """Test that mixed success/error calls don't trigger cascade."""
        session_id = "test_session_no_cascade"

        self.agathos.register_session(
            {
                "session_id": session_id,
                "session_type": "manual",
                "task_description": "Test session",
            }
        )

        from datetime import timezone

        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Interleaved errors and successes — no cascade
        for i in range(6):
            success = i % 2 == 0
            self.agathos.cursor.execute(
                """
                INSERT INTO tool_calls (session_id, tool_name, success, error_message, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    "terminal",
                    success,
                    None if success else "exit 1",
                    utc_now,
                ),
            )

        self.agathos.conn.commit()

        detections = _entropy.detect_error_cascade(self.agathos.cursor, session_id)
        self.assertEqual(len(detections), 0)

    def test_detect_tool_error_heuristic(self):
        """Test detect_tool_error matches agent display behavior."""
        # Terminal: exit code 0 = success
        is_err, _ = _entropy.detect_tool_error(
            "terminal", '{"exit_code": 0, "output": "ok"}'
        )
        self.assertFalse(is_err)

        # Terminal: exit code 1 = error
        is_err, detail = _entropy.detect_tool_error(
            "terminal", '{"exit_code": 1, "output": "fail"}'
        )
        self.assertTrue(is_err)
        self.assertIn("exit 1", detail)

        # Terminal: non-JSON = success (parse failure isn't an error)
        is_err, _ = _entropy.detect_tool_error("terminal", "raw output text")
        self.assertFalse(is_err)

        # Generic: "error" in JSON
        is_err, _ = _entropy.detect_tool_error(
            "read_file", '{"error": "file not found"}'
        )
        self.assertTrue(is_err)

        # Generic: starts with "Error"
        is_err, _ = _entropy.detect_tool_error("write_file", "Error: permission denied")
        self.assertTrue(is_err)

        # Generic: normal result
        is_err, _ = _entropy.detect_tool_error("read_file", "file contents here")
        self.assertFalse(is_err)

        # Memory: full
        is_err, _ = _entropy.detect_tool_error(
            "memory", '{"success": false, "error": "exceed the limit"}'
        )
        self.assertTrue(is_err)

        # Empty/None
        is_err, _ = _entropy.detect_tool_error("terminal", "")
        self.assertFalse(is_err)


if __name__ == "__main__":
    unittest.main()
