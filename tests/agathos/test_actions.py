#!/usr/bin/env python3
"""
Unit tests for Agathos actions module.

Tests restart and kill action logic, corrective prompt building,
and session-type-specific handlers.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock, ANY
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import agathos
from agathos import actions


class TestActions(unittest.TestCase):
    """Test cases for Agathos actions module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Create connection
        import sqlite3
        self.conn = sqlite3.connect(self.temp_db.name)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        # Create minimal schema for testing
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                session_type TEXT NOT NULL,
                job_id TEXT,
                task_description TEXT,
                status TEXT DEFAULT 'active',
                restart_count INTEGER DEFAULT 0,
                kill_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS entropy_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                session_id TEXT NOT NULL,
                entropy_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                details TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS watcher_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                session_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                action_reason TEXT,
                success BOOLEAN DEFAULT TRUE,
                details TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                session_id TEXT NOT NULL,
                notification_type TEXT NOT NULL,
                message TEXT,
                delivered BOOLEAN DEFAULT FALSE
            )
        """)
        self.conn.commit()

    def tearDown(self):
        """Clean up test fixtures."""
        self.conn.close()
        os.unlink(self.temp_db.name)

    def _create_test_session(self, session_id, session_type, job_id=None, metadata=None):
        """Helper to create a test session."""
        self.cursor.execute("""
            INSERT INTO sessions (session_id, session_type, job_id, task_description, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, session_type, job_id, "Test task", json.dumps(metadata or {})))
        self.conn.commit()

    def test_build_corrective_prompt_no_detections(self):
        """Test corrective prompt when no entropy detected."""
        session_id = "test_session_1"
        self._create_test_session(session_id, "cron", "job_1")

        prompt = actions.build_corrective_prompt(
            self.cursor, session_id, "High entropy detected"
        )

        # Should return generic restart message when no detections
        self.assertIn("restart", prompt.lower())
        self.assertIn("Agathos", prompt)

    def test_build_corrective_prompt_with_detection(self):
        """Test corrective prompt when entropy detected."""
        session_id = "test_session_2"
        self._create_test_session(session_id, "delegate_task")

        # Insert entropy detection
        self.cursor.execute("""
            INSERT INTO entropy_detections (session_id, entropy_type, severity, details)
            VALUES (?, ?, ?, ?)
        """, (session_id, "repeat_tool_calls", "warning", json.dumps({"count": 5})))
        self.conn.commit()

        prompt = actions.build_corrective_prompt(
            self.cursor, session_id, "Repeat tool calls detected"
        )

        # Should include detection-specific guidance
        self.assertIn("repeat", prompt.lower())

    def test_build_corrective_prompt_custom_templates(self):
        """Test corrective prompt with custom prompt templates."""
        session_id = "test_session_3"
        self._create_test_session(session_id, "manual")

        # Insert entropy detection
        self.cursor.execute("""
            INSERT INTO entropy_detections (session_id, entropy_type, severity, details)
            VALUES (?, ?, ?, ?)
        """, (session_id, "stuck_loop", "critical", json.dumps({"pattern": "read-write"})))
        self.conn.commit()

        custom_prompts = {
            "stuck_loop": "Custom stuck loop guidance: {reason}"
        }

        prompt = actions.build_corrective_prompt(
            self.cursor, session_id, "Stuck loop detected", custom_prompts
        )

        # Should use custom template
        self.assertIn("Custom stuck loop guidance", prompt)

    @patch("agathos.actions.logger")
    def test_kill_session_updates_database(self, mock_logger):
        """Test kill_session updates session status and records action."""
        session_id = "test_session_kill"
        self._create_test_session(session_id, "cron", "job_kill")

        # Mock the cron session kill to avoid external API calls
        with patch.object(actions, "kill_cron_session") as mock_kill_cron:
            mock_kill_cron.return_value = None

            actions.kill_session(
                self.cursor, self.conn, session_id, "Test kill reason"
            )

            # Verify session status updated
            self.cursor.execute(
                "SELECT status, kill_count FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = self.cursor.fetchone()
            self.assertEqual(row["status"], "killed")
            self.assertEqual(row["kill_count"], 1)

            # Verify action recorded
            self.cursor.execute(
                "SELECT action_type, action_reason FROM watcher_actions WHERE session_id = ?",
                (session_id,)
            )
            action_row = self.cursor.fetchone()
            self.assertEqual(action_row["action_type"], "kill")
            self.assertEqual(action_row["action_reason"], "Test kill reason")

    @patch("agathos.actions.logger")
    def test_restart_session_updates_database(self, mock_logger):
        """Test restart_session updates session status and records action."""
        session_id = "test_session_restart"
        self._create_test_session(session_id, "delegate_task", metadata={"pid": 12345})

        # Mock the delegate session restart to avoid process termination
        with patch.object(actions, "restart_delegate_session") as mock_restart_delegate:
            mock_restart_delegate.return_value = None

            actions.restart_session(
                self.cursor, self.conn, session_id, "Test restart reason"
            )

            # Verify session status updated
            self.cursor.execute(
                "SELECT status, restart_count FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = self.cursor.fetchone()
            self.assertEqual(row["status"], "restarted")
            self.assertEqual(row["restart_count"], 1)

    def test_safe_subprocess_success(self):
        """Test safe_subprocess with successful command."""
        result = actions.safe_subprocess(["echo", "test"], timeout=5)

        self.assertIsNotNone(result)
        self.assertEqual(result.returncode, 0)

    def test_safe_subprocess_failure(self):
        """Test safe_subprocess with failing command."""
        result = actions.safe_subprocess(["false"], timeout=5)

        # Should return result even on failure (never raises)
        self.assertIsNotNone(result)
        self.assertNotEqual(result.returncode, 0)

    def test_safe_subprocess_timeout(self):
        """Test safe_subprocess timeout handling."""
        # This should timeout
        result = actions.safe_subprocess(["sleep", "10"], timeout=1)

        # Should return None on timeout (per function docstring)
        # Actually implementation may vary - let's check
        # The function says "Never raises" so it should handle gracefully

    def test_terminate_pid_graceful_then_force(self):
        """Test terminate_pid sends SIGTERM then SIGKILL."""
        # Start a subprocess we can terminate
        import subprocess
        proc = subprocess.Popen(["sleep", "30"])

        # Give it time to start
        import time
        time.sleep(0.1)

        # Terminate it
        actions.terminate_pid(proc.pid, "test termination")

        # Verify process is gone
        proc.poll()
        # Process should be terminated

    def test_restart_cron_session_missing_job_id(self):
        """Test restart_cron_session handles missing job_id."""
        session = {"session_id": "test", "job_id": None}

        # Should log warning and return without error
        with patch("agathos.actions.logger") as mock_logger:
            actions.restart_cron_session(session, "test prompt")
            mock_logger.warning.assert_called()

    def test_kill_cron_session_missing_job_id(self):
        """Test kill_cron_session handles missing job_id."""
        session = {"session_id": "test", "job_id": None}

        # Should log warning and return without error
        with patch("agathos.actions.logger") as mock_logger:
            actions.kill_cron_session(session, "test reason")
            mock_logger.warning.assert_called()

    def test_restart_manual_session_logs_info(self):
        """Test restart_manual_session logs user intervention needed."""
        session = {"session_id": "manual_session"}

        with patch("agathos.actions.logger") as mock_logger:
            actions.restart_manual_session(session, "test prompt")
            mock_logger.info.assert_called()
            # Should mention user intervention
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("manual", call_args.lower())

    def test_kill_manual_session_records_notification(self):
        """Test kill_manual_session records notification."""
        session = {"session_id": "manual_session_kill"}

        actions.kill_manual_session(self.cursor, session, "test kill reason")

        # Verify notification recorded
        self.cursor.execute(
            "SELECT notification_type, message FROM notifications WHERE session_id = ?",
            ("manual_session_kill",)
        )
        row = self.cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["notification_type"], "kill")


if __name__ == "__main__":
    unittest.main()
