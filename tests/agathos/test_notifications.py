#!/usr/bin/env python3
"""
Unit tests for Agathos notifications module.

Tests message formatting and notification delivery logic.
Uses mocked HTTP requests to avoid actual API calls.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock, ANY
from urllib.error import HTTPError, URLError

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import agathos
from agathos import notifications


class TestNotifications(unittest.TestCase):
    """Test cases for Agathos notifications module."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear any cached env state
        notifications._dotenv_loaded = False

    def tearDown(self):
        """Clean up test fixtures."""
        pass

    def test_format_notification_message_structure(self):
        """Test notification message has correct structure."""
        session_id = "test_session_123"
        session = {
            "session_type": "cron",
            "task_description": "Test cron job",
            "job_id": "job_456"
        }

        message = notifications.format_notification_message(
            session_id, session, "restart", "High entropy detected"
        )

        # Verify structure
        self.assertIn("Agent Watcher Alert", message)
        self.assertIn(session_id, message)
        self.assertIn("cron", message)
        self.assertIn("Test cron job", message)
        self.assertIn("RESTART", message)  # notification_type is uppercased
        self.assertIn("High entropy detected", message)

    def test_format_notification_message_delegate_task(self):
        """Test notification formatting for delegate_task sessions."""
        session_id = "delegate_123"
        session = {
            "session_type": "delegate_task",
            "task_description": "Code review"
        }

        message = notifications.format_notification_message(
            session_id, session, "kill", "Circuit breaker tripped"
        )

        self.assertIn("delegate_task", message)
        self.assertIn("Code review", message)
        self.assertIn("KILL", message)  # notification_type is uppercased in output

    def test_http_post_success(self):
        """Test _http_post with successful response."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"ok": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            delivered, error = notifications._http_post(
                "https://api.test.com/send", {"message": "test"}
            )

        self.assertTrue(delivered)
        self.assertIsNone(error)

    def test_http_post_http_error(self):
        """Test _http_post with HTTP error."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = HTTPError(
                "https://api.test.com/send", 429, "Too Many Requests", {}, None
            )
            delivered, error = notifications._http_post(
                "https://api.test.com/send", {"message": "test"}
            )

        self.assertFalse(delivered)
        self.assertIn("429", error)

    def test_http_post_url_error(self):
        """Test _http_post with connection error."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("Connection refused")
            delivered, error = notifications._http_post(
                "https://api.test.com/send", {"message": "test"}
            )

        self.assertFalse(delivered)
        self.assertIn("Connection", error)

    def test_send_telegram_not_configured(self):
        """Test send_telegram returns error when not configured."""
        with patch.object(notifications, "_env", return_value=None):
            delivered, error = notifications.send_telegram("test message")

        self.assertFalse(delivered)
        self.assertIn("not configured", error)

    def test_send_telegram_success(self):
        """Test send_telegram with valid config."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"ok": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        def mock_env(key):
            if "TOKEN" in key:
                return "test_token_123"
            if "CHANNEL" in key:
                return "test_channel_456"
            return None

        with patch.object(notifications, "_env", side_effect=mock_env):
            with patch("urllib.request.urlopen", return_value=mock_response):
                with patch("agathos.notifications.logger"):
                    delivered, error = notifications.send_telegram("test message")

        self.assertTrue(delivered)
        self.assertIsNone(error)

    def test_send_discord_not_configured(self):
        """Test send_discord returns error when not configured."""
        with patch.object(notifications, "_env", return_value=None):
            delivered, error = notifications.send_discord("test message")

        self.assertFalse(delivered)
        self.assertIn("not configured", error)

    def test_send_slack_not_configured(self):
        """Test send_slack returns error when not configured."""
        with patch.object(notifications, "_env", return_value=None):
            delivered, error = notifications.send_slack("test message")

        self.assertFalse(delivered)
        self.assertIn("not configured", error)

    @patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "DISCORD_BOT_TOKEN": "test_token",
    }, clear=False)
    def test_discover_platforms_finds_configured(self):
        """Test discover_platforms finds configured platforms."""
        platforms = notifications.discover_platforms()

        # Should find Telegram and Discord
        platform_names = [p[0] for p in platforms]
        self.assertIn("telegram", platform_names)
        self.assertIn("discord", platform_names)

    def test_discover_platforms_empty_when_no_config(self):
        """Test discover_platforms returns empty list when nothing configured."""
        with patch.object(notifications, "_env", return_value=None):
            platforms = notifications.discover_platforms()

        self.assertEqual(len(platforms), 0)

    def test_send_to_all_platforms_no_config(self):
        """Test send_to_all_platforms with no platforms configured."""
        with patch.object(notifications, "discover_platforms", return_value=[]):
            results = notifications.send_to_all_platforms("test message")

        # Should return empty results
        self.assertEqual(len(results), 0)

    def test_send_to_all_platforms_with_gateway(self):
        """Test send_to_all_platforms prefers gateway when configured."""
        with patch.object(notifications, "send_via_gateway") as mock_gateway:
            mock_gateway.return_value = (True, None)

            results = notifications.send_to_all_platforms(
                "test message", prefer_gateway=True
            )

        self.assertIn("gateway", results)
        self.assertTrue(results["gateway"][0])

    def test_send_to_all_platforms_gateway_fails_fallback(self):
        """Test send_to_all_platforms falls back to direct when gateway fails."""
        with patch.object(notifications, "send_via_gateway") as mock_gateway:
            mock_gateway.return_value = (False, "Gateway error")

            with patch.object(notifications, "discover_platforms", return_value=[]):
                results = notifications.send_to_all_platforms(
                    "test message", prefer_gateway=True
                )

        # Should record gateway failure
        self.assertIn("gateway", results)
        self.assertFalse(results["gateway"][0])

    def test_send_notification_records_to_db(self):
        """Test send_notification records delivery in database."""
        import sqlite3

        # Create temp database
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()

        conn = sqlite3.connect(temp_db.name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Create schema
        cursor.execute("""
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                session_type TEXT,
                task_description TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE notifications (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                notification_type TEXT,
                message TEXT,
                delivered BOOLEAN,
                delivery_error TEXT
            )
        """)

        # Insert test session
        cursor.execute(
            "INSERT INTO sessions VALUES (1, ?, ?, ?)",
            ("test_session", "cron", "Test task")
        )
        conn.commit()

        # Mock platform sends
        with patch.object(notifications, "send_to_all_platforms") as mock_send:
            mock_send.return_value = {"telegram": (True, None)}

            with patch("agathos.notifications.logger"):
                notifications.send_notification(
                    cursor, conn, "test_session", "restart", "Test message"
                )

        # Verify notification recorded
        cursor.execute(
            "SELECT notification_type, delivered FROM notifications WHERE session_id = ?",
            ("test_session",)
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["notification_type"], "restart")

        conn.close()
        os.unlink(temp_db.name)

    def test_send_via_gateway_unavailable(self):
        """Test send_via_gateway when gateway module not available."""
        # Force ImportError by patching import
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "gateway" in name:
                raise ImportError("No module named 'gateway'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            with patch("agathos.notifications.logger"):
                delivered, error = notifications.send_via_gateway("test")

        self.assertFalse(delivered)
        self.assertIn("not available", error.lower())


if __name__ == "__main__":
    unittest.main()
