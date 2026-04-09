#!/usr/bin/env python3
"""Tests for ToolCallMonitor and check_session_entropy."""

import os
import sys
import json
import time
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wal_monitor import ToolCallMonitor, check_session_entropy, ToolCallEvent


class TestToolCallMonitorInternals(unittest.TestCase):
    """Test ToolCallMonitor internal methods."""

    def test_ts_float_with_number(self):
        self.assertEqual(ToolCallMonitor._ts_float(1700000000.0), 1700000000.0)
        self.assertEqual(ToolCallMonitor._ts_float(42), 42.0)

    def test_ts_float_with_iso_string(self):
        result = ToolCallMonitor._ts_float('2026-04-08T20:00:00')
        self.assertGreater(result, 0)

    def test_ts_float_with_invalid(self):
        self.assertEqual(ToolCallMonitor._ts_float(None), 0.0)
        self.assertEqual(ToolCallMonitor._ts_float(''), 0.0)
        self.assertEqual(ToolCallMonitor._ts_float('not-a-date'), 0.0)

    def test_enqueue_increments_cursor(self):
        monitor = ToolCallMonitor()
        event = ToolCallEvent(cursor=0, session_id='s1', event_type='tool_call', tool_name='read_file')
        monitor._enqueue(event)
        self.assertEqual(event.cursor, 1)
        self.assertEqual(len(monitor._queue), 1)

        event2 = ToolCallEvent(cursor=0, session_id='s1', event_type='tool_call', tool_name='write_file')
        monitor._enqueue(event2)
        self.assertEqual(event2.cursor, 2)
        self.assertEqual(len(monitor._queue), 2)

    def test_get_events_drains_queue(self):
        monitor = ToolCallMonitor()
        for i in range(5):
            monitor._enqueue(ToolCallEvent(cursor=0, session_id='s1', event_type='tool_call'))

        events = monitor.get_events(limit=3)
        self.assertEqual(len(events), 3)
        self.assertEqual(len(monitor._queue), 2)

        events2 = monitor.get_events()
        self.assertEqual(len(events2), 2)
        self.assertEqual(len(monitor._queue), 0)

    def test_queue_trimmed_at_1000(self):
        monitor = ToolCallMonitor()
        for i in range(1100):
            monitor._enqueue(ToolCallEvent(cursor=0, session_id='s1', event_type='tool_call'))
        self.assertEqual(len(monitor._queue), 1000)


class TestCheckSessionEntropy(unittest.TestCase):
    """Test one-shot entropy check."""

    @patch('wal_monitor._HERMES_INTERNALS_AVAILABLE', False)
    def test_returns_error_when_hermes_unavailable(self):
        result = check_session_entropy('test_session')
        self.assertIn('error', result)


class TestToolCallMonitorThreading(unittest.TestCase):
    """Test threading behavior."""

    def test_start_stop_without_hermes(self):
        """Monitor should not start when hermes internals unavailable."""
        with patch('wal_monitor._HERMES_INTERNALS_AVAILABLE', False):
            monitor = ToolCallMonitor()
            monitor.start()
            self.assertFalse(monitor._running)
            monitor.stop()  # Should not hang

    def test_stop_is_idempotent(self):
        """Calling stop multiple times should not hang."""
        monitor = ToolCallMonitor()
        monitor.stop()
        monitor.stop()


if __name__ == '__main__':
    unittest.main(verbosity=2)
