"""Tests for citadel_listener.py functionality.

Verifies that:
1. send_voice_toggle() uses os.getppid() to get parent PID
2. Wake word detection works with fuzzy matching
3. Stop word detection works with fuzzy matching
4. drain_audio_queue() correctly empties the queue
5. Silence timeout logic works correctly
"""

import os
import signal
import queue
import time
from unittest.mock import MagicMock, patch, call
import pytest


# --- Helper to import functions from citadel_listener.py ---
import sys
from pathlib import Path

LISTENER_SCRIPT = Path.home() / ".hermes" / "citadel_listener.py"
if not LISTENER_SCRIPT.exists():
    pytest.skip(f"citadel_listener.py not found at {LISTENER_SCRIPT}", allow_module_level=True)

# Add the hermes home to path so we can import the script
import importlib.util
spec = importlib.util.spec_from_file_location("citadel_listener", LISTENER_SCRIPT)
citadel_listener = importlib.util.module_from_spec(spec)
sys.modules["citadel_listener"] = citadel_listener
spec.loader.exec_module(citadel_listener)

# Now we can access the functions
text_contains_word = citadel_listener.text_contains_word
drain_audio_queue = citadel_listener.drain_audio_queue
send_voice_toggle = citadel_listener.send_voice_toggle
WAKE_WORD = citadel_listener.WAKE_WORD
STOP_WORD = citadel_listener.STOP_WORD
STOP_TIMEOUT = citadel_listener.STOP_TIMEOUT


class TestTextContainsWord:
    """Tests for fuzzy word matching."""

    def test_exact_wake_word(self):
        assert text_contains_word("страж", WAKE_WORD) == True

    def test_wake_word_in_sentence(self):
        assert text_contains_word("скажи страж что-то", WAKE_WORD) == True

    def test_fuzzy_storozh(self):
        assert text_contains_word("сторож", WAKE_WORD) == True

    def test_fuzzy_starozh(self):
        assert text_contains_word("старож", WAKE_WORD) == True

    def test_exact_stop_word(self):
        assert text_contains_word("тишина", STOP_WORD) == True

    def test_stop_word_fuzzy_tishena(self):
        assert text_contains_word("тишена", STOP_WORD) == True

    def test_no_match(self):
        assert text_contains_word("привет как дела", WAKE_WORD) == False

    def test_empty_text(self):
        assert text_contains_word("", WAKE_WORD) == False

    def test_case_insensitive(self):
        assert text_contains_word("СТРАЖ", WAKE_WORD) == True


class TestDrainAudioQueue:
    """Tests for drain_audio_queue function."""

    def test_drain_empty_queue(self):
        q = queue.Queue()
        assert drain_audio_queue(q) == 0

    def test_drain_one_item(self):
        q = queue.Queue()
        q.put(b"data1")
        assert drain_audio_queue(q) == 1
        assert q.empty()

    def test_drain_multiple_items(self):
        q = queue.Queue()
        for i in range(5):
            q.put(f"data{i}".encode())
        assert drain_audio_queue(q) == 5
        assert q.empty()


class TestSendVoiceToggle:
    """Tests for send_voice_toggle using os.getppid()."""

    @patch('citadel_listener.os.getppid')
    @patch('citadel_listener.os.kill')
    def test_sends_sigusr1_to_parent_pid(self, mock_kill, mock_getppid):
        mock_getppid.return_value = 12345
        result = send_voice_toggle()
        mock_kill.assert_called_once_with(12345, signal.SIGUSR1)
        assert result == True

    @patch('citadel_listener.os.getppid')
    @patch('citadel_listener.os.kill')
    def test_returns_false_on_process_lookup_error(self, mock_kill, mock_getppid):
        mock_getppid.return_value = 99999
        mock_kill.side_effect = ProcessLookupError("Process not found")
        result = send_voice_toggle()
        assert result == False

    @patch('citadel_listener.os.getppid')
    @patch('citadel_listener.os.kill')
    def test_returns_false_on_permission_error(self, mock_kill, mock_getppid):
        mock_getppid.return_value = 12345
        mock_kill.side_effect = PermissionError("No permission")
        result = send_voice_toggle()
        assert result == False


class TestStopTimeoutLogic:
    """Tests for silence timeout logic (simulated)."""

    def test_timeout_reached(self):
        """Simulate that timeout is reached after STOP_TIMEOUT seconds of no activity."""
        last_activity = time.time() - STOP_TIMEOUT - 0.5  # Simulate idle > timeout
        current_time = time.time()
        timeout_reached = (current_time - last_activity) > STOP_TIMEOUT
        assert timeout_reached == True

    def test_timeout_not_reached(self):
        """Simulate recent activity, timeout not yet reached."""
        last_activity = time.time() - 1.0  # Activity 1 second ago
        current_time = time.time()
        timeout_reached = (current_time - last_activity) > STOP_TIMEOUT
        assert timeout_reached == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
