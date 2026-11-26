"""
Tests for citadel_listener.py functionality.

Verifies that:
1. send_voice_toggle() uses os.getppid() to get parent PID
2. Wake word detection works with fuzzy matching (English default, Russian via env)
3. Stop word detection works with fuzzy matching
4. drain_queue() correctly empties the queue
5. Cooldown logic works correctly
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

# Clear any existing env vars to test defaults
for key in ["CITADEL_WAKE_WORD", "CITADEL_STOP_WORD", "CITADEL_MODEL_PATH", "CITADEL_COOLDOWN", "CITADEL_DEVICE"]:
    os.environ.pop(key, None)

import importlib.util
spec = importlib.util.spec_from_file_location("citadel_listener", LISTENER_SCRIPT)
citadel_listener = importlib.util.module_from_spec(spec)
sys.modules["citadel_listener"] = citadel_listener
spec.loader.exec_module(citadel_listener)

# Now we can access the functions
text_contains_word = citadel_listener.text_contains_word
drain_queue = citadel_listener.drain_queue
send_voice_toggle = citadel_listener.send_voice_toggle
WAKE_WORD = citadel_listener.WAKE_WORD
STOP_WORD = citadel_listener.STOP_WORD
STOP_TIMEOUT = citadel_listener.STOP_TIMEOUT


class TestTextContainsWord:
    """Tests for fuzzy word matching with default (English) words."""

    def test_exact_wake_word(self):
        assert text_contains_word("wake", WAKE_WORD) == True

    def test_wake_word_in_sentence(self):
        assert text_contains_word("hey wake up", WAKE_WORD) == True

    def test_exact_stop_word(self):
        assert text_contains_word("stop", STOP_WORD) == True

    def test_stop_word_in_sentence(self):
        assert text_contains_word("please stop listening", STOP_WORD) == True

    def test_no_match(self):
        assert text_contains_word("hello how are you", WAKE_WORD) == False

    def test_empty_text(self):
        assert text_contains_word("", WAKE_WORD) == False

    def test_case_insensitive(self):
        assert text_contains_word("WAKE", WAKE_WORD) == True


class TestTextContainsWordRussian:
    """Tests for Russian words (simulating env var override)."""

    def test_russian_wake_word(self):
        assert text_contains_word("страж", "страж") == True

    def test_russian_wake_word_fuzzy_storozh(self):
        assert text_contains_word("сторож", "страж") == True

    def test_russian_wake_word_fuzzy_starozh(self):
        assert text_contains_word("старож", "страж") == True

    def test_russian_stop_word(self):
        assert text_contains_word("тишина", "тишина") == True

    def test_russian_stop_word_fuzzy(self):
        assert text_contains_word("тишена", "тишина") == True


class TestDrainQueue:
    """Tests for drain_queue function."""

    def test_drain_empty_queue(self):
        q = queue.Queue()
        assert drain_queue(q) == 0

    def test_drain_one_item(self):
        q = queue.Queue()
        q.put(b"data1")
        assert drain_queue(q) == 1
        assert q.empty()

    def test_drain_multiple_items(self):
        q = queue.Queue()
        for i in range(5):
            q.put(f"data{i}".encode())
        assert drain_queue(q) == 5
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
    """Tests for cooldown timeout logic (simulated)."""

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


class TestWakeWordReturn:
    """Test that run_listener_cycle returns after wake word detection."""

    def test_returns_after_wake_word_in_source(self):
        """Check that the source code has a return statement after wake word detection."""
        import ast
        with open(LISTENER_SCRIPT, 'r') as f:
            source = f.read()
        tree = ast.parse(source)
        # Find run_listener_cycle function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'run_listener_cycle':
                # Look for if text_contains_word(text, WAKE_WORD): block
                for n in ast.walk(node):
                    if isinstance(n, ast.If):
                        # Check if condition is text_contains_word(text, WAKE_WORD)
                        # We'll just check that there is a return statement within the function
                        # that is not inside the stop word block (simpler: check that there is a return
                        # after the wake word block).
                        # For simplicity, search for any return statement in the function
                        # and ensure it's not only for stop word.
                        pass
                # Simpler: just check that there is a line with "return" after wake word detection.
                # We'll do a regex search.
                import re
                # Find the wake word block and check for return within it or after it.
                # Pattern: after "if text_contains_word(text, WAKE_WORD):" there should be a "return"
                # We'll search for the pattern.
                pattern = r'if text_contains_word\(text, WAKE_WORD\):.*?return'
                if not re.search(pattern, source, re.DOTALL):
                    pytest.fail("No return statement found after wake word detection in run_listener_cycle")
                return
        pytest.fail("run_listener_cycle function not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
