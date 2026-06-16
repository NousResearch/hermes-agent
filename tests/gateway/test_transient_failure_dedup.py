"""Tests for transient-failure user-message dedup (#47237).

When the gateway persists a user message after a transient provider failure
(429, timeout, auth error), subsequent retries of the same message must NOT
stack duplicate user turns in the transcript.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _dedup_should_skip(history, message_text, message_id=None):
    """Replicate the dedup guard from _handle_message_with_agent (line ~9207).

    Returns True when the persistence should be SKIPPED (message already
    present as the last entry in history).
    """
    if not history:
        return False
    _last = history[-1]
    if _last.get("role") == "user" and _last.get("content") == message_text:
        _last_mid = str(_last.get("message_id", ""))
        _curr_mid = str(message_id or "")
        if not _curr_mid or _last_mid == _curr_mid:
            return True
    return False


class TestTransientFailureDedup:
    """Verify the dedup guard prevents stacking duplicate user turns."""

    def test_empty_history_does_not_skip(self):
        """First attempt: history is empty → must persist."""
        assert _dedup_should_skip([], "Hello") is False

    def test_same_content_at_end_skips(self):
        """Second attempt: last history entry matches → skip."""
        history = [
            {"role": "user", "content": "Hello", "message_id": "m1"},
        ]
        assert _dedup_should_skip(history, "Hello", "m1") is True

    def test_same_content_no_message_id_skips(self):
        """When no message_id on either side, content match alone is enough."""
        history = [
            {"role": "user", "content": "Hello"},
        ]
        assert _dedup_should_skip(history, "Hello") is True

    def test_different_content_does_not_skip(self):
        """Different message content → must persist."""
        history = [
            {"role": "user", "content": "Hello", "message_id": "m1"},
        ]
        assert _dedup_should_skip(history, "World", "m2") is False

    def test_different_message_id_does_not_skip(self):
        """Same content but different message_id → distinct message, persist."""
        history = [
            {"role": "user", "content": "Hello", "message_id": "m1"},
        ]
        assert _dedup_should_skip(history, "Hello", "m2") is False

    def test_last_entry_is_assistant_does_not_skip(self):
        """Last entry is assistant → different message, must persist."""
        history = [
            {"role": "user", "content": "Hello", "message_id": "m1"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert _dedup_should_skip(history, "Hello", "m1") is False

    def test_current_no_message_id_matches_any_last_mid(self):
        """When current message has no id, match by content only."""
        history = [
            {"role": "user", "content": "Hello", "message_id": "m1"},
        ]
        assert _dedup_should_skip(history, "Hello", None) is True

    def test_last_no_message_id_current_has_id_does_not_skip(self):
        """Last has no id, current has id → can't confirm same message."""
        history = [
            {"role": "user", "content": "Hello"},
        ]
        # content matches but _last_mid="" != _curr_mid="m1"
        # and _curr_mid is truthy → skip condition fails
        assert _dedup_should_skip(history, "Hello", "m1") is False

    def test_three_retries_only_first_persists(self):
        """Simulate 3 retries: first persists, 2nd and 3rd are skipped."""
        history = []

        # First attempt: history empty → persist
        assert _dedup_should_skip(history, "Hello", "m1") is False
        history.append({"role": "user", "content": "Hello", "message_id": "m1"})

        # Second attempt: history has the message → skip
        assert _dedup_should_skip(history, "Hello", "m1") is True

        # Third attempt: still skip (history unchanged)
        assert _dedup_should_skip(history, "Hello", "m1") is True

    def test_mixed_history_only_checks_last(self):
        """Only the LAST entry matters — earlier duplicates don't affect."""
        history = [
            {"role": "user", "content": "Hello", "message_id": "m1"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Hello", "message_id": "m1"},
        ]
        # Last entry is user with same content+id → skip
        assert _dedup_should_skip(history, "Hello", "m1") is True
