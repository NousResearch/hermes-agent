"""Tests for session-topics topic_detection integration.

Tests the _TopicDriftTracker dynamic threshold behavior and the
_check_topic_shift_from_history bridge method.

These tests focus on the interaction between topic_detection and the
existing hysteresis mechanism — not the detection logic itself (covered
by tests/test_topic_detection.py).
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import importlib.util
import sys

# Load run_agent to access _TopicDriftTracker and AIAgent._check_topic_shift_from_history
# We can import directly since this branch has run_agent.py at the repo root


class TestTopicDriftTrackerThreshold:
    """Test the new dynamic threshold methods on _TopicDriftTracker."""

    def test_base_threshold_preserved(self):
        from run_agent import _TopicDriftTracker
        tracker = _TopicDriftTracker(threshold=2)
        assert tracker._base_threshold == 2
        assert tracker._threshold == 2

    def test_set_threshold(self):
        from run_agent import _TopicDriftTracker
        tracker = _TopicDriftTracker(threshold=2)
        tracker.set_threshold(1)
        assert tracker._threshold == 1

    def test_set_threshold_floors_at_1(self):
        from run_agent import _TopicDriftTracker
        tracker = _TopicDriftTracker(threshold=2)
        tracker.set_threshold(0)
        tracker.set_threshold(-5)
        assert tracker._threshold == 1

    def test_reset_threshold(self):
        from run_agent import _TopicDriftTracker
        tracker = _TopicDriftTracker(threshold=3)
        tracker.set_threshold(1)
        assert tracker._threshold == 1
        tracker.reset_threshold()
        assert tracker._threshold == 3

    def test_lowered_threshold_allows_single_confirmation(self):
        """With threshold=1, a single feed for a new name should confirm."""
        from run_agent import _TopicDriftTracker
        tracker = _TopicDriftTracker(threshold=2)
        tracker.set_threshold(1)
        # First feed for "new-topic" should confirm immediately
        result = tracker.feed("new-topic")
        assert result == "new-topic"

    def test_default_threshold_still_requires_two(self):
        """Without lowering, two consecutive signals are still needed."""
        from run_agent import _TopicDriftTracker
        tracker = _TopicDriftTracker(threshold=2)
        # First signal: no confirmation
        result1 = tracker.feed("new-topic")
        assert result1 is None
        # Second signal: confirmed
        result2 = tracker.feed("new-topic")
        assert result2 == "new-topic"

    def test_reset_threshold_after_turn(self):
        """After reset_threshold, the original hysteresis is restored."""
        from run_agent import _TopicDriftTracker
        tracker = _TopicDriftTracker(threshold=2)
        tracker.set_threshold(1)
        tracker.feed("topic-a")  # confirms immediately (threshold=1)
        tracker.reset_threshold()
        # Now needs 2 again
        result = tracker.feed("topic-b")
        assert result is None  # first signal, not enough


class TestCheckTopicShiftFromHistory:
    """Test the _check_topic_shift_from_history bridge method."""

    def _make_agent(self, history=None):
        """Create a minimal mock with _check_topic_shift_from_history."""
        from run_agent import AIAgent
        agent = MagicMock(spec=AIAgent)
        agent.conversation_history = history or []
        # Bind the real method
        agent._check_topic_shift_from_history = (
            AIAgent._check_topic_shift_from_history.__get__(agent, AIAgent)
        )
        return agent

    def test_insufficient_history(self):
        """Less than 2 user messages → None."""
        agent = self._make_agent(history=[
            {"role": "user", "content": "only one message"},
        ])
        result = agent._check_topic_shift_from_history()
        assert result is None

    def test_empty_history(self):
        """No messages → None."""
        agent = self._make_agent(history=[])
        result = agent._check_topic_shift_from_history()
        assert result is None

    def test_no_user_messages(self):
        """Only assistant messages → None."""
        agent = self._make_agent(history=[
            {"role": "assistant", "content": "response 1"},
            {"role": "assistant", "content": "response 2"},
        ])
        result = agent._check_topic_shift_from_history()
        assert result is None

    def test_two_user_messages_calls_detect(self):
        """With 2 user messages, detect_topic_shift should be called."""
        agent = self._make_agent(history=[
            {"role": "user", "content": "help me with docker"},
            {"role": "assistant", "content": "sure"},
            {"role": "user", "content": "now write a novel"},
        ])
        with patch("agent.topic_detection.detect_topic_shift") as mock_detect:
            mock_detect.return_value = {
                "topic_continuation": False,
                "method": "s6_only",
            }
            result = agent._check_topic_shift_from_history()
            assert result is not None
            assert result["topic_continuation"] is False
            mock_detect.assert_called_once()
            # Verify it was called with the last two user messages
            call_args = mock_detect.call_args
            assert call_args[0][0] == "now write a novel"  # current_msg
            assert call_args[1]["prev_msg"] == "help me with docker"

    def test_handles_non_string_content(self):
        """Non-string content in history should be skipped."""
        agent = self._make_agent(history=[
            {"role": "user", "content": None},
            {"role": "user", "content": "valid message"},
        ])
        with patch("agent.topic_detection.detect_topic_shift") as mock_detect:
            result = agent._check_topic_shift_from_history()
            # Only one valid user message → None
            assert result is None

    def test_exception_returns_none(self):
        """If detect_topic_shift raises, return None."""
        agent = self._make_agent(history=[
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "msg2"},
        ])
        with patch("agent.topic_detection.detect_topic_shift",
                   side_effect=Exception("boom")):
            result = agent._check_topic_shift_from_history()
            assert result is None
