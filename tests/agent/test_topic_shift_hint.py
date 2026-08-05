"""Tests for _build_topic_shift_hint in agent/turn_context.

Tests the hint injection: when topic_detection detects a shift, a hint
string is returned that prompts the LLM to emit a new TOPIC: line.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestBuildTopicShiftHint:
    """Test the hint generation logic."""

    def _import(self):
        from agent.turn_context import _build_topic_shift_hint
        return _build_topic_shift_hint

    def _make_agent(self):
        return MagicMock()

    def test_insufficient_history(self):
        """Less than 2 user messages → None."""
        hint_fn = self._import()
        agent = self._make_agent()
        result = hint_fn(
            agent, "current message",
            [{"role": "user", "content": "only one"}],
        )
        assert result is None

    def test_empty_history(self):
        """No messages → None."""
        hint_fn = self._import()
        agent = self._make_agent()
        result = hint_fn(agent, "msg", [])
        assert result is None

    def test_no_user_messages(self):
        """Only assistant messages → None."""
        hint_fn = self._import()
        agent = self._make_agent()
        result = hint_fn(agent, "msg", [
            {"role": "assistant", "content": "response 1"},
            {"role": "assistant", "content": "response 2"},
        ])
        assert result is None

    def test_same_topic_returns_none(self):
        """When detect_topic_shift says same topic → None."""
        hint_fn = self._import()
        agent = self._make_agent()
        with patch("agent.topic_detection.detect_topic_shift") as mock_detect:
            mock_detect.return_value = {
                "topic_continuation": True,
                "method": "s6_only",
                "confidence": 0.89,
            }
            result = hint_fn(
                agent, "same topic message",
                [
                    {"role": "user", "content": "previous message"},
                    {"role": "assistant", "content": "response"},
                    {"role": "user", "content": "same topic message"},
                ],
            )
            assert result is None

    def test_shift_returns_hint(self):
        """When detect_topic_shift says shift → hint string."""
        hint_fn = self._import()
        agent = self._make_agent()
        with patch("agent.topic_detection.detect_topic_shift") as mock_detect:
            mock_detect.return_value = {
                "topic_continuation": False,
                "method": "s6_only",
                "confidence": 0.89,
            }
            result = hint_fn(
                agent, "completely different topic",
                [
                    {"role": "user", "content": "docker setup"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "completely different topic"},
                ],
            )
            assert result is not None
            assert "Topic shift detected" in result
            assert "TOPIC:" in result
            assert "english-kebab-case-name" in result

    def test_fallback_method_returns_none(self):
        """When detect_topic_shift returns fallback → None."""
        hint_fn = self._import()
        agent = self._make_agent()
        with patch("agent.topic_detection.detect_topic_shift") as mock_detect:
            mock_detect.return_value = {
                "topic_continuation": False,
                "method": "fallback",
                "confidence": 0.0,
            }
            result = hint_fn(
                agent, "msg",
                [
                    {"role": "user", "content": "prev"},
                    {"role": "user", "content": "msg"},
                ],
            )
            assert result is None

    def test_exception_returns_none(self):
        """If detect_topic_shift raises → None."""
        hint_fn = self._import()
        agent = self._make_agent()
        with patch("agent.topic_detection.detect_topic_shift",
                   side_effect=Exception("boom")):
            result = hint_fn(
                agent, "msg",
                [
                    {"role": "user", "content": "prev"},
                    {"role": "user", "content": "msg"},
                ],
            )
            assert result is None

    def test_non_string_content_skipped(self):
        """Non-string content should be skipped, not crash."""
        hint_fn = self._import()
        agent = self._make_agent()
        with patch("agent.topic_detection.detect_topic_shift") as mock_detect:
            mock_detect.return_value = {
                "topic_continuation": False,
                "method": "s6_only",
            }
            result = hint_fn(
                agent, "msg",
                [
                    {"role": "user", "content": None},
                    {"role": "user", "content": 123},
                    {"role": "user", "content": "valid"},
                    {"role": "user", "content": "msg"},
                ],
            )
            # Should have called detect with "valid" as prev
            mock_detect.assert_called_once()
            call_kwargs = mock_detect.call_args
            assert call_kwargs[1]["prev_msg"] == "valid"

    def test_hint_is_english_only(self):
        """Hint must not contain Chinese format instructions."""
        hint_fn = self._import()
        agent = self._make_agent()
        with patch("agent.topic_detection.detect_topic_shift") as mock_detect:
            mock_detect.return_value = {
                "topic_continuation": False,
                "method": "s6_only",
            }
            result = hint_fn(
                agent, "new topic",
                [
                    {"role": "user", "content": "old topic"},
                    {"role": "user", "content": "new topic"},
                ],
            )
            assert result is not None
            # No Chinese characters in the hint
            assert "主题" not in result
            assert "话题" not in result
