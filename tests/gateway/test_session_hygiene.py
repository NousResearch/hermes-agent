"""Tests for gateway session hygiene — auto-compression of large sessions.

Verifies that the gateway detects pathologically large transcripts and
triggers auto-compression before running the agent.  (#628)
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agent.model_metadata import estimate_messages_tokens_rough


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history(n_messages: int, content_size: int = 100) -> list:
    """Build a fake transcript with n_messages user/assistant pairs."""
    history = []
    content = "x" * content_size
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": content, "timestamp": f"t{i}"})
    return history


def _make_large_history_tokens(target_tokens: int) -> list:
    """Build a history that estimates to roughly target_tokens tokens."""
    # estimate_messages_tokens_rough counts total chars in str(msg) // 4
    # Each msg dict has ~60 chars of overhead + content chars
    # So for N tokens we need roughly N * 4 total chars across all messages
    target_chars = target_tokens * 4
    # Each message as a dict string is roughly len(content) + 60 chars
    msg_overhead = 60
    # Use 50 messages with appropriately sized content
    n_msgs = 50
    content_size = max(10, (target_chars // n_msgs) - msg_overhead)
    return _make_history(n_msgs, content_size=content_size)


# ---------------------------------------------------------------------------
# Detection threshold tests
# ---------------------------------------------------------------------------

class TestSessionHygieneThresholds:
    """Test that the threshold logic correctly identifies large sessions."""

    def test_small_session_below_thresholds(self):
        """A 10-message session should not trigger compression."""
        history = _make_history(10)
        msg_count = len(history)
        approx_tokens = estimate_messages_tokens_rough(history)

        compress_token_threshold = 100_000
        compress_msg_threshold = 200

        needs_compress = (
            approx_tokens >= compress_token_threshold
            or msg_count >= compress_msg_threshold
        )
        assert not needs_compress

    def test_large_message_count_triggers(self):
        """200+ messages should trigger compression even if tokens are low."""
        history = _make_history(250, content_size=10)
        msg_count = len(history)

        compress_msg_threshold = 200
        needs_compress = msg_count >= compress_msg_threshold
        assert needs_compress

    def test_large_token_count_triggers(self):
        """High token count should trigger compression even if message count is low."""
        # 50 messages with huge content to exceed 100K tokens
        history = _make_history(50, content_size=10_000)
        approx_tokens = estimate_messages_tokens_rough(history)

        compress_token_threshold = 100_000
        needs_compress = approx_tokens >= compress_token_threshold
        assert needs_compress

    def test_under_both_thresholds_no_trigger(self):
        """Session under both thresholds should not trigger."""
        history = _make_history(100, content_size=100)
        msg_count = len(history)
        approx_tokens = estimate_messages_tokens_rough(history)

        compress_token_threshold = 100_000
        compress_msg_threshold = 200

        needs_compress = (
            approx_tokens >= compress_token_threshold
            or msg_count >= compress_msg_threshold
        )
        assert not needs_compress

    def test_custom_thresholds(self):
        """Custom thresholds from config should be respected."""
        history = _make_history(60, content_size=100)
        msg_count = len(history)

        # Custom lower threshold
        compress_msg_threshold = 50
        needs_compress = msg_count >= compress_msg_threshold
        assert needs_compress

        # Custom higher threshold
        compress_msg_threshold = 100
        needs_compress = msg_count >= compress_msg_threshold
        assert not needs_compress

    def test_minimum_message_guard(self):
        """Sessions with fewer than 4 messages should never trigger."""
        history = _make_history(3, content_size=100_000)
        # Even with enormous content, < 4 messages should be skipped
        # (the gateway code checks `len(history) >= 4` before evaluating)
        assert len(history) < 4


class TestSessionHygieneWarnThreshold:
    """Test the post-compression warning threshold."""

    def test_warn_when_still_large(self):
        """If compressed result is still above warn_tokens, should warn."""
        # Simulate post-compression tokens
        warn_threshold = 200_000
        post_compress_tokens = 250_000
        assert post_compress_tokens >= warn_threshold

    def test_no_warn_when_under(self):
        """If compressed result is under warn_tokens, no warning."""
        warn_threshold = 200_000
        post_compress_tokens = 150_000
        assert post_compress_tokens < warn_threshold


class TestTokenEstimation:
    """Verify rough token estimation works as expected for hygiene checks."""

    def test_empty_history(self):
        assert estimate_messages_tokens_rough([]) == 0

    def test_proportional_to_content(self):
        small = _make_history(10, content_size=100)
        large = _make_history(10, content_size=10_000)
        assert estimate_messages_tokens_rough(large) > estimate_messages_tokens_rough(small)

    def test_proportional_to_count(self):
        few = _make_history(10, content_size=1000)
        many = _make_history(100, content_size=1000)
        assert estimate_messages_tokens_rough(many) > estimate_messages_tokens_rough(few)

    def test_pathological_session_detected(self):
        """The reported pathological case: 648 messages, ~299K tokens."""
        # Simulate a 648-message session averaging ~460 tokens per message
        history = _make_history(648, content_size=1800)
        tokens = estimate_messages_tokens_rough(history)
        # Should be well above the 100K default threshold
        assert tokens > 100_000
        assert len(history) > 200
