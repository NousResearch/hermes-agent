"""Tests for cross-session prompt cache optimization (#68191).

Validates that the system prompt is split into stable/volatile content
blocks so the stable prefix can hit cross-session cache even when
context/volatile bytes differ between sessions.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.prompt_caching import apply_anthropic_cache_control, _build_marker


def _make_system_msg(content):
    """Helper to create a system message."""
    return {"role": "system", "content": content}


def _make_user_msg(content):
    """Helper to create a user message."""
    return {"role": "user", "content": content}


class TestContentBlockLayout:
    """Test that content blocks with cache_control are preserved."""

    def test_content_blocks_with_cache_control_preserved(self):
        """System message with content blocks carrying cache_control markers
        should not be re-marked by apply_anthropic_cache_control."""
        blocks = [
            {"type": "text", "text": "stable prefix", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "volatile suffix"},
        ]
        messages = [
            {"role": "system", "content": blocks},
            _make_user_msg("hello"),
        ]
        result = apply_anthropic_cache_control(messages)
        sys_content = result[0]["content"]
        # The stable block should still have its cache_control
        assert sys_content[0]["cache_control"] == {"type": "ephemeral"}
        # The volatile block should NOT have cache_control
        assert "cache_control" not in sys_content[1]

    def test_content_blocks_get_breakpoint_budget(self):
        """When system message already has cache_control in content blocks,
        apply_anthropic_cache_control should use remaining breakpoints on
        non-system messages."""
        blocks = [
            {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "volatile"},
        ]
        messages = [
            {"role": "system", "content": blocks},
            _make_user_msg("msg1"),
            {"role": "assistant", "content": "reply1"},
            _make_user_msg("msg2"),
            {"role": "assistant", "content": "reply2"},
            _make_user_msg("msg3"),
        ]
        result = apply_anthropic_cache_control(messages)
        # System already marked -> 1 breakpoint used
        # Should mark last 3 non-system messages (remaining budget = 3)
        non_sys = [m for m in result if m.get("role") != "system"]
        marked = [m for m in non_sys if _has_cache_control(m)]
        assert len(marked) == 3

    def test_plain_string_system_gets_legacy_layout(self):
        """Plain string system message should get the legacy single-breakpoint
        layout (system_and_3)."""
        messages = [
            _make_system_msg("full system prompt"),
            _make_user_msg("msg1"),
            {"role": "assistant", "content": "reply1"},
            _make_user_msg("msg2"),
            {"role": "assistant", "content": "reply2"},
            _make_user_msg("msg3"),
        ]
        result = apply_anthropic_cache_control(messages)
        # System should have cache_control
        assert _has_cache_control(result[0])
        # Last 3 non-system messages should have cache_control
        non_sys = [m for m in result if m.get("role") != "system"]
        marked = [m for m in non_sys if _has_cache_control(m)]
        assert len(marked) == 3


def _has_cache_control(msg):
    """Check if a message has cache_control in its content or top-level."""
    if "cache_control" in msg:
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and "cache_control" in b
            for b in content
        )
    return False
