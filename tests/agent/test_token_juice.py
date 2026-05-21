"""Tests for agent/token_juice.py — compression safety gates."""

import pytest
from unittest.mock import patch, MagicMock

from agent.token_juice import (
    should_passthrough,
    classify_messages,
    compression_safe_to_apply,
    TOKENJUICE_PASSTHROUGH_BYTES,
    TOKENJUICE_PASSTHROUGH_COMMANDS,
    TOKENJUICE_MIN_COMPRESSION_RATIO,
)


class TestShouldPassthrough:
    """Gates 1 & 2: size threshold and file-inspection tool passthrough."""

    def test_tiny_output_passthrough(self):
        msg = {"role": "tool", "content": "short"}
        assert should_passthrough(msg) is True

    def test_above_threshold_not_passthrough(self):
        msg = {"role": "tool", "content": "x" * TOKENJUICE_PASSTHROUGH_BYTES}
        assert should_passthrough(msg) is False

    def test_non_tool_message_false(self):
        msg = {"role": "user", "content": "hello"}
        assert should_passthrough(msg) is False

    def test_terminal_cat_passthrough(self):
        msg = {"role": "tool", "content": "file contents " * 100}
        assert should_passthrough(
            msg,
            tool_name="terminal",
            tool_args='{"command": "cat ./file.txt"}',
        ) is True

    def test_terminal_tail_passthrough(self):
        msg = {"role": "tool", "content": "last 10 lines " * 100}
        assert should_passthrough(
            msg,
            tool_name="terminal",
            tool_args='{"command": "tail -n 50 /var/log/syslog"}',
        ) is True

    def test_terminal_head_passthrough(self):
        msg = {"role": "tool", "content": "first 10 lines " * 100}
        assert should_passthrough(
            msg,
            tool_name="terminal",
            tool_args='{"command": "head -20 data.csv"}',
        ) is True

    def test_terminal_bat_passthrough(self):
        msg = {"role": "tool", "content": "syntax highlighted " * 100}
        assert should_passthrough(
            msg,
            tool_name="terminal",
            tool_args='{"command": "bat README.md"}',
        ) is True

    def test_terminal_batcat_passthrough(self):
        msg = {"role": "tool", "content": "syntax highlighted " * 100}
        assert should_passthrough(
            msg,
            tool_name="terminal",
            tool_args='{"command": "batcat config.toml"}',
        ) is True

    def test_read_file_passthrough(self):
        """read_file tool output should be in passthrough set."""
        msg = {"role": "tool", "content": "file contents " * 100}
        assert should_passthrough(
            msg,
            tool_name="read_file",
            tool_args='{"path": "README.md"}',
        ) is True

    def test_terminal_non_inspection_command_not_passthrough(self):
        msg = {"role": "tool", "content": "build output " * 100}
        assert should_passthrough(
            msg,
            tool_name="terminal",
            tool_args='{"command": "cargo build --release"}',
        ) is False

    def test_non_terminal_tool_above_threshold_not_passthrough(self):
        msg = {"role": "tool", "content": "web search results " * 100}
        assert should_passthrough(
            msg,
            tool_name="web_search",
            tool_args='{"query": "test"}',
        ) is False


class TestClassifyMessages:
    def test_splits_protected_and_compressible(self):
        call_id_to_tool = {
            "call_1": ("terminal", '{"command": "cat file.txt"}'),
            "call_2": ("terminal", '{"command": "npm install"}'),
        }
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "big" * 200},
            {"role": "tool", "tool_call_id": "call_2", "content": "big" * 200},
        ]
        protected, compressible = classify_messages(messages, call_id_to_tool)
        assert len(protected) == 1  # cat output protected
        assert len(compressible) == 1  # npm install compressible

    def test_all_protected(self):
        messages = [
            {"role": "tool", "content": "tiny"},
            {"role": "tool", "content": "also small"},
        ]
        protected, compressible = classify_messages(messages)
        assert len(protected) == 2
        assert len(compressible) == 0


class TestGate3CompressionSafeToApply:
    """Gate 3: reject summaries that didn't achieve meaningful compression."""

    def test_good_compression_accepted(self):
        """50% compression → accepted."""
        assert compression_safe_to_apply(1000, 500) is True

    def test_borderline_rejected(self):
        """96% of original → rejected (>= 0.95 threshold)."""
        assert compression_safe_to_apply(1000, 960) is False

    def test_exact_threshold_rejected(self):
        """Exactly at threshold → rejected."""
        threshold_len = int(1000 * TOKENJUICE_MIN_COMPRESSION_RATIO)
        assert compression_safe_to_apply(1000, threshold_len) is False

    def test_just_below_threshold_accepted(self):
        """Just under 0.95 → accepted."""
        threshold_len = int(1000 * TOKENJUICE_MIN_COMPRESSION_RATIO) - 1
        assert compression_safe_to_apply(1000, max(threshold_len, 1)) is True

    def test_zero_original_rejected(self):
        assert compression_safe_to_apply(0, 0) is False

    def test_negative_original_rejected(self):
        assert compression_safe_to_apply(-1, 100) is False

    def test_short_summary_accepted(self):
        """Short summary of long content — well below threshold."""
        assert compression_safe_to_apply(5000, 200) is True

    def test_low_benefit_summary_rejected(self):
        """Summary nearly as long as original → rejected."""
        assert compression_safe_to_apply(500, 490) is False


class TestPassthroughCommandsSet:
    """read_file must be in the passthrough commands set."""

    def test_read_file_in_passthrough_commands(self):
        assert "read_file" in TOKENJUICE_PASSTHROUGH_COMMANDS

    def test_cat_family_in_passthrough_commands(self):
        for cmd in ["cat", "tail", "head", "bat", "batcat"]:
            assert cmd in TOKENJUICE_PASSTHROUGH_COMMANDS
