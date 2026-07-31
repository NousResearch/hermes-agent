"""Tests for agent/trajectory.py — trajectory saving utilities and static helpers.

Covers Bug #36322: bring module coverage to >=70%.
"""

import json
import logging
import os

import pytest

from agent.trajectory import (
    convert_scratchpad_to_think,
    has_incomplete_scratchpad,
    save_trajectory,
)


class TestConvertScratchpadToThink:
    """Lines 16-22 of agent/trajectory.py.

    Behavior summary: the function checks for the opening tag
    <REASONING_SCRATCHPAD> in the content. If not present (or content is
    falsy), it returns the input unchanged. Otherwise it replaces both
    opening and closing tags via two global str.replace calls.

    Note: '</REASONING_SCRATCHPAD>' does NOT contain '<REASONING_SCRATCHPAD>'
    as a substring (because of the leading '</' vs '<'), so a closing tag
    alone is NOT converted. This is implementation-level behavior, not a
    documented contract.
    """

    def test_returns_empty_string_unchanged(self):
        """Empty string is a no-op (early return on falsy)."""
        assert convert_scratchpad_to_think("") == ""

    def test_returns_none_unchanged(self):
        """None is falsy too — early returns the input unchanged."""
        assert convert_scratchpad_to_think(None) is None

    def test_returns_text_unchanged_when_no_tags(self):
        """Plain text without tags passes through."""
        text = "Hello, world! Nothing to convert here."
        assert convert_scratchpad_to_think(text) == text

    def test_converts_opening_tag(self):
        """<REASONING_SCRATCHPAD> -> <think>"""
        assert convert_scratchpad_to_think("<REASONING_SCRATCHPAD>hello") == "<think>hello"

    def test_does_not_convert_closing_tag_alone(self):
        """Closing tag without opening: the function's check is for the opening
        substring, which 'hello</REASONING_SCRATCHPAD>' does NOT contain
        (because '</' != '<'). So early return; no conversion."""
        text = "hello</REASONING_SCRATCHPAD>"
        assert convert_scratchpad_to_think(text) == "hello</REASONING_SCRATCHPAD>"

    def test_converts_both_tags_when_opening_present(self):
        """When opening tag is present, both opening and closing get replaced."""
        text = "<REASONING_SCRATCHPAD>thinking</REASONING_SCRATCHPAD>"
        assert convert_scratchpad_to_think(text) == "<think>thinking</think>"

    def test_converts_all_opening_tags(self):
        """str.replace is global — all opening tags get converted."""
        text = "<REASONING_SCRATCHPAD>a<REASONING_SCRATCHPAD>b"
        assert convert_scratchpad_to_think(text) == "<think>a<think>b"

    def test_converts_only_opening_when_no_closing(self):
        """Opening tag without matching closing tag: opening still converts,
        closing is absent so the second replace is a no-op."""
        text = "<REASONING_SCRATCHPAD>still thinking..."
        assert convert_scratchpad_to_think(text) == "<think>still thinking..."

    def test_idempotent_after_first_conversion(self):
        """Calling twice is safe — second call sees no opening tags to convert."""
        once = convert_scratchpad_to_think("<REASONING_SCRATCHPAD>hello</REASONING_SCRATCHPAD>")
        twice = convert_scratchpad_to_think(once)
        assert once == twice == "<think>hello</think>"


class TestHasIncompleteScratchpad:
    """Lines 25-28 of agent/trajectory.py.

    Behavior: returns True iff the opening tag is present AND the closing
    tag is not present. This is a presence check, NOT a balance check.
    """

    def test_returns_false_for_empty_string(self):
        """Empty string is not incomplete."""
        assert has_incomplete_scratchpad("") is False

    def test_returns_false_for_none(self):
        """None is not incomplete."""
        assert has_incomplete_scratchpad(None) is False

    def test_returns_false_for_plain_text(self):
        """Plain text without tags is not incomplete."""
        assert has_incomplete_scratchpad("just plain text") is False

    def test_returns_false_for_properly_paired_tags(self):
        """A complete pair (both opening and closing) is not incomplete."""
        text = "<REASONING_SCRATCHPAD>thinking</REASONING_SCRATCHPAD>"
        assert has_incomplete_scratchpad(text) is False

    def test_returns_false_for_text_with_only_closing_tag(self):
        """Closing tag without opening: not 'incomplete' (no opening to be incomplete)."""
        text = "thought</REASONING_SCRATCHPAD>"
        assert has_incomplete_scratchpad(text) is False

    def test_returns_true_for_opening_without_closing(self):
        """An opening tag with no closing is incomplete."""
        text = "<REASONING_SCRATCHPAD>still thinking..."
        assert has_incomplete_scratchpad(text) is True

    def test_returns_false_for_two_opening_one_closing(self):
        """Multiple opening + one closing: the closing IS present, so not incomplete.
        This is a presence check, not a balance check."""
        text = "<REASONING_SCRATCHPAD>a<REASONING_SCRATCHPAD>b</REASONING_SCRATCHPAD>"
        assert has_incomplete_scratchpad(text) is False


class TestSaveTrajectory:
    """Lines 41-56 of agent/trajectory.py."""

    def test_writes_to_completed_file_when_completed_true(self, tmp_path):
        """completed=True writes to trajectory_samples.jsonl by default."""
        traj = [{"from": "human", "value": "hello"}]
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            save_trajectory(traj, "gpt-4", completed=True)
            assert os.path.exists("trajectory_samples.jsonl")
            with open("trajectory_samples.jsonl", "r", encoding="utf-8") as f:
                line = f.readline().strip()
            entry = json.loads(line)
            assert entry["conversations"] == traj
            assert entry["model"] == "gpt-4"
            assert entry["completed"] is True
            assert "timestamp" in entry
        finally:
            os.chdir(old_cwd)

    def test_writes_to_failed_file_when_completed_false(self, tmp_path):
        """completed=False writes to failed_trajectories.jsonl by default."""
        traj = [{"from": "human", "value": "oops"}]
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            save_trajectory(traj, "gpt-4", completed=False)
            assert os.path.exists("failed_trajectories.jsonl")
            with open("failed_trajectories.jsonl", "r", encoding="utf-8") as f:
                line = f.readline().strip()
            entry = json.loads(line)
            assert entry["completed"] is False
        finally:
            os.chdir(old_cwd)

    def test_respects_explicit_filename_override(self, tmp_path):
        """Explicit filename parameter overrides the default selection."""
        traj = [{"from": "user", "value": "test"}]
        custom_path = tmp_path / "custom.jsonl"
        save_trajectory(traj, "model-x", completed=True, filename=str(custom_path))
        assert custom_path.exists()
        with open(custom_path, "r", encoding="utf-8") as f:
            entry = json.loads(f.readline().strip())
        assert entry["model"] == "model-x"

    def test_appends_multiple_entries(self, tmp_path):
        """Multiple calls append (one per line, JSONL format)."""
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            save_trajectory([{"i": 1}], "m", completed=True)
            save_trajectory([{"i": 2}], "m", completed=True)
            save_trajectory([{"i": 3}], "m", completed=True)
            with open("trajectory_samples.jsonl", "r", encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 3
            assert json.loads(lines[0].strip())["conversations"] == [{"i": 1}]
            assert json.loads(lines[1].strip())["conversations"] == [{"i": 2}]
            assert json.loads(lines[2].strip())["conversations"] == [{"i": 3}]
        finally:
            os.chdir(old_cwd)

    def test_logs_warning_on_failure(self, caplog):
        """Failed writes log a warning rather than raise."""
        invalid_path = "/nonexistent_dir_xyz_definitely_does_not_exist/trajectory.jsonl"
        with caplog.at_level(logging.WARNING, logger="agent.trajectory"):
            # Should not raise; should log a warning
            save_trajectory([{"i": 1}], "m", completed=True, filename=invalid_path)
        assert any("Failed to save trajectory" in r.message for r in caplog.records)

    def test_handles_unicode_in_trajectory(self, tmp_path):
        """Non-ASCII characters in the trajectory are preserved (ensure_ascii=False)."""
        traj = [{"from": "human", "value": "héllo wörld 你好"}]
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            save_trajectory(traj, "m", completed=True)
            with open("trajectory_samples.jsonl", "r", encoding="utf-8") as f:
                content = f.read()
            # ensure_ascii=False means the actual unicode chars are in the file
            assert "héllo" in content
            assert "你好" in content
        finally:
            os.chdir(old_cwd)
