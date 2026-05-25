"""Tests for agent.trajectory — static helpers and save_trajectory()."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.trajectory import (
    convert_scratchpad_to_think,
    has_incomplete_scratchpad,
    save_trajectory,
)


# ============================================================================
# convert_scratchpad_to_think()
# ============================================================================
class TestConvertScratchpadToThink:
    def test_no_tags_returns_unchanged(self):
        assert convert_scratchpad_to_think("hello world") == "hello world"

    def test_empty_string(self):
        assert convert_scratchpad_to_think("") == ""

    def test_single_pair(self):
        result = convert_scratchpad_to_think(
            "<REASONING_SCRATCHPAD>some thoughts</REASONING_SCRATCHPAD>"
        )
        assert result == "<think>some thoughts</think>"

    def test_multiple_pairs(self):
        result = convert_scratchpad_to_think(
            "<REASONING_SCRATCHPAD>first</REASONING_SCRATCHPAD> text "
            "<REASONING_SCRATCHPAD>second</REASONING_SCRATCHPAD>"
        )
        assert result == "<think>first</think> text <think>second</think>"

    def test_nested_like_content(self):
        """Sequential opening/closing tags should all convert."""
        result = convert_scratchpad_to_think(
            "<REASONING_SCRATCHPAD>a</REASONING_SCRATCHPAD>"
            "<REASONING_SCRATCHPAD>b</REASONING_SCRATCHPAD>"
        )
        assert result == "<think>a</think><think>b</think>"

    def test_only_opening_no_closing(self):
        """Only opening tag present — still converted (partial)."""
        result = convert_scratchpad_to_think(
            "<REASONING_SCRATCHPAD>unclosed"
        )
        assert result == "<think>unclosed"

    def test_only_closing_no_opening(self):
        """Closing tag alone — guard clause returns early (no opening tag present)."""
        result = convert_scratchpad_to_think("trailing</REASONING_SCRATCHPAD>")
        # Guard: "<REASONING_SCRATCHPAD>" not in content → returns unchanged
        assert result == "trailing</REASONING_SCRATCHPAD>"

    def test_uppercase_tags(self):
        """Tags are case-sensitive — uppercase not converted."""
        result = convert_scratchpad_to_think(
            "<REASONING_SCRATCHPAD>hello</REASONING_SCRATCHPAD>"
        )
        # This IS the exact case of the tag, so it converts
        assert result == "<think>hello</think>"

    def test_partial_tag_not_converted(self):
        """Text that looks like, but is not exactly, the tag."""
        result = convert_scratchpad_to_think("REASONING_SCRATCHPAD without brackets")
        assert result == "REASONING_SCRATCHPAD without brackets"


# ============================================================================
# has_incomplete_scratchpad()
# ============================================================================
class TestHasIncompleteScratchpad:
    def test_complete_pair_returns_false(self):
        assert has_incomplete_scratchpad(
            "<REASONING_SCRATCHPAD>done</REASONING_SCRATCHPAD>"
        ) is False

    def test_no_tags_returns_false(self):
        assert has_incomplete_scratchpad("plain text") is False

    def test_empty_string_returns_false(self):
        assert has_incomplete_scratchpad("") is False

    def test_opening_without_closing(self):
        assert has_incomplete_scratchpad(
            "<REASONING_SCRATCHPAD>in progress..."
        ) is True

    def test_closing_without_opening(self):
        """Closing without opening — not incomplete (just a stray closing tag)."""
        assert has_incomplete_scratchpad(
            "stray</REASONING_SCRATCHPAD>"
        ) is False

    def test_multiple_complete(self):
        assert has_incomplete_scratchpad(
            "<REASONING_SCRATCHPAD>a</REASONING_SCRATCHPAD>"
            "<REASONING_SCRATCHPAD>b</REASONING_SCRATCHPAD>"
        ) is False

    def test_mixed_complete_and_incomplete(self):
        """has_incomplete only checks if closing tag exists at all.
        Even with an unclosed second opening, the closing tag from the
        first pair satisfies the check — returns False."""
        assert has_incomplete_scratchpad(
            "<REASONING_SCRATCHPAD>done</REASONING_SCRATCHPAD>"
            "<REASONING_SCRATCHPAD>still going"
        ) is False

    def test_closing_before_opening(self):
        """Closing tag appears before opening tag."""
        assert has_incomplete_scratchpad(
            "</REASONING_SCRATCHPAD><REASONING_SCRATCHPAD>"
        ) is False


# ============================================================================
# save_trajectory()
# ============================================================================
class TestSaveTrajectory:
    def test_save_completed_default_filename(self, tmp_path: Path):
        """Completed=True writes to trajectory_samples.jsonl."""
        cwd = tmp_path
        save_trajectory(
            [{"role": "user", "content": "hi"}],
            model="test-model",
            completed=True,
            filename=str(cwd / "trajectory_samples.jsonl"),
        )
        assert (cwd / "trajectory_samples.jsonl").exists()
        lines = (cwd / "trajectory_samples.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["model"] == "test-model"
        assert entry["completed"] is True
        assert entry["conversations"] == [{"role": "user", "content": "hi"}]
        assert "timestamp" in entry

    def test_save_failed_default_filename(self, tmp_path: Path):
        """Completed=False writes to failed_trajectories.jsonl."""
        cwd = tmp_path
        save_trajectory(
            [{"role": "assistant", "content": "oops"}],
            model="gpt-4",
            completed=False,
            filename=str(cwd / "failed_trajectories.jsonl"),
        )
        assert (cwd / "failed_trajectories.jsonl").exists()
        entry = json.loads((cwd / "failed_trajectories.jsonl").read_text().strip())
        assert entry["completed"] is False
        assert entry["model"] == "gpt-4"

    def test_appends_to_existing_file(self, tmp_path: Path):
        """Multiple calls append, not overwrite."""
        f = tmp_path / "out.jsonl"
        save_trajectory([{"role": "user", "content": "one"}], "m", True, str(f))
        save_trajectory([{"role": "user", "content": "two"}], "m", True, str(f))
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["conversations"][0]["content"] == "one"
        assert json.loads(lines[1])["conversations"][0]["content"] == "two"

    def test_custom_filename(self, tmp_path: Path):
        """Explicit filename override works."""
        f = tmp_path / "custom.jsonl"
        save_trajectory([], "any", True, str(f))
        assert f.exists()

    def test_multiple_conversations(self, tmp_path: Path):
        f = tmp_path / "t.jsonl"
        save_trajectory(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            model="multi",
            completed=True,
            filename=str(f),
        )
        entry = json.loads(f.read_text().strip())
        assert len(entry["conversations"]) == 3

    def test_timestamp_is_isoformat(self, tmp_path: Path):
        f = tmp_path / "t.jsonl"
        save_trajectory([], "m", True, str(f))
        entry = json.loads(f.read_text().strip())
        ts = entry["timestamp"]
        # ISO format contains 'T'
        assert "T" in ts

    def test_filename_none_completed(self, tmp_path, monkeypatch):
        """When filename is None and completed=True, defaults to trajectory_samples.jsonl."""
        monkeypatch.chdir(tmp_path)
        save_trajectory([{"role": "user", "content": "x"}], "m", completed=True)
        assert (tmp_path / "trajectory_samples.jsonl").exists()

    def test_filename_none_failed(self, tmp_path, monkeypatch):
        """When filename is None and completed=False, defaults to failed_trajectories.jsonl."""
        monkeypatch.chdir(tmp_path)
        save_trajectory([{"role": "user", "content": "x"}], "m", completed=False)
        assert (tmp_path / "failed_trajectories.jsonl").exists()

    def test_empty_trajectory_list(self, tmp_path: Path):
        """Empty conversation list is valid."""
        f = tmp_path / "e.jsonl"
        save_trajectory([], "e", True, str(f))
        entry = json.loads(f.read_text().strip())
        assert entry["conversations"] == []
