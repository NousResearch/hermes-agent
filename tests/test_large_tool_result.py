"""Tests for _save_oversized_tool_result() — the large tool response handler.

When a tool returns more than _LARGE_RESULT_CHARS characters, the full content
is saved to a file and the model receives a preview + file path instead.
"""

import os
import re

import pytest

from run_agent import (
    _save_oversized_tool_result,
    _LARGE_RESULT_CHARS,
    _LARGE_RESULT_PREVIEW_CHARS,
)


class TestSaveOversizedToolResult:
    """Unit tests for the large tool result handler."""

    def test_small_result_returned_unchanged(self):
        """Results under the threshold pass through untouched."""
        small = "x" * 1000
        assert _save_oversized_tool_result("terminal", small) is small

    def test_exactly_at_threshold_returned_unchanged(self):
        """Results exactly at the threshold pass through."""
        exact = "y" * _LARGE_RESULT_CHARS
        assert _save_oversized_tool_result("terminal", exact) is exact

    def test_oversized_result_saved_to_file(self, tmp_path, monkeypatch):
        """Results over the threshold are written to a file."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        os.makedirs(tmp_path / ".hermes", exist_ok=True)

        big = "A" * (_LARGE_RESULT_CHARS + 500)
        result = _save_oversized_tool_result("terminal", big)

        # Should contain the preview
        assert result.startswith("A" * _LARGE_RESULT_PREVIEW_CHARS)
        # Should mention the file path
        assert "Full output saved to:" in result
        # Should mention original size
        assert f"{len(big):,}" in result

        # Extract the file path and verify the file exists with full content
        match = re.search(r"Full output saved to: (.+?)\n", result)
        assert match, f"No file path found in result: {result[:300]}"
        filepath = match.group(1)
        assert os.path.isfile(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            saved = f.read()
        assert saved == big
        assert len(saved) == _LARGE_RESULT_CHARS + 500

    def test_file_placed_in_cache_tool_responses(self, tmp_path, monkeypatch):
        """Saved file lives under HERMES_HOME/cache/tool_responses/."""
        hermes_home = str(tmp_path / ".hermes")
        monkeypatch.setenv("HERMES_HOME", hermes_home)
        os.makedirs(hermes_home, exist_ok=True)

        big = "B" * (_LARGE_RESULT_CHARS + 1)
        result = _save_oversized_tool_result("web_search", big)

        match = re.search(r"Full output saved to: (.+?)\n", result)
        filepath = match.group(1)
        expected_dir = os.path.join(hermes_home, "cache", "tool_responses")
        assert filepath.startswith(expected_dir)

    def test_filename_contains_tool_name(self, tmp_path, monkeypatch):
        """The saved filename includes a sanitized version of the tool name."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        os.makedirs(tmp_path / ".hermes", exist_ok=True)

        big = "C" * (_LARGE_RESULT_CHARS + 1)
        result = _save_oversized_tool_result("browser_navigate", big)

        match = re.search(r"Full output saved to: (.+?)\n", result)
        filename = os.path.basename(match.group(1))
        assert filename.startswith("browser_navigate_")
        assert filename.endswith(".txt")

    def test_tool_name_sanitized(self, tmp_path, monkeypatch):
        """Special characters in tool names are replaced in the filename."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        os.makedirs(tmp_path / ".hermes", exist_ok=True)

        big = "D" * (_LARGE_RESULT_CHARS + 1)
        result = _save_oversized_tool_result("mcp:some/weird tool", big)

        match = re.search(r"Full output saved to: (.+?)\n", result)
        filename = os.path.basename(match.group(1))
        # No slashes or colons in filename
        assert "/" not in filename
        assert ":" not in filename

    def test_fallback_on_write_failure(self, tmp_path, monkeypatch):
        """When file write fails, falls back to destructive truncation."""
        # Point HERMES_HOME to a path that will fail (file, not directory)
        bad_path = str(tmp_path / "not_a_dir.txt")
        with open(bad_path, "w") as f:
            f.write("I'm a file, not a directory")
        monkeypatch.setenv("HERMES_HOME", bad_path)

        big = "E" * (_LARGE_RESULT_CHARS + 50_000)
        result = _save_oversized_tool_result("terminal", big)

        # Should still contain data (fallback truncation)
        assert len(result) > 0
        assert result.startswith("E" * 1000)
        # Should mention the failure
        assert "File save failed" in result
        # Should be truncated to approximately _LARGE_RESULT_CHARS + error msg
        assert len(result) < len(big)

    def test_preview_length_capped(self, tmp_path, monkeypatch):
        """The inline preview is capped at _LARGE_RESULT_PREVIEW_CHARS."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        os.makedirs(tmp_path / ".hermes", exist_ok=True)

        # Use distinct chars so we can measure the preview
        big = "Z" * (_LARGE_RESULT_CHARS + 5000)
        result = _save_oversized_tool_result("terminal", big)

        # The preview section is the content before the "[Large tool response:" marker
        marker_pos = result.index("[Large tool response:")
        preview_section = result[:marker_pos].rstrip()
        assert len(preview_section) == _LARGE_RESULT_PREVIEW_CHARS

    def test_guidance_message_mentions_tools(self, tmp_path, monkeypatch):
        """The replacement message tells the model how to access the file."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        os.makedirs(tmp_path / ".hermes", exist_ok=True)

        big = "F" * (_LARGE_RESULT_CHARS + 1)
        result = _save_oversized_tool_result("terminal", big)

        assert "read_file" in result
        assert "search_files" in result

    def test_empty_result_passes_through(self):
        """Empty strings are not oversized."""
        assert _save_oversized_tool_result("terminal", "") == ""

    def test_unicode_content_preserved(self, tmp_path, monkeypatch):
        """Unicode content is fully preserved in the saved file."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        os.makedirs(tmp_path / ".hermes", exist_ok=True)

        # Mix of ASCII and multi-byte unicode to exceed threshold
        unit = "Hello 世界! 🎉 " * 100  # ~1400 chars per repeat
        big = unit * ((_LARGE_RESULT_CHARS // len(unit)) + 1)
        assert len(big) > _LARGE_RESULT_CHARS

        result = _save_oversized_tool_result("terminal", big)
        match = re.search(r"Full output saved to: (.+?)\n", result)
        filepath = match.group(1)

        with open(filepath, "r", encoding="utf-8") as f:
            saved = f.read()
        assert saved == big
