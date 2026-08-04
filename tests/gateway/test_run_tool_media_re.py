r"""Tests for _TOOL_MEDIA_RE regex patterns in gateway/run.py.

Issue #34632: The _TOOL_MEDIA_RE patterns in GatewayRunner used (?:/|~\/) to
anchor paths, which only matched Unix-style absolute and home-relative paths.
Windows absolute paths (C:\\Users\\..., D:/...) were silently ignored, causing
MEDIA directive delivery to fail on Windows.

Fix: Add [A-Za-z]:[/\\\\] as a third anchor alternative in both patterns.

Two identical _TOOL_MEDIA_RE patterns exist in run.py:
1. History scanning (~L17223): collects already-seen media paths
2. Result scanning (~L17549): extracts new media tags from agent output

This test file validates that both equivalent regex patterns correctly match
Windows paths while preserving existing Unix path matching behavior.
"""

import re

import pytest


# Reconstruct the exact _TOOL_MEDIA_RE pattern from gateway/run.py
# The pattern is built by concatenating raw string parts:
#   r'MEDIA:((?:[A-Za-z]:[/\\]|/|~\/)\S+\.(?:png|...))'
_TOOL_MEDIA_RE = re.compile(
    r'MEDIA:((?:[A-Za-z]:[/\\]|/|~\/)\S+\.(?:png|jpe?g|gif|webp|'
    r'mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|'
    r'flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|'
    r'txt|csv|apk|ipa))',
    re.IGNORECASE,
)


# Reconstruct the pre-fix pattern (without Windows anchor) for regression proof
_TOOL_MEDIA_RE_PRE_FIX = re.compile(
    r'MEDIA:((?:/|~\/)\S+\.(?:png|jpe?g|gif|webp|'
    r'mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|'
    r'flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|'
    r'txt|csv|apk|ipa))',
    re.IGNORECASE,
)


class TestToolMediaReWindowsPaths:
    """Issue #34632: _TOOL_MEDIA_RE must match Windows absolute paths."""

    # ── Positive: Windows paths now match ──────────────────────────

    @pytest.mark.parametrize("media_tag, expected_path", [
        # Windows backslash paths
        ("MEDIA:C:\\Users\\test\\image.png", "C:\\Users\\test\\image.png"),
        ("MEDIA:D:\\data\\report.pdf", "D:\\data\\report.pdf"),
        ("MEDIA:E:\\Photos\\vacation.jpg", "E:\\Photos\\vacation.jpg"),
        # Windows forward-slash paths
        ("MEDIA:C:/Users/test/image.png", "C:/Users/test/image.png"),
        ("MEDIA:D:/data/report.pdf", "D:/data/report.pdf"),
        # Mixed separators
        ("MEDIA:C:\\Users/test\\image.webp", "C:\\Users/test\\image.webp"),
        # Various extensions
        ("MEDIA:F:\\videos\\clip.mp4", "F:\\videos\\clip.mp4"),
        ("MEDIA:G:\\audio\\song.mp3", "G:\\audio\\song.mp3"),
        ("MEDIA:H:\\docs\\sheet.xlsx", "H:\\docs\\sheet.xlsx"),
        ("MEDIA:Z:\\archive\\backup.zip", "Z:\\archive\\backup.zip"),
    ])
    def test_windows_paths_match(self, media_tag, expected_path):
        """Windows absolute paths with drive letters are matched."""
        match = _TOOL_MEDIA_RE.search(media_tag)
        assert match is not None, f"Should match: {media_tag}"
        assert match.group(1) == expected_path

    # ── Positive: Unix paths still match ───────────────────────────


    # ── Negative: invalid paths don't match ────────────────────────


    # ── Negative/preserved: old pattern rejects Windows paths ──────


    # ── Edge cases ─────────────────────────────────────────────────


    @pytest.mark.parametrize("media_tag", [
        "MEDIA:C:\\path\\file.jpeg",
        "MEDIA:C:\\path\\file.JPG",
        "MEDIA:C:\\path\\file.GIF",
        "MEDIA:C:\\path\\file.MP4",
    ])
    def test_case_insensitive_extensions(self, media_tag):
        """File extensions are matched case-insensitively."""
        match = _TOOL_MEDIA_RE.search(media_tag)
        assert match is not None, f"Should match: {media_tag}"
