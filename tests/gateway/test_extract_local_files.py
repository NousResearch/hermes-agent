"""
Tests for extract_local_files() — auto-detection of bare local file paths
in model response text for native media delivery.

Covers: path matching, code-block exclusion, URL rejection, tilde expansion,
deduplication, text cleanup, and extension routing.

Based on PR #1636 by sudoingX (salvaged + hardened).
"""

import os
from unittest.mock import patch

import pytest

from gateway.platforms.base import BasePlatformAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract(content: str, existing_files: set[str] | None = None):
    """
    Run extract_local_files with os.path.isfile mocked to return True
    for any path in *existing_files* (expanded form).  If *existing_files*
    is None every path passes.
    """
    existing = existing_files

    def fake_isfile(p):
        if existing is None:
            return True
        return p in existing

    def fake_expanduser(p):
        if p.startswith("~/"):
            return "/home/user" + p[1:]
        return p

    with patch("os.path.isfile", side_effect=fake_isfile), \
         patch("os.path.expanduser", side_effect=fake_expanduser):
        return BasePlatformAdapter.extract_local_files(content)


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------

class TestBasicDetection:

    def test_absolute_path_image(self):
        paths, cleaned = _extract("Here is the screenshot /root/screenshots/game.png enjoy")
        assert paths == ["/root/screenshots/game.png"]
        assert "/root/screenshots/game.png" not in cleaned
        assert "Here is the screenshot" in cleaned

    def test_tilde_path_image(self):
        paths, cleaned = _extract("Check out ~/photos/cat.jpg for the cat")
        assert paths == ["/home/user/photos/cat.jpg"]
        assert "~/photos/cat.jpg" not in cleaned

    def test_video_extensions(self):
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            text = f"Video at /tmp/clip{ext} here"
            paths, _ = _extract(text)
            assert len(paths) == 1, f"Failed for {ext}"
            assert paths[0] == f"/tmp/clip{ext}"

    def test_image_extensions(self):
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            text = f"Image at /tmp/pic{ext} here"
            paths, _ = _extract(text)
            assert len(paths) == 1, f"Failed for {ext}"
            assert paths[0] == f"/tmp/pic{ext}"

    def test_case_insensitive_extension(self):
        paths, _ = _extract("See /tmp/PHOTO.PNG and /tmp/vid.MP4 now")
        assert len(paths) == 2

    def test_multiple_paths(self):
        text = "First /tmp/a.png then /tmp/b.jpg and /tmp/c.mp4 done"
        paths, cleaned = _extract(text)
        assert len(paths) == 3
        assert "/tmp/a.png" in paths
        assert "/tmp/b.jpg" in paths
        assert "/tmp/c.mp4" in paths
        for p in paths:
            assert p not in cleaned

    def test_path_at_line_start(self):
        paths, _ = _extract("/var/data/image.png")
        assert paths == ["/var/data/image.png"]

    def test_path_at_end_of_line(self):
        paths, _ = _extract("saved to /var/data/image.png")
        assert paths == ["/var/data/image.png"]

    def test_path_with_dots_in_directory(self):
        paths, _ = _extract("See /opt/my.app/assets/logo.png here")
        assert paths == ["/opt/my.app/assets/logo.png"]

    def test_path_with_hyphens(self):
        paths, _ = _extract("File at /tmp/my-screenshot-2024.png done")
        assert paths == ["/tmp/my-screenshot-2024.png"]


# ---------------------------------------------------------------------------
# Non-existent files are skipped
# ---------------------------------------------------------------------------

class TestIsfileGuard:

    def test_nonexistent_path_skipped(self):
        """Paths that don't exist on disk are not extracted."""
        paths, cleaned = _extract(
            "See /tmp/nope.png here",
            existing_files=set(),  # nothing exists
        )
        assert paths == []
        assert "/tmp/nope.png" in cleaned  # not stripped

    def test_only_existing_paths_extracted(self):
        """Mix of existing and non-existing — only existing are returned."""
        paths, cleaned = _extract(
            "A /tmp/real.png and /tmp/fake.jpg end",
            existing_files={"/tmp/real.png"},
        )
        assert paths == ["/tmp/real.png"]
        assert "/tmp/real.png" not in cleaned
        assert "/tmp/fake.jpg" in cleaned


# ---------------------------------------------------------------------------
# URL false-positive prevention
# ---------------------------------------------------------------------------

class TestURLRejection:

    def test_https_url_not_matched(self):
        """Paths embedded in HTTP URLs must not be extracted."""
        paths, cleaned = _extract("Visit https://example.com/images/photo.png for details")
        # The regex lookbehind should prevent matching the URL's path segment
        # Even if it did match, isfile would be False for /images/photo.png
        # (we mock isfile to True-for-all here, so the lookbehind is the guard)
        assert paths == []
        assert "https://example.com/images/photo.png" in cleaned

    def test_http_url_not_matched(self):
        paths, _ = _extract("See http://cdn.example.com/assets/banner.jpg here")
        assert paths == []

    def test_file_url_not_matched(self):
        paths, _ = _extract("Open file:///home/user/doc.png in browser")
        # file:// has :// before /home so lookbehind blocks it
        assert paths == []


# ---------------------------------------------------------------------------
# Code block exclusion
# ---------------------------------------------------------------------------

class TestCodeBlockExclusion:

    def test_fenced_code_block_skipped(self):
        text = "Here's how:\n```python\nimg = open('/tmp/image.png')\n```\nDone."
        paths, cleaned = _extract(text)
        assert paths == []
        assert "/tmp/image.png" in cleaned  # not stripped

    def test_inline_code_skipped(self):
        text = "Use the path `/tmp/image.png` in your config"
        paths, cleaned = _extract(text)
        assert paths == []
        assert "`/tmp/image.png`" in cleaned

    def test_path_outside_code_block_still_matched(self):
        text = (
            "```\ncode: /tmp/inside.png\n```\n"
            "But this one is real: /tmp/outside.png"
        )
        paths, _ = _extract(text, existing_files={"/tmp/outside.png"})
        assert paths == ["/tmp/outside.png"]

    def test_mixed_inline_code_and_bare_path(self):
        text = "Config uses `/etc/app/bg.png` but output is /tmp/result.jpg"
        paths, cleaned = _extract(text, existing_files={"/tmp/result.jpg"})
        assert paths == ["/tmp/result.jpg"]
        assert "`/etc/app/bg.png`" in cleaned
        assert "/tmp/result.jpg" not in cleaned

    def test_multiline_fenced_block(self):
        text = (
            "```bash\n"
            "cp /source/a.png /dest/b.png\n"
            "mv /source/c.mp4 /dest/d.mp4\n"
            "```\n"
            "Files are ready."
        )
        paths, _ = _extract(text)
        assert paths == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_duplicate_paths_deduplicated(self):
        text = "See /tmp/img.png and also /tmp/img.png again"
        paths, _ = _extract(text)
        assert paths == ["/tmp/img.png"]

    def test_tilde_and_expanded_same_file(self):
        """~/photos/a.png and /home/user/photos/a.png are the same file."""
        text = "See ~/photos/a.png and /home/user/photos/a.png here"
        paths, _ = _extract(text, existing_files={"/home/user/photos/a.png"})
        assert len(paths) == 1
        assert paths[0] == "/home/user/photos/a.png"


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

class TestTextCleanup:

    def test_path_removed_from_text(self):
        paths, cleaned = _extract("Before /tmp/x.png after")
        assert "Before" in cleaned
        assert "after" in cleaned
        assert "/tmp/x.png" not in cleaned

    def test_excessive_blank_lines_collapsed(self):
        text = "Before\n\n\n/tmp/x.png\n\n\nAfter"
        _, cleaned = _extract(text)
        assert "\n\n\n" not in cleaned

    def test_no_paths_text_unchanged(self):
        text = "This is a normal response with no file paths."
        paths, cleaned = _extract(text)
        assert paths == []
        assert cleaned == text

    def test_tilde_form_cleaned_from_text(self):
        """The raw ~/... form should be removed, not the expanded /home/user/... form."""
        text = "Output saved to ~/result.png for review"
        paths, cleaned = _extract(text)
        assert paths == ["/home/user/result.png"]
        assert "~/result.png" not in cleaned

    def test_only_path_in_text(self):
        """If the response is just a path, cleaned text is empty."""
        paths, cleaned = _extract("/tmp/screenshot.png")
        assert paths == ["/tmp/screenshot.png"]
        assert cleaned == ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_string(self):
        paths, cleaned = _extract("")
        assert paths == []
        assert cleaned == ""

    def test_no_media_extensions(self):
        """Non-media extensions should not be matched."""
        paths, _ = _extract("See /tmp/data.csv and /tmp/script.py and /tmp/notes.txt")
        assert paths == []

    def test_path_with_spaces_not_matched(self):
        """Paths with spaces are intentionally not matched (avoids false positives)."""
        paths, _ = _extract("File at /tmp/my file.png here")
        assert paths == []

    def test_windows_path_not_matched(self):
        """Windows-style paths should not match."""
        paths, _ = _extract("See C:\\Users\\test\\image.png")
        assert paths == []

    def test_relative_path_not_matched(self):
        """Relative paths like ./image.png should not match."""
        paths, _ = _extract("File at ./screenshots/image.png here")
        assert paths == []

    def test_bare_filename_not_matched(self):
        """Just 'image.png' without a path should not match."""
        paths, _ = _extract("Open image.png to see")
        assert paths == []

    def test_path_followed_by_punctuation(self):
        """Path followed by comma, period, paren should still match."""
        for suffix in [",", ".", ")", ":", ";"]:
            text = f"See /tmp/img.png{suffix} details"
            paths, _ = _extract(text)
            assert len(paths) == 1, f"Failed with suffix '{suffix}'"

    def test_path_in_parentheses(self):
        paths, _ = _extract("(see /tmp/img.png)")
        assert paths == ["/tmp/img.png"]

    def test_path_in_quotes(self):
        paths, _ = _extract('The file is "/tmp/img.png" right here')
        assert paths == ["/tmp/img.png"]

    def test_deep_nested_path(self):
        paths, _ = _extract("At /a/b/c/d/e/f/g/h/image.png end")
        assert paths == ["/a/b/c/d/e/f/g/h/image.png"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
