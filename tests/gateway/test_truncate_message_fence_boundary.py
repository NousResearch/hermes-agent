"""Regression tests for truncate_message fence-boundary detection.

Issue #55292: lines starting with ``` followed by non-whitespace text
inside a code block are ordinary content, not closing fences (CommonMark).
"""

import pytest

from gateway.platforms.base import BasePlatformAdapter


# ---------------------------------------------------------------------------
# Unit tests for the fence scanner (extracted logic from truncate_message)
# ---------------------------------------------------------------------------

def _scan_fences(lines: list[str], in_code: bool = False, lang: str = ""):
    """Replicate the fence-detection loop from truncate_message."""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                after_backticks = stripped[3:]
                if not after_backticks or after_backticks.isspace():
                    in_code = False
                    lang = ""
            else:
                in_code = True
                tag = stripped[3:].strip()
                lang = tag.split()[0] if tag else ""
    return in_code, lang


class TestFenceDetection:
    """Direct unit tests for the fence-detection invariant."""

    def test_bare_close_fence(self):
        ic, lang = _scan_fences(["aaa", "```"], in_code=True, lang="text")
        assert ic is False and lang == ""

    def test_close_fence_with_trailing_spaces(self):
        ic, lang = _scan_fences(["aaa", "```   "], in_code=True, lang="text")
        assert ic is False and lang == ""

    def test_close_fence_with_trailing_tab(self):
        ic, lang = _scan_fences(["aaa", "```\t"], in_code=True, lang="text")
        assert ic is False and lang == ""

    def test_fence_with_trailing_text_is_not_close(self):
        ic, lang = _scan_fences(
            ["aaa", "``` not a close", "bbb"], in_code=True, lang="text",
        )
        assert ic is True and lang == "text"

    def test_fence_with_info_string_is_not_close(self):
        ic, lang = _scan_fences(
            ["aaa", "```python"], in_code=True, lang="text",
        )
        assert ic is True and lang == "text"

    def test_fence_with_info_string_and_text_is_not_close(self):
        ic, lang = _scan_fences(
            ["aaa", "```python extra"], in_code=True, lang="text",
        )
        assert ic is True and lang == "text"

    def test_opening_fence_sets_lang(self):
        ic, lang = _scan_fences(["```ruby", "x = 1", "```"])
        assert ic is False and lang == ""

    def test_opening_fence_split_before_close(self):
        ic, lang = _scan_fences(["```ruby", "x = 1"])
        assert ic is True and lang == "ruby"


# ---------------------------------------------------------------------------
# End-to-end tests for truncate_message
# ---------------------------------------------------------------------------

class TestTruncateMessageE2E:
    """End-to-end tests verifying truncate_message chunk behavior."""

    @staticmethod
    def _truncate(content: str, max_length: int = 200) -> list[str]:
        return BasePlatformAdapter.truncate_message(content, max_length)

    def test_bare_close_fence_allows_proper_close(self):
        """A bare ``` closes the block; continuation should NOT reopen."""
        content = "pad\n```text\naaa\n```\noutside\n" + "B" * 300
        chunks = self._truncate(content, max_length=120)
        assert len(chunks) >= 2
        # Block properly closed — continuation should NOT reopen.
        assert all(not c.startswith("```text\n") for c in chunks)

    def test_fence_with_trailing_text_keeps_block_open(self):
        """``` followed by non-whitespace keeps the block open."""
        content = "pad\n```ruby\nfirst\n``` not a close\nlast\n" + "A" * 300
        chunks = self._truncate(content, max_length=120)
        assert len(chunks) >= 2
        # Block is still open — continuation must reopen with ```ruby.
        assert any(c.startswith("```ruby\n") for c in chunks[1:])

    def test_short_content_unchanged(self):
        content = "```text\n``` not a close\nstill code\n```"
        result = self._truncate(content, max_length=9999)
        assert result == [content]
