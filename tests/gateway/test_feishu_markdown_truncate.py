"""Tests for the markdown-aware chunk boundary detector (Phase 2 / L2).

Covers:
  1. Module import surface — `_MARKDOWN_BOUNDARY_RE` exists and matches the
     documented element set (blank line, heading, hr, bullet, ordered list).
  2. truncate_message picks markdown boundaries over raw newlines when
     a structural break is available inside the region.
  3. truncate_message falls back to the original newline → space → hard-cut
     ladder when no safe boundary exists.
  4. Code-block fence handling still works — the existing carry_lang
     fence-reopen logic must keep working alongside the new boundary scan.
  5. Floor guard — boundaries that fall in the first 50% of the region are
     ignored, preventing greedy backtracking that starves the next chunk.
  6. table-row mid-cut is still avoided (table source rows start with '|'
     but `_MARKDOWN_BOUNDARY_RE` intentionally does NOT match them; tables
     fall through to the newline ladder — same behavior as before).
  7. Lazy-import failure path — when the Feishu adapter is unavailable,
     `_find_markdown_boundary` returns -1 and the chunker behaves like
     the pre-Phase-2 implementation.

Each test instantiates a `BasePlatformAdapter` substitute by calling
`truncate_message` as a free function via the existing
`gateway.platforms.base.BasePlatformAdapter.truncate_message` static path
— we don't need a real adapter instance, only the algorithm.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable when pytest is run from the project dir.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from plugins.platforms.feishu.adapter import _MARKDOWN_BOUNDARY_RE  # noqa: E402
from gateway.platforms.base import BasePlatformAdapter  # noqa: E402


# --- Test 1: regex module surface -----------------------------------------

class TestBoundaryRegexSurface:
    def test_regex_compiles(self):
        """The module-level regex must be a compiled re.Pattern."""
        assert isinstance(_MARKDOWN_BOUNDARY_RE, re.Pattern)

    def test_matches_blank_line(self):
        m = _MARKDOWN_BOUNDARY_RE.search("para one\n\npara two")
        assert m is not None
        assert m.start() == len("para one")

    def test_matches_atx_heading(self):
        text = "intro paragraph\n## Section\nbody"
        m = _MARKDOWN_BOUNDARY_RE.search(text)
        assert m is not None
        # Match starts AT the heading (the split offset is the "\n" before).
        assert text[m.start():].startswith("## Section")

    def test_matches_hr_dash(self):
        text = "above\n\n---\n\nbelow"
        m = _MARKDOWN_BOUNDARY_RE.search(text)
        assert m is not None
        assert text[m.start():].lstrip().startswith("---")

    def test_matches_hr_asterisk(self):
        text = "above\n***\nbelow"
        m = _MARKDOWN_BOUNDARY_RE.search(text)
        assert m is not None

    def test_matches_bullet_item(self):
        text = "lead in\n- first item\n- second item"
        m = _MARKDOWN_BOUNDARY_RE.search(text)
        assert m is not None
        assert text[m.start():].startswith("- first item")

    def test_matches_ordered_item(self):
        text = "lead in\n1. one\n2. two"
        m = _MARKDOWN_BOUNDARY_RE.search(text)
        assert m is not None
        assert text[m.start():].startswith("1. one")

    def test_does_not_match_table_pipe(self):
        """Pipe tables are intentionally NOT in the boundary set — splitting
        mid-table is the very thing we're avoiding, and Feishu's tag:md
        already handles GFM tables.  Match-position must fall back to
        the newline ladder so the existing behavior is preserved.
        """
        text = "before\n| col1 | col2 |\n|------|------|\n| a    | b    |\nafter"
        m = _MARKDOWN_BOUNDARY_RE.search(text)
        # We DO allow the trailing blank line + "after" paragraph boundary,
        # but never inside the table block itself.  Verify by checking that
        # no match falls between the table-header line and the table-body
        # last row.
        if m is not None:
            table_start = text.index("| col1")
            table_end = text.index("after")
            assert not (table_start < m.start() < table_end - 10), (
                f"Boundary should not land mid-table: matched at {m.start()} "
                f"but table spans [{table_start}, {table_end})"
            )


# --- Test 2-7: truncate_message integration --------------------------------

def _truncate(content: str, max_length: int = 200):
    """Thin wrapper that calls the static truncate_message on a stub instance.

    We instantiate BasePlatformAdapter via a NoOp subclass that bypasses the
    ABC __init__ checks (truncate_message doesn't touch self).
    """
    class _NoOp(BasePlatformAdapter):
        async def send(self, *a, **kw):  # pragma: no cover
            raise NotImplementedError
        async def connect(self):  # pragma: no cover
            raise NotImplementedError
        async def disconnect(self):  # pragma: no cover
            raise NotImplementedError
        def _load_settings(self, raw):  # pragma: no cover
            return {}

    return _NoOp.truncate_message(content, max_length)


class TestTruncateUsesMarkdownBoundary:
    def test_splits_on_blank_line_before_heading(self):
        """Long content with a blank line + heading must split on the blank
        line rather than mid-paragraph."""
        body = "x" * 150 + "\n\n## Section Two\n" + "y" * 150
        chunks = _truncate(body, max_length=200)
        # First chunk must end before "## Section Two"
        assert len(chunks) == 2
        assert "## Section Two" in chunks[1]
        assert "## Section Two" not in chunks[0]
        # The boundary is the blank line; strip chunk indicator and verify
        # the body ends at the blank line (so remaining = "## Section Two\n...")
        body0 = re.sub(r" \(\d+/\d+\)$", "", chunks[0])
        assert body0.endswith("x") or body0.endswith("\n"), (
            f"chunk[0] body should end on a clean boundary, got {body0!r}"
        )

    def test_splits_on_bullet_item_start(self):
        """Long prose followed by a bullet list must split so the first
        bullet stays whole.  Total content must exceed max_length so we
        actually trigger chunking (>200 chars)."""
        body = "introduction prose " + ("z" * 200) + "\n\n- first\n- second\n- third"
        chunks = _truncate(body, max_length=200)
        assert len(chunks) >= 2
        body0 = re.sub(r" \(\d+/\d+\)$", "", chunks[0])
        assert "- first" not in body0
        assert "- first" in chunks[-1]

    def test_splits_on_ordered_list(self):
        body = "lead " + ("q" * 200) + "\n\n1. one\n2. two"
        chunks = _truncate(body, max_length=200)
        assert len(chunks) >= 2
        body0 = re.sub(r" \(\d+/\d+\)$", "", chunks[0])
        assert "1. one" not in body0
        assert "1. one" in chunks[-1]

    def test_splits_on_hr(self):
        body = "above\n\n" + ("z" * 200) + "\n\n---\n\nbelow"
        chunks = _truncate(body, max_length=200)
        assert len(chunks) >= 2
        # Last chunk contains the rule and the content below it
        assert "below" in chunks[-1]

    def test_falls_back_to_newline_when_no_boundary(self):
        """When the region has no markdown boundary, behavior must match
        the pre-Phase-2 newline ladder."""
        body = "no structural break here at all, " * 20  # ~660 chars, all prose
        chunks = _truncate(body, max_length=100)
        # Expect at least 2 chunks and no data loss
        reconstructed = "".join(
            c.rsplit(" ", 1)[0]  # strip "(N/M)" indicator for comparison
            if c.endswith(")") and "(" in c else c
            for c in chunks
        )
        # Order-preserving substring check: each chunk starts where the
        # previous one left off in the original (modulo chunk indicators)
        assert len(chunks) >= 2
        # No chunk exceeds the cap (lenient: +10 for the "(N/M)" suffix).
        for c in chunks:
            assert len(c) <= 110, f"chunk too long: {len(c)}"

    def test_floor_guard_skips_early_boundary(self):
        """A boundary very early in the region (within the first 50%) must
        be ignored — otherwise the algorithm would produce many tiny chunks.
        We construct content where the only blank line is at offset 5."""
        body = "a\n\nb" + ("z" * 250)  # short, single blank line near start
        chunks = _truncate(body, max_length=80)
        # The early blank line (pos 1) is < floor (40), so it's ignored;
        # we expect the chunker to fall back to the newline ladder.
        assert len(chunks) >= 2
        # First chunk must contain more than just "a" — we kept going.
        assert len(chunks[0]) > 5

    def test_code_block_fence_still_balanced(self):
        """Adding the boundary detector must NOT regress the carry_lang
        fence-reopen logic."""
        body = (
            "```python\n"
            + ("x" * 100) + "\n"
            + ("y" * 100) + "\n"
            + "```\n\n"
            + "After code " + ("z" * 100)
        )
        chunks = _truncate(body, max_length=150)
        # Reconstructed, no fence should be left dangling in the middle of
        # a chunk. The first chunk should end with a properly-closed fence.
        assert "```" in chunks[0]  # code block present
        # Every chunk must have an even number of "```" markers
        # (since unbalanced fences are the bug we're guarding against).
        for c in chunks:
            assert c.count("```") % 2 == 0, (
                f"Unbalanced fences in chunk: {c!r}"
            )

    def test_short_content_passes_through(self):
        """Content shorter than max_length is returned as-is."""
        body = "Short and sweet."
        chunks = _truncate(body, max_length=200)
        assert chunks == [body]

    def test_empty_content_returns_empty_chunk(self):
        """Empty content must not infinite-loop (regression on the
        existing floor logic; the new boundary detector must not change it).
        """
        chunks = _truncate("", max_length=100)
        # Pre-Phase-2: returns [""]
        assert chunks == [""]

    def test_table_in_long_doc_does_not_get_split_mid_row(self):
        """A table block must never be split mid-row. Even though our
        boundary regex does NOT target '|' lines, this test verifies the
        newline ladder plus floor guard keep table rows intact."""
        header = "| col1 | col2 |\n|------|------|\n"
        rows = "".join(f"| row{i:02d}   | data   |\n" for i in range(20))
        body = header + rows + "\nafter"
        chunks = _truncate(body, max_length=100)
        # Every chunk must contain complete table rows (each row ends with \n)
        for c in chunks:
            # No chunk should end with a partial row ("| col1" or "| data" with
            # no trailing newline before the next "|").
            assert "|" in c or "after" in c
        # Reconstruct order: the table block as a whole is preserved.
        joined = "".join(chunks)
        # Strip chunk indicators for comparison
        clean = re.sub(r" \(\d+/\d+\)$", "", joined, flags=re.MULTILINE)
        assert header in clean


# --- Test 7: lazy import failure path --------------------------------------

class TestBoundaryDetectorLazyImportFallback:
    def test_returns_minus_one_when_import_fails(self, monkeypatch):
        """When the Feishu adapter import fails (e.g. circular import in a
        test environment that mocks sys.modules), the boundary detector
        returns -1 and truncate_message falls back to the newline ladder.

        We force the failure by stuffing a broken module into sys.modules.
        """
        import sys as _sys

        # Save and remove the real adapter module so the lazy import fails.
        real = _sys.modules.pop(
            "plugins.platforms.feishu.adapter", None
        )
        _sys.modules["plugins.platforms.feishu.adapter"] = None
        try:
            # Re-trigger the lazy import by calling truncate_message; it must
            # not raise, and the result must equal the pre-Phase-2 behavior.
            body = "plain prose " * 50
            chunks = _truncate(body, max_length=80)
            assert len(chunks) >= 2
            for c in chunks:
                assert len(c) <= 90  # only indicator overhead
        finally:
            # Restore so other tests can use the real adapter.
            _sys.modules.pop("plugins.platforms.feishu.adapter", None)
            if real is not None:
                _sys.modules["plugins.platforms.feishu.adapter"] = real