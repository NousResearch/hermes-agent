"""Tests for truncate_message Markdown structure preservation.

Covers: code blocks, inline code, math blocks ($$), tables, ordered lists,
and combinations.
"""

from gateway.platforms.base import BasePlatformAdapter

truncate = BasePlatformAdapter.truncate_message


def _assert_chunks(content, max_len, expected_count=None):
    """Split and return chunks.  Optionally verify count."""
    chunks = truncate(content, max_len)
    if expected_count is not None:
        assert len(chunks) == expected_count, \
            f"Expected {expected_count} chunks, got {len(chunks)}"
    return chunks


# ---- Code blocks ----

def test_code_block_preserved():
    chunks = _assert_chunks(
        "A\n\n```py\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n```\n\nB", 30, 3)
    for c in chunks:
        assert c.count("```") % 2 == 0, f"Unbalanced backticks in {c[:40]!r}"
    # Each chunk except maybe the last should carry the lang tag
    assert "```py" in chunks[1], f"Missing lang reopen in {chunks[1][:40]!r}"


def test_code_block_no_tag():
    chunks = _assert_chunks(
        "```\na\nb\nc\nd\ne\nf\ng\nh\n```", 20)
    for c in chunks:
        assert c.count("```") % 2 == 0


def test_short_message_unchanged():
    chunks = truncate("Hello", 100)
    assert chunks == ["Hello"]


# ---- Inline code ----

def test_inline_code_not_split():
    # A long line with backtick span inside
    text = "Use `some_function(x, y)` which " + "does stuff " * 15 + "here"
    chunks = _assert_chunks(text, 80)
    for c in chunks:
        bt = c.count("`")
        assert bt % 2 == 0, f"Unbalanced backticks in {c[:40]!r}"


# ---- Math blocks ($$) ----

def test_math_block_reopened():
    """$$...$$ spanning chunks gets closed and reopened."""
    content = "Before\n\n$$\nE = mc^2\n\\int_0^\\infty f(x) dx\n\\sum_{i=1}^n i\n$$\n\nAfter"
    chunks = _assert_chunks(content, 30)
    for c in chunks:
        assert c.count("$$") % 2 == 0, f"Unbalanced $$ in {c[:40]!r}"
    # The continuation chunk should start with $$
    assert any(c.startswith("$$") for c in chunks[1:]), \
        "No chunk reopens math fence"


def test_math_and_code_independent():
    """$$ and ``` tracking shouldn't interfere."""
    content = (
        "A\n\n```\ncode line 1\ncode line 2\ncode line 3\n"
        "code line 4\ncode line 5\ncode line 6\n```\n\n"
        "$$\nmath line 1\nmath line 2\nmath line 3\n"
        "math line 4\n$$\n\nB"
    )
    chunks = _assert_chunks(content, 30)
    for c in chunks:
        assert c.count("```") % 2 == 0
        assert c.count("$$") % 2 == 0


def test_math_in_code_is_literal():
    """$$ inside a code block must NOT toggle math mode."""
    content = "```\n$$\nE = mc^2\n$$\n```\n"
    # Short enough to fit in one chunk
    chunks = truncate(content, 200)
    assert len(chunks) == 1
    assert chunks[0].count("$$") == 2  # literal, not toggling
    assert chunks[0].count("```") == 2


# ---- Tables ----

def test_table_stays_whole_when_possible():
    """If there's room, the whole table fits in one chunk."""
    table = "Intro\n\n| H1 | H2 |\n|---|---|\n| A | B |\n| C | D |\n\nOutro"
    chunks = truncate(table, 200)
    assert len(chunks) == 1


def test_table_boundary_preferred():
    """When the table is too large, split before the table."""
    table = ("Some text.\n\n| Col A | Col B |\n|---|---|\n"
             + "| a | b |\n" * 20 + "\n\nMore text.")
    chunks = _assert_chunks(table, 80)
    for c in chunks:
        pipes = [l for l in c.split("\n") if l.startswith("|")]
        if pipes:
            # Every chunk with pipe lines must have a separator
            assert any("---" in l for l in pipes), f"No separator in {c[:50]!r}"


def test_table_header_reconstructed():
    """When a table spans chunks, each gets a header."""
    text = ("| A | B |\n|---|---|\n"
            + "| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\n"
            + "| 7 | 8 |\n| 9 | 10 |\n| 11 | 12 |\n"
            + "| 13 | 14 |\n| 15 | 16 |\n\nOutro")
    chunks = _assert_chunks(text, 60)
    for i, c in enumerate(chunks):
        pipes = [l for l in c.split("\n") if l.startswith("|")]
        if pipes:
            # First pipe must be a header, not a separator
            first = pipes[0]
            assert "---" not in first, f"Chunk {i} starts with separator: {first}"
            # Must have a separator too
            assert any("---" in l for l in pipes), f"Chunk {i} missing separator"
            assert first == "| A | B |", \
                f"Chunk {i} has wrong header: {first!r}"


# ---- Ordered lists ----

def test_ordered_list_boundary():
    """Split before an ordered list when mid-list."""
    text = "A paragraph with enough text to push the list boundary far enough.\n\n1. first\n2. second\n3. third\n4. fourth\n5. fifth\n\nOutro"
    chunks = _assert_chunks(text, 55)
    for c in chunks:
        items = [l.strip() for l in c.split("\n") if l.strip()[:2].isdigit() and ". " in l.strip()[:5]]
        if items:
            assert items[0].startswith("1."), f"List doesn't start at 1: {items[0]!r}"


# ---- Everything fits ----

def test_short_fits():
    chunks = truncate("Short", 50)
    assert chunks == ["Short"]


def test_exact_fit():
    chunks = truncate("x" * 50, 55)
    assert len(chunks) == 1


# ---- Edge: mixed constructions ----

def test_code_then_table_then_math():
    """All three fence types in sequence, each requiring a split."""
    content = (
        "```\ncode\n```\n\n"
        "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\n"
        "| 7 | 8 |\n| 9 | 10 |\n\n"
        "$$\nmath\n$$\n\nEnd"
    )
    chunks = _assert_chunks(content, 40)
    fences_ok = all(c.count("```") % 2 == 0 and c.count("$$") % 2 == 0
                    for c in chunks)
    assert fences_ok, "Unbalanced fences in mixed content"
