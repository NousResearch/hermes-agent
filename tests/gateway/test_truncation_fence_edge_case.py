"""
Tests for truncated message code fence edge case.

When a message with multiple code blocks (e.g. reasoning outer fence +
response code block) is truncated by truncate_message(), the split
chunks should each have balanced code fences.

This reproduces the scenario where:
- A message contains 4 ``` markers: reasoning outer open + close,
  response code block open + close
- The split falls inside the response code block (between 3rd and 4th)
- Each resulting chunk must have even number of ``` markers
"""

import pytest
from gateway.platforms.base import BasePlatformAdapter


def ensure_balanced_fences(text: str) -> str:
    """Append closing fence if odd count of triple-backtick."""
    if text.count("```") % 2 == 1:
        return text.rstrip("\n") + "\n```"
    return text


def _build_long_quad_message(max_length: int = 400) -> str:
    """Build a message with 4 ``` markers, long enough to split at max_length."""
    return (
        "💭 **Reasoning:**\n```\n"
        + "A" * 80
        + "\n```\n\n"
        + "B" * 80
        + "\n\n```python\n"
        + "C" * 80
        + "\n```\n\n"
        + "D" * max_length  # <- padding to force truncation
    )


class TestTruncationFenceEdgeCase:
    """When truncate_message splits content with multiple code blocks,
    each chunk must have balanced fences."""

    def test_chunks_have_even_fence_count(self):
        """Each truncated chunk must have even fence count."""
        msg = _build_long_quad_message(max_length=400)
        chunks = BasePlatformAdapter.truncate_message(msg, max_length=400)
        assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
        for i, chunk in enumerate(chunks):
            clean = chunk.rsplit("(", 1)[0].strip() if " (" in chunk else chunk
            assert clean.count("```") % 2 == 0, (
                f"Chunk {i+1}/{len(chunks)} has odd ``` count"
            )

    def test_no_false_positives_on_small_message(self):
        """A short message should stay as one balanced chunk."""
        msg = "simple `text` without ``` code block"
        chunks = BasePlatformAdapter.truncate_message(msg, max_length=500)
        assert len(chunks) == 1

    def test_all_split_points_balanced(self):
        """Edge: split happens at various fence boundaries."""
        for pad in range(50, 500, 50):
            msg = _build_long_quad_message(max_length=pad)
            chunks = BasePlatformAdapter.truncate_message(msg, max_length=400)
            for i, chunk in enumerate(chunks):
                clean = chunk.rsplit("(", 1)[0].strip() if " (" in chunk else chunk
                assert clean.count("```") % 2 == 0, (
                    f"pad={pad}, chunk {i+1} has odd fences"
                )

    def test_ensure_closed_fences_on_chunks_is_idempotent(self):
        """Running ensure_balanced_fences on already-fixed chunks is no-op."""
        msg = _build_long_quad_message(max_length=400)
        chunks = BasePlatformAdapter.truncate_message(msg, max_length=400)
        for chunk in chunks:
            clean = chunk.rsplit("(", 1)[0].strip() if " (" in chunk else chunk
            result = ensure_balanced_fences(clean)
            assert result.count("```") == clean.count("```")

    def test_reasoning_contains_unescaped_fences(self):
        """Reasoning block with unescaped ``` inside, plus outer fence."""
        msg = (
            "💭 **Reasoning:**\n```\n"
            "I used \\`\\`\\`python\\nprint(1)\\n\\`\\`\\` in thinking\n"
            "```\n\n"
            + "x" * 500
        )
        chunks = BasePlatformAdapter.truncate_message(msg, max_length=400)
        for i, chunk in enumerate(chunks):
            clean = chunk.rsplit("(", 1)[0].strip() if " (" in chunk else chunk
            assert clean.count("```") % 2 == 0, (
                f"Chunk {i+1}/{len(chunks)} has odd ``` count"
            )

    def test_single_code_block_spanning_split(self):
        """Single code block that spans across truncation boundary."""
        code_body = "\n".join(f"line_{i} = {i}" for i in range(50))
        msg = f"Some text\n```python\n{code_body}\n```\nEnd text"
        chunks = BasePlatformAdapter.truncate_message(msg, max_length=200)
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks):
            clean = chunk.rsplit("(", 1)[0].strip() if " (" in chunk else chunk
            assert clean.count("```") % 2 == 0, (
                f"Chunk {i+1}/{len(chunks)} has odd ``` count"
            )
