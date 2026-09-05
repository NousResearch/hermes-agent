"""Invariant tests for the shared fence-aware markdown chunker core.

The core lives in ``gateway.platforms.helpers`` and was extracted from the
Yuanbao ``MarkdownProcessor`` (the richest of the four formerly duplicated
implementations).  These tests assert the core's *contract* (invariants),
not exact snapshots, so they survive internal tweaks that preserve behavior.
"""

import pytest

from gateway.platforms.helpers import (
    balance_fences_across_chunks,
    greedy_pack_blocks,
    infer_block_separator,
    merge_streaming_fences,
    split_at_paragraph_boundary,
    split_markdown_atoms,
    split_markdown_table_row,
    split_text_fence_aware,
    text_has_unclosed_fence,
)


def utf16_len(s: str) -> int:
    return sum(2 if ord(c) > 0xFFFF else 1 for c in s)


LONG_PARAS = ("This is a sentence that goes on for a while. " * 8 + "\n\n") * 6
FENCED = (
    "Header line\n\n```js\n"
    + "\n".join(f"console.log({i}); // padding padding" for i in range(30))
    + "\n```\n\nTail paragraph after the code block ends here."
)
UNCLOSED = (
    "Some text before.\n\n```bash\necho one\n"
    + "echo more stuff here to pad the line out\n" * 10
)
TABLE = (
    "Intro paragraph.\n\n| col1 | col2 |\n|------|------|\n"
    + "\n".join(f"| value{i} | data{i} |" for i in range(40))
    + "\n\nDone."
)
CJK = "这是一个中文段落，包含了很多字符。用于测试宽字符处理。这句话结束了。\n\n" * 10
MIXED = (
    "Start.\n\n```sql\nSELECT * FROM t;\n```\n\n| x | y |\n|---|---|\n| 1 | 2 |\n\n"
    "End paragraph with some extra words to pad things out."
)

SAMPLES = [LONG_PARAS, FENCED, UNCLOSED, TABLE, CJK, MIXED, "x" * 500]


# ── split_text_fence_aware (paragraph mode: yuanbao-derived) ─────────────────


@pytest.mark.parametrize("text", SAMPLES)
@pytest.mark.parametrize("limit", [80, 200, 400])
def test_paragraph_mode_chunks_within_limit_unless_atomic(text, limit):
    chunks = split_text_fence_aware(text, limit, prefer_paragraphs=True)
    atoms = split_markdown_atoms(text)
    oversize_atom = any(len(a) > limit for a in atoms)
    for chunk in chunks:
        # A chunk may exceed the limit only when a single indivisible atom
        # (code block / table) itself exceeds it.
        if len(chunk) > limit:
            assert oversize_atom, (
                f"chunk of {len(chunk)} > {limit} without an oversize atom"
            )


# ── split_text_fence_aware (newline mode + balancing: stream_consumer) ───────


@pytest.mark.parametrize("text", SAMPLES)
def test_newline_mode_balanced_fences_every_chunk(text):
    chunks = split_text_fence_aware(
        text, 100, prefer_paragraphs=False, balance_fences=True
    )
    for chunk in chunks:
        # Every delivered chunk must render standalone: even fence count.
        assert chunk.count("\n```") % 2 == 0 or not text_has_unclosed_fence(chunk)
        assert not text_has_unclosed_fence(chunk)


# ── split_at_paragraph_boundary ──────────────────────────────────────────────


@pytest.mark.parametrize("text", SAMPLES)
def test_split_at_paragraph_boundary_head_plus_tail(text):
    head, tail = split_at_paragraph_boundary(text, 100)
    assert head + tail == text
    assert len(head) <= 100 or "\n" not in text[:100]


# ── atoms ────────────────────────────────────────────────────────────────────


def test_atoms_fence_kept_whole():
    atoms = split_markdown_atoms(FENCED)
    fence_atoms = [a for a in atoms if a.lstrip().startswith("```")]
    assert len(fence_atoms) == 1
    assert fence_atoms[0].rstrip().endswith("```")


# ── streaming merge + separators ─────────────────────────────────────────────


def test_merge_streaming_fences_rejoins_split_fence():
    chunks = ["intro\n```py\ncode line", "more code\n```\ntail"]
    merged = merge_streaming_fences(chunks)
    assert len(merged) == 1
    assert not text_has_unclosed_fence(merged[0])


def test_infer_block_separator_rules():
    assert infer_block_separator("text\n```", "next") == "\n"
    assert infer_block_separator("text", "```py\nx") == "\n"
    assert infer_block_separator("| a | b |", "| c | d |") == "\n"
    assert infer_block_separator("plain", "plain") == "\n\n"


# ── balance_fences_across_chunks ─────────────────────────────────────────────


def test_balance_closes_and_reopens():
    out = balance_fences_across_chunks(["a\n```go\nx", "y\n```\nb"])
    assert out[0].endswith("\n```")
    assert out[1].startswith("```go\n")
    assert all(not text_has_unclosed_fence(c) for c in out)


# ── greedy_pack_blocks ───────────────────────────────────────────────────────


def test_greedy_pack_overflow_callback():
    calls = []

    def overflow(block):
        calls.append(block)
        return [block[:50], block[50:]]

    packed = greedy_pack_blocks(["x" * 120], 60, overflow=overflow)
    assert calls and packed == ["x" * 50, "x" * 70]


# ── canonical table-row splitter delegation ──────────────────────────────────




# ── UTF-16 length units at the Telegram limit (#55844) ──────────────────────


def test_emoji_heavy_split_respects_utf16_limit():
    """Telegram's 4096 limit is UTF-16 code units — emoji count as 2.

    Regression for #55844: the fallback splitter must measure chunks with
    the platform len_fn, not Python codepoints, or emoji-heavy messages
    ship oversized chunks that Telegram rejects with MESSAGE_TOO_LONG.
    """
    limit = 3996  # gateway's safe limit: 4096 - 100 headroom
    text = "🚀" * 3000  # 3000 codepoints == 6000 UTF-16 units
    chunks = split_text_fence_aware(
        text, limit, utf16_len, prefer_paragraphs=False, balance_fences=True
    )
    assert len(chunks) > 1
    assert all(utf16_len(c) <= limit for c in chunks)
    assert "".join(chunks) == text


def test_mixed_emoji_newline_text_utf16_limit():
    text = "emoji 🎉🎊 line with some more words attached\n" * 400
    chunks = split_text_fence_aware(
        text, 500, utf16_len, prefer_paragraphs=False, balance_fences=True
    )
    assert all(utf16_len(c) <= 500 for c in chunks)
    assert all(c for c in chunks)
