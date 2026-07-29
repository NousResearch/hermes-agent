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


@pytest.mark.parametrize("text", [FENCED, MIXED])
def test_paragraph_mode_never_splits_closed_fences(text):
    for limit in (80, 150, 300):
        chunks = split_text_fence_aware(text, limit, prefer_paragraphs=True)
        for chunk in chunks:
            assert not text_has_unclosed_fence(chunk), (
                f"fence split mid-block at limit={limit}: {chunk!r}"
            )


@pytest.mark.parametrize("text", SAMPLES)
def test_paragraph_mode_no_empty_chunks(text):
    for limit in (80, 400):
        chunks = split_text_fence_aware(text, limit, prefer_paragraphs=True)
        assert all(c for c in chunks)


def test_paragraph_mode_short_text_single_chunk():
    assert split_text_fence_aware("hello", 100) == ["hello"]
    assert split_text_fence_aware("", 100) == []


def test_paragraph_mode_utf16_len_fn():
    chunks = split_text_fence_aware(CJK, 120, utf16_len, prefer_paragraphs=True)
    assert chunks
    assert all(utf16_len(c) <= 120 for c in chunks)


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


def test_newline_mode_balancing_reopens_language_tag():
    text = "before\n\n```python\n" + "print(1)\n" * 20 + "```\nafter"
    chunks = split_text_fence_aware(
        text, 80, prefer_paragraphs=False, balance_fences=True
    )
    assert len(chunks) > 1
    # Some tail chunk must reopen the python fence.
    assert any(c.startswith("```python\n") for c in chunks[1:])


def test_newline_mode_content_preserved_without_fences():
    text = "\n".join(f"line {i} with several words in it" for i in range(40))
    chunks = split_text_fence_aware(text, 120, prefer_paragraphs=False)
    joined = "\n".join(chunks)
    # Newline-mode splitting only removes leading newlines at boundaries.
    assert joined.replace("\n", "") == text.replace("\n", "")


# ── split_at_paragraph_boundary ──────────────────────────────────────────────


@pytest.mark.parametrize("text", SAMPLES)
def test_split_at_paragraph_boundary_head_plus_tail(text):
    head, tail = split_at_paragraph_boundary(text, 100)
    assert head + tail == text
    assert len(head) <= 100 or "\n" not in text[:100]


def test_split_at_paragraph_boundary_prefers_blank_line():
    text = "para one\n\npara two\n\npara three " + "x" * 200
    head, _ = split_at_paragraph_boundary(text, 60)
    assert head.endswith("\n\n")


def test_split_at_paragraph_boundary_cjk_sentence():
    text = "第一句话。\n第二句话！\n" + "第三句话没有结束标点一直写下去" * 20
    head, tail = split_at_paragraph_boundary(text, 30)
    assert head + tail == text
    assert head.endswith(("。\n", "！\n"))


# ── atoms ────────────────────────────────────────────────────────────────────


def test_atoms_fence_kept_whole():
    atoms = split_markdown_atoms(FENCED)
    fence_atoms = [a for a in atoms if a.lstrip().startswith("```")]
    assert len(fence_atoms) == 1
    assert fence_atoms[0].rstrip().endswith("```")


def test_atoms_table_kept_whole():
    atoms = split_markdown_atoms(TABLE)
    table_atoms = [a for a in atoms if a.split("\n")[0].strip().startswith("|")]
    assert len(table_atoms) == 1
    assert table_atoms[0].count("\n") == 41  # header + rule + 40 rows


def test_atoms_nonempty_and_no_blank_lines():
    for text in SAMPLES:
        for atom in split_markdown_atoms(text):
            assert atom.strip()


# ── streaming merge + separators ─────────────────────────────────────────────


def test_merge_streaming_fences_rejoins_split_fence():
    chunks = ["intro\n```py\ncode line", "more code\n```\ntail"]
    merged = merge_streaming_fences(chunks)
    assert len(merged) == 1
    assert not text_has_unclosed_fence(merged[0])


def test_merge_streaming_fences_leaves_balanced_alone():
    chunks = ["one", "```\nx\n```", "three"]
    assert merge_streaming_fences(chunks) == chunks
    assert merge_streaming_fences([]) == []


def test_infer_block_separator_rules():
    assert infer_block_separator("text\n```", "next") == "\n"
    assert infer_block_separator("text", "```py\nx") == "\n"
    assert infer_block_separator("| a | b |", "| c | d |") == "\n"
    assert infer_block_separator("plain", "plain") == "\n\n"


# ── balance_fences_across_chunks ─────────────────────────────────────────────


def test_balance_single_chunk_untouched():
    chunks = ["```py\nunclosed"]
    assert balance_fences_across_chunks(chunks) == chunks


def test_balance_closes_and_reopens():
    out = balance_fences_across_chunks(["a\n```go\nx", "y\n```\nb"])
    assert out[0].endswith("\n```")
    assert out[1].startswith("```go\n")
    assert all(not text_has_unclosed_fence(c) for c in out)


# ── greedy_pack_blocks ───────────────────────────────────────────────────────


def test_greedy_pack_respects_limit_and_order():
    blocks = [f"block {i} " + "w" * 30 for i in range(10)]
    packed = greedy_pack_blocks(blocks, 90)
    assert all(len(p) <= 90 for p in packed)
    assert "\n\n".join(packed).replace("\n\n", "|").count("|") >= 0
    # Order/content preserved
    assert "".join(packed).replace("\n\n", "") == "".join(blocks)


def test_greedy_pack_overflow_callback():
    calls = []

    def overflow(block):
        calls.append(block)
        return [block[:50], block[50:]]

    packed = greedy_pack_blocks(["x" * 120], 60, overflow=overflow)
    assert calls and packed == ["x" * 50, "x" * 70]


# ── canonical table-row splitter delegation ──────────────────────────────────


def test_table_row_splitters_are_unified():
    from agent.markdown_tables import split_table_row
    from gateway.platforms.weixin import _split_table_row

    rows = ["| a | b | c |", "a | b | c", "|配置|状态|", "  | x |  ", "||"]
    for row in rows:
        expected = split_table_row(row)
        assert split_markdown_table_row(row) == expected
        assert _split_table_row(row) == expected
