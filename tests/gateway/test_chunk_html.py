# -*- coding: utf-8 -*-
"""TKT-0040 — repro tests for the HTML-leak / mid-span chunk-split defect.

King chunking test (2026-08-19) proved a naive `truncate_message` split of a
giant single-tag span produces chunks with UNBALANCED tags → Telegram rejects
("can't parse entities") → patch falls back → original bug returns. These tests
assert the SHARED `chunk_html` / `detect_parse_mode` utility keeps every chunk's
tags balanced and stays under the UTF-16 limit.
"""
import re

import pytest

from gateway.platforms.helpers import (
    chunk_html,
    detect_parse_mode,
)


def utf16_len(s: str) -> int:
    return sum(2 if ord(c) > 0xFFFF else 1 for c in s)

MAX = 4096


def _tags_balanced(chunk: str) -> bool:
    """True if every opened non-void tag in the chunk is closed within it.
    Uses the same tag regex as the production module (closers included)."""
    void = {"br", "hr", "img", "wbr", "input", "meta", "link"}
    stack = []
    for m in re.finditer(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)((?:\s+[^<>]*)?)\s*(/?)>", chunk):
        if m.group(1) == "/":
            if stack and stack[-1] == m.group(2).lower():
                stack.pop()
        elif not m.group(0).rstrip().endswith("/>") and m.group(2).lower() not in void:
            stack.append(m.group(2).lower())
    return not stack


# ── detect_parse_mode ────────────────────────────────────────────────────────

def test_detect_parse_mode_html():
    assert detect_parse_mode("<b>bold</b>") == "HTML"
    assert detect_parse_mode('see <a href="https://x">link</a>') == "HTML"
    assert detect_parse_mode("line one<br>line two") == "HTML"
    # KR-022: matched pair required; unmatched opener alone is not enough
    assert detect_parse_mode("use <b>bold</b> for emphasis") == "HTML"


def test_detect_parse_mode_markdown():
    assert detect_parse_mode("**bold** and `code`") == "MarkdownV2"
    assert detect_parse_mode("plain text, no tags") == "MarkdownV2"
    assert detect_parse_mode("") == "MarkdownV2"
    assert detect_parse_mode("a < b and c > d (math, not tags)") == "MarkdownV2"
    # KR-022: comparison text without spaces must NOT select HTML
    assert detect_parse_mode("if a<b and c>d then x") == "MarkdownV2"
    assert detect_parse_mode("x<y>z") == "MarkdownV2"


# ── chunk_html: the King's failing cases ─────────────────────────────────────

def test_short_html_unchanged():
    s = "<b>short</b>"
    assert chunk_html(s, max_length=MAX, len_fn=utf16_len) == [s]


def test_giant_single_span_stays_balanced():
    """King case B: one giant <b> span (4.2K cp) split mid-span → must not
    produce an unbalanced chunk."""
    body = "word " * 900  # ~4500 codepoints inside a single <b>
    html = f"<b>{body}</b>"
    chunks = chunk_html(html, max_length=MAX, len_fn=utf16_len)
    assert len(chunks) > 1, "expected the span to be split"
    for i, c in enumerate(chunks):
        assert _tags_balanced(c), f"chunk {i} has unbalanced tags: {c[:60]}…{c[-60:]}"


def test_tight_tag_boundary_stays_balanced():
    """King case C: tag boundary tight against the split point."""
    pre = "x" * 4090
    html = f"{pre}<b>tail content here</b>"
    chunks = chunk_html(html, max_length=MAX, len_fn=utf16_len)
    for i, c in enumerate(chunks):
        assert _tags_balanced(c), f"chunk {i} unbalanced: {c[:40]}…{c[-40:]}"


def test_newline_rich_card_balanced_and_content_preserved():
    """King case A (was already passing) — regression guard + content fidelity."""
    lines = "\n".join(f"<b>Header {i}</b>: value {i}" for i in range(120))
    chunks = chunk_html(lines, max_length=MAX, len_fn=utf16_len)
    for c in chunks:
        assert _tags_balanced(c)
    # every chunk must respect the UTF-16 cap once rebalanced
    for c in chunks:
        assert utf16_len(c) <= MAX + 64  # allow closer/reopen overhead slack


def test_nested_tags_reopen_on_next_chunk():
    """Nested <b><i>…</i></b> split mid-way: next chunk re-opens both, in order."""
    inner = "data " * 1000
    html = f"<b><i>{inner}</i></b>"
    chunks = chunk_html(html, max_length=MAX, len_fn=utf16_len)
    assert len(chunks) > 1
    for c in chunks:
        assert _tags_balanced(c)
    # the continuation chunk must re-open the outer tag before the inner
    if len(chunks) > 1:
        assert chunks[1].lstrip().startswith("<b>")


def test_emoji_heavy_respects_utf16_limit():
    """The utf16 regression: emoji = 2 UTF-16 units; chunks must not blow 4096."""
    html = "<b>" + ("😀" * 3000) + "</b>"  # ~6000 UTF-16 units
    chunks = chunk_html(html, max_length=MAX, len_fn=utf16_len)
    for c in chunks:
        assert utf16_len(c) <= MAX + 64
        assert _tags_balanced(c)


def test_deep_nesting_budget():
    """KR-022: 40-deep nested <span>s must not blow the 4096 limit when the
    synthetic close+reopen overhead is large. The chunker must shrink the raw
    piece to fit within budget."""
    # Build 40 nested spans with attributes (realistic overhead)
    inner = "x" * 100  # small payload, deep nesting
    for i in range(40):
        inner = f'<span class="level-{i}" data-depth="{i}">{inner}</span>'
    # Now wrap in a long body that forces a split
    body = "word " * 2000  # ~10K chars
    html = f"<b>{body}{inner}</b>"
    chunks = chunk_html(html, max_length=MAX, len_fn=utf16_len)
    for c in chunks:
        assert utf16_len(c) <= MAX, f"chunk exceeds 4096: {utf16_len(c)}"
        assert _tags_balanced(c)
