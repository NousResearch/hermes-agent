"""Shared Signal formatting helpers: Markdown → Signal native formatting lives here so both the live
adapter and the standalone send paths emit the same bodyRanges."""

from __future__ import annotations

import re

from agent.markdown_tables import is_table_divider, realign_markdown_tables

# Signal has no fixed-width client area; this budgets for a phone screen in the app's
# monospace face. Tables wider than this fall back to realign_markdown_tables()'s
# vertical Key: value rendering rather than soft-wrapping mid-cell.
_TABLE_WIDTH = 40

_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_INLINE_PATTERNS = [
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), "BOLD"),
    (re.compile(r"__(.+?)__", re.DOTALL), "BOLD"),
    (re.compile(r"~~(.+?)~~", re.DOTALL), "STRIKETHROUGH"),
    (re.compile(r"`(.+?)`"), "MONOSPACE"),
    (re.compile(r"(?<!\*)\*(?!\*| )(.+?)(?<!\*)\*(?!\*)"), "ITALIC"),
    (re.compile(r"(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)"), "ITALIC")]


def _utf16_len(s: str) -> int:
    """Length of *s* in UTF-16 code units."""
    return len(s.encode("utf-16-le")) // 2


def _normalize_bullet_markers(source: str) -> str:
    """Replace Markdown bullet markers with plain Unicode bullets (Signal renders ``- item`` literally).
    Fenced code blocks are kept byte-for-byte: list-looking lines inside code are code, not bullets."""
    parts = re.split(r"(```.*?```)", source, flags=re.DOTALL)
    return "".join(re.sub(r"(?m)^([ \t]{0,3})[-*+]\s+", r"\1• ", part) if idx % 2 == 0 else part
                   for idx, part in enumerate(parts))


def _fence_tables(source: str) -> str:
    """Re-align every GFM table outside a code fence and wrap it in a fence, so the code-block pass
    in markdown_to_signal() records one MONOSPACE range over the aligned block and shifts every
    later range for it. Fenced regions are left alone: a pipe table inside a code block is code."""
    parts = re.split(r"(```.*?```)", source, flags=re.DOTALL)
    return "".join(part if idx % 2 else _fence_unfenced_tables(part) for idx, part in enumerate(parts))


def _fence_unfenced_tables(text: str) -> str:
    if "|" not in text:
        return text
    lines, out = text.split("\n"), []
    i, n = 0, len(lines)
    while i < n:
        # Same block rule as realign_markdown_tables(): header row, divider, contiguous pipe rows.
        if "|" in lines[i] and i + 1 < n and is_table_divider(lines[i + 1]):
            j = i + 2
            while j < n and "|" in lines[j] and lines[j].strip():
                j += 1
            out += ["```", realign_markdown_tables("\n".join(lines[i:j]), _TABLE_WIDTH), "```"]
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _process_inline(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """strip inline markdown markers and return plain text with relative style spans."""
    all_matches: list[tuple[int, int, int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern, style in _INLINE_PATTERNS:
        for match in pattern.finditer(text):
            ms, me = match.start(), match.end()
            if not any(ms < oe and me > os for os, oe in occupied):
                all_matches.append((ms, me, match.start(1), match.end(1), style))
                occupied.append((ms, me))
    all_matches.sort()
    result = ""
    last_end = 0
    inline_styles: list[tuple[int, int, str]] = []
    for ms, me, g1s, g1e, style in all_matches:
        result += text[last_end:ms]
        start_offset = len(result)
        inner = text[g1s:g1e]
        inline_styles.append((start_offset, len(inner), style))
        result += inner
        last_end = me
    result += text[last_end:]
    return result, inline_styles


def _process_plain_markdown(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """process headings and inline styles in non-code markdown segments."""
    result = ""
    styles: list[tuple[int, int, str]] = []
    last_end = 0
    for match in _HEADING_RE.finditer(text):
        before = text[last_end:match.start()]
        if before:
            clean_before, before_styles = _process_inline(before)
            base_offset = len(result)
            for s, l, st in before_styles:
                styles.append((base_offset + s, l, st))
            result += clean_before

        eol = text.find("\n", match.end())
        if eol == -1:
            eol = len(text)
        heading_raw = text[match.end():eol]
        clean_heading, heading_inline = _process_inline(heading_raw)
        base_offset = len(result)
        styles.append((base_offset, len(clean_heading), "BOLD"))
        for s, l, st in heading_inline:
            styles.append((base_offset + s, l, st))
        result += clean_heading
        last_end = eol

    after = text[last_end:]
    if after:
        clean_after, after_styles = _process_inline(after)
        base_offset = len(result)
        for s, l, st in after_styles:
            styles.append((base_offset + s, l, st))
        result += clean_after

    return result, styles


def markdown_to_signal(text: str) -> tuple[str, list[str]]:
    """convert markdown to plain text + signal textstyles list."""
    text = _fence_tables(_normalize_bullet_markers(re.sub(r"\n{3,}", "\n\n", text).strip()))
    final_text = ""
    raw_styles: list[tuple[int, int, str]] = []
    last_end = 0
    for match in _CODE_BLOCK_RE.finditer(text):
        before = text[last_end:match.start()]
        if before:
            clean_before, before_styles = _process_plain_markdown(before)
            base_offset = len(final_text)
            for s, l, st in before_styles:
                raw_styles.append((base_offset + s, l, st))
            final_text += clean_before

        inner = match.group(1).rstrip("\n")
        base_offset = len(final_text)
        raw_styles.append((base_offset, len(inner), "MONOSPACE"))
        final_text += inner
        last_end = match.end()

    after = text[last_end:]
    if after:
        clean_after, after_styles = _process_plain_markdown(after)
        base_offset = len(final_text)
        for s, l, st in after_styles:
            raw_styles.append((base_offset + s, l, st))
        final_text += clean_after

    style_strings: list[str] = []
    for cp_start, cp_len, style_type in sorted(raw_styles):
        if 0 <= cp_start and cp_start + cp_len <= len(final_text) and cp_len > 0:
            u16_start = _utf16_len(final_text[:cp_start])
            u16_len = _utf16_len(final_text[cp_start : cp_start + cp_len])
            style_strings.append(f"{u16_start}:{u16_len}:{style_type}")
    return final_text, style_strings

