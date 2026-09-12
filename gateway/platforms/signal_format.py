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


def markdown_to_signal(text: str) -> tuple[str, list[str]]:
    """Convert markdown to plain text + Signal textStyles list. Signal uses ``bodyRanges`` (signal-cli
    ``textStyle`` / ``textStyles`` params) as ``start:length:STYLE`` with positions in UTF-16 code units.
    Supported styles: BOLD, ITALIC, STRIKETHROUGH, MONOSPACE."""
    text = _fence_tables(_normalize_bullet_markers(re.sub(r"\n{3,}", "\n\n", text).strip()))
    styles: list[tuple[int, int, str]] = []
    while match := _CODE_BLOCK_RE.search(text):
        inner = match.group(1).rstrip("\n")
        styles.append((match.start(), len(inner), "MONOSPACE"))
        text = text[: match.start()] + inner + text[match.end() :]
    new_text, last_end = "", 0
    for match in _HEADING_RE.finditer(text):
        new_text += text[last_end : match.start()]
        eol = text.find("\n", match.end())
        if eol == -1:
            eol = len(text)
        heading_text = text[match.end() : eol]
        styles.append((len(new_text), len(heading_text), "BOLD"))
        new_text += heading_text
        last_end = eol
    text = new_text + text[last_end:]
    # Inline markers: first pattern to claim a span wins; later overlapping matches are dropped.
    all_matches: list[tuple[int, int, int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern, style in _INLINE_PATTERNS:
        for match in pattern.finditer(text):
            ms, me = match.start(), match.end()
            if not any(ms < oe and me > os for os, oe in occupied):
                all_matches.append((ms, me, match.start(1), match.end(1), style))
                occupied.append((ms, me))
    all_matches.sort()
    # Strip the markers, recording (pos, len) removals so earlier block/heading ranges can be
    # shifted, and capturing inline ranges in the stripped text.
    result, last_end = "", 0
    removals: list[tuple[int, int]] = []
    inline_styles: list[tuple[int, int, str]] = []
    for ms, me, g1s, g1e, style in all_matches:
        if g1s > ms:
            removals.append((ms, g1s - ms))
        if me > g1e:
            removals.append((g1e, me - g1e))
        result += text[last_end:ms]
        inner = text[g1s:g1e]
        inline_styles.append((len(result), len(inner), style))
        result += inner
        last_end = me
    removals.sort()

    def _adjust(pos: int) -> int:
        shift = 0
        for remove_pos, remove_len in removals:
            if remove_pos >= pos:
                break
            shift += min(remove_len, pos - remove_pos)
        return pos - shift

    adjusted_prior = [(_adjust(start), _adjust(start + length) - _adjust(start), style)
                      for start, length, style in styles if _adjust(start + length) > _adjust(start)]
    text = result + text[last_end:]
    style_strings: list[str] = []
    for cp_start, cp_len, style_type in sorted(adjusted_prior + inline_styles):
        if 0 <= cp_start and cp_start + cp_len <= len(text):
            u16_start, u16_len = _utf16_len(text[:cp_start]), _utf16_len(text[cp_start : cp_start + cp_len])
            style_strings.append(f"{u16_start}:{u16_len}:{style_type}")
    return text, style_strings
