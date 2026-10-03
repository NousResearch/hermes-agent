"""Shared Signal formatting helpers: Markdown → Signal native formatting lives here so both the live
adapter and the standalone send paths emit the same bodyRanges."""

from __future__ import annotations

import bisect
import itertools
import re

from agent.markdown_tables import is_table_divider, realign_markdown_tables

# Signal has no fixed-width client area; this budgets for a phone screen in the app's
# monospace face. Tables wider than this fall back to realign_markdown_tables()'s
# vertical Key: value rendering rather than soft-wrapping mid-cell.
_TABLE_WIDTH = 40

_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`(.+?)`")
_INLINE_PATTERNS = [
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), "BOLD"),
    (re.compile(r"__(.+?)__", re.DOTALL), "BOLD"),
    (re.compile(r"~~(.+?)~~", re.DOTALL), "STRIKETHROUGH"),
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
    # Headings and inline markers are resolved in this text's coordinates and every marker they strip
    # is one (pos, len) removal, so a single _adjust maps each range (code blocks included) onto the
    # final text. Code is verbatim: a '#' line or '*', '_', '`' inside it is code, not formatting.
    code_spans = [(start, start + length) for start, length, _ in styles]
    removals: list[tuple[int, int]] = []
    for match in _HEADING_RE.finditer(text):
        if any(match.start() < ce and match.end() > cs for cs, ce in code_spans):
            continue
        eol = text.find("\n", match.end())
        if eol == -1:
            eol = len(text)
        removals.append((match.start(), match.end() - match.start()))
        styles.append((match.end(), eol - match.end(), "BOLD"))
    masked = _mask(text, code_spans)
    all_matches = [(m.start(), m.end(), m.start(1), m.end(1), "MONOSPACE") for m in _INLINE_CODE_RE.finditer(masked)]
    masked = _mask(masked, [(ms, me) for ms, me, *_ in all_matches])
    # Other inline markers: first pattern to claim a span wins; later overlapping matches are dropped.
    # Claimed spans never overlap, so they stay sorted by start and by end: only the last one starting
    # before a match can overlap it.
    occupied_starts: list[int] = []
    occupied_ends: list[int] = []
    for pattern, style in _INLINE_PATTERNS:
        for match in pattern.finditer(masked):
            ms, me = match.start(), match.end()
            i = bisect.bisect_left(occupied_starts, me)
            if not (i and occupied_ends[i - 1] > ms):
                all_matches.append((ms, me, match.start(1), match.end(1), style))
                occupied_starts.insert(i, ms)
                occupied_ends.insert(i, me)
    for ms, me, g1s, g1e, style in all_matches:
        removals += [(ms, g1s - ms), (g1e, me - g1e)]
        styles.append((g1s, g1e - g1s, style))
    removals.sort()
    # Every pass below is linear in the text (plus a log factor per range): a reply with thousands of
    # spans must not block the gateway on per-span whole-string work.
    removal_starts = [remove_pos for remove_pos, _ in removals]
    removed_before = list(itertools.accumulate((remove_len for _, remove_len in removals), initial=0))

    def _adjust(pos: int) -> int:
        # Removals never overlap, so only the last one starting before pos can straddle it.
        if not (i := bisect.bisect_left(removal_starts, pos)):
            return pos
        return pos - removed_before[i - 1] - min(removals[i - 1][1], pos - removal_starts[i - 1])

    kept, last_end = [], 0
    for remove_pos, remove_len in removals:
        kept.append(text[last_end:remove_pos])
        last_end = remove_pos + remove_len
    text = "".join(kept) + text[last_end:]
    adjusted = [(_adjust(start), _adjust(start + length) - _adjust(start), style)
                for start, length, style in styles if _adjust(start + length) > _adjust(start)]
    style_strings: list[str] = []
    cp_done = u16_done = 0  # sorted starts: advance the UTF-16 offset instead of re-encoding each prefix
    for cp_start, cp_len, style_type in sorted(adjusted):
        if 0 <= cp_start and cp_start + cp_len <= len(text):
            u16_done += _utf16_len(text[cp_done:cp_start])
            cp_done = cp_start
            style_strings.append(f"{u16_done}:{_utf16_len(text[cp_start : cp_start + cp_len])}:{style_type}")
    return text, style_strings


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    """Blank *spans* (keeping newlines, so line-bound patterns still see line breaks) so no marker
    pattern can match inside them; positions are unchanged."""
    parts, last_end = [], 0
    for start, end in sorted(spans):  # spans never overlap
        parts += [text[last_end:start], re.sub(r"[^\n]", "\x00", text[start:end])]
        last_end = end
    return "".join(parts) + text[last_end:]
