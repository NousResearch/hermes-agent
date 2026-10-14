"""Pure fenced-code-block parser for Markdown.

Equivalent to the TypeScript ``codeFence.ts`` used by CopyBlox.
Recognises backtick and tilde fences (length ≥ 3), requires matching
closer character and minimum length, extracts info string / language,
and returns the raw content between the fences.
"""

from __future__ import annotations

import re

_FENCE_OPENER_RE = re.compile(r'^\s*(`{3,}|~{3,})(.*)$')
_FENCE_CLOSER_RE = re.compile(r'^\s*(`{3,}|~{3,})\s*$')


def parse_code_fences(source: str) -> list[dict]:
    """Parse all fenced code blocks from raw source text.

    Returns a list of dicts with keys:

    * ``closed`` (bool) — whether a matching closer was found
    * ``open_line_index`` (int) — 0-based line index of the opener
    * ``end_line_index`` (int) — line index of the closer, or ``-1``
    * ``fence_char`` (``'`'`` or ``'~'``)
    * ``fence_length`` (int)
    * ``info_string`` (str) — raw info from the opener line
    * ``language`` (str) — normalised display language
    * ``raw_content`` (str) — exact text between fences
    """
    lines = _line_boundaries(source)
    fences: list[dict] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        match = _FENCE_OPENER_RE.match(line)

        if not match:
            i += 1
            continue

        fence_char = match.group(1)[0]
        fence_length = len(match.group(1))
        info_string = match.group(2).strip()
        open_line_index = i
        i += 1

        content_parts: list[str] = []
        closer_line = -1

        for scan in range(i, len(lines)):
            close_match = _FENCE_CLOSER_RE.match(lines[scan])
            if (
                close_match
                and close_match.group(1)[0] == fence_char
                and len(close_match.group(1)) >= fence_length
            ):
                closer_line = scan
                break
            # Collect content lines
            content_parts.append(lines[scan])

        closed = closer_line >= 0

        if closed:
            i = closer_line + 1  # skip past the closer
        else:
            i = len(lines)  # reached end of source

        raw_content = '\n'.join(content_parts)
        language = _parse_language(info_string, fence_char, raw_content)

        fences.append({
            'closed': closed,
            'open_line_index': open_line_index,
            'end_line_index': closer_line,
            'fence_char': fence_char,
            'fence_length': fence_length,
            'info_string': info_string,
            'language': language,
            'raw_content': raw_content,
        })

    return fences


def _line_boundaries(source: str) -> list[str]:
    """Split *source* into lines on ``\\n``."""
    lines: list[str] = []
    start = 0

    for idx, ch in enumerate(source):
        if ch == '\n':
            lines.append(source[start:idx])
            start = idx + 1

    # Last line (no trailing newline)
    if start <= len(source):
        lines.append(source[start:])

    return lines


def _parse_language(info_string: str, _fence_char: str, raw_content: str) -> str:
    """Extract normalised display language from the info string.

    Falls back to ``'diff'`` when content looks like a unified diff,
    otherwise returns ``'text'``.
    """
    if info_string:
        first_token = info_string.split()[0]
        normalised = first_token.lower().replace('language:', '').replace('language=', '')
        return normalised

    if raw_content.startswith('--- ') or raw_content.startswith('+++ '):
        return 'diff'

    return 'text'
