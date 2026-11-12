"""hashline-guard core: strict-match pre-check on patch `old_string` anchors.

verify_anchor(file_text, old_string) -> ('ok', None) | ('block', reason)
context_hash(file_text, old_string, window=2) -> SHA-256 hex string

FAIL-OPEN: any IO/setup error is logged and skipped, never blocks.

CANONICALIZATION
----------------
- Newline normalization: CRLF/LFCR/CR are normalized to LF before matching/hashing.
- Trailing whitespace: kept byte-exact on each line; do NOT strip.
- Hash payload version tag: 'hashline:v1:<windowed_text>' so future format
  changes don't collide with prior hashline values.
- Window size is part of the computed text shape; changing window changes the hash.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_EMPTY_REASON = "old_string must be non-empty"
_VERSION_TAG = "hashline:v1"


def verify_anchor(file_text: str, old_string: str) -> Tuple[str, Optional[str]]:
    """Return ('ok', None) when old_string occurs exactly once, else ('block', reason).

    Matching is tolerant of trailing-whitespace differences: an anchor that
    matches after rstrip()-ing each line is accepted (the patch tool applies
    its own exact match; this guard's job is catching stale/ambiguous anchors,
    not pedantic whitespace). When no match is found, the reason includes a
    snippet of the file near the closest line so the caller can see the drift.
    """
    if not old_string:
        return "block", _EMPTY_REASON

    canonical = _canonicalize(file_text)
    anchor = _canonicalize(old_string)
    count = canonical.count(anchor)
    if count == 1:
        return "ok", None
    if count > 1:
        return "block", f"old_string is ambiguous: found {count} times in live file"

    # Tolerant pass: rstrip each line, then retry.
    stripped_text = "\n".join(line.rstrip() for line in canonical.split("\n"))
    stripped_anchor = "\n".join(line.rstrip() for line in anchor.split("\n"))
    scount = stripped_text.count(stripped_anchor)
    if scount == 1:
        return "ok", None
    if scount > 1:
        return "block", f"old_string is ambiguous after whitespace normalization: found {scount} times"

    # No match at all — produce a helpful snippet.
    snippet = _nearest_snippet(canonical, anchor)
    if snippet:
        return "block", f"old_string not found in live file — anchor drifted. Nearest lines:\n{snippet}"
    return "block", "old_string not found in live file — anchor drifted"


def context_hash(file_text: str, old_string: str, window: int = 2) -> str:
    """SHA-256 of old_string + up to `window` surrounding lines (foundation for full-hashline primitive)."""
    if not old_string:
        return hashlib.sha256(b"").hexdigest()

    return compute_hashline(file_text, old_string, 0, window=window)


def find_all(file_text: str, old_string: str) -> List[Tuple[int, int, int]]:
    """Return list of (start_idx, end_idx, line_number) for every occurrence of old_string."""
    if not old_string:
        return []
    text = _canonicalize(file_text)
    anchor = _canonicalize(old_string)
    out = []
    start = 0
    while True:
        idx = text.find(anchor, start)
        if idx == -1:
            break
        end_idx = idx + len(anchor)
        line_number = text[:idx].count("\n") + 1
        out.append((idx, end_idx, line_number))
        start = end_idx
    return out


def compute_hashline(file_text: str, old_string: str, occurrence_index: int, window: int = 2) -> str:
    """SHA-256 hex digest of a versioned window around the `occurrence_index`-th match."""
    if not old_string:
        return hashlib.sha256(b"").hexdigest()
    matches = find_all(file_text, old_string)
    if not matches or occurrence_index < 0 or occurrence_index >= len(matches):
        return hashlib.sha256(old_string.encode("utf-8")).hexdigest()

    lines = _canonicalize(file_text).splitlines()
    anchor_line = matches[occurrence_index][2] - 1
    start = max(0, anchor_line - window)
    end = min(len(lines), anchor_line + old_string.count("\n") + 1 + window)
    window_lines = lines[start:end]
    payload = f"{_VERSION_TAG}:{occurrence_index}:" + "\n".join(window_lines)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_anchor_by_hash(file_text: str, old_string: str, expected_hashline: str, window: int = 2) -> Tuple[str, Tuple]:
    """Pin an exact occurrence by hashline.

    Returns:
      ('ok', occurrence_index)
      ('block', {'reason': str, 'found': [hashline, ...], 'lines': [line_number, ...]})
    """
    if not old_string:
        return "block", {"reason": _EMPTY_REASON, "found": [], "lines": []}

    matches = find_all(file_text, old_string)
    count = len(matches)
    if count == 0:
        return "block", {"reason": "old_string not found in live file — anchor drifted", "found": [], "lines": []}

    found = []
    lines = []
    for idx in range(count):
        lines.append(matches[idx][2])
        found.append(compute_hashline(file_text, old_string, idx, window=window))

    if count == 1:
        if found[0] == expected_hashline:
            return "ok", 0
        return "block", {"reason": "single occurrence hashline mismatch", "found": found, "lines": lines}

    exact_matches = [i for i, h in enumerate(found) if h == expected_hashline]
    if len(exact_matches) == 1:
        return "ok", exact_matches[0]
    if not exact_matches:
        reason = f"hashline did not match any occurrence; found {count} anchors"
    else:
        reason = f"hashline is ambiguous: matched {len(exact_matches)} occurrences"
    return "block", {"reason": reason, "found": found, "lines": lines}


def _nearest_snippet(file_text: str, anchor: str, radius: int = 3) -> str:
    """Return a few lines of file_text around the line containing the first
    word of anchor, to help diagnose drift. Empty string if no candidate."""
    if not anchor.strip():
        return ""
    first_line = anchor.splitlines()[0].strip() if anchor.splitlines() else ""
    if not first_line:
        return ""
    # Find the line in file_text whose stripped form shares the most
    # leading tokens with the anchor's first line.
    anchor_tokens = first_line.split()[:4]
    best_idx = -1
    best_score = 0
    for i, line in enumerate(file_text.splitlines()):
        ls = line.strip().split()[:4]
        score = sum(1 for a, b in zip(anchor_tokens, ls) if a == b)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx < 0 or best_score == 0:
        return ""
    start = max(0, best_idx - radius)
    end = min(len(file_text.splitlines()), best_idx + radius + 1)
    lines = file_text.splitlines()[start:end]
    return "\n".join(f"{start + j + 1}: {ln}" for j, ln in enumerate(lines))


def _canonicalize(text: str) -> str:
    if not text:
        return ""
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    # normalize NEL/line-separator/paragraph-separator if present
    out = out.replace("\u0085", "\n").replace("\u2029", "\n").replace("\u2028", "\n")
    return out


def raw_offsets(file_text: str, canonical_start: int, canonical_end: int) -> Tuple[int, int]:
    """Map canonical (LF-normalized) offsets back to raw byte offsets.

    ``find_all``/``compute_hashline`` match against the canonicalized (CRLF/CR ->
    LF) text, so the offsets they return index the normalized string, which is
    SHORTER than the raw file when it contains CRLF/CR line endings. Splicing
    the raw file at those canonical offsets corrupts it (a ``\\r`` gets orphaned
    before the anchor). This translates a canonical [start, end) range into the
    equivalent raw byte range so the caller can splice the original text.

    Canonicalization only removes bytes (never reorders or adds), so a single
    left-to-right pass recording the raw index of each canonical char is exact.
    """
    if not file_text:
        return 0, 0
    can = _canonicalize(file_text)
    if len(can) == len(file_text):
        return canonical_start, canonical_end

    # raw_index_of[i] = raw byte index of canonical char i.
    raw_index_of: List[int] = []
    ci = 0
    for ri, ch in enumerate(file_text):
        if ci < len(can) and ch == can[ci]:
            raw_index_of.append(ri)
            ci += 1

    def _raw_start(canon_idx: int) -> int:
        """Raw offset of the first char of the canonical match (a preserved char)."""
        if canon_idx >= len(raw_index_of):
            return len(file_text)
        return raw_index_of[canon_idx]

    def _raw_end(canon_idx: int) -> int:
        """Raw exclusive end: one past the raw byte of the last preserved match char.

        The last canonical match char is at canonical_end - 1 and is always a
        preserved char (canonicalization only removes CR/separator bytes, never
        a match char), so its raw index + 1 is the exact raw boundary.
        """
        if canon_idx <= 0:
            return 0
        last_preserved = raw_index_of[canon_idx - 1] if canon_idx - 1 < len(raw_index_of) else len(file_text)
        return last_preserved + 1

    return _raw_start(canonical_start), _raw_end(canonical_end)
