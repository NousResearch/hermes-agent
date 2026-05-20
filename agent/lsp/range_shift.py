"""Diff-aware line-shift map for cross-edit LSP delta filtering.

When an edit deletes or inserts lines in the middle of a file, every
diagnostic below the edit point shifts to a new line number.  The
LSPService delta filter subtracts the pre-edit baseline from the
post-edit diagnostics keyed on ``(severity, code, source, message,
range)`` — without an adjustment, the shifted-but-otherwise-identical
diagnostics look brand-new and the agent gets flooded with noise.

The fix used here is the same trick git's blame and unified diff use:
build a piecewise-linear map from pre-edit line numbers to post-edit
line numbers, then apply that map to baseline diagnostics before the
set-difference.  Diagnostics whose pre-edit line is in a region the
edit deleted return ``None`` and are dropped from the baseline (they
genuinely no longer apply).

Trade-off vs. dropping range from the key entirely (the previous
fix): preserves the "new instance of an identical error at a
different line" signal — if the model introduces a second instance
of the same error class at a different location, that one will be
surfaced as new instead of swallowed by content-only dedup.

The map is derived from ``difflib.SequenceMatcher.get_opcodes()`` and
exposed as a single callable so callers don't have to reason about
diff regions.
"""
from __future__ import annotations

import difflib
from typing import Any, Callable, Dict, List, Optional


def build_line_shift(pre_text: str, post_text: str) -> Callable[[int], Optional[int]]:
    """Build a function mapping pre-edit line numbers to post-edit line numbers.

    Lines are 0-indexed to match the LSP wire format
    (``range.start.line`` is 0-indexed).

    The returned callable takes a pre-edit 0-indexed line number and
    returns the corresponding post-edit 0-indexed line number, or
    ``None`` if that line was deleted by the edit (no post-edit
    counterpart exists).

    Cost: one ``SequenceMatcher.get_opcodes()`` call up front; the
    returned closure is O(log n) per call (binary search over opcode
    regions).  Cheap enough to call once per write/patch and apply to
    every baseline diagnostic.
    """
    pre_lines = pre_text.splitlines() if pre_text else []
    post_lines = post_text.splitlines() if post_text else []

    # Trivial case: identical content or no content — identity map.
    if pre_lines == post_lines:
        return lambda line: line

    # SequenceMatcher.get_opcodes() returns a list of
    # (tag, i1, i2, j1, j2) where tag is 'equal', 'replace', 'delete',
    # or 'insert'.  i1:i2 is the range in pre, j1:j2 is the range in
    # post.  We build a list of (i1, i2, j1, j2, tag) tuples and
    # binary-search by i for each lookup.
    sm = difflib.SequenceMatcher(a=pre_lines, b=post_lines, autojunk=False)
    opcodes = sm.get_opcodes()

    def shift(line: int) -> Optional[int]:
        # Find the opcode region whose i1 <= line < i2.
        # Linear scan is fine — typical opcode count is small (single
        # digits for a typical patch-tool edit).
        for tag, i1, i2, j1, j2 in opcodes:
            if i1 <= line < i2:
                if tag == "equal":
                    # Pre-line N → post-line (N - i1 + j1).
                    return line - i1 + j1
                if tag == "delete":
                    # Pre-line is in a deleted region — no post counterpart.
                    return None
                if tag == "replace":
                    # Replace == delete + insert; the pre-line has no
                    # post counterpart in any meaningful sense.  Drop.
                    return None
                # 'insert' has i1 == i2 so line < i2 can't be hit.
            if line < i1:
                # Past the relevant region — handled in earlier iteration.
                break
        # Past the last opcode region (line >= len(pre_lines)).
        # Anchor at end of post.
        return max(0, len(post_lines) - 1) if post_lines else None

    return shift


def shift_diagnostic_range(diag: Dict[str, Any],
                           shift: Callable[[int], Optional[int]]) -> Optional[Dict[str, Any]]:
    """Return a copy of ``diag`` with its line range remapped through ``shift``.

    Returns ``None`` if the diagnostic's start line maps to ``None``
    (the line was deleted by the edit) — caller drops it from the
    baseline since the diagnostic no longer applies.

    Both ``start.line`` and ``end.line`` are remapped independently;
    when only the end maps to ``None`` (rare, multi-line diagnostic
    straddling the edit boundary) we collapse to a single-line range
    at the shifted start to keep the diagnostic in the baseline.

    The original ``diag`` is not mutated.
    """
    rng = diag.get("range") or {}
    start = rng.get("start") or {}
    end = rng.get("end") or {}

    pre_start_line = int(start.get("line", 0))
    pre_end_line = int(end.get("line", pre_start_line))

    new_start_line = shift(pre_start_line)
    if new_start_line is None:
        return None

    new_end_line = shift(pre_end_line)
    if new_end_line is None:
        # Diagnostic straddled the deletion — collapse to start.
        new_end_line = new_start_line

    shifted = dict(diag)
    shifted["range"] = {
        "start": {
            "line": new_start_line,
            "character": int(start.get("character", 0)),
        },
        "end": {
            "line": new_end_line,
            "character": int(end.get("character", 0)),
        },
    }
    return shifted


def shift_baseline(baseline: List[Dict[str, Any]],
                   shift: Callable[[int], Optional[int]]) -> List[Dict[str, Any]]:
    """Apply ``shift`` to every diagnostic in ``baseline``, dropping deleted entries."""
    out: List[Dict[str, Any]] = []
    for d in baseline:
        if not isinstance(d, dict):
            continue
        shifted = shift_diagnostic_range(d, shift)
        if shifted is not None:
            out.append(shifted)
    return out


__all__ = ["build_line_shift", "shift_diagnostic_range", "shift_baseline"]
