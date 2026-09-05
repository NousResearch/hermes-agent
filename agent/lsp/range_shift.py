"""Diff-aware line-shift map for cross-edit LSP delta filtering.

An edit that inserts/deletes lines moves every diagnostic below it, and the delta
filter keys on ``(severity, code, source, message, range)`` — so shifted-but-identical
diagnostics would look brand-new.  We build a pre→post line map from
``difflib.SequenceMatcher.get_opcodes()`` and apply it to the baseline before the
set-difference; diagnostics in a deleted region map to ``None`` and drop out.
Keeping range in the key preserves the "same error at a new line" signal.
"""
from __future__ import annotations

import difflib
from typing import Any, Callable, Dict, List, Optional

_Shift = Callable[[int], Optional[int]]


def build_line_shift(pre_text: str, post_text: str) -> _Shift:
    """Return ``shift(pre_line) -> post_line | None`` over 0-indexed lines (LSP convention).

    ``None`` means the line was deleted.  One ``get_opcodes()`` call up front;
    the closure scans the (small) opcode list per lookup.
    """
    pre_lines = pre_text.splitlines() if pre_text else []
    post_lines = post_text.splitlines() if post_text else []
    if pre_lines == post_lines:
        return lambda line: line
    # Opcodes are (tag, i1, i2, j1, j2): i-range in pre, j-range in post.
    opcodes = difflib.SequenceMatcher(a=pre_lines, b=post_lines, autojunk=False).get_opcodes()

    def shift(line: int) -> Optional[int]:
        for tag, i1, i2, j1, _j2 in opcodes:
            if i1 <= line < i2:
                # 'equal' maps by offset; 'delete'/'replace' lines have no
                # post counterpart.  'insert' has i1 == i2 and can't match.
                return line - i1 + j1 if tag == "equal" else None
            if line < i1:
                break
        # Past the last pre line: anchor at end of post.
        return max(0, len(post_lines) - 1) if post_lines else None

    return shift


def shift_diagnostic_range(diag: Dict[str, Any], shift: _Shift) -> Optional[Dict[str, Any]]:
    """Copy of ``diag`` with its line range remapped; ``None`` if the start line was deleted.

    A multi-line diagnostic whose end straddles the deletion collapses to a
    single-line range at the shifted start so it stays in the baseline.
    """
    rng = diag.get("range") or {}
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    pre_start_line = int(start.get("line", 0))
    new_start_line = shift(pre_start_line)
    if new_start_line is None:
        return None
    new_end_line = shift(int(end.get("line", pre_start_line)))
    if new_end_line is None:
        new_end_line = new_start_line
    return {
        **diag,
        "range": {
            "start": {"line": new_start_line, "character": int(start.get("character", 0))},
            "end": {"line": new_end_line, "character": int(end.get("character", 0))},
        },
    }


def shift_baseline(baseline: List[Dict[str, Any]], shift: _Shift) -> List[Dict[str, Any]]:
    """Apply ``shift`` to every diagnostic in ``baseline``, dropping deleted entries."""
    shifted = (shift_diagnostic_range(d, shift) for d in baseline if isinstance(d, dict))
    return [s for s in shifted if s is not None]


__all__ = ["build_line_shift", "shift_diagnostic_range", "shift_baseline"]
