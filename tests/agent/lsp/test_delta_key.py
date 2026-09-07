"""Tests for cross-edit LSP delta filtering.

The delta-filter contract spans three pieces:

  1. ``agent.lsp.manager._diag_key`` — strict equality key including
     the diagnostic's position range.  Two diagnostics with the same
     content but different lines are NOT equal under this key (they
     are genuinely different diagnostics).
  2. ``agent.lsp.range_shift.build_line_shift`` — derives a function
     mapping pre-edit line numbers to post-edit line numbers from a
     pre/post text pair.
  3. ``agent.lsp.manager.LSPService.get_diagnostics_sync(line_shift=…)``
     — applies the shift to baseline diagnostics before computing the
     set-difference, so pre-existing errors at shifted lines hash
     equal to their post-edit counterparts and get filtered out.

These tests exercise the contract at the unit level; the E2E case
(real LSP server, real shift) is covered in test_service.py.
"""
from __future__ import annotations

from agent.lsp.client import _diagnostic_key
from agent.lsp.manager import _diag_key
from agent.lsp.range_shift import (
    build_line_shift,
    shift_baseline,
    shift_diagnostic_range,
)


def _diag(*, line: int, message: str = "Undefined variable",
          severity: int = 1, code: str = "reportUndefinedVariable",
          source: str = "Pyright", end_line: int | None = None) -> dict:
    if end_line is None:
        end_line = line
    return {
        "severity": severity,
        "code": code,
        "source": source,
        "message": message,
        "range": {
            "start": {"line": line, "character": 0},
            "end": {"line": end_line, "character": 10},
        },
    }


# ----------------------------------------------------------------------
# _diag_key: strict equality (with range)
# ----------------------------------------------------------------------



def test_diag_key_matches_client_key_for_shifted_baseline():
    """When a baseline diagnostic is remapped through a shift, its
    _diag_key must match the corresponding post-edit diagnostic's key
    at the same coordinates.  This is the contract the delta filter
    relies on."""
    pre = _diag(line=200)
    # Edit deletes 14 lines above line 200, so the same error now
    # appears at line 186 post-edit.
    shift = lambda L: L - 14 if L >= 14 else L
    shifted = shift_diagnostic_range(pre, shift)
    assert shifted is not None
    post = _diag(line=186)
    assert _diag_key(shifted) == _diag_key(post)








def test_diag_key_matches_client_key_byte_for_byte():
    """The manager-side and client-side keys must agree on diagnostic
    identity — they're used by two layers that need to round-trip the
    same diagnostics through dedup and delta filtering."""
    d = _diag(line=42)
    assert _diag_key(d) == _diagnostic_key(d)


# ----------------------------------------------------------------------
# build_line_shift
# ----------------------------------------------------------------------







def test_shift_replacement_in_middle():
    """Replace 2 lines in the middle with 1 line.  Lines above
    unchanged; lines below shift up by 1."""
    pre = "a\nb\nc\nd\ne\n"
    post = "a\nb\nX\ne\n"  # replaced lines 2,3 (c,d) with X
    shift = build_line_shift(pre, post)
    assert shift(0) == 0  # a → a
    assert shift(1) == 1  # b → b
    assert shift(2) is None  # c → deleted
    assert shift(3) is None  # d → deleted
    assert shift(4) == 3  # e → post line 3






# ----------------------------------------------------------------------
# shift_diagnostic_range
# ----------------------------------------------------------------------

def test_shift_diag_remaps_start_and_end():
    pre = "a\nb\nc\nd\n"
    post = "X\na\nb\nc\nd\n"  # one line inserted at top
    shift = build_line_shift(pre, post)
    d = _diag(line=2, end_line=2)
    remapped = shift_diagnostic_range(d, shift)
    assert remapped is not None
    assert remapped["range"]["start"]["line"] == 3
    assert remapped["range"]["end"]["line"] == 3






def test_shift_baseline_drops_deleted_and_remaps_rest():
    pre = "a\nb\nc\nd\ne\n"
    post = "a\ne\n"  # deleted b,c,d
    shift = build_line_shift(pre, post)
    baseline = [
        _diag(line=0, message="err on a"),
        _diag(line=1, message="err on b"),  # → deleted
        _diag(line=2, message="err on c"),  # → deleted
        _diag(line=4, message="err on e"),
    ]
    out = shift_baseline(baseline, shift)
    assert [d["message"] for d in out] == ["err on a", "err on e"]
    assert out[0]["range"]["start"]["line"] == 0
    assert out[1]["range"]["start"]["line"] == 1


# ----------------------------------------------------------------------
# End-to-end: simulate the delta-filter pipeline
# ----------------------------------------------------------------------



def test_pipeline_preserves_new_instance_at_different_line():
    """The case content-only keys would miss: the model introduces a
    SECOND instance of the same error class at a new location.  The
    new instance must surface."""
    pre = "good\ngood\ngood\n"
    post = "good\nbad\ngood\nbad\n"  # added 2 new error lines
    shift = build_line_shift(pre, post)

    baseline = [_diag(line=0, message="bad style")]  # pre-existing
    post_diags = [
        _diag(line=0, message="bad style"),  # pre-existing
        _diag(line=1, message="bad style"),  # NEW — different line
        _diag(line=3, message="bad style"),  # NEW — different line
    ]

    shifted_baseline = shift_baseline(baseline, shift)
    seen = {_diag_key(d) for d in shifted_baseline}
    new_diags = [d for d in post_diags if _diag_key(d) not in seen]

    # Two genuinely new instances must be surfaced.
    assert len(new_diags) == 2
    assert {d["range"]["start"]["line"] for d in new_diags} == {1, 3}
