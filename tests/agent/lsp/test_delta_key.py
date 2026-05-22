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

def test_diag_key_treats_shifted_diagnostics_as_distinct():
    """Two diagnostics with the same message but at different lines hash
    differently — they are genuinely different diagnostics.  The shift
    map is what makes them equal AFTER remapping; the key itself stays
    strict."""
    a = _diag(line=100)
    b = _diag(line=200)
    assert _diag_key(a) != _diag_key(b)


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


def test_diag_key_distinguishes_message():
    a = _diag(line=100, message="foo")
    b = _diag(line=100, message="bar")
    assert _diag_key(a) != _diag_key(b)


def test_diag_key_distinguishes_severity():
    a = _diag(line=100, severity=1)
    b = _diag(line=100, severity=2)
    assert _diag_key(a) != _diag_key(b)


def test_diag_key_distinguishes_source():
    a = _diag(line=100, source="Pyright")
    b = _diag(line=100, source="Ruff")
    assert _diag_key(a) != _diag_key(b)


def test_diag_key_matches_client_key_byte_for_byte():
    """The manager-side and client-side keys must agree on diagnostic
    identity — they're used by two layers that need to round-trip the
    same diagnostics through dedup and delta filtering."""
    d = _diag(line=42)
    assert _diag_key(d) == _diagnostic_key(d)


# ----------------------------------------------------------------------
# build_line_shift
# ----------------------------------------------------------------------

def test_shift_identity_for_identical_content():
    shift = build_line_shift("a\nb\nc\n", "a\nb\nc\n")
    assert shift(0) == 0
    assert shift(1) == 1
    assert shift(2) == 2


def test_shift_pure_deletion_above_line():
    """Delete 2 lines at the top; everything below shifts up by 2."""
    pre = "line0\nline1\nline2\nline3\nline4\n"
    post = "line2\nline3\nline4\n"  # deleted lines 0-1
    shift = build_line_shift(pre, post)
    # Pre lines 0,1 → deleted → None
    assert shift(0) is None
    assert shift(1) is None
    # Pre line 2 → post line 0
    assert shift(2) == 0
    # Pre line 4 → post line 2
    assert shift(4) == 2


def test_shift_pure_insertion_above_line():
    """Insert 3 lines at the top; everything below shifts down by 3."""
    pre = "line0\nline1\nline2\n"
    post = "new0\nnew1\nnew2\nline0\nline1\nline2\n"
    shift = build_line_shift(pre, post)
    # Pre lines unchanged in identity, shifted by 3
    assert shift(0) == 3
    assert shift(1) == 4
    assert shift(2) == 5


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


def test_shift_handles_empty_pre():
    """First write of a file: pre is empty, post has content.  Nothing
    to shift, so the function should be well-defined for empty pre."""
    shift = build_line_shift("", "hello\nworld\n")
    # Any pre line falls past the end of an empty pre — anchor at end of post
    assert shift(0) == 1


def test_shift_handles_empty_post():
    """File deleted to empty.  Every pre line returns None."""
    shift = build_line_shift("line0\nline1\n", "")
    assert shift(0) is None
    assert shift(1) is None


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


def test_shift_diag_drops_diagnostic_in_deleted_region():
    pre = "a\nb\nc\nd\n"
    post = "a\nd\n"  # deleted lines 1,2 (b,c)
    shift = build_line_shift(pre, post)
    d = _diag(line=1)
    assert shift_diagnostic_range(d, shift) is None


def test_shift_diag_does_not_mutate_original():
    pre = "a\nb\n"
    post = "X\na\nb\n"
    shift = build_line_shift(pre, post)
    d = _diag(line=0)
    original_line = d["range"]["start"]["line"]
    _ = shift_diagnostic_range(d, shift)
    assert d["range"]["start"]["line"] == original_line


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

def test_pipeline_filters_shifted_baseline_under_strict_key():
    """The exact scenario the bug fix is for: an edit deletes lines,
    every diagnostic below shifts, and the delta filter (strict key
    + shifted baseline) correctly identifies them as pre-existing."""
    pre = "line0\nline1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\n"
    # Delete lines 2,3,4 — pre-existing errors at lines 7,8 should
    # appear at lines 4,5 post-edit and be filtered out.
    post = "line0\nline1\nline5\nline6\nline7\nline8\nline9\n"
    shift = build_line_shift(pre, post)

    baseline = [_diag(line=7, message="X"), _diag(line=8, message="Y")]
    post_diags = [_diag(line=4, message="X"), _diag(line=5, message="Y")]

    shifted_baseline = shift_baseline(baseline, shift)
    seen = {_diag_key(d) for d in shifted_baseline}
    new_diags = [d for d in post_diags if _diag_key(d) not in seen]

    # Both errors were pre-existing — filtered out.
    assert new_diags == []


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
