"""Tests for agent.lsp.range_shift — build_line_shift, shift_diagnostic_range, shift_baseline."""

from __future__ import annotations

import pytest

from agent.lsp.range_shift import (
    build_line_shift,
    shift_baseline,
    shift_diagnostic_range,
)


# ============================================================================
# build_line_shift
# ============================================================================
class TestBuildLineShift:
    # -- identity ------------------------------------------------------------
    def test_identical_text_identity_map(self):
        text = "line0\nline1\nline2"
        shift = build_line_shift(text, text)
        assert shift(0) == 0
        assert shift(1) == 1
        assert shift(2) == 2

    def test_both_empty_identity_map(self):
        shift = build_line_shift("", "")
        assert shift(0) == 0
        assert shift(5) == 5

    # -- empty pre-text ------------------------------------------------------
    def test_empty_pre_text_is_identity(self):
        """Empty pre_text → pre_lines=[], equal to post_lines if both empty."""
        shift = build_line_shift("", "line0\nline1")
        # pre_lines=[] != post_lines=["line0","line1"] → goes to difflib
        # Opcodes for empty pre: [('insert', 0,0, 0,2)]
        # Any line query: no i1<=line<i2 match → falls past last opcode
        # → returns max(0, len(post)-1) = 1
        assert shift(0) == 1
        assert shift(10) == 1

    def test_empty_post_text(self):
        shift = build_line_shift("a\nb\nc", "")
        # pre_lines=["a","b","c"], post_lines=[]
        # Opcode: [('delete', 0,3, 0,0)]
        # line 0 → in delete region → None
        # line 3+ → falls past last opcode → post_lines empty → None
        assert shift(0) is None
        assert shift(1) is None
        assert shift(2) is None
        assert shift(3) is None

    # -- insertion only ------------------------------------------------------
    def test_single_insertion_at_start(self):
        """Pre: '' → Post: 'new\nline0\nline1'"""
        pre = "line0\nline1"
        post = "new\nline0\nline1"
        shift = build_line_shift(pre, post)

        # 'insert' has i1==i2, so line 0 doesn't hit insert; hits 'equal' at i1=0
        # Actually: opcodes for this: [('insert', 0,0,0,1), ('equal', 0,2,1,3)]
        # line 0 → hits equal: 0-0+1=1. Wait, let me think...
        # Actually 'line0' is line 0 in pre, but in post it's line 1 (after 'new')
        # Pre: ["line0","line1"], Post: ["new","line0","line1"]
        # Opcodes: [('insert',0,0,0,1), ('equal',0,2,1,3)]
        # line 0 → hits equal region (i1=0 <= 0 < i2=2): return 0-0+1 = 1
        assert shift(0) == 1  # pre line 0 → post line 1
        assert shift(1) == 2  # pre line 1 → post line 2

    def test_single_insertion_in_middle(self):
        pre = "a\nc"
        post = "a\nb\nc"
        shift = build_line_shift(pre, post)
        # Pre ["a","c"], Post ["a","b","c"]
        # Opcodes: [('equal',0,1,0,1), ('insert',1,1,1,2), ('equal',1,2,2,3)]
        assert shift(0) == 0  # 'a' stays at 0
        # line 1: hits equal (i1=1 <= 1 < i2=2): 1-1+2 = 2
        assert shift(1) == 2  # 'c' shifts from 1→2

    # -- deletion ------------------------------------------------------------
    def test_single_deletion(self):
        pre = "a\nb\nc"
        post = "a\nc"
        shift = build_line_shift(pre, post)
        # line 0 ('a') → 0
        assert shift(0) == 0
        # line 1 ('b') → in delete region → None
        assert shift(1) is None
        # line 2 ('c') → shifted to post line 1
        assert shift(2) == 1

    # -- replacement ---------------------------------------------------------
    def test_single_replacement(self):
        pre = "hello\nworld"
        post = "hello\nthere"
        shift = build_line_shift(pre, post)
        # line 0 equal → 0
        assert shift(0) == 0
        # line 1 in replace region → None
        assert shift(1) is None

    # -- line past end -------------------------------------------------------
    def test_line_past_end_of_post(self):
        pre = "a\nb"
        post = "a"
        shift = build_line_shift(pre, post)
        # line 0 equal → 0
        assert shift(0) == 0
        # line 1 deleted → None
        assert shift(1) is None
        # line 2+ past end → max(0, len(post_lines)-1) = 0
        assert shift(2) == 0
        assert shift(100) == 0

    def test_line_past_end_when_post_empty(self):
        shift = build_line_shift("a", "")
        assert shift(5) is None  # post_lines empty → None

    # -- complex edits -------------------------------------------------------
    def test_multiple_edits(self):
        pre = "keep\nold\nalso_old\nend"
        post = "keep\nnew_middle\nend"
        shift = build_line_shift(pre, post)
        assert shift(0) == 0  # 'keep' stays
        assert shift(1) is None  # 'old' replaced
        assert shift(2) is None  # 'also_old' replaced
        assert shift(3) == 2  # 'end' shifts to 2


# ============================================================================
# shift_diagnostic_range
# ============================================================================
class TestShiftDiagnosticRange:
    def _diag(self, start_line=5, end_line=7, start_char=0, end_char=10):
        return {
            "range": {
                "start": {"line": start_line, "character": start_char},
                "end": {"line": end_line, "character": end_char},
            },
            "severity": 2,
            "message": "test diagnostic",
        }

    def test_normal_shift(self):
        diag = self._diag(start_line=3, end_line=5)
        # map 3→13, 5→15
        shift = lambda x: x + 10
        result = shift_diagnostic_range(diag, shift)
        assert result is not None
        assert result["range"]["start"]["line"] == 13
        assert result["range"]["end"]["line"] == 15
        # original not mutated
        assert diag["range"]["start"]["line"] == 3

    def test_start_maps_to_none_returns_none(self):
        diag = self._diag(start_line=3)
        shift = lambda x: None if x == 3 else x
        result = shift_diagnostic_range(diag, shift)
        assert result is None

    def test_end_maps_to_none_collapses_to_start(self):
        diag = self._diag(start_line=3, end_line=5)
        shift = lambda x: None if x == 5 else x + 10
        result = shift_diagnostic_range(diag, shift)
        assert result is not None
        assert result["range"]["start"]["line"] == 13
        assert result["range"]["end"]["line"] == 13  # collapsed

    def test_missing_range_defaults(self):
        diag = {"severity": 1, "message": "no range"}
        shift = lambda x: x + 1
        result = shift_diagnostic_range(diag, shift)
        assert result is not None
        assert result["range"]["start"]["line"] == 1  # default line 0 + 1
        assert result["range"]["end"]["line"] == 1

    def test_missing_start_defaults(self):
        diag = {"range": {"end": {"line": 5}}}
        shift = lambda x: x + 10
        result = shift_diagnostic_range(diag, shift)
        assert result["range"]["start"]["line"] == 10  # default 0+10

    def test_missing_end_defaults_to_start(self):
        diag = {"range": {"start": {"line": 7}}}
        shift = lambda x: x + 10
        result = shift_diagnostic_range(diag, shift)
        # end line defaults to start line (7), shifted → 17
        assert result["range"]["end"]["line"] == 17

    def test_preserves_other_fields(self):
        diag = self._diag()
        shift = lambda x: x + 1
        result = shift_diagnostic_range(diag, shift)
        assert result["severity"] == 2
        assert result["message"] == "test diagnostic"

    def test_original_not_mutated(self):
        diag = self._diag(start_line=0)
        original = dict(diag)
        shift = lambda x: x + 100
        shift_diagnostic_range(diag, shift)
        assert diag == original

    def test_preserves_characters(self):
        diag = self._diag(start_line=1, end_line=3, start_char=4, end_char=20)
        shift = lambda x: x + 10
        result = shift_diagnostic_range(diag, shift)
        assert result["range"]["start"]["character"] == 4
        assert result["range"]["end"]["character"] == 20


# ============================================================================
# shift_baseline
# ============================================================================
class TestShiftBaseline:
    def test_all_shifted(self):
        diags = [
            {"range": {"start": {"line": 1}, "end": {"line": 1}}},
            {"range": {"start": {"line": 3}, "end": {"line": 5}}},
        ]
        shift = lambda x: x + 10
        result = shift_baseline(diags, shift)
        assert len(result) == 2
        assert result[0]["range"]["start"]["line"] == 11
        assert result[1]["range"]["start"]["line"] == 13

    def test_some_dropped(self):
        diags = [
            {"range": {"start": {"line": 1}, "end": {"line": 1}}},  # stays
            {"range": {"start": {"line": 2}, "end": {"line": 2}}},  # dropped
            {"range": {"start": {"line": 3}, "end": {"line": 3}}},  # stays
        ]
        shift = lambda x: None if x == 2 else x + 10
        result = shift_baseline(diags, shift)
        assert len(result) == 2
        assert result[0]["range"]["start"]["line"] == 11
        assert result[1]["range"]["start"]["line"] == 13

    def test_empty_list(self):
        assert shift_baseline([], lambda x: x) == []

    def test_non_dict_entries_skipped(self):
        diags = [
            {"range": {"start": {"line": 0}, "end": {"line": 0}}},
            "not a dict",
            42,
            {"range": {"start": {"line": 1}, "end": {"line": 1}}},
        ]
        shift = lambda x: x + 1
        result = shift_baseline(diags, shift)
        assert len(result) == 2

    def test_original_not_mutated(self):
        diags = [
            {"range": {"start": {"line": 1}, "end": {"line": 1}}, "msg": "a"},
        ]
        original = [dict(d) for d in diags]
        shift_baseline(diags, lambda x: x + 10)
        assert diags == original
