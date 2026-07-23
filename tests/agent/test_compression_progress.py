"""Regression: detect compression progress by tokens, not just rows.

Issue #39548: preflight compression in the turn prologue was checking
``len(messages) >= _orig_len`` to decide "Cannot compress further". This
false-positives when a pass summarises message contents — reducing the
estimated request token count without removing any rows — and surfaces a
spurious ``Context length exceeded`` failure followed by an auto-reset of
an otherwise healthy session.

These tests pin the contract of ``_compression_made_progress``: a
row-count reduction OR a *material* (>5%) token-count reduction counts as
progress.
"""

from __future__ import annotations

from agent.turn_context import (
    _compression_made_progress,
    _compression_warrants_another_preflight_pass,
)


class TestCompressionMadeProgress:
    def test_rows_reduced_counts_as_progress(self):
        """Removing message rows is the obvious progress signal."""
        assert _compression_made_progress(
            orig_len=10, new_len=5, orig_tokens=1000, new_tokens=1000
        ) is True

    def test_tokens_reduced_without_row_change_counts_as_progress(self):
        """Issue #39548: 220 → 220 rows, 288k → 183k tokens IS progress."""
        assert _compression_made_progress(
            orig_len=220, new_len=220, orig_tokens=288_028, new_tokens=183_180
        ) is True

    def test_both_reduced_counts_as_progress(self):
        """Common case: summarising drops some rows and shrinks the rest."""
        assert _compression_made_progress(
            orig_len=220, new_len=180, orig_tokens=288_028, new_tokens=150_000
        ) is True

    def test_neither_moved_means_no_progress(self):
        """The genuine "stuck" case — same rows, same tokens, give up."""
        assert _compression_made_progress(
            orig_len=10, new_len=10, orig_tokens=1000, new_tokens=1000
        ) is False

    def test_rows_grew_and_tokens_grew_means_no_progress(self):
        """Pathological: the pass made the request larger — definitely stuck."""
        assert _compression_made_progress(
            orig_len=10, new_len=12, orig_tokens=1000, new_tokens=1200
        ) is False

    def test_rows_grew_but_tokens_dropped_is_progress(self):
        """Edge: summary rows may expand the row count while shrinking tokens.

        Token reduction alone is sufficient to keep the loop going.
        """
        assert _compression_made_progress(
            orig_len=10, new_len=11, orig_tokens=1000, new_tokens=600
        ) is True

    def test_tokens_grew_but_rows_dropped_is_progress(self):
        """Edge: row reduction alone is sufficient even if tokens nominally
        creep up (e.g. summary verbosity).  Row-count reduction is a hard
        signal that the transcript actually shrank.
        """
        assert _compression_made_progress(
            orig_len=10, new_len=5, orig_tokens=1000, new_tokens=1100
        ) is True

    def test_sub_5pct_token_drop_is_not_progress(self):
        """A token reduction below the 5% material floor does NOT count as
        progress — matching the overflow-handler retry path (#39550) so a
        marginal wobble can't keep the multi-pass loop spinning."""
        # 1000 -> 970 is a 3% drop, below the 5% floor.
        assert _compression_made_progress(
            orig_len=10, new_len=10, orig_tokens=1000, new_tokens=970
        ) is False
        # 1000 -> 940 is a 6% drop, above the floor.
        assert _compression_made_progress(
            orig_len=10, new_len=10, orig_tokens=1000, new_tokens=940
        ) is True

    def test_zero_orig_tokens_is_not_progress(self):
        """Degenerate estimate (0 tokens) must not be read as a token win."""
        assert _compression_made_progress(
            orig_len=10, new_len=10, orig_tokens=0, new_tokens=0
        ) is False


class TestCompressionWarrantsAnotherPreflightPass:
    def test_material_reduction_above_threshold_allows_another_pass(self):
        assert _compression_warrants_another_preflight_pass(
            orig_tokens=400_000,
            new_tokens=350_000,
            threshold_tokens=272_000,
        ) is True

    def test_marginal_reduction_above_threshold_stops(self):
        assert _compression_warrants_another_preflight_pass(
            orig_tokens=350_000,
            new_tokens=345_000,
            threshold_tokens=272_000,
        ) is False

    def test_clearing_threshold_needs_no_additional_pass(self):
        assert _compression_warrants_another_preflight_pass(
            orig_tokens=280_000,
            new_tokens=250_000,
            threshold_tokens=272_000,
        ) is False
