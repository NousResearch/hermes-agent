"""Regression: detect compression progress by tokens, not just rows.

Issue #39548: preflight compression in the turn prologue was checking
``len(messages) >= _orig_len`` to decide "Cannot compress further". This
false-positives when a pass summarises message contents — reducing the
estimated request token count without removing any rows — and surfaces a
spurious ``Context length exceeded`` failure followed by an auto-reset of
an otherwise healthy session.

These tests pin the contract of ``compression_made_progress``: a
row-count reduction OR a *material* (>5%) token-count reduction counts as
progress.
"""

from __future__ import annotations

from agent.turn_context import (
    compression_made_progress,
    _compression_warrants_another_preflight_pass,
)


class TestCompressionMadeProgress:
    def test_rows_reduced_counts_as_progress(self):
        """Removing message rows is the obvious progress signal."""
        assert compression_made_progress(
            orig_len=10, new_len=5, orig_tokens=1000, new_tokens=1000
        ) is True



    def test_neither_moved_means_no_progress(self):
        """The genuine "stuck" case — same rows, same tokens, give up."""
        assert compression_made_progress(
            orig_len=10, new_len=10, orig_tokens=1000, new_tokens=1000
        ) is False




    def test_sub_5pct_token_drop_is_not_progress(self):
        """A token reduction below the 5% material floor does NOT count as
        progress — matching the overflow-handler retry path (#39550) so a
        marginal wobble can't keep the multi-pass loop spinning."""
        # 1000 -> 970 is a 3% drop, below the 5% floor.
        assert compression_made_progress(
            orig_len=10, new_len=10, orig_tokens=1000, new_tokens=970
        ) is False
        # 1000 -> 940 is a 6% drop, above the floor.
        assert compression_made_progress(
            orig_len=10, new_len=10, orig_tokens=1000, new_tokens=940
        ) is True



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

