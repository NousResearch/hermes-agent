"""Tests for the reuse tracker."""

from __future__ import annotations

from agent.signal_sources.reuse_tracker import (
    ReuseEntry,
    lookup_reuse_outcome,
    mark_invocation,
    merge_history,
)


T0 = 1_700_000_000.0


class TestMarkInvocation:
    def test_appends_entry(self):
        h: list = []
        h2 = mark_invocation(h, timestamp=T0, success=True)
        assert len(h2) == 1
        assert h2[0].timestamp == T0
        assert h2[0].success is True

    def test_does_not_mutate_input(self):
        h: list = []
        mark_invocation(h, timestamp=T0)
        assert h == []


class TestLookupReuseOutcome:
    def test_no_entries_returns_none(self):
        assert lookup_reuse_outcome([], after_timestamp=T0) is None

    def test_no_entries_after_returns_none(self):
        h = [ReuseEntry(T0 - 100, True)]
        assert lookup_reuse_outcome(h, after_timestamp=T0) is None

    def test_immediate_only_returns_next(self):
        h = [
            ReuseEntry(T0 - 50, False),
            ReuseEntry(T0 - 25, True),
            ReuseEntry(T0 - 10, False),
        ]
        assert lookup_reuse_outcome(h, after_timestamp=T0 - 100) is False

    def test_majority_outcome(self):
        h = [
            ReuseEntry(T0 - 50, True),
            ReuseEntry(T0 - 30, True),
            ReuseEntry(T0 - 20, False),
        ]
        assert (
            lookup_reuse_outcome(h, after_timestamp=T0 - 100, immediate_only=False)
            is True
        )

    def test_majority_tie_returns_none(self):
        h = [
            ReuseEntry(T0 - 50, True),
            ReuseEntry(T0 - 30, False),
        ]
        assert (
            lookup_reuse_outcome(h, after_timestamp=T0 - 100, immediate_only=False)
            is None
        )


class TestMergeHistory:
    def test_sorts_by_timestamp(self):
        a = [ReuseEntry(50.0, True), ReuseEntry(10.0, False)]
        b = [ReuseEntry(30.0, True)]
        merged = merge_history(a, b)
        assert [e.timestamp for e in merged] == [10.0, 30.0, 50.0]

    def test_empty(self):
        assert merge_history() == []
        assert merge_history([]) == []
