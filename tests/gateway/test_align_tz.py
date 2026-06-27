"""Tests for ``gateway.run._align_tz`` and the resume-pending tz-mismatch
defensive path.

Background: ``_schedule_resume_pending_sessions`` compares ``datetime.now()``
against ``entry.last_resume_marked_at`` (or ``entry.updated_at``).  If a
deployment has a half-applied naive→aware migration — or a ``sessions.json``
written by an older naive build is loaded by a newer aware build — the
subtraction raises ``TypeError: can't subtract offset-naive and offset-aware
datetimes`` and the gateway boot loop wedges on every restart.

A single legacy entry crashing the entire pass is the wrong failure mode.
``_align_tz`` makes the comparison resilient and the loop logs+skips the bad
entry instead.
"""

from datetime import datetime, timezone

import pytest

from gateway.run import _align_tz


class TestAlignTz:
    def test_both_naive_returns_reference_unchanged(self):
        ref = datetime(2026, 6, 15, 12, 0, 0)
        other = datetime(2026, 6, 15, 11, 0, 0)
        out = _align_tz(ref, other)
        assert out is ref
        assert out.tzinfo is None

    def test_both_aware_returns_reference_unchanged(self):
        ref = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        other = datetime(2026, 6, 15, 11, 0, 0, tzinfo=timezone.utc)
        out = _align_tz(ref, other)
        assert out is ref
        assert out.tzinfo is timezone.utc

    def test_aware_reference_naive_other_strips_tz(self):
        # Reference is aware (e.g. datetime.now(tz=timezone.utc)), other is
        # naive (e.g. legacy persisted marker).  Result must be naive so the
        # subsequent subtraction does not raise.
        ref = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        other = datetime(2026, 6, 15, 11, 0, 0)
        out = _align_tz(ref, other)
        assert out.tzinfo is None
        # Wall-clock preserved — UTC reference, naive other interpreted as UTC.
        delta = (out - other).total_seconds()
        assert delta == 3600.0

    def test_naive_reference_aware_other_assumes_utc(self):
        # Reference is naive, other is aware.  Promote naive to UTC so the
        # subtraction does not raise and wall-clock semantics hold.
        ref = datetime(2026, 6, 15, 12, 0, 0)
        other = datetime(2026, 6, 15, 11, 0, 0, tzinfo=timezone.utc)
        out = _align_tz(ref, other)
        assert out.tzinfo is timezone.utc
        delta = (out - other).total_seconds()
        assert delta == 3600.0

    def test_aware_to_naive_round_trip_preserves_instant(self):
        # An aware UTC reference and a naive marker representing the same
        # instant should compare as zero delta after alignment.
        instant = datetime(2026, 6, 15, 12, 0, 0)
        ref = instant.replace(tzinfo=timezone.utc)
        out = _align_tz(ref, instant)
        assert (out - instant).total_seconds() == 0.0

    def test_non_utc_aware_other_normalizes_to_utc(self):
        # Reference aware-UTC, other aware in a non-UTC zone.  Both branches
        # of the helper hit only on tz-awareness mismatch; same-awareness
        # values pass through unchanged.  This documents the helper does
        # NOT cross-zone-convert when both are aware.
        from datetime import timedelta
        non_utc = timezone(timedelta(hours=7))
        ref = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        other = datetime(2026, 6, 15, 19, 0, 0, tzinfo=non_utc)
        out = _align_tz(ref, other)
        assert out is ref  # both aware → no transform
        # The two values represent the same instant; subtraction is safe.
        assert (out - other).total_seconds() == 0.0
