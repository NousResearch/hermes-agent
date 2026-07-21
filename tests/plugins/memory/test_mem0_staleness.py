"""Unit tests for _apply_staleness_weight in plugins.memory.mem0.

The function applies a time-decay blend to mem0 search results so that
older memories score lower than recent ones. Vacation periods freeze or
reset decay while the user is away.

Patch target: ``plugins.memory.mem0.datetime``
(the module does ``from datetime import datetime, timedelta, timezone``,
so we replace the *name* ``datetime`` inside the module namespace).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

from plugins.memory.mem0 import _apply_staleness_weight
from plugins.memory.mem0 import _DECAY_ALPHA, _DECAY_HALF_LIFE_DAYS, _DECAY_GRACE_DAYS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc

# "now" used in every test unless overridden.
NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=_UTC)


def _patch_now(fake_now: datetime):
    """Return a context manager that freezes datetime.now() inside the module."""
    mock_dt = MagicMock(wraps=datetime)
    mock_dt.now.return_value = fake_now
    # Keep class methods that the function calls directly
    mock_dt.fromisoformat = datetime.fromisoformat
    mock_dt.fromtimestamp = datetime.fromtimestamp
    return patch("plugins.memory.mem0.datetime", mock_dt)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _result(score: float, ts: str | None = None, *, use_updated_at=True) -> dict:
    """Construct a minimal result dict."""
    r: dict = {"score": score}
    if ts is not None:
        if use_updated_at:
            r["updated_at"] = ts
        else:
            r["created_at"] = ts
    return r


# ---------------------------------------------------------------------------
# 1. No vacations — sorted by decay score descending
# ---------------------------------------------------------------------------

class TestNoVacations:

    def test_sorted_descending(self):
        """Results are re-sorted highest adjusted score first."""
        recent = _iso(NOW - timedelta(days=1))
        old = _iso(NOW - timedelta(days=120))
        results = [
            _result(0.8, old),    # will get heavy penalty
            _result(0.9, recent), # nearly no penalty
        ]
        with _patch_now(NOW):
            out = _apply_staleness_weight(results)
        assert out[0]["score"] >= out[1]["score"], "should be sorted descending"
        # The recent result should rank first because it keeps most of its score.
        # Verify by identity of the memory timestamp.
        assert out[0].get("updated_at") == recent

    # -----------------------------------------------------------------------
    # 2. Missing timestamp passes through unchanged
    # -----------------------------------------------------------------------

    def test_missing_timestamp_passes_through(self):
        """A result without updated_at/created_at gets adjusted = base_score."""
        r = {"score": 0.75, "memory": "no timestamp here"}
        with _patch_now(NOW):
            out = _apply_staleness_weight([r])
        assert out[0]["score"] == pytest.approx(0.75)

    # -----------------------------------------------------------------------
    # 3. Unix float timestamp (int/float isinstance branch)
    # -----------------------------------------------------------------------

    def test_unix_float_timestamp(self):
        """A numeric (Unix epoch float) timestamp is handled via fromtimestamp."""
        ts_unix = NOW.timestamp() - 1.0  # 1 second ago → within grace
        r = {"score": 0.9, "updated_at": ts_unix}
        with _patch_now(NOW):
            out = _apply_staleness_weight([r])
        # 1 second old → well within grace → decay_factor=1.0 → no penalty
        expected = 0.9  # (1-0.3)*0.9 + 0.3*1.0*0.9 = 0.9
        assert out[0]["score"] == pytest.approx(expected, rel=1e-6)

    def test_unix_int_timestamp(self):
        """An integer Unix timestamp is also handled correctly."""
        ts_unix = int((NOW - timedelta(days=1)).timestamp())
        r = {"score": 0.8, "updated_at": ts_unix}
        with _patch_now(NOW):
            out = _apply_staleness_weight([r])
        # 1 day old → within default grace of 14 days → decay_factor=1.0
        expected = 0.8
        assert out[0]["score"] == pytest.approx(expected, rel=1e-6)

    # -----------------------------------------------------------------------
    # 4. Within grace_days → decay_factor = 1.0 (no penalty)
    # -----------------------------------------------------------------------

    def test_within_grace_no_penalty(self):
        """Memory created exactly at the grace boundary incurs no penalty."""
        grace = _DECAY_GRACE_DAYS
        ts = _iso(NOW - timedelta(days=grace - 0.001))  # just inside grace
        r = _result(0.7, ts)
        with _patch_now(NOW):
            out = _apply_staleness_weight([r], grace_days=grace)
        # adjusted = (1-alpha)*base + alpha*1.0*base = base
        assert out[0]["score"] == pytest.approx(0.7, rel=1e-6)

    # -----------------------------------------------------------------------
    # 5. Beyond grace_days → decay < 1.0, score reduced
    # -----------------------------------------------------------------------

    def test_beyond_grace_score_reduced(self):
        """Memory older than grace_days gets a fractional decay_factor."""
        grace = _DECAY_GRACE_DAYS
        ts = _iso(NOW - timedelta(days=grace + 30))  # 30 days past grace
        r = _result(1.0, ts)
        with _patch_now(NOW):
            out = _apply_staleness_weight([r], grace_days=grace)
        assert out[0]["score"] < 1.0

    # -----------------------------------------------------------------------
    # 6. Exact decay formula verification
    # -----------------------------------------------------------------------

    def test_decay_formula_exact(self):
        """adjusted = (1 - alpha)*base + alpha * decay_factor * base."""
        alpha = _DECAY_ALPHA
        half_life = _DECAY_HALF_LIFE_DAYS
        grace = _DECAY_GRACE_DAYS
        base = 0.8
        days_past_grace = 15.0
        days_old = grace + days_past_grace
        ts = _iso(NOW - timedelta(days=days_old))
        r = _result(base, ts)
        decay_factor = 0.5 ** (days_past_grace / half_life)
        expected = (1 - alpha) * base + alpha * decay_factor * base
        with _patch_now(NOW):
            out = _apply_staleness_weight([r], grace_days=grace)
        assert out[0]["score"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. Active vacation — no decay, sorted by base score
# ---------------------------------------------------------------------------

class TestActiveVacation:

    def test_active_vacation_no_decay(self):
        """While NOW is inside a vacation period, results are returned sorted
        by base score without any time-decay adjustment."""
        vstart = _iso(NOW - timedelta(days=5))
        vend = _iso(NOW + timedelta(days=10))
        vacations = [{"start": vstart, "end": vend}]

        # Old timestamps that would normally get heavy penalties
        old_ts = _iso(NOW - timedelta(days=200))
        results = [
            _result(0.5, old_ts),
            _result(0.9, old_ts),
            _result(0.7, old_ts),
        ]
        with _patch_now(NOW):
            out = _apply_staleness_weight(results, vacations=vacations)

        scores = [r["score"] for r in out]
        # Must be sorted descending by BASE score (no decay applied)
        assert scores == sorted(scores, reverse=True)
        # The scores must equal the originals (no blending)
        assert scores == pytest.approx([0.9, 0.7, 0.5])

    # -----------------------------------------------------------------------
    # 8. Active vacation, no start date — start defaults to epoch
    # -----------------------------------------------------------------------

    def test_active_vacation_no_start_defaults_to_epoch(self):
        """A period without 'start' uses epoch as start; any NOW during the
        period (including NOW far from epoch) still triggers early return."""
        vend = _iso(NOW + timedelta(days=10))
        vacations = [{"end": vend}]  # no "start" key

        old_ts = _iso(NOW - timedelta(days=300))
        r = _result(0.6, old_ts)
        with _patch_now(NOW):
            out = _apply_staleness_weight([r], vacations=vacations)
        # Early return keeps original score
        assert out[0]["score"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 9. Post-vacation, within grace — effective_now set to vacation_end
# ---------------------------------------------------------------------------

class TestPostVacationWithinGrace:

    def test_within_grace_after_vacation_full_score(self):
        """Days since return <= grace_days → effective_now = vacation_end.

        Memory created just before vacation_end has a very small days_old
        (measured from effective_now=vacation_end), so it gets decay_factor=1.0.
        """
        vacation_end = NOW - timedelta(days=5)   # returned 5 days ago
        vacation_start = vacation_end - timedelta(days=14)
        vacations = [{"start": _iso(vacation_start), "end": _iso(vacation_end)}]

        # Memory created 1 day before vacation ended
        mem_ts = _iso(vacation_end - timedelta(days=1))
        r = _result(0.8, mem_ts)

        with _patch_now(NOW):
            out = _apply_staleness_weight([r], vacations=vacations)

        # days_old measured from effective_now=vacation_end: 1 day → within grace
        expected = 0.8  # decay_factor=1.0, no penalty
        assert out[0]["score"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 10. Post-vacation, beyond grace — effective_now stays as real now
# ---------------------------------------------------------------------------

class TestPostVacationBeyondGrace:

    def test_beyond_grace_after_vacation_normal_decay(self):
        """Days since return > grace_days → effective_now stays = now.
        Normal decay applies as if no vacation had happened."""
        grace = _DECAY_GRACE_DAYS
        vacation_end = NOW - timedelta(days=grace + 5)  # returned well past grace
        vacation_start = vacation_end - timedelta(days=10)
        vacations = [{"start": _iso(vacation_start), "end": _iso(vacation_end)}]

        # Memory from a very long time ago
        old_ts = _iso(NOW - timedelta(days=200))
        r = _result(1.0, old_ts)

        with _patch_now(NOW):
            out = _apply_staleness_weight([r], vacations=vacations, grace_days=grace)

        # score should be < 1.0 since effective_now=now and memory is very old
        assert out[0]["score"] < 1.0


# ---------------------------------------------------------------------------
# 11. Multiple past vacations — most recent vacation_end wins
# ---------------------------------------------------------------------------

class TestMultiplePastVacations:

    def test_most_recent_past_end_wins(self):
        """When several vacations have ended, the largest vacation_end
        (most recent) is chosen as best_past_end, minimising days_old."""
        older_end = NOW - timedelta(days=10)
        newer_end = NOW - timedelta(days=3)
        vacations = [
            {"start": _iso(older_end - timedelta(days=7)), "end": _iso(older_end)},
            {"start": _iso(newer_end - timedelta(days=7)), "end": _iso(newer_end)},
        ]

        # Memory created 1 day before the NEWER vacation ended
        mem_ts = _iso(newer_end - timedelta(days=1))
        r = _result(0.9, mem_ts)

        with _patch_now(NOW):
            out = _apply_staleness_weight([r], vacations=vacations)

        # effective_now = newer_end → days_old ≈ 1 day → within grace → no decay
        assert out[0]["score"] == pytest.approx(0.9, rel=1e-6)

    def test_older_end_would_have_penalised(self):
        """Confirm that if we used the older vacation_end instead, the score
        would be lower — proving that the newest-end selection matters."""
        older_end = NOW - timedelta(days=20)
        # Only one past vacation whose end is 20 days ago
        vacations = [
            {"start": _iso(older_end - timedelta(days=5)), "end": _iso(older_end)},
        ]
        # Memory created right at vacation_end
        mem_ts = _iso(older_end)
        r = _result(1.0, mem_ts)

        # 20 days since return > default grace_days (14) → effective_now stays as now
        with _patch_now(NOW):
            out = _apply_staleness_weight([r], vacations=vacations)

        # days_old measured from NOW: 20 days → beyond grace → decay < 1.0
        assert out[0]["score"] < 1.0


# ---------------------------------------------------------------------------
# 12. Multiple vacations, one active — triggers early return
# ---------------------------------------------------------------------------

class TestMultipleVacationsOneActive:

    def test_active_period_triggers_early_return(self):
        """If ANY vacation period is currently active, return early with base
        scores regardless of past periods that have ended."""
        past_end = NOW - timedelta(days=30)
        active_start = NOW - timedelta(days=2)
        active_end = NOW + timedelta(days=5)
        vacations = [
            {"start": _iso(past_end - timedelta(days=7)), "end": _iso(past_end)},
            {"start": _iso(active_start), "end": _iso(active_end)},
        ]

        old_ts = _iso(NOW - timedelta(days=300))
        results = [_result(0.4, old_ts), _result(0.9, old_ts)]

        with _patch_now(NOW):
            out = _apply_staleness_weight(results, vacations=vacations)

        # Early return: sorted by base score, values unchanged
        assert out[0]["score"] == pytest.approx(0.9)
        assert out[1]["score"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 13. Malformed period — missing 'end' key is skipped silently
# ---------------------------------------------------------------------------

class TestMalformedPeriods:

    def test_missing_end_key_skipped(self):
        """A period dict without an 'end' key is silently skipped; the
        remaining logic (no-vacation path) proceeds normally."""
        vacations = [{"start": _iso(NOW - timedelta(days=5))}]  # no 'end'
        recent_ts = _iso(NOW - timedelta(days=1))
        r = _result(0.8, recent_ts)

        with _patch_now(NOW):
            out = _apply_staleness_weight([r], vacations=vacations)

        # No vacation effect; memory is recent → decay_factor=1.0 → score unchanged
        assert out[0]["score"] == pytest.approx(0.8, rel=1e-6)

    # -----------------------------------------------------------------------
    # 14. Malformed period — non-parseable end date string is skipped
    # -----------------------------------------------------------------------

    def test_non_parseable_end_date_skipped(self):
        """A period whose 'end' value cannot be parsed as ISO is skipped;
        rest of the logic (no-vacation path) still runs."""
        vacations = [{"start": _iso(NOW - timedelta(days=5)), "end": "NOT-A-DATE"}]
        recent_ts = _iso(NOW - timedelta(days=1))
        r = _result(0.8, recent_ts)

        with _patch_now(NOW):
            out = _apply_staleness_weight([r], vacations=vacations)

        # Malformed period skipped → no vacation → normal decay → within grace
        assert out[0]["score"] == pytest.approx(0.8, rel=1e-6)


# ---------------------------------------------------------------------------
# 15. Empty vacations list — identical to no vacations
# ---------------------------------------------------------------------------

class TestEmptyVacations:

    def test_empty_list_same_as_none(self):
        """vacations=[] and vacations=None produce identical results."""
        recent_ts = _iso(NOW - timedelta(days=1))
        results = [_result(0.7, recent_ts)]

        with _patch_now(NOW):
            out_none = _apply_staleness_weight(results, vacations=None)
        with _patch_now(NOW):
            out_empty = _apply_staleness_weight(results, vacations=[])

        assert out_none[0]["score"] == pytest.approx(out_empty[0]["score"])


# ---------------------------------------------------------------------------
# 16. Sort order — output sorted highest-to-lowest adjusted score
# ---------------------------------------------------------------------------

class TestSortOrder:

    def test_sort_order_across_multiple_results(self):
        """With several results having different ages and base scores, the
        output list is always in descending order of adjusted score."""
        results = [
            _result(0.5, _iso(NOW - timedelta(days=200))),  # old, low base
            _result(0.95, _iso(NOW - timedelta(days=180))), # old, high base
            _result(0.6, _iso(NOW - timedelta(days=2))),    # new, medium base
            _result(0.85, _iso(NOW - timedelta(days=1))),   # new, high base
            _result(0.4, _iso(NOW - timedelta(days=50))),   # mid-age, low base
        ]

        with _patch_now(NOW):
            out = _apply_staleness_weight(results)

        scores = [r["score"] for r in out]
        assert scores == sorted(scores, reverse=True), (
            f"Output not sorted descending: {scores}"
        )

    def test_sort_handles_mixed_timestamp_presence(self):
        """Results with and without timestamps are all included and sorted."""
        results = [
            {"score": 0.6},                                      # no ts
            _result(0.9, _iso(NOW - timedelta(days=1))),         # new
            _result(0.3, _iso(NOW - timedelta(days=300))),       # very old
        ]
        with _patch_now(NOW):
            out = _apply_staleness_weight(results)

        scores = [r["score"] for r in out]
        assert scores == sorted(scores, reverse=True)
