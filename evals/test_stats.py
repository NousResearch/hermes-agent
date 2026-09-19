"""Checks for evals/_stats.py.

The tripwire that matters is the last one: on a synthetic no-effect arm the harness
must decline to call a win. A statistics helper that always finds a difference is
worse than no helper, because it launders noise into a scorecard.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stats import (  # noqa: E402
    bootstrap_ci,
    fmt_ci,
    min_detectable_difference,
    paired_delta_ci,
    separable,
    stdev,
    wilson,
)

REPS = 2000  # enough for these assertions, fast enough for the per-file runner


def test_wilson_matches_the_longhand_form():
    """Recomputed from the closed form, so a bug in _stats cannot pass its own test."""
    for k, n in ((14, 15), (0, 15), (15, 15), (30, 60)):
        z = 1.959963984540054
        p = k / n
        a = 2 * n * p + z * z
        b = z * math.sqrt(z * z + 4 * n * p * (1 - p))
        d = 2 * (n + z * z)
        lo, hi = wilson(k, n)
        assert abs(lo - (a - b) / d) < 1e-12
        assert abs(hi - (a + b) / d) < 1e-12


def test_wilson_stays_inside_the_unit_interval():
    assert wilson(0, 15)[0] == 0.0
    assert wilson(15, 15)[1] == 1.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_paired_delta_is_exact_when_every_item_agrees():
    """No spread between items means no width: the interval collapses onto the point."""
    a = [2.0] * 15
    b = [1.0] * 15
    point, lo, hi = paired_delta_ci(a, b, reps=REPS)
    assert point == 1.0
    assert lo == hi == 1.0


def test_paired_delta_refuses_mismatched_arms():
    """Two arms scored on different items are not a paired comparison."""
    try:
        paired_delta_ci([1.0, 2.0], [1.0])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("mismatched arms must raise, not truncate")


def test_paired_delta_is_deterministic():
    """Same input, same interval — a scorecard has to be reproducible."""
    a = [2, 0, 2, 2, 0, 0, 0, 2, 0, 1, 0, 2, 0, 1, 0]
    b = [0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    first = paired_delta_ci(a, b, reps=REPS)
    second = paired_delta_ci(a, b, reps=REPS)
    assert first == second


def test_the_no_effect_arm_is_not_called_a_win():
    """THE TRIPWIRE. Two arms drawn from the same distribution must not separate.

    Built from a real shape: 15 questions scored 0/1/2, the arms differing only by
    which questions happened to land. A harness that reports a winner here is the
    failure this module exists to prevent.
    """
    a = [2, 0, 2, 0, 1, 0, 2, 0, 0, 1, 2, 0, 0, 1, 0]
    b = [0, 2, 0, 2, 0, 1, 0, 2, 1, 0, 0, 2, 0, 0, 1]
    assert sum(a) == sum(b), "fixture should be a true tie in total score"
    assert not separable(a, b, reps=REPS)
    point, lo, hi = paired_delta_ci(a, b, reps=REPS)
    assert lo < 0 < hi, f"a tied pair must straddle zero, got [{lo}, {hi}]"


def test_a_real_effect_is_still_detected():
    """The counterpart: the tripwire must not be a test that never fires."""
    a = [2] * 14 + [0]
    b = [0] * 14 + [2]
    assert separable(a, b, reps=REPS)


def test_recall_scale_lands_in_percentage_points():
    """compaction reports 100*sum/(2n), so the scale for a 0/1/2 item is 50."""
    a = [2] * 15
    b = [0] * 15
    point, _, _ = paired_delta_ci(a, b, scale=50.0, reps=REPS)
    assert abs(point - 100.0) < 1e-9


def test_bootstrap_brackets_the_sample_mean():
    vals = [2, 0, 2, 2, 0, 0, 0, 2, 0, 1, 0, 2, 0, 1, 0]
    lo, hi = bootstrap_ci(vals, reps=REPS)
    assert lo <= sum(vals) / len(vals) <= hi


def test_mdd_shrinks_with_n_and_grows_with_spread():
    assert min_detectable_difference(60, 1.0) < min_detectable_difference(15, 1.0)
    assert min_detectable_difference(15, 2.0) > min_detectable_difference(15, 1.0)
    # doubling n divides the detectable difference by sqrt(2)
    assert abs(
        min_detectable_difference(30, 1.0) - min_detectable_difference(15, 1.0) / math.sqrt(2)
    ) < 1e-12
    assert min_detectable_difference(0, 1.0) == float("inf")


def test_stdev_and_formatting():
    assert stdev([1.0]) == 0.0
    assert abs(stdev([2.0, 4.0]) - math.sqrt(2)) < 1e-12
    assert fmt_ci(1.0, 2.0) == "[1.0, 2.0]"
    assert fmt_ci(-1.0, 2.0) == "[-1.0, +2.0]"
    assert fmt_ci(float("nan"), 1.0) == "[n/a]"
