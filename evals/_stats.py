"""Uncertainty for the eval harnesses in this tree. Standard library only.

Three of the harnesses here already say in prose what this module computes.
``evals/readtool/README.md`` states the rule outright — *"3 reps minimum; single-run
deltas within ±3% are noise, not wins"* — and nothing in the tree decides it. The
compaction runner already writes the per-question ``scores`` array
(``evals/compaction/runner.py``) and the reporter reads only the summary percentage,
so the paired data an interval needs is on disk and thrown away.

Everything here is paired, because that is what these harnesses produce: the same
question bank across every policy, the same task list across every arm. A paired
interval is strictly tighter than comparing two independent ones, and it is the
only form that answers "did this change help", as opposed to "are these two numbers
different".

No numpy, no scipy — the tree takes no new dependencies for `evals/`, and an
interval nobody can re-run is worth less than one anybody can.
"""

from __future__ import annotations

import math
import random

__all__ = [
    "paired_delta_ci",
    "bootstrap_ci",
    "wilson",
    "min_detectable_difference",
    "separable",
    "fmt_ci",
]

# Resampling is seeded so a scorecard is reproducible: the same results directory
# must print the same interval on two machines and in CI. 10k resamples puts the
# Monte-Carlo error on a 95% bound well under 0.1 pp, which is finer than any
# number these harnesses report.
DEFAULT_REPS = 10_000
DEFAULT_SEED = 0


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 1]. Input must be sorted."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap_ci(
    values: list[float],
    *,
    scale: float = 1.0,
    reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``, times ``scale``.

    Resamples items, not runs. For the compaction harness an item is a question, so
    this answers "how much would this policy's recall move on a different question
    bank of the same size" — which is the question a 15-question exam raises and the
    scorecard currently answers by not asking.
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (values[0] * scale, values[0] * scale)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(reps):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n * scale)
    means.sort()
    return (_percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2))


def paired_delta_ci(
    a: list[float],
    b: list[float],
    *,
    scale: float = 1.0,
    reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Interval on ``mean(a) - mean(b)`` for two arms scored on the SAME items.

    Returns ``(point, lo, hi)``, all multiplied by ``scale``.

    Bootstraps the per-item DIFFERENCE, so any item that is simply hard for both
    arms contributes nothing to the width — that cancellation is the whole reason
    to keep the pairing rather than compare two separate intervals.

    Raises on a length mismatch: two arms that were not scored on the same items
    cannot be compared this way, and silently truncating would produce a confident
    number about the wrong thing.
    """
    if len(a) != len(b):
        raise ValueError(f"paired arms must be the same length: {len(a)} vs {len(b)}")
    deltas = [x - y for x, y in zip(a, b)]
    point = (sum(deltas) / len(deltas)) * scale if deltas else float("nan")
    lo, hi = bootstrap_ci(deltas, scale=scale, reps=reps, seed=seed, alpha=alpha)
    return (point, lo, hi)


def separable(a: list[float], b: list[float], **kw) -> bool:
    """True when the paired interval on ``a - b`` excludes zero.

    "Not separable" is not "the same" — it means this many items cannot tell them
    apart. Report it that way.
    """
    _, lo, hi = paired_delta_ci(a, b, **kw)
    return lo > 0 or hi < 0


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a rate, e.g. an ok-rate over tasks.

    Wilson rather than Wald because Wald misbehaves near 0 and 1, and a pass-rate
    table's interesting rows are exactly the ones near the ends.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def min_detectable_difference(n: int, sd_of_differences: float) -> float:
    """Smallest paired difference this many items could detect, 80% power, α=0.05.

    The prospective number. An interval says whether *these* two arms separated;
    this says how big a gap would have to be before the harness could ever see one,
    which is what tells you whether to add items or stop comparing.
    """
    if n <= 0 or sd_of_differences <= 0:
        return float("inf")
    z_a = 1.959963984540054  # two-sided 0.05
    z_b = 0.8416212335729143  # power 0.80
    return (z_a + z_b) * sd_of_differences / math.sqrt(n)


def stdev(values: list[float]) -> float:
    """Sample standard deviation; 0.0 for fewer than two values."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def fmt_ci(lo: float, hi: float, unit: str = "") -> str:
    """``[lo, hi]`` for a table cell, or ``[n/a]`` when the interval is undefined."""
    if any(math.isnan(v) for v in (lo, hi)):
        return "[n/a]"
    return f"[{lo:+.1f}, {hi:+.1f}]{unit}" if lo < 0 or hi < 0 else f"[{lo:.1f}, {hi:.1f}]{unit}"
