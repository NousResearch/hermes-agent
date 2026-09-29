"""An unusable CPU count is absence of evidence for starvation, so it must EXIT, not HOLD.

Caught in review on #117431 by @ashishsinghbora. ``evaluate_liveness_miss`` gated the starvation
predicate on the load average being available, but fell back to ``cores = 1`` whenever
``os.cpu_count()`` returned ``None``. That fallback is the most PERMISSIVE default, not a
conservative one: on a 64-core host with ``load1=20`` it evaluated ``20 > max(2 * 1, 8)`` and
entered the HOLD path, where the real threshold is ``max(2 * 64, 8) = 128`` and the correct
classification is WEDGED.

The PR's stated invariant is "absence of evidence for starvation must not create a hold", which is
why an unavailable ``os.getloadavg`` classifies as wedged. ``ncpu`` is the other term of the same
predicate and gets the same treatment.

Companion to ``test_liveness_starvation_hold.py``; load is always MOCKED (passed as arguments to
the pure classifier) because generating real load is the incident under test.
"""

from __future__ import annotations

import pytest

from gateway.shutdown_watchdog import evaluate_liveness_miss

# max(2 * 1, 8) = 8 under the old cores=1 fallback, so this load HELD.
# max(2 * 64, 8) = 128 with the count known, so it EXITS. One load, two verdicts.
_LOAD_THAT_EXPOSED_THE_FALLBACK = 20.0


@pytest.mark.parametrize(
    "ncpu",
    [None, 0, -1, "8", 2.5, True],
    ids=["none", "zero", "negative", "string", "float", "bool"],
)
def test_unusable_cpu_count_is_wedged_even_with_a_live_load_average(ncpu):
    decision = evaluate_liveness_miss(
        strikes=3,
        strikes_limit=3,
        load1=_LOAD_THAT_EXPOSED_THE_FALLBACK,
        ncpu=ncpu,
        load_factor=2.0,
        starved_since=None,
        now=1000.0,
        max_hold_s=900.0,
    )
    assert decision.action == "exit", f"ncpu={ncpu!r} is not evidence of starvation"
    assert decision.phase is None
    assert decision.starved_since is None
    assert decision.strikes == 3, "the wedge path must leave the strike counter untouched"


def test_the_reviewers_64_core_example_exits():
    """@ashishsinghbora's exact counterexample: 64 cores, load1=20, cpu_count() -> None."""
    unknown = evaluate_liveness_miss(
        strikes=3, strikes_limit=3, load1=20.0, ncpu=None, load_factor=2.0,
        starved_since=None, now=1000.0, max_hold_s=900.0,
    )
    known = evaluate_liveness_miss(
        strikes=3, strikes_limit=3, load1=20.0, ncpu=64, load_factor=2.0,
        starved_since=None, now=1000.0, max_hold_s=900.0,
    )
    assert unknown.action == "exit"
    # The host really is not starved at 20/64, so the KNOWN-count verdict is the same. The bug was
    # that the unknown-count verdict disagreed with it.
    assert known.action == "exit"
    assert unknown == known


def test_a_known_cpu_count_still_holds_at_the_same_load():
    """Control: the only thing that changes is ncpu becoming KNOWN and genuinely small.

    Pins that the fix narrowed the predicate's INPUTS, not the starvation behaviour itself — a
    2-core host at load 20 is still starved and still holds.
    """
    decision = evaluate_liveness_miss(
        strikes=3,
        strikes_limit=3,
        load1=_LOAD_THAT_EXPOSED_THE_FALLBACK,
        ncpu=2,  # max(2 * 2, 8) = 8, and 20 > 8
        load_factor=2.0,
        starved_since=None,
        now=1000.0,
        max_hold_s=900.0,
    )
    assert decision.action == "hold"
    assert decision.phase == "liveness_starved"
    assert decision.starved_since == 1000.0
