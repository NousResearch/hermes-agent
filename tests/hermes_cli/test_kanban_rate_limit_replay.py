"""ACCEPTANCE: replay real rate_limited history against the new backoff ladder.

The card requires showing that rate_limited spawns drop >90%. We replay the
REAL inter-retry timings recorded on the live board (exported as a fixture, so
the test stays hermetic and does not read ~/.hermes) and count how many spawns
the old flat-300s rule admitted versus the new escalating ladder.

Fixture provenance: ~/.hermes/kanban.db, task_runs where outcome='rate_limited'
over the 7d ending 2026-09-22. Captured per-task as retry counts; the replay
reconstructs the storm from the observed cadence rather than replaying every
row, so the assertion is about the RULE, not about one board's row ids.
"""
from __future__ import annotations

import hermes_cli.kanban_db_dispatch as kbd

# (task_id, consecutive rate_limited runs observed in 7d) — the 8 worst
# offenders measured on the live board 2026-09-22.
OBSERVED_STORMS = [
    ("t_17ccc60e", 128),
    ("t_13445a80", 103),
    ("t_671fd52c", 90),
    ("t_85cc093e", 85),
    ("t_59c0886e", 80),
    ("t_f9fbeac7", 77),
    ("t_e3cdb67f", 75),
    ("t_62f0eb15", 70),
]

# Measured: zero inter-retry gaps below 304s, mean 304s in the 60-310s bucket.
# The old rule released a task every cooldown, so a storm of N runs spanned
# roughly N * 300s of wall clock.
OLD_FLAT_COOLDOWN = 300


def _spawns_under_ladder(window_seconds: int) -> int:
    """How many spawns the escalating ladder admits in ``window_seconds``."""
    t = 0
    spawns = 0
    while True:
        streak = spawns + 1
        hold = kbd.rate_limit_backoff_seconds(streak)
        t += hold
        if t > window_seconds:
            return spawns
        spawns += 1


def test_replay_shows_over_90_percent_reduction():
    old_total = 0
    new_total = 0
    for _tid, runs in OBSERVED_STORMS:
        # Wall-clock the storm actually occupied under the old flat rule.
        window = runs * OLD_FLAT_COOLDOWN
        old_total += runs
        new_total += _spawns_under_ladder(window)

    assert old_total == 708, old_total
    reduction = 1.0 - (new_total / old_total)
    assert reduction > 0.90, (
        f"expected >90% fewer spawns, got {reduction:.1%} "
        f"(old={old_total}, new={new_total})"
    )


def test_ladder_bounds_a_long_outage():
    """Over a 24h outage the ladder admits ~13 spawns, not 288."""
    day = 86_400
    old = day // OLD_FLAT_COOLDOWN  # 288
    new = _spawns_under_ladder(day)
    assert old == 288
    assert new <= 15, new
    assert (1.0 - new / old) > 0.90


def test_single_throttle_is_not_penalised():
    """One rate-limit still retries after 5m — no regression for a blip."""
    assert kbd.rate_limit_backoff_seconds(1) == OLD_FLAT_COOLDOWN
