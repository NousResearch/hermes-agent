"""A planned stop must let the gateway finish its configured drain before SIGKILL.

The gateway answers SIGTERM by draining in-flight turns for ``agent.restart_drain_timeout``,
giving interrupted agents 5 s to unwind, and only then tearing adapters / stores / browsers down.
The CLI waited a fixed 5 s and killed the process, so every planned stop and restart cut a running
turn and the lifecycle ledger recorded an unclean exit — the drain setting had no effect on stops.
"""

from __future__ import annotations

import math

from gateway.restart import planned_stop_wait


def test_planned_stop_covers_the_drain_and_the_teardown_after_it():
    budget = planned_stop_wait(30.0)

    assert budget["force_after"] == 50.0  # 30 s drain + the 20 s unwind/teardown reserve
    assert budget["timeout"] > budget["force_after"]


def test_planned_stop_keeps_a_floor_without_a_drain():
    assert planned_stop_wait(0.0)["force_after"] == 20.0


def test_planned_stop_tolerates_a_non_numeric_drain():
    assert planned_stop_wait("not-a-number")["force_after"] == 20.0  # type: ignore[arg-type]


def test_planned_stop_never_forces_an_indefinite_drain():
    assert math.isinf(planned_stop_wait(math.inf)["force_after"])
