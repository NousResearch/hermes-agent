"""The relaunch liveness budget has to start when the replacement process appears.

Regression for the Desktop dialog *"The update finished, but needs one more step (gateway_restart)"*.

``_spawn_gateway_restart_watcher`` respawns the gateway only once the paused PID is gone, and that
graceful exit takes ~25-35 s on a busy update. The shared 30 s ``deadline`` in ``_poll_until_ready``
is computed when the resume starts, so the wait for the *old* process could consume the whole budget:
in both occurrences on this host the replacement was first seen 6-7 s before the deadline expired or
just after it (``gateway-exit-diag.log``: restart 04:50:30 -> first ``gateway.start`` 04:51:03 ->
follow-up recorded 04:51:10). A cold gateway then needs >11 s before its control socket answers - the
hand-off printed "not published yet (its control socket has not answered)" 11 s after the retry's own
spawn. ``_verify_relaunched_gateways_alive`` raised, the updater committed a ``gateway_restart``
follow-up, its in-run retry 20 s later succeeded, and nothing revisited the verdict - so the Desktop
replayed the stale debt at the next boot gate.

These cases drive the poll with a fake clock, so none of them depends on real Windows timing.
"""

from __future__ import annotations

import pytest

from hermes_cli import update_cmd_windows as ucw

# Measured on the reported host: the replacement appears this far into a 30 s budget (restart stage
# 04:50:30 -> first gateway.start 04:51:03, budget origin ~04:50:40).
LATE_RESPAWN_S = 33.0


class _Clock:
    """``_time`` stand-in: only ``sleep`` advances it, so the poll runs as fast as the CPU allows."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture()
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(ucw, "_time", c)
    return c


def _target(clock: _Clock, appears_at: float, pid: int = 4242, profile: str = "default"):
    """A profile target whose replacement process becomes visible at ``appears_at``."""

    def probe(_scan):
        return [pid] if clock.t >= appears_at else []

    return (("profile", profile), probe, True)


def test_a_late_replacement_is_still_verified(clock):
    """A respawn that lands after the shared deadline is given its own budget and is credited."""
    ready = ucw._poll_until_ready(
        [_target(clock, appears_at=LATE_RESPAWN_S)], 30.0, set(), respawn_window=30.0)

    assert ready.get(("profile", "default")) == [4242]
    # It still had to be stable for the confirmation window before being credited.
    assert clock.t >= LATE_RESPAWN_S + ucw._READY_CONFIRM_S
    # ...and the extension is bounded: one budget after the sighting, not later.
    assert clock.t <= LATE_RESPAWN_S + 30.0 + ucw._READY_CONFIRM_S


def test_shared_budget_times_out_when_the_replacement_is_late(clock):
    """Negative control: the pre-fix behaviour must fail exactly this way, or the bug was never real.

    Without the per-target budget the target is dropped by the ``now < deadline`` filter before its
    process exists, so ``_verify_relaunched_gateways_alive`` reports it unready - the reported bug.
    """
    ready = ucw._poll_until_ready([_target(clock, appears_at=LATE_RESPAWN_S)], 30.0, set())

    assert ready == {}
    assert clock.t >= 30.0


def test_a_replacement_inside_the_original_window_is_unaffected(clock):
    """No overreach: a respawn that lands inside the shared budget behaves exactly as before."""
    ready = ucw._poll_until_ready([_target(clock, appears_at=23.0)], 30.0, set())

    assert ready.get(("profile", "default")) == [4242]
    assert clock.t <= 30.0


def test_a_replacement_that_never_appears_still_times_out(clock):
    """Fail-closed: the extension cannot keep a relaunch that never happened alive.

    One grace window is granted (the spawn may not have happened yet), then the target is reported
    unready — the verdict is late, never wrong.
    """
    ready = ucw._poll_until_ready(
        [_target(clock, appears_at=float("inf"))], 30.0, set(), respawn_window=30.0)

    assert ready == {}
    assert 2 * 30.0 <= clock.t <= 2 * 30.0 + ucw._READY_CONFIRM_S + 1.0


def test_one_late_target_does_not_cost_the_early_one_its_pass(clock):
    """The extension belongs to the late target only; a target visible at t=0 is credited as before."""
    early = _target(clock, appears_at=0.0, pid=1111, profile="default")
    late = _target(clock, appears_at=LATE_RESPAWN_S, pid=2222, profile="other")

    ready = ucw._poll_until_ready([early, late], 30.0, set(), respawn_window=30.0)

    assert ready[("profile", "default")] == [1111]
    assert ready[("profile", "other")] == [2222]
    assert clock.t <= LATE_RESPAWN_S + 30.0 + ucw._READY_CONFIRM_S
