"""One monotonic restart handoff deadline, armed at restart admission.

``begin()`` establishes the single handoff context the first time an explicit
restart is requested; duplicate requests reuse it and can never extend it. The
deadline covers the after-turn wait, the stop drain and the teardown tail in
one bound. ``arm=True`` starts a daemon OS-thread watchdog that hard-exits the
process (exit 75, the restart exit code) when the deadline passes even if the
event loop is wedged — a loop-level timer cannot fire on a frozen loop.
A maintenance (external drain marker) never creates a deadline: it stays
reversible until cleared or TTL, by design.
"""
from __future__ import annotations

import os
import time

WATCHDOG_EXIT_CODE = 75


class RestartHandoff:
    def __init__(self, deadline: float, stop_budget: float = 0.0):
        self.deadline = deadline
        self.stop_budget = stop_budget
        self._done: "threading.Event | None" = None

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def phase_budget(self, budget: float, reserve: float = 0.0) -> float:
        """Clamp one phase to the total budget left after ``reserve`` (teardown tail)."""
        return max(0.0, min(float(budget), self.remaining() - float(reserve)))

    def disarm(self) -> None:
        done, self._done = self._done, None
        if done is not None:
            done.set()


def begin(runner, *, stop_budget: float, arm: bool = True) -> RestartHandoff:
    """Establish (or return the existing) handoff context. Never extends a deadline."""
    existing = getattr(runner, "_restart_handoff", None)
    if existing is not None:
        return existing
    after_turn = float(getattr(runner, "_restart_after_turn_timeout", 0.0) or 0.0)
    drain = float(getattr(runner, "_restart_drain_timeout", 0.0) or 0.0)
    handoff = RestartHandoff(time.monotonic() + float(stop_budget) + after_turn + drain,
                             stop_budget=float(stop_budget))
    runner._restart_handoff = handoff
    if arm and not os.environ.get("PYTEST_CURRENT_TEST"):
        # The in-house backstop: dumps all thread stacks before hard-exit, observes a
        # late disarm within ~1s via its done-event, and no-ops on a nonpositive delay.
        from gateway.shutdown_watchdog import arm_shutdown_watchdog
        handoff._done = arm_shutdown_watchdog(handoff.remaining(), exit_code=WATCHDOG_EXIT_CODE,
                                              name="gateway-restart-handoff-watchdog")
    return handoff


def snapshot(runner) -> dict:
    """Admission readiness separate from process liveness (diagnostics stay readable)."""
    if getattr(runner, "_external_drain_active", False):
        return {"admission_ready": False, "drain_reason": "external_drain"}
    handoff = getattr(runner, "_restart_handoff", None)
    if handoff is not None:
        return {"admission_ready": False, "drain_reason": "restart",
                "deadline_in_seconds": round(handoff.remaining(), 3)}
    return {"admission_ready": True, "drain_reason": ""}
