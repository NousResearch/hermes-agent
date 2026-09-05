"""A compacting CLI looks identical to a wedged one, and got killed for it.

Between the PreCompact hook and the compact_boundary system message the CLI
emits NOTHING. _TurnWatch has no way to tell that silence from a wedge, so the
post-tool quiet watchdog tripped, interrupted the CLI mid-compaction, and the
turn's terminal ResultMessage never arrived -- surfacing next turn as
"discarding N stale unsolicited text(s)" and, to the user, as a turn that
simply died at a compaction event.

Measured on a live deployment 2026-08-16/17 (CEST):
  03:57:17  compaction started (trigger=auto)
  03:58:48  "no compact_boundary seen; emitting completion at turn end"  <- 91s, DIED
  03:59:15  compaction started (trigger=auto)
  04:01:02  compact_boundary trigger=auto                                <- 107s, survived
  05:04:07  compaction started (trigger=auto)                            } post-fix,
  05:06:12  compact_boundary trigger=auto                                } 125s, survived

91s is one 90s post_tool_quiet timeout after the last tool result. The LONGER
compaction survived, so duration alone is not the trigger: it is whether
compaction begins while post_tool_armed is set -- i.e. right after a tool
call. That is why it hits tool-heavy turns and looks intermittent.

The suspension is bounded on purpose. compact_boundary is not guaranteed (the
03:57 case never produced one), so an unbounded gate would trade a killed turn
for a hung one.
"""

from __future__ import annotations

import pytest

from agent.transports.claude_agent_sdk_session import (
    _COMPACTION_MAX_SUSPEND,
    _TurnWatch,
)

QUIET = 90.0
BUDGET = 600.0


def _armed_watch(monkeypatch, t0=1000.0):
    """A watch that has just taken a tool result -- the vulnerable state."""
    clock = {"now": t0}
    monkeypatch.setattr(
        "agent.transports.claude_agent_sdk_session.time.monotonic",
        lambda: clock["now"],
    )
    watch = _TurnWatch()
    watch.arm_post_tool()
    return watch, clock


class TestTheRegression:
    def test_silent_compaction_used_to_look_exactly_like_a_wedge(self, monkeypatch):
        """Baseline: without the fix this is the kill. Pin the mechanism."""
        watch, clock = _armed_watch(monkeypatch)
        clock["now"] += QUIET + 1
        assert watch.check(budget=BUDGET, quiet=QUIET) == "post_tool_quiet"

    def test_compaction_survives_past_the_quiet_timeout(self, monkeypatch):
        """The actual fix: 91s of compaction silence must not trip."""
        watch, clock = _armed_watch(monkeypatch)
        watch.compaction_begin()
        clock["now"] += 91.0
        assert watch.check(budget=BUDGET, quiet=QUIET) is None

    def test_the_125s_production_case_survives(self, monkeypatch):
        """The post-fix compaction measured at 05:04:07 -> 05:06:12."""
        watch, clock = _armed_watch(monkeypatch)
        watch.compaction_begin()
        clock["now"] += 125.0
        assert watch.check(budget=BUDGET, quiet=QUIET) is None

    def test_budget_rule_is_suspended_during_compaction_too(self, monkeypatch):
        """Compaction late in a long turn must not trip the budget rule."""
        watch, clock = _armed_watch(monkeypatch)
        clock["now"] += BUDGET - 10
        watch.compaction_begin()
        clock["now"] += 60
        assert watch.check(budget=BUDGET, quiet=QUIET) is None


class TestResumingAfterCompaction:
    def test_the_quiet_window_restarts_after_the_boundary(self, monkeypatch):
        """compaction_end must restamp, not inherit a pre-compaction stamp.

        Without the tick() in compaction_end, a turn that compacted for 91s
        would resume already 91s idle and trip on the very next poll -- the
        same kill, one poll later.
        """
        watch, clock = _armed_watch(monkeypatch)
        watch.compaction_begin()
        clock["now"] += 91.0
        watch.compaction_end()
        clock["now"] += 1.0
        assert watch.check(budget=BUDGET, quiet=QUIET) is None

    def test_a_genuine_wedge_after_compaction_still_trips(self, monkeypatch):
        """The watchdog must not be permanently disarmed by the fix."""
        watch, clock = _armed_watch(monkeypatch)
        watch.compaction_begin()
        clock["now"] += 91.0
        watch.compaction_end()
        clock["now"] += QUIET + 1
        assert watch.check(budget=BUDGET, quiet=QUIET) == "post_tool_quiet"


class TestTheSuspensionIsBounded:
    def test_a_boundary_that_never_arrives_cannot_hang_the_turn(self, monkeypatch):
        """The 03:57 case: compaction began, no boundary ever came.

        Trading a killed turn for a turn hung until the gateway's 1800s
        inactivity ceiling would not be a fix.
        """
        watch, clock = _armed_watch(monkeypatch)
        watch.compaction_begin()
        clock["now"] += _COMPACTION_MAX_SUSPEND + 1
        assert watch.check(budget=BUDGET, quiet=QUIET) == "post_tool_quiet"

    def test_reentrant_precompact_cannot_extend_the_ceiling(self, monkeypatch):
        """A second PreCompact must not restart the bounding clock.

        Advance past BOTH the ceiling (measured from the FIRST begin) and the
        quiet window, so the two behaviours are distinguishable: with
        earliest-start the gate is lifted and the idle time trips; if re-entry
        restarted the clock the gate would still hold and this returns None.
        """
        watch, clock = _armed_watch(monkeypatch)
        watch.compaction_begin()
        clock["now"] += _COMPACTION_MAX_SUSPEND - 5
        watch.compaction_begin()  # re-entry, earliest start wins
        clock["now"] += QUIET + 1
        assert watch.check(budget=BUDGET, quiet=QUIET) == "post_tool_quiet"

    def test_unbalanced_end_cannot_drive_the_counter_negative(self, monkeypatch):
        watch, _ = _armed_watch(monkeypatch)
        watch.compaction_end()
        watch.compaction_end()
        assert watch.compaction_active == 0


class TestExistingGatesUnchanged:
    def test_outstanding_tools_still_suspend(self, monkeypatch):
        watch, clock = _armed_watch(monkeypatch)
        watch.note_tools_issued(1)
        clock["now"] += QUIET + 1
        assert watch.check(budget=BUDGET, quiet=QUIET) is None

    def test_approvals_still_suspend(self, monkeypatch):
        watch, clock = _armed_watch(monkeypatch)
        watch.approval_begin()
        clock["now"] += QUIET + 1
        assert watch.check(budget=BUDGET, quiet=QUIET) is None

    def test_unarmed_idle_turn_still_respects_the_budget(self, monkeypatch):
        watch, clock = _armed_watch(monkeypatch)
        watch.disarm_post_tool()
        clock["now"] += BUDGET + 1
        assert watch.check(budget=BUDGET, quiet=QUIET) == "budget"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
