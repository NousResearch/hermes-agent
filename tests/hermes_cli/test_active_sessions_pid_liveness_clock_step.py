"""Regression: _pid_liveness must tolerate a wall-clock step (NTP/sleep/DST).

Before #105714 the create_time comparison used a 1 ms tolerance, so a ~1 s wall
clock step (NTP correction, laptop sleep/wake, the DST shift on 2026-10-25)
made psutil report a create_time ~1 s away from the value recorded when the
lease was taken. A live process was silently deemed dead and its lease pruned
from the active session registry on the next read/write — nothing errored, the
state file just stopped describing reality.

The fix widens the tolerance to ``_PID_START_TOLERANCE_S`` (2.0 s): a clock
step under ~2 s no longer false-prunes, while a recycled PID is still caught
(a reused PID's create_time differs by the process's whole age — seconds to
days, never ~1 s).
"""
from unittest.mock import patch

from hermes_cli import active_sessions as a
import gateway.status as gs


PID = 4242
RECORDED = 1_757_000_000.0


def test_no_clock_step_live_process_stays_alive():
    """Baseline: recorded start matches current start -> live."""
    with patch.object(gs, "_pid_exists", return_value=True), \
         patch.object(a, "_process_start_time", return_value=RECORDED):
        assert a._pid_liveness(PID, RECORDED) is True


def test_clock_step_does_not_false_prune_live_session():
    """The same live process after a 1 s wall-clock step must still be alive.

    This is the #105714 regression: a 1 ms tolerance returned False here; the
    2 s tolerance returns True.
    """
    with patch.object(gs, "_pid_exists", return_value=True), \
         patch.object(a, "_process_start_time", return_value=RECORDED + 1.0):
        assert a._pid_liveness(PID, RECORDED) is True


def test_recycled_pid_is_still_caught():
    """A PID reused by a fresh process (create_time an hour away) is still pruned.

    2 s still catches recycling: a reused PID's create_time differs by the whole
    process age (seconds-days), never ~1 s.
    """
    with patch.object(gs, "_pid_exists", return_value=True), \
         patch.object(a, "_process_start_time", return_value=RECORDED + 3600.0):
        assert a._pid_liveness(PID, RECORDED) is False


def test_prune_dead_keeps_clock_stepped_lease():
    """End-to-end: a clock-stepped live session survives _prune_dead."""
    entry = {
        "pid": PID,
        "process_start_time": RECORDED,
        "lease_id": "x",
        "session_id": "s",
        "surface": "cli",
    }
    with patch.object(gs, "_pid_exists", return_value=True), \
         patch.object(a, "_process_start_time", return_value=RECORDED + 1.0):
        survivors = a._prune_dead([entry])
    assert len(survivors) == 1
    assert survivors[0]["session_id"] == "s"


def test_prune_dead_drops_a_genuinely_dead_session():
    """Regression guard: a process whose PID no longer exists is still pruned."""
    entry = {
        "pid": PID,
        "process_start_time": RECORDED,
        "lease_id": "x",
        "session_id": "s",
        "surface": "cli",
    }
    with patch.object(gs, "_pid_exists", return_value=False):
        survivors = a._prune_dead([entry])
    assert survivors == []
