"""A retiring gateway must get long enough to finish its WAL checkpoint.

Regression for the 2026-08-31 corruption. ``gateway run --replace`` reaped the
previous instance with SIGTERM, waited **5 seconds**, then SIGKILLed any
survivor. A gateway closing a 500MB WAL store runs a PASSIVE checkpoint in
``SessionDB.close()``; on a WAL that had grown to 3936 pages (4x the 1000-page
autocheckpoint threshold) that does not reliably finish in 5s. A SIGKILL
landing mid-checkpoint leaves half-written b-tree pages — macOS ``fsync``
guarantees neither data-on-platter nor write ordering, which is exactly why
``hermes_state_wal._enforce_macos_synchronous_full`` exists.

The port-rebinding reason the 5s deadline was introduced still holds, so the
force-kill stays — it just must not fire on a process that is still shutting
down normally, and when it does fire it must say so.
"""
from __future__ import annotations

import pytest

import hermes_state_wal

from hermes_cli.gateway import (
    _ORPHAN_EXIT_GRACE_SECONDS,
    _await_gateway_exit,
)


def _exits_after(polls: int):
    """pid_exists() that reports the process gone after *polls* checks."""
    seen = {"n": 0}

    def pid_exists(_pid: int) -> bool:
        seen["n"] += 1
        return seen["n"] <= polls

    return pid_exists


def test_a_slow_but_healthy_shutdown_is_not_force_killed() -> None:
    """12s of checkpointing — well past the old 5s budget — must survive."""
    survivors = _await_gateway_exit(
        [4242],
        pid_exists=_exits_after(60),  # 60 polls x 0.2s = 12s
        sleep=lambda _s: None,
    )
    assert survivors == []


def test_a_truly_hung_process_is_still_reported_for_force_kill() -> None:
    survivors = _await_gateway_exit(
        [4242],
        pid_exists=lambda _pid: True,
        sleep=lambda _s: None,
    )
    assert survivors == [4242]


def test_an_already_dead_process_returns_immediately() -> None:
    slept = []
    survivors = _await_gateway_exit(
        [4242],
        pid_exists=lambda _pid: False,
        sleep=lambda s: slept.append(s),
    )
    assert survivors == []
    assert slept == [], "waited on a process that was already gone"


def test_the_grace_period_covers_a_large_wal_checkpoint() -> None:
    """5s was the value that let the SIGKILL land mid-checkpoint."""
    assert _ORPHAN_EXIT_GRACE_SECONDS >= 20.0


def test_force_kill_is_logged_so_the_next_incident_has_evidence(caplog) -> None:
    import logging

    from hermes_cli import gateway as gw

    killed = []
    with caplog.at_level(logging.WARNING):
        gw._force_kill_survivors([4242], kill=lambda pid, sig: killed.append(pid))

    assert killed == [4242]
    assert any("4242" in r.getMessage() for r in caplog.records), (
        "a force-kill left no trace in the log"
    )


def test_a_pid_that_vanishes_in_the_final_interval_is_not_killed() -> None:
    """The last sleep must still be followed by a check.

    Without a re-check after the final sleep the process is reported as a
    survivor and gets SIGKILL. That is normally a harmless
    ProcessLookupError — but if the PID has already been recycled the signal
    lands on an unrelated process.
    """
    polls = int(_ORPHAN_EXIT_GRACE_SECONDS / 0.2)  # every in-loop check says "alive"
    survivors = _await_gateway_exit(
        [4242],
        pid_exists=_exits_after(polls),
        sleep=lambda _s: None,
    )
    assert survivors == [], "a process that exited during the last interval was killed"
