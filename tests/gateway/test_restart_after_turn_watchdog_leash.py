"""Shutdown watchdog must not fire during a legitimate after-turn wait.

In-band restart (``/restart``, SIGUSR1, fleet restart) defers ``stop()`` and
waits up to ``agent.restart_after_turn_timeout`` (default 1800s) for in-flight
turns to finish (#77184). ``stop()`` arms a thread watchdog whose leash is
``resolve_shutdown_watchdog_delay(restart_drain_timeout)`` —
``restart_drain_timeout`` defaults to **0**, so the leash is just the 60s
grace.

When something enters ``stop()`` while the after-turn wait is still running,
the watchdog kills a process that is behaving correctly.

Observed 2026-09-03 on Shane's gateway::

    12:35:57 Restart requested with 4 active work unit(s); deferring stop() (cap=1800s)
    12:36:58 Stopping gateway for restart...        <- stop() entered, watchdog armed
    12:36:58 Restart deferred: waiting on 4 ... (1739s remaining)
    12:37:56 Restart deferred: waiting on 4 ... (1680s remaining)
    12:37:58 CRITICAL shutdown watchdog fired after 60s — forcing process exit

The force-exit killed the Windows Task Scheduler supervisor's blocked
``shell.Run``, leaving the gateway with no external supervision.

The leash must cover whatever remains of the after-turn budget.
"""

from __future__ import annotations

import time

import pytest

from gateway.run import GatewayRunner
from gateway.shutdown_watchdog import (
    DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S,
    resolve_shutdown_watchdog_delay,
)


class _Runner:
    """Minimal double exposing only what the leash calculation touches."""

    def __init__(self, *, drain_timeout: float = 0.0, remaining: float = 0.0):
        self._restart_drain_timeout = drain_timeout
        # Mirrors the real attribute published by
        # ``_await_active_work_before_restart``: a monotonic deadline, or None.
        self._restart_after_turn_deadline = (
            time.monotonic() + remaining if remaining > 0 else None
        )
        self._remaining = remaining


def test_leash_covers_remaining_after_turn_wait():
    """The reported bug: 60s leash vs 1739s of legitimate remaining wait."""
    runner = _Runner(drain_timeout=0.0, remaining=1739.0)

    leash = GatewayRunner._resolve_shutdown_leash(runner)

    assert leash >= 1739.0, (
        f"watchdog leash {leash}s is shorter than the {runner._remaining}s of "
        "after-turn budget still legitimately in flight — this force-kills a "
        "healthy gateway and orphans its supervisor"
    )


def test_leash_keeps_grace_on_top_of_remaining_wait():
    runner = _Runner(drain_timeout=0.0, remaining=100.0)
    leash = GatewayRunner._resolve_shutdown_leash(runner)
    # Wall-clock elapses between constructing the double and reading the
    # deadline, so assert the contract (grace on top, bounded) not an exact value.
    assert 100.0 < leash <= 100.0 + DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S


def test_leash_unchanged_when_no_after_turn_wait_is_active():
    """Normal shutdown keeps the existing drain+grace contract."""
    runner = _Runner(drain_timeout=180.0, remaining=0.0)
    assert GatewayRunner._resolve_shutdown_leash(runner) == pytest.approx(
        resolve_shutdown_watchdog_delay(180.0)
    )


def test_leash_uses_whichever_budget_is_larger():
    """A long drain budget must not be shortened by a small remaining wait."""
    runner = _Runner(drain_timeout=600.0, remaining=30.0)
    assert GatewayRunner._resolve_shutdown_leash(runner) == pytest.approx(
        resolve_shutdown_watchdog_delay(600.0)
    )


def test_leash_survives_a_malformed_deadline():
    """A corrupt deadline must degrade to the old behaviour, never crash stop()."""

    class _Bad:
        _restart_drain_timeout = 0.0
        _restart_after_turn_deadline = "not-a-number"

    assert GatewayRunner._resolve_shutdown_leash(_Bad()) == pytest.approx(
        resolve_shutdown_watchdog_delay(0.0)
    )


def test_leash_reads_the_deadline_directly_not_via_a_helper():
    """Regression: the leash must not depend on a helper method being present.

    An earlier implementation called ``self._remaining_after_turn_wait()``
    inside a bare ``except Exception``. Any object without that method — or any
    bug inside it — silently fell back to the 60s leash, reintroducing the
    force-kill while every unit test still passed.
    """

    class _NoHelper:
        """Has the deadline but deliberately no ``_remaining_after_turn_wait``."""

        _restart_drain_timeout = 0.0

        def __init__(self, remaining: float):
            self._restart_after_turn_deadline = time.monotonic() + remaining

    leash = GatewayRunner._resolve_shutdown_leash(_NoHelper(1739.0))
    assert leash > 1739.0, (
        f"leash collapsed to {leash}s — the deadline is being read through a "
        "helper whose absence is silently swallowed"
    )


# --- the deadline probe itself -----------------------------------------


def test_remaining_is_zero_when_no_restart_wait_is_running():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._restart_after_turn_deadline = None
    assert GatewayRunner._remaining_after_turn_wait(runner) == 0.0


def test_remaining_is_zero_after_the_deadline_passes():
    import time

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._restart_after_turn_deadline = time.monotonic() - 5.0
    assert GatewayRunner._remaining_after_turn_wait(runner) == 0.0


def test_remaining_reports_time_left_while_waiting():
    import time

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._restart_after_turn_deadline = time.monotonic() + 120.0
    remaining = GatewayRunner._remaining_after_turn_wait(runner)
    assert 100.0 < remaining <= 120.0
