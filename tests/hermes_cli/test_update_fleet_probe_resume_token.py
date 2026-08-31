"""Regression for #93406 (residual) — the Windows pause/resume token is NOT a
fleet runtime and must not be counted by ``_fleet_probe_expected_runtimes``.

The first #93406 guard counted the ``_windows_gateway_resume`` token
(``profiles`` / ``unmapped`` entries) as an "expected fleet rows" signal. But
the token is pause/resume *bookkeeping*, not a runtime inventory:

* ``unmapped`` entries (Scheduled-Task gateways) never publish
  ``gateway_state.json`` rows at all, and
* a paused-then-resumed profile gateway relaunches DETACHED and may not
  republish its identity within the probe's window,

so ``collect_fleet_versions()`` can legitimately return zero rows for a
perfectly healthy Windows update. With the token counted as an expected
runtime, ``_fleet_rows_expected`` is True, the verification loop silently
waits out its polling window (~14 min wall clock on an end-user report with
the retry loop), prints "Fleet version check returned no rows", and ``hermes
update`` exits 1 — for an update that succeeded.

The invariant this file pins: ``_fleet_probe_expected_runtimes`` may only
return True for signals that correspond to rows ``collect_fleet_versions()``
is actually capable of returning (restart-phase bookkeeping, the pre-restart
PID snapshot, the pre-update plan inventory). A genuinely live pre-update
Windows gateway is already covered by ``pre_restart_pids`` and the plan
inventory — the token adds no row-capable information on top.

Counterfactual: every test in ``TestResumeTokenIsNotARuntime`` FAILS on the
pre-fix ``_fleet_probe_expected_runtimes`` (which returns True for a
token-only signal).
"""

from __future__ import annotations

import types

from hermes_cli.main import _fleet_probe_expected_runtimes


def _plan(runtimes):
    return types.SimpleNamespace(runtimes=runtimes)


class TestResumeTokenIsNotARuntime:
    """Token-only signals must NOT mark fleet rows as expected (#93406)."""

    def test_token_profiles_alone_do_not_expect_rows(self):
        # A paused/resumed profile gateway relaunches detached; its row is
        # not guaranteed within the probe window. Token-only == no rows
        # expected, so zero rows stays exit 0 instead of a false failure.
        token = {"resume_needed": False, "profiles": {"default": 4321}}
        assert (
            _fleet_probe_expected_runtimes(None, [], token, [], set()) is False
        )

    def test_token_unmapped_alone_does_not_expect_rows(self):
        # Scheduled-Task gateways (token["unmapped"]) never publish
        # gateway_state.json rows — collect_fleet_versions() CANNOT return a
        # row for them, so they must not be counted as expected rows.
        token = {"resume_needed": False, "unmapped": [{"pid": 99, "argv": ["x"]}]}
        assert (
            _fleet_probe_expected_runtimes(None, [], token, [], set()) is False
        )

    def test_token_with_empty_pid_snapshot_is_still_not_expected(self):
        # Even alongside an affirmatively-empty PID snapshot and an empty
        # plan, the token alone must not flip the expectation.
        token = {"resume_needed": True, "profiles": {"work": 777}, "unmapped": []}
        assert (
            _fleet_probe_expected_runtimes(_plan([]), [], token, [], set())
            is False
        )


class TestRowCapableSignalsStillCount:
    """The row-capable liveness signals are unaffected by the exclusion."""

    def test_pre_restart_pids_still_expect_rows_alongside_token(self):
        # A live pre-update gateway is covered by the PID snapshot — the
        # row-capable signal — regardless of the token riding along.
        token = {"resume_needed": False, "profiles": {"default": 4321}}
        assert (
            _fleet_probe_expected_runtimes(None, [4321], token, [], set())
            is True
        )

    def test_plan_inventory_still_expects_rows_alongside_token(self):
        token = {"resume_needed": False, "unmapped": [{"pid": 99, "argv": ["x"]}]}
        assert (
            _fleet_probe_expected_runtimes(_plan([object()]), [], token, [], set())
            is True
        )

    def test_unreadable_pre_state_still_expects_rows(self):
        assert _fleet_probe_expected_runtimes(None, None, None, [], set()) is True
