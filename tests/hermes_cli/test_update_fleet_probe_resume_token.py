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
from hermes_cli.update_inventory import RuntimeRecord
from types import SimpleNamespace
import pytest


def _plan(runtimes):
    return types.SimpleNamespace(runtimes=runtimes)


class TestResumeTokenIsNotARuntime:
    """Token-only signals must NOT mark fleet rows as expected (#93406)."""



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
            _fleet_probe_expected_runtimes(
                _plan([RuntimeRecord(kind="gateway", profile="default")]),
                [],
                token,
                [],
                set(),
            )
            is True
        )


@pytest.mark.parametrize('plan,pids,token,services,killed,expected', [
    (None, [], {'profiles': {'default': 4321}}, [], set(), False),
    (None, [], {'unmapped': [{'pid': 99}]}, [], set(), False),
    ([], [], {'profiles': {'work': 777}}, [], set(), False),
    (None, [], {'services': ['HermesGateway']}, [], set(), False),
    (None, [], {}, [], set(), False), (None, [], None, [], set(), False),
    ([], [], None, [], set(), False), (None, None, None, [], set(), True),
    (None, [4321], {'profiles': {'default': 4321}}, [], set(), True),
    (['gateway'], [], {'unmapped': [{'pid': 99}]}, [], set(), True),
    (['serve', 'dashboard'], [], None, [], set(), False),
    (['serve', 'gateway'], [], None, [], set(), True),
    (None, [], None, ['hermes-gateway'], set(), True),
    (None, [], None, [], {4321}, True),
])
def test_expected_rows(plan, pids, token, services, killed, expected):
    inventory = None if plan is None else SimpleNamespace(
        runtimes=[RuntimeRecord(kind=kind, profile='default') for kind in plan])
    assert _fleet_probe_expected_runtimes(inventory, pids, token, services, killed) is expected
