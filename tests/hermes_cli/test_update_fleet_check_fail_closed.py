"""Regression for #93406 — post-update fleet version check must fail closed.

``collect_fleet_versions()`` swallows every probe failure via
``logger.debug()`` and ``print_fleet_version_matrix([])`` early-returns
``False``, so an empty fleet snapshot used to read as "healthy fleet" and
``hermes update`` exited 0 with zero rows — even when a gateway was
verifiably live before the update.

The first guard (PR #93410) keyed on ``(restarted_services or killed_pids)``,
which never fires on Windows: ``_pause_windows_gateways_for_update`` /
``_resume_windows_gateways_after_update`` populate neither list.  The fix
hoists the "should the probe have produced rows?" decision into
``_fleet_probe_expected_runtimes`` and keys it on the ROW-CAPABLE pre-update
liveness signals: restart-phase bookkeeping, the pre-restart PID snapshot,
and the pre-update plan inventory.  The Windows pause/resume token is
deliberately NOT a signal — it is bookkeeping, not a runtime inventory, and
its entries have no corresponding ``collect_fleet_versions()`` rows (see
``test_update_fleet_probe_resume_token.py``).  The same condition gates the
2.0s settle sleep.
"""

from __future__ import annotations

import inspect
import types

from hermes_cli.main import _fleet_probe_expected_runtimes


def _plan(runtimes):
    return types.SimpleNamespace(runtimes=runtimes)


class TestEmptySnapshotFailClosed:
    """Signals under which zero fleet rows means verification failure."""

    def test_incomplete_when_pre_update_plan_saw_runtimes(self):
        # (a) The plan inventoried a live runtime pre-update but the restart
        # phase's POSIX bookkeeping is empty (e.g. Windows, or an
        # externally-supervised gateway). Zero rows must fail closed.
        assert (
            _fleet_probe_expected_runtimes(
                _plan([object()]),
                [],  # pre_restart_pids: probe saw nothing
                None,  # no Windows resume token
                [],  # restarted_services
                set(),  # killed_pids
            )
            is True
        )

    def test_windows_resume_token_alone_is_not_expected(self):
        # (c) The Windows pause/resume token is EXCLUDED from the expectation
        # (#93406 residual): it is pause/resume bookkeeping, not a runtime
        # inventory, and collect_fleet_versions() cannot return rows for its
        # entries (unmapped Scheduled-Task gateways never publish
        # gateway_state.json; a resumed profile gateway relaunches detached).
        # Counting it made a healthy Windows update wait out the probe window
        # and exit 1 on zero rows. Full coverage lives in
        # test_update_fleet_probe_resume_token.py.
        token = {"resume_needed": False, "profiles": {"default": 4321}}
        assert (
            _fleet_probe_expected_runtimes(None, [], token, [], set()) is False
        )
        token = {"resume_needed": False, "unmapped": [{"pid": 99, "argv": ["x"]}]}
        assert (
            _fleet_probe_expected_runtimes(None, [], token, [], set()) is False
        )

    def test_windows_resume_token_services_do_not_demand_rows(self):
        # Deliberately inverted from the original pin (#93406/#95589): SCM
        # services the updater itself paused/resumed produce NO probe rows —
        # counting them as "expected runtimes" made every healthy Windows
        # desktop update stall ~14min in fleet verification and exit 1.
        # The token is excluded wholesale; restart-phase and pre-restart
        # signals below still fail closed.
        token = {
            "resume_needed": False,
            "profiles": {},
            "unmapped": [],
            "services": ["HermesGateway"],
        }
        assert (
            _fleet_probe_expected_runtimes(None, [], token, [], set()) is False
        )

    def test_incomplete_when_restart_phase_touched_gateways(self):
        # The original #93410 signal still counts.
        assert (
            _fleet_probe_expected_runtimes(None, [], None, ["hermes-gateway"], set())
            is True
        )
        assert _fleet_probe_expected_runtimes(None, [], None, [], {4321}) is True

    def test_incomplete_when_pre_restart_pids_seen(self):
        assert _fleet_probe_expected_runtimes(None, [4321], None, [], set()) is True

    def test_incomplete_when_pre_restart_state_unreadable(self):
        # None means the pre-state could not be read — cannot prove nothing
        # was running, same contract as _restart_phase_failure_is_incomplete.
        assert _fleet_probe_expected_runtimes(None, None, None, [], set()) is True


class TestEmptySnapshotGenuinelyIdle:
    def test_success_when_nothing_was_running_pre_update(self):
        # (b) Positive control: no plan runtimes, empty PID snapshot, no
        # Windows token, no restart bookkeeping — zero rows stays a success.
        assert (
            _fleet_probe_expected_runtimes(_plan([]), [], None, [], set()) is False
        )

    def test_success_with_no_plan_at_all(self):
        assert _fleet_probe_expected_runtimes(None, [], None, [], set()) is False

    def test_success_with_empty_windows_token(self):
        # A token that paused nothing (e.g. Windows host with no gateways)
        # is not a liveness signal.
        token = {"resume_needed": False, "profiles": {}, "unmapped": []}
        assert _fleet_probe_expected_runtimes(None, [], token, [], set()) is False


class TestCallSiteWiring:
    """The guard AND the settle sleep must both key on the shared signal.

    Sabotage-proof for the wiring itself: reverting the call site to the
    pre-fix ``(restarted_services or killed_pids)`` condition — while leaving
    the helper in place — makes these fail.
    """

    def _impl_source(self):
        from hermes_cli import update_cmd

        return inspect.getsource(update_cmd._cmd_update_impl)

    def test_settle_sleep_gated_on_expected_runtimes(self):
        src = self._impl_source()
        assert "_fleet_rows_expected = _m()._fleet_probe_expected_runtimes(" in src
        # The 2.0s settle window must key on the cross-platform signal, so a
        # resumed Windows gateway gets its settle window too (#93406).
        assert "if _fleet_rows_expected:\n" in src
        assert "if restarted_services or killed_pids:\n                _time.sleep" not in src

    def test_zero_row_guard_gated_on_expected_runtimes(self):
        src = self._impl_source()
        assert "elif not _fleet_snapshot and _fleet_rows_expected:" in src
        assert (
            "elif not _fleet_snapshot and (restarted_services or killed_pids):"
            not in src
        )
