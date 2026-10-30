"""The relaunch verdict is written before the post-restart matrix exists, so let the matrix retract it.

Regression for the Desktop dialog *"The update finished, but needs one more step (gateway_restart)"*.

``_verify_relaunched_gateways_alive`` polls PID files inside a fixed liveness budget
(``_RELAUNCH_VERIFY_TIMEOUT_S``). A respawn that lands after that budget, or one a parent Job Object
kills during updater teardown (#48820), raises *"Windows gateway relaunch after update was not
verified alive"* — and the resulting ``gateway_restart`` follow-up is committed and never revisited.
The same run's ``fleet`` matrix, collected a few seconds later, reports the gateway ``current`` on the
new checkout, so the debt the Desktop re-dialogs is false and its remedy (``hermes gateway restart``)
answers "already running — nothing to start".

Both tests below drive the production ``_verify_fleet_after_update`` (only external probes and the
receipt writer are stubbed), because the defect is the ORDER — discharge before ``record_followup`` —
not the predicate alone.

Evidence used as the fixture is the reported host's own receipt
(``logs/update_receipts/update_20261028_171315_*.json``): plan runtime gateway/default pid 55992,
``gateway_restart.phase_error`` = the relaunch timeout, ``fleet`` = one ``current`` row (pid 5544) on
the pulled sha, ``followups`` = ``gateway_restart``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd_fleet as fleet
from hermes_cli import update_cmd_fleet_verify as fv
from hermes_cli.update_cmd_fleet import _GatewayRestartOutcome

#: What the reported host's restart phase recorded (see module docstring). Spelled out rather than
#: imported from the module under test, so this suite still collects against the pre-fix code and
#: the negative controls fail on the real defect instead of on a missing attribute.
RELAUNCH_ERROR = "Windows gateway relaunch after update was not verified alive: default"


def _restart(**overrides) -> _GatewayRestartOutcome:
    """Restart bookkeeping exactly as the Windows resume leaves it after a relaunch timeout."""
    fields = dict(
        incomplete=True,
        phase_errors=[RELAUNCH_ERROR],
        pre_restart_gateway_pids=[55992],
        restarted_services=[],
        failed_or_stale_units=[],
        relaunched_profiles=[],
        externally_supervised_profiles=[],
        killed_pids=set(),
    )
    fields.update(overrides)
    return _GatewayRestartOutcome(**fields)


def _plan(*runtimes) -> SimpleNamespace:
    """A pre-update plan; the reported host's one gateway row by default."""
    rows = runtimes or (SimpleNamespace(kind="gateway", profile="default", pid=55992),)
    return SimpleNamespace(runtimes=list(rows))


def _row(profile: str = "default", state: str = "current", pid: int = 5544) -> dict:
    return {"profile": profile, "state": state, "pid": pid, "code_sha": "190c0b7f65e"}


@pytest.fixture()
def verified(monkeypatch):
    """``_verify_fleet_after_update`` wired to a fleet matrix of *rows*, with every probe stubbed.

    Returns the list ``record_followup`` was called with, so a test can assert the debt was NOT
    committed (and, in the negative controls, that the path really does reach that call).
    """
    followups: list = []

    def wire(rows: list) -> list:
        monkeypatch.setattr(fv, "_print_legacy_units_warning", lambda: None)
        monkeypatch.setattr(fv, "_named_gateways_still_owed", lambda: False)
        monkeypatch.setattr(fv, "_collect_fleet_snapshot", lambda restart, expected: list(rows))
        monkeypatch.setattr(fv, "_settle_units_proven_current", lambda restart, rows: False)
        monkeypatch.setattr(fleet, "_warn_incomplete_gateway_fleet_restart", lambda units: None)
        monkeypatch.setattr(fleet, "_clear_fleet_restart_pending_marker", lambda: None)
        monkeypatch.setattr("hermes_cli.update_cmd_maint._refresh_dashboard_after_update",
                            lambda already_restarted_units=None: set())
        monkeypatch.setattr("hermes_cli.update_cmd._surviving_pre_update_serve_runtimes", lambda plan: [])
        monkeypatch.setattr("hermes_cli.update_cmd._warn_stale_serve_runtimes", lambda rows: None)
        monkeypatch.setattr("hermes_cli.update_cmd._fleet_probe_expected_runtimes", lambda *a, **k: True)
        # ``_m()`` resolves to ``hermes_cli.main`` (the historical patch surface), not update_cmd.
        monkeypatch.setattr("hermes_cli.main._fleet_probe_expected_runtimes", lambda *a, **k: True)
        monkeypatch.setattr("hermes_cli.update_receipt.print_fleet_version_matrix", lambda rows: False)
        monkeypatch.setattr("hermes_cli.update_receipt.row_is_external", lambda row: False)
        monkeypatch.setattr("hermes_cli.update_receipt.finalize_update_receipt", lambda *a, **k: None)
        monkeypatch.setattr("hermes_cli.update_inventory.match_runtime_outcomes",
                            lambda plan, **kwargs: [{"kind": r.kind, "profile": r.profile, "pid": r.pid,
                                                     "outcome": "restarted"} for r in plan.runtimes])
        monkeypatch.setattr("hermes_cli.update_inventory.report_unaccounted_runtimes", lambda outcomes: False)
        monkeypatch.setattr("hermes_cli.update_serve_obligations.defer_manual_serve", lambda *a, **k: False)
        monkeypatch.setattr("hermes_cli.update_receipt.record_followup",
                            lambda step, reason: followups.append((step, reason)))
        monkeypatch.setattr("hermes_cli.update_host_obligation.mark_host_restart_completed", lambda sha: None)
        return followups

    return wire


def _verify(restart, plan) -> None:
    fv._verify_fleet_after_update(
        restart, _pre_update_plan=plan, _windows_gateway_resume=None, update_complete=False)


def test_a_relaunch_timeout_the_fleet_disproves_records_no_followup(verified, monkeypatch, capsys):
    """The reported run: relaunch timeout + a ``current`` row for the planned gateway ⇒ no debt."""
    followups = verified([_row()])
    recorded: list = []
    restart = _restart()
    monkeypatch.setattr(restart, "record_receipt", lambda **extra: recorded.append(extra))

    _verify(restart, _plan())

    assert followups == [], f"a false gateway_restart debt was committed: {followups}"
    assert restart.incomplete is False
    assert restart.phase_errors == []
    # The receipt must stop reporting the debt too, or the next run re-arms it.
    assert recorded, "the restart result was not re-recorded after the discharge"
    out = capsys.readouterr().out
    assert "Relaunch timeout discharged" in out and "default" in out


def test_without_the_discharge_the_false_debt_is_recorded(verified, monkeypatch):
    """Negative control: with only the discharge call disabled, the SAME run commits a false debt.

    This is the pre-fix behaviour (the code path that produced the reported receipt): the matrix
    proves the planned gateway ``current`` and the run still records ``gateway_restart`` — the state
    the Desktop re-dialogs. It fails if the discharge call site is ever dropped, and it proves the
    green test above is green because of that call and not because the follow-up path is dead.
    """
    followups = verified([_row()])
    restart = _restart()
    monkeypatch.setattr(restart, "record_receipt", lambda **extra: None)
    monkeypatch.setattr(fv, "_discharge_relaunch_verification_by_fleet", lambda *a, **k: False,
                        raising=False)

    _verify(restart, _plan())

    assert [step for step, _reason in followups] == ["gateway_restart"]
    assert restart.incomplete is True
    assert restart.phase_errors == [RELAUNCH_ERROR]


def test_a_stale_row_still_owes_the_restart(verified, monkeypatch):
    """Negative control: the same run with a row that is NOT current keeps its follow-up."""
    followups = verified([_row(state="stale")])
    restart = _restart()
    monkeypatch.setattr(restart, "record_receipt", lambda **extra: None)

    _verify(restart, _plan())

    assert [step for step, _reason in followups] == ["gateway_restart"]
    assert restart.incomplete is True
    assert restart.phase_errors == [RELAUNCH_ERROR]


def test_a_missing_row_still_owes_the_restart(verified, monkeypatch):
    """A profile the plan expected with no row at all is absence of evidence, not evidence."""
    followups = verified([_row(profile="critic")])
    restart = _restart()
    monkeypatch.setattr(restart, "record_receipt", lambda **extra: None)

    _verify(restart, _plan(SimpleNamespace(kind="gateway", profile="default", pid=55992)))

    assert [step for step, _reason in followups] == ["gateway_restart"]


def test_an_empty_matrix_still_owes_the_restart(verified, monkeypatch):
    followups = verified([])
    restart = _restart()
    monkeypatch.setattr(restart, "record_receipt", lambda **extra: None)

    _verify(restart, _plan())

    assert [step for step, _reason in followups] == ["gateway_restart"]


# --- the predicate on its own: every other owed source survives a healthy matrix ----------------


def test_a_second_phase_error_is_never_laundered():
    restart = _restart(phase_errors=[RELAUNCH_ERROR, "Could not restart every paused Windows gateway: pid 1"])

    assert fv._discharge_relaunch_verification_by_fleet(restart, _plan(), [_row()]) is False
    assert restart.incomplete is True
    assert len(restart.phase_errors) == 2


def test_a_failed_unit_is_never_laundered():
    restart = _restart(failed_or_stale_units=["hermes-gateway"])

    assert fv._discharge_relaunch_verification_by_fleet(restart, _plan(), [_row()]) is False
    assert restart.incomplete is True


def test_a_gateway_stopped_without_a_successor_is_never_laundered():
    restart = _restart(stopped_unmapped_pids={4242})

    assert fv._discharge_relaunch_verification_by_fleet(restart, _plan(), [_row()]) is False
    assert restart.incomplete is True


def test_a_non_relaunch_phase_error_does_nothing():
    """Nothing to discharge: the relaunch prefix is what identifies THIS verdict."""
    restart = _restart(phase_errors=["Could not restart every paused Windows gateway: pid 1"])
    before = list(restart.phase_errors)

    assert fv._discharge_relaunch_verification_by_fleet(restart, _plan(), [_row()]) is False
    assert restart.phase_errors == before
    assert restart.incomplete is True


def test_a_serve_only_plan_proves_nothing_about_gateways():
    """A plan with no gateway row cannot vouch for a gateway restart."""
    restart = _restart()

    assert fv._discharge_relaunch_verification_by_fleet(
        restart, _plan(SimpleNamespace(kind="serve", profile="default", pid=1)), [_row()]) is False
    assert restart.incomplete is True


def test_every_expected_profile_needs_its_own_row():
    """One current row never vouches for a profile the plan also expected."""
    restart = _restart()
    plan = _plan(SimpleNamespace(kind="gateway", profile="default", pid=55992),
                 SimpleNamespace(kind="gateway", profile="critic", pid=55993))

    assert fv._discharge_relaunch_verification_by_fleet(restart, plan, [_row()]) is False
    assert restart.incomplete is True
