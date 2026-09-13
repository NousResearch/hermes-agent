"""SQLite completion and fleet verification remain independent update outcomes."""

from contextlib import nullcontext
import json

import pytest

from hermes_cli import update_cmd, update_cmd_fleet, update_cmd_maint, update_receipt
from hermes_constants import get_hermes_home


@pytest.mark.parametrize(
    "update_complete,state",
    [(True, "current"), (False, "current"), (True, "stale"), (True, "down"), (True, None)],
)
def test_fleet_completion_preserves_runtime_verdict_and_restart_obligation(
    update_complete, state, monkeypatch,
):
    refreshed, migrated = [], []
    snapshot = [{"profile": "default", "pid": 1234, "state": state}] if state else []
    restart = update_cmd_fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[1234],
        restarted_services=["hermes-gateway"], failed_or_stale_units=[],
        relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
    )
    monkeypatch.setattr(update_cmd_fleet, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(
        update_cmd_maint, "_refresh_dashboard_after_update",
        lambda **kwargs: refreshed.append(kwargs),
    )
    monkeypatch.setattr(update_receipt, "_code_identity", lambda **kwargs: {})
    monkeypatch.setattr(
        "hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: migrated.append(True),
    )

    def collect(outcome, rows_expected):
        assert outcome is restart
        assert rows_expected is True
        return snapshot

    monkeypatch.setattr(update_cmd_fleet, "_collect_fleet_snapshot", collect)
    update_cmd_fleet._write_fleet_restart_pending_marker()
    marker = update_cmd_fleet._fleet_restart_pending_marker_path()
    assert marker.exists()
    healthy = update_complete and state == "current"
    with update_receipt.update_receipt_scope():
        update_receipt.begin_update_receipt()
        with nullcontext() if healthy else pytest.raises(SystemExit) as exc:
            update_cmd_fleet._verify_fleet_after_update(
                restart, _pre_update_plan=None, _windows_gateway_resume=None,
                update_complete=update_complete,
            )
        if not healthy:
            assert exc.value.code == 1

    receipt = json.loads((get_hermes_home() / "logs/update_receipts/latest.json").read_text())
    assert receipt["outcome"] == ("success" if healthy else "partial")
    assert receipt["fleet"] == snapshot
    assert restart.incomplete is (state != "current")
    assert marker.exists() is (state != "current")
    assert migrated == ([True] if healthy else [])
    assert refreshed == [{"already_restarted_units": {"hermes-gateway"}}]
