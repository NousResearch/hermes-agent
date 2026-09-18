"""A historical receipt cannot outweigh complete, identity-matched live evidence."""

import json

import pytest

from hermes_cli import update_cmd, update_cmd_fleet, update_receipt
from hermes_constants import get_hermes_home


@pytest.mark.parametrize("profiles", [["alpha"], ["alpha", "beta"]])
def test_current_successors_settle_historical_obligations(monkeypatch, profiles):
    home = get_hermes_home()
    directory = home / "logs" / "update_receipts"
    directory.mkdir(parents=True)
    receipt = {
        "outcome": "failed",
        "plan": {
            "runtimes": [
                {"kind": "gateway", "profile": p, "pid": 1, "code_sha": "old"}
                for p in profiles
            ]
        },
    }
    path = directory / "latest.json"
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "new")
    monkeypatch.setattr(
        update_receipt,
        "collect_fleet_versions",
        lambda: [
            {"profile": p, "pid": 2, "state": "current", "code_sha": "new"}
            for p in profiles
        ],
    )
    assert not update_cmd_fleet._pending_fleet_restart_needed()
    assert (
        json.loads(path.read_text()) == receipt
    )  # Historical failure remains truthful.


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "unknown",
        "down",
        "stale",
        "wrong-sha",
        "wrong-kind",
        "unknown-profile",
        "marker",
    ],
)
def test_every_owed_identity_requires_current_evidence(monkeypatch, bad):
    home = get_hermes_home()
    directory = home / "logs" / "update_receipts"
    directory.mkdir(parents=True)
    if bad == "marker":
        # Obligation for an SHA the fleet does not serve: no verified discharge.
        (home / "fleet_restart_pending").write_text("expected_sha=future\n")
    owed = {"kind": "gateway", "profile": "beta", "code_sha": "old"}
    if bad == "wrong-kind":
        # An unrecognized runtime kind is still unvouchable evidence: stay conservative.
        owed["kind"] = "worker"
    if bad == "unknown-profile":
        owed["profile"] = "unknown"
    (directory / "latest.json").write_text(
        json.dumps({
            "outcome": "failed",
            "plan": {
                "runtimes": [
                    {"kind": "gateway", "profile": "alpha", "code_sha": "old"},
                    owed,
                ]
            },
        })
    )
    rows = [{"profile": "alpha", "state": "current", "code_sha": "new"}]
    if bad != "missing":
        rows.append({
            "profile": "beta",
            "state": bad if bad in {"unknown", "down", "stale"} else "current",
            "code_sha": "old" if bad == "wrong-sha" else "new",
        })
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "new")
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda: rows)
    assert update_cmd_fleet._pending_fleet_restart_needed()


def test_dashboard_runtime_does_not_veto_gateway_discharge(monkeypatch):
    """A serve/dashboard row is outside the gateway matrix's evidence, not against it.

    A host running `hermes dashboard` has a dashboard row in every receipt; the blanket
    veto made the fleet-restart warning permanently undischargeable there (#115090).
    """
    home = get_hermes_home()
    directory = home / "logs" / "update_receipts"
    directory.mkdir(parents=True)
    (directory / "latest.json").write_text(json.dumps({
        "outcome": "failed",
        "plan": {
            "runtimes": [
                {"kind": "gateway", "profile": "default", "pid": 1, "code_sha": "old"},
                {"kind": "dashboard", "profile": "default", "pid": 9, "code_sha": None},
            ]
        },
    }))
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "new")
    monkeypatch.setattr(
        update_receipt,
        "collect_fleet_versions",
        lambda: [{"profile": "default", "pid": 2, "state": "current", "code_sha": "new"}],
    )
    assert not update_cmd_fleet._pending_fleet_restart_needed()
    assert not update_cmd_fleet._update_owes_fleet_restart()


def test_dashboard_runtime_does_not_mask_stale_gateway(monkeypatch):
    """Skipping the dashboard row must not swallow a genuinely stale gateway (#115090)."""
    home = get_hermes_home()
    directory = home / "logs" / "update_receipts"
    directory.mkdir(parents=True)
    (directory / "latest.json").write_text(json.dumps({
        "outcome": "failed",
        "plan": {
            "runtimes": [
                {"kind": "gateway", "profile": "default", "pid": 1, "code_sha": "old"},
                {"kind": "dashboard", "profile": "default", "pid": 9, "code_sha": None},
            ]
        },
    }))
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "new")
    monkeypatch.setattr(
        update_receipt,
        "collect_fleet_versions",
        lambda: [{"profile": "default", "pid": 2, "state": "stale", "code_sha": "old"}],
    )
    assert update_cmd_fleet._pending_fleet_restart_needed()
    assert update_cmd_fleet._update_owes_fleet_restart()
