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
        # Recorded obligation for an SHA the fleet does not serve: no verified discharge.
        inventory = json.dumps({"version": 1, "runtimes": [{"kind": "gateway", "profile": "beta"}]})
        (home / "fleet_restart_pending").write_text(f"expected_sha=future\ninventory={inventory}\n")
    owed = {"kind": "gateway", "profile": "beta", "code_sha": "old"}
    if bad == "wrong-kind":
        owed["kind"] = "serve"
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


@pytest.mark.parametrize("supervisor", ["desktop", "launchd", "systemd"])
def test_supervised_serve_row_never_vetoes_gateway_discharge(monkeypatch, supervisor):
    """A host that also runs a supervised ``hermes dashboard``/``serve`` carries that row in every
    receipt; it must not make gateway coverage unverifiable (#115090). The gateways restarted at
    the pulled SHA and the fleet has since moved on to a newer checkout: nothing is owed.
    """
    home = get_hermes_home()
    directory = home / "logs" / "update_receipts"
    directory.mkdir(parents=True)
    (directory / "latest.json").write_text(json.dumps({
        "outcome": "failed",
        "post_update": {"sha": "pulled"},
        "gateway_restart": {"restarted_services": ["hermes-gateway"], "incomplete": False},
        "plan": {"runtimes": [
            {"kind": "gateway", "profile": "default", "code_sha": "old", "supervisor": "manual"},
            {"kind": "dashboard", "profile": "default", "code_sha": None, "supervisor": supervisor, "restart_via": "respawn-argv"},
        ]},
    }))
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "new")
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda: [{"profile": "default", "state": "current", "code_sha": "new"}])
    assert not update_cmd_fleet._update_owes_fleet_restart()
    assert not update_cmd_fleet._pending_fleet_restart_needed()
