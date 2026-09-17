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


def test_non_gateway_runtime_neither_owes_nor_vetoes_gateway_coverage(monkeypatch):
    """A recorded serve/dashboard is not a gateway identity: it does not veto the gateway question.

    Replaces the old ``wrong-kind`` case, which asserted the opposite. The receipt written by a
    deferred ``hermes update --no-gateway-restart`` records the managed dashboard next to the
    gateway, and that one entry used to return ``None`` for the whole receipt — leaving the warning
    armed on a fleet that is provably current (#107402, #107817).
    """
    home = get_hermes_home()
    (home / "fleet_restart_pending").unlink(missing_ok=True)
    directory = home / "logs" / "update_receipts"
    directory.mkdir(parents=True)
    (directory / "latest.json").write_text(
        json.dumps({
            "outcome": "failed",
            "plan": {
                "runtimes": [
                    {"kind": "gateway", "profile": "alpha", "code_sha": "old"},
                    {"kind": "serve", "profile": "alpha", "pid": 5555, "code_sha": None},
                    {"kind": "dashboard", "profile": "alpha", "pid": 5556, "code_sha": None},
                ]
            },
        })
    )
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "new")
    monkeypatch.setattr(
        update_receipt,
        "collect_fleet_versions",
        lambda: [{"profile": "alpha", "state": "current", "code_sha": "new"}],
    )
    assert not update_cmd_fleet._pending_fleet_restart_needed()
