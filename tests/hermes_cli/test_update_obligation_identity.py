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
        (home / "fleet_restart_pending").write_text("expected_sha=new\n")
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
    if bad == "marker":
        # REVERSED (lease-with-verification): a stranded marker whose expected_sha matches a
        # fully-current live fleet is DISCHARGED by that evidence — live truth outranks the
        # breadcrumb. The old assertion (marker beats live evidence) pinned the 26-day warning bug.
        try:
            assert not update_cmd_fleet._pending_fleet_restart_needed()
        finally:
            update_cmd_fleet._clear_fleet_restart_pending_marker()
    else:
        assert update_cmd_fleet._pending_fleet_restart_needed()


def test_unverifiable_row_routes_to_age_ceiling_not_forever_pin(monkeypatch):
    """M1 (wizred): a row with state='unknown'/code_sha=None is UNVERIFIABLE evidence, not
    provably stale — it must yield verdict 'unknown' (age-ceiling route), never a forever
    'stale' pin. Contrast: a present-and-different code_sha stays provably stale."""
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: "new")
    # Unverifiable row: no sha to compare.
    monkeypatch.setattr(
        update_receipt, "collect_fleet_versions",
        lambda: [{"profile": "beta", "state": "unknown", "code_sha": None}],
    )
    assert update_cmd_fleet._fleet_restart_verdict("new") == "unknown"
    # Inverse: present-and-different sha is PROVABLY stale (restart is the cure, not time).
    monkeypatch.setattr(
        update_receipt, "collect_fleet_versions",
        lambda: [{"profile": "beta", "state": "stale", "code_sha": "old"}],
    )
    assert update_cmd_fleet._fleet_restart_verdict("new") == "stale"
