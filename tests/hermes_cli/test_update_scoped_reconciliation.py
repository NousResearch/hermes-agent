"""Marker settlement uses its own inventory, never a historical receipt's ownership."""

import json

import pytest

from hermes_cli import process_identity, update_cmd_fleet as fleet, update_receipt
from hermes_constants import get_hermes_home

MANUAL = {"kind": "serve", "profile": "work", "pid": 900, "supervisor": "manual-serve", "restart_via": "respawn-argv", "code_sha": "old", "detail": {"create_time": 1000.0}}
CURRENT = {"profile": "alpha", "state": "current", "code_sha": "new"}
GATEWAY = {"kind": "gateway", "profile": "alpha", "code_sha": "old"}

CASES = [
    ("receipt-successor", {"outcome": "failed", "plan": {"runtimes": [GATEWAY]}}, None, [CURRENT], False),
    ("marker-external-restart", {}, "new", [CURRENT], False),
    ("missing-sibling", {"outcome": "failed", "plan": {"runtimes": [GATEWAY, dict(GATEWAY, profile="beta")]}}, "new", [CURRENT], True),
    ("stale-successor", {"outcome": "failed", "plan": {"runtimes": [GATEWAY]}}, None, [dict(CURRENT, state="stale", code_sha="old")], True),
    ("unknown-successor", {"outcome": "failed", "plan": {"runtimes": [GATEWAY]}}, None, [dict(CURRENT, state="unknown")], True),
    ("marker-no-sha", {}, "", [CURRENT], True),
    ("checkout-moved", {}, "old", [CURRENT], True),
    ("marker-empty-no-receipt", {}, "new", [], True),
    ("markerless-stamped-manual", {"outcome": "partial", "plan": {"runtimes": [MANUAL]}}, None, [], False),
    ("old-manual-new-marker", {"outcome": "success", "post_update": {"sha": "old"}, "plan": {"runtimes": [MANUAL]}, "fleet": []}, "new", [], True),
    ("same-sha-not-ownership", {"outcome": "success", "post_update": {"sha": "new"}, "plan": {"runtimes": [MANUAL]}, "fleet": []}, "new", [], True),
    ("mixed-receipt-successors", {"outcome": "partial", "plan": {"runtimes": [GATEWAY, MANUAL]}, "fleet": [dict(CURRENT, state="stale", code_sha="old")]}, None, [CURRENT], False),
]


def seed(monkeypatch, old, marker, live, alive=True):
    root = get_hermes_home() / "logs" / "update_receipts"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "latest.json"
    target.write_text(json.dumps(old))
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a: alive)
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "new")
    monkeypatch.setattr("hermes_cli.update_cmd._current_checkout_sha", lambda: "new")
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **k: live)
    if marker is not None:
        fleet._write_fleet_restart_pending_marker(expected_sha=marker)
    return target


@pytest.mark.parametrize("name,old,marker,live,pending", CASES, ids=[case[0] for case in CASES])
def test_scoped_reconciliation_matrix(monkeypatch, capsys, name, old, marker, live, pending):
    live = list(live)
    target = seed(monkeypatch, old, marker, live)
    if name in ("marker-external-restart", "missing-sibling", "checkout-moved"):
        runtimes = [GATEWAY, dict(GATEWAY, profile="beta")] if name == "missing-sibling" else [GATEWAY]
        fleet._write_fleet_restart_pending_marker(expected_sha=marker, runtimes=runtimes)
    before = target.read_bytes()
    assert fleet._pending_fleet_restart_needed() is pending
    fleet._warn_pending_fleet_restart_on_startup()
    assert ("hermes gateway restart" in capsys.readouterr().err) is pending
    # Deferred catch-up rides the ordinary completion owner under PM; the marker
    # lifecycle is what the startup warning reflects here.
    assert target.read_bytes() == before
    assert fleet._fleet_restart_pending_marker_path().exists() is (marker is not None and pending)
    if name == "missing-sibling":
        live.append(dict(CURRENT, profile="beta"))
        assert not fleet._pending_fleet_restart_needed()
        assert not fleet._fleet_restart_pending_marker_path().exists()
        assert target.read_bytes() == before


@pytest.mark.parametrize("suffix", ["inventory=not-json", "expected_sha=", "inventory={}", "stray=1"])
def test_malformed_marker_stays_pending(monkeypatch, suffix):
    seed(monkeypatch, {}, "new", [CURRENT])
    marker = fleet._fleet_restart_pending_marker_path()
    with marker.open("a") as stream:
        stream.write(suffix + "\n")
    assert fleet._pending_fleet_restart_needed()
    assert marker.exists()


def test_marker_reconciliation_collects_one_live_snapshot(monkeypatch):
    seed(monkeypatch, {}, "new", [CURRENT])
    fleet._write_fleet_restart_pending_marker(expected_sha="new", runtimes=[GATEWAY])
    probes = []

    def collect(**kwargs):
        probes.append(True)
        return [CURRENT] if len(probes) == 1 else []

    monkeypatch.setattr(update_receipt, "collect_fleet_versions", collect)
    assert not fleet._pending_fleet_restart_needed()
    assert len(probes) == 1
