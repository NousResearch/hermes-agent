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
    # These unfinished receipts require the same evidence for startup and deferred catch-up.
    fleet._apply_pending_fleet_restart_catchup(defer=True)
    assert ("fleet restart deferred" in capsys.readouterr().out) is pending
    assert target.read_bytes() == before
    assert fleet._fleet_restart_pending_marker_path().exists() is (marker is not None and pending)
    if name == "missing-sibling":
        live.append(dict(CURRENT, profile="beta"))
        assert not fleet._pending_fleet_restart_needed()
        assert not fleet._fleet_restart_pending_marker_path().exists()
        assert target.read_bytes() == before


@pytest.mark.parametrize("alive", [True, False, None], ids=["alive", "dead", "unknown"])
@pytest.mark.parametrize("sha", ["old", "new", None], ids=["different-sha", "same-sha", "no-sha"])
def test_empty_marker_never_inherits_receipt_ownership(monkeypatch, capsys, alive, sha):
    old = {"outcome": "success", "plan": {"runtimes": [MANUAL]}, "post_update": {"sha": sha}}
    target = seed(monkeypatch, old, "new", [], alive)
    before = target.read_bytes()
    fleet._warn_pending_fleet_restart_on_startup()
    warning = capsys.readouterr().err
    assert "hermes gateway restart" in warning
    assert ("serve [work] pid 900" in warning) is (alive is not False)
    assert fleet._fleet_restart_pending_marker_path().exists()
    assert target.read_bytes() == before


@pytest.mark.parametrize("consumer", ["predicate", "startup"])
def test_reconciliation_uses_one_receipt_snapshot(monkeypatch, capsys, consumer):
    old = {"outcome": "partial", "plan": {"runtimes": [MANUAL]}, "fleet": []}
    seed(monkeypatch, old, None, [])
    reads = []

    def read_rotating_receipt():
        reads.append(True)
        return old if len(reads) == 1 else {"outcome": "failed", "plan": {"runtimes": [GATEWAY]}}

    monkeypatch.setattr(update_receipt, "read_latest_receipt", read_rotating_receipt)
    if consumer == "predicate":
        assert not fleet._pending_fleet_restart_needed()
    else:
        fleet._warn_pending_fleet_restart_on_startup()
        warning = capsys.readouterr().err
        assert "hermes gateway restart" not in warning
        assert "serve [work] pid 900" in warning
    assert len(reads) == 1
    assert list((get_hermes_home() / "serve_restart_pending").glob("*.json"))


@pytest.mark.parametrize("marker", [None, "new"])
@pytest.mark.parametrize("blocked_storage", [False, True])
def test_probe_exception_does_not_hide_manual_warning(monkeypatch, capsys, marker, blocked_storage):
    old = {"outcome": "partial", "plan": {"runtimes": [GATEWAY, MANUAL]}}
    target = seed(monkeypatch, old, marker, [])
    if marker is not None:
        fleet._write_fleet_restart_pending_marker(expected_sha=marker, runtimes=[GATEWAY])
    before = target.read_bytes()
    if blocked_storage:
        (get_hermes_home() / "serve_restart_pending").write_text("not a directory")

    def unavailable(**kwargs):
        raise OSError("gateway probe unavailable")

    monkeypatch.setattr(update_receipt, "collect_fleet_versions", unavailable)
    fleet._warn_pending_fleet_restart_on_startup()
    warning = capsys.readouterr().err
    assert "hermes gateway restart" in warning
    assert "serve [work] pid 900" in warning
    assert ("could not be saved" in warning) is blocked_storage
    assert target.read_bytes() == before


@pytest.mark.parametrize("completed_restart", [False, True])
def test_new_marker_cannot_borrow_old_alpha_receipt(monkeypatch, capsys, completed_restart):
    """N owns alpha; N+1 owns alpha and beta but dies before writing its receipt."""
    old = {"outcome": "failed", "plan": {"runtimes": [GATEWAY]}}
    if completed_restart:
        old.update(post_update={"sha": "new"}, gateway_restart={"incomplete": False})
    live = [CURRENT]
    target = seed(monkeypatch, old, "new", live)
    marker = fleet._fleet_restart_pending_marker_path()
    with marker.open("a") as stream:
        stream.write("inventory=" + json.dumps({"version": 1, "runtimes": [GATEWAY, dict(GATEWAY, profile="beta")]}) + "\n")
    receipt_before, marker_before = target.read_bytes(), marker.read_bytes()
    fleet._warn_pending_fleet_restart_on_startup()
    assert "hermes gateway restart" in capsys.readouterr().err
    assert fleet._pending_fleet_restart_needed()
    fleet._apply_pending_fleet_restart_catchup(defer=True)
    assert "fleet restart deferred" in capsys.readouterr().out
    assert marker.read_bytes() == marker_before
    assert target.read_bytes() == receipt_before
    live.append(dict(CURRENT, profile="beta"))
    assert not fleet._pending_fleet_restart_needed()
    assert not marker.exists()
    assert target.read_bytes() == receipt_before


@pytest.mark.parametrize("completed_restart", [False, True])
def test_legacy_marker_discharges_on_live_fleet_evidence_without_receipt(monkeypatch, capsys, completed_restart):
    """An inventory-less N+1 marker settles on live-fleet evidence alone (#115638).

    It never borrows the old receipt's ownership: the receipt is left intact and the
    marker discharges only because every live row is current at its expected SHA.
    """
    old = {"outcome": "failed", "plan": {"runtimes": [GATEWAY]}}
    if completed_restart:
        old.update(post_update={"sha": "new"}, gateway_restart={"incomplete": False})
    live = [CURRENT]
    target = seed(monkeypatch, old, "new", live)
    marker = fleet._fleet_restart_pending_marker_path()
    receipt_before = target.read_bytes()
    fleet._warn_pending_fleet_restart_on_startup()
    assert "hermes gateway restart" not in capsys.readouterr().err
    assert not fleet._pending_fleet_restart_needed()
    assert not marker.exists()
    assert target.read_bytes() == receipt_before


# Explicit `inventory=null` records no obligation either, so it settles on live-fleet
# evidence like a missing line (#115638); the variants below stay fail-closed.
@pytest.mark.parametrize("inventory", [{}, [], {"version": 2, "runtimes": [GATEWAY]}, {"version": 1, "runtimes": []}, {"version": 1, "runtimes": [GATEWAY, dict(MANUAL, detail={})]}, {"version": 1, "runtimes": [None]}, {"version": 1, "runtimes": [{"kind": "gateway", "profile": "unknown"}]}, {"version": 1, "runtimes": [{"kind": "gateway", "profile": []}]}])
def test_unverified_marker_inventory_stays_pending(monkeypatch, inventory):
    seed(monkeypatch, {"outcome": "success", "plan": {"runtimes": [GATEWAY]}}, "new", [CURRENT])
    marker = fleet._fleet_restart_pending_marker_path()
    with marker.open("a") as stream:
        stream.write("inventory=" + json.dumps(inventory) + "\n")
    before = marker.read_bytes()
    assert fleet._pending_fleet_restart_needed()
    monkeypatch.setattr("hermes_cli.update_cmd._run_pending_fleet_restart", lambda: True)
    with pytest.raises(SystemExit) as exc:
        fleet._apply_pending_fleet_restart_catchup()
    assert exc.value.code == 1
    assert marker.read_bytes() == before


@pytest.mark.parametrize("suffix", ['inventory={', 'inventory=null\ninventory={"version":1,"runtimes":[]}', 'broken-line'])
def test_malformed_marker_stays_pending(monkeypatch, suffix):
    seed(monkeypatch, {}, "new", [CURRENT])
    marker = fleet._fleet_restart_pending_marker_path()
    with marker.open("a") as stream:
        stream.write(suffix + "\n")
    assert fleet._pending_fleet_restart_needed()
    assert marker.exists()


def test_pulled_update_marker_owns_pre_update_inventory(monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import update_cmd
    from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan

    old = {"outcome": "failed", "plan": {"runtimes": [GATEWAY]}}
    target = seed(monkeypatch, old, None, [CURRENT])
    before = target.read_bytes()
    plan = UpdatePlan(runtimes=[RuntimeRecord(kind="gateway", profile=p) for p in ("alpha", "beta")])
    monkeypatch.setattr(update_cmd, "_invalidate_update_cache", lambda: None)
    monkeypatch.setattr(update_cmd, "_verify_head_after_pull", lambda *a, **k: "new")

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(update_cmd, "_sweep_bytecode_after_update", interrupt)
    with pytest.raises(KeyboardInterrupt):
        update_cmd._apply_pulled_update([], "main", "old", SimpleNamespace(in_place_update=True), None, gateway_mode=False, is_fork=False, desktop_dir=None, had_desktop_app_before_update=False, pre_update_snapshot_id=None, _pre_update_plan=plan, _windows_gateway_resume=None, args=SimpleNamespace())
    marker = fleet._fleet_restart_pending_marker_path()
    fields = dict(line.split("=", 1) for line in marker.read_text().splitlines())
    assert fields["expected_sha"] == "new"
    assert json.loads(fields["inventory"]) == {"version": 1, "runtimes": plan.to_dict()["runtimes"]}
    assert fleet._pending_fleet_restart_needed()
    assert target.read_bytes() == before


@pytest.mark.parametrize("successor", ["missing", "stale", "unknown", "probe-error", "current"])
def test_catchup_verifies_owned_fleet_after_restart(monkeypatch, successor):
    from hermes_cli import update_cmd

    old = {"outcome": "failed", "plan": {"runtimes": [GATEWAY]}}
    live = [CURRENT]
    target = seed(monkeypatch, old, "new", live)
    fleet._write_fleet_restart_pending_marker(expected_sha="new", runtimes=[GATEWAY, dict(GATEWAY, profile="beta")])
    marker = fleet._fleet_restart_pending_marker_path()
    receipt_before, marker_before = target.read_bytes(), marker.read_bytes()
    restarted = []

    def restart():
        restarted.append("alpha")
        live[:] = [CURRENT]
        if successor not in ("missing", "probe-error"):
            live.append(dict(CURRENT, profile="beta", state=successor, code_sha="old" if successor == "stale" else "new"))
        return True

    def collect(**kwargs):
        if restarted and successor == "probe-error":
            raise OSError("fleet unavailable")
        return live

    monkeypatch.setattr(update_cmd, "_run_pending_fleet_restart", restart)
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", collect)
    if successor == "current":
        fleet._apply_pending_fleet_restart_catchup()
        assert not marker.exists()
    else:
        with pytest.raises(SystemExit) as exc:
            fleet._apply_pending_fleet_restart_catchup()
        assert exc.value.code == 1
        assert marker.read_bytes() == marker_before
        live[:] = [CURRENT, dict(CURRENT, profile="beta")]
        monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **k: live)
        fleet._apply_pending_fleet_restart_catchup()
        assert not marker.exists()
    assert restarted == ["alpha"]
    assert target.read_bytes() == receipt_before


@pytest.mark.parametrize("blocked_storage", [False, True])
def test_verified_restart_surviving_marker_preserves_manual_debt(monkeypatch, capsys, blocked_storage):
    from hermes_cli import update_cmd

    manual = [MANUAL, dict(MANUAL, pid=901)]
    old = {"outcome": "partial", "post_update": {"sha": "new"}, "gateway_restart": {"incomplete": False, "phase_error": ""}, "plan": {"runtimes": [GATEWAY, *manual]}, "fleet": [CURRENT]}
    target = seed(monkeypatch, old, "new", [CURRENT])
    fleet._write_fleet_restart_pending_marker(expected_sha="new", runtimes=[GATEWAY, *manual])
    marker = fleet._fleet_restart_pending_marker_path()
    receipt_before, marker_before = target.read_bytes(), marker.read_bytes()
    directory = get_hermes_home() / "serve_restart_pending"
    if blocked_storage:
        directory.write_text("not a directory")
    restarted = []
    monkeypatch.setattr(update_cmd, "_run_pending_fleet_restart", lambda: restarted.append(True) or True)
    if blocked_storage:
        with pytest.raises(SystemExit) as exc:
            fleet._apply_pending_fleet_restart_catchup()
        assert exc.value.code == 1
        assert marker.read_bytes() == marker_before
        directory.unlink()
        restarted.clear()
    fleet._apply_pending_fleet_restart_catchup()
    assert not restarted
    assert not marker.exists()
    assert {json.loads(path.read_text())["pid"] for path in directory.glob("*.json")} == {900, 901}
    fleet._warn_pending_fleet_restart_on_startup()
    warning = capsys.readouterr().err
    assert "hermes gateway restart" not in warning
    assert "pid 900" in warning and "pid 901" in warning
    assert target.read_bytes() == receipt_before


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
