"""The gateway restart marker must not erase a concurrent update's debt."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import update_cmd, update_cmd_fleet as fleet, update_host_obligation as host, update_lock


def _isolate_paths(monkeypatch, tmp_path):
    marker = tmp_path / "fleet_restart_pending"
    update_marker = tmp_path / "update-running"
    monkeypatch.setattr(host, "host_obligation_path", lambda: tmp_path / "host.json")
    monkeypatch.setattr(fleet, "_fleet_restart_pending_marker_path", lambda: marker)
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=tmp_path))
    monkeypatch.setattr(update_lock, "update_marker_path", lambda: update_marker)
    return marker, update_marker


def test_host_discharge_does_not_remove_a_replaced_obligation(monkeypatch, tmp_path):
    _isolate_paths(monkeypatch, tmp_path)
    assert host.write_host_obligation(expected_sha="old", runtimes=[])
    snapshot = (host.host_obligation_path(), host.read_host_obligation())
    assert host.write_host_obligation(expected_sha="new", runtimes=[{"kind": "gateway", "profile": "work"}])

    assert not fleet._clear_judged_fleet_marker(snapshot)
    assert host.read_host_obligation()["expected_sha"] == "new"


def test_verdict_does_not_mix_terms_and_snapshot_from_different_records(monkeypatch, tmp_path):
    _isolate_paths(monkeypatch, tmp_path)
    assert host.write_host_obligation(expected_sha="old", runtimes=[])
    monkeypatch.setattr(fleet, "_obligation_fields", lambda: {"expected_sha": "new", "inventory": "null"})

    assert fleet._marker_verdict_inputs() is None
    assert host.read_host_obligation()["expected_sha"] == "old"


def test_fallback_discharge_keeps_replaced_marker_and_new_host_record(monkeypatch, tmp_path):
    marker, _ = _isolate_paths(monkeypatch, tmp_path)
    marker.write_bytes(b"started=1\npid=1\nexpected_sha=old\n")
    snapshot = marker.read_bytes()
    # Exercise the actual fallback writer, which uses the same marker mutex as discharge.
    monkeypatch.setattr(host, "write_host_obligation", lambda **kw: False)
    monkeypatch.setattr(host, "_named_profiles_exist", lambda: False)
    assert host.arm_host_obligation(marker, expected_sha="new")
    assert not fleet._clear_judged_fleet_marker(None, snapshot)
    assert b"expected_sha=new" in marker.read_bytes()

    # A host record appearing after the fallback was read must not be hidden by a stale discharge.
    snapshot = marker.read_bytes()
    host._write_record(host.host_obligation_path(), {"version": 1, "expected_sha": "host"})
    assert not fleet._clear_judged_fleet_marker(None, snapshot)
    assert marker.exists()
    assert host.read_host_obligation()["expected_sha"] == "host"


def test_fallback_discharge_waits_for_historical_update_claim(monkeypatch, tmp_path):
    marker, update_marker = _isolate_paths(monkeypatch, tmp_path)
    marker.write_bytes(b"started=1\npid=1\nexpected_sha=old\n")
    snapshot = marker.read_bytes()
    code = """from pathlib import Path
import sys
from hermes_cli.update_lock import UpdateLock
lock = UpdateLock(path=Path(sys.argv[1]), install_root=Path(sys.argv[2]))
assert lock.acquire()
print('ready', flush=True)
sys.stdin.readline()
lock.release()
"""
    proc = subprocess.Popen([sys.executable, "-c", code, str(update_marker), str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[2], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        assert not fleet._clear_judged_fleet_marker(None, snapshot)
        assert marker.read_bytes() == snapshot
    finally:
        proc.communicate("stop\n", timeout=10)
    assert proc.returncode == 0
    assert fleet._clear_judged_fleet_marker(None, snapshot)
    assert not marker.exists()


def test_fallback_discharge_can_join_its_own_update_claim(monkeypatch, tmp_path):
    marker, update_marker = _isolate_paths(monkeypatch, tmp_path)
    marker.write_bytes(b"started=1\npid=1\nexpected_sha=old\n")
    outer = update_lock.UpdateLock(path=update_marker, install_root=tmp_path)
    assert outer.acquire()
    try:
        assert fleet._clear_judged_fleet_marker(None, marker.read_bytes())
        assert not marker.exists()
        assert update_marker.exists()  # The outer update still owns its claim.
    finally:
        outer.release()
