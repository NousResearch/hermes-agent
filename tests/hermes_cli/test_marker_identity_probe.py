"""A marker's profile coverage cannot hide its still-running recorded process (#118013)."""

import json
import subprocess
import sys
from types import SimpleNamespace

import psutil

from hermes_cli import update_cmd_fleet, update_host_obligation, update_receipt
from hermes_constants import get_hermes_home


def test_live_original_gateway_keeps_marker_despite_current_successor(monkeypatch):
    old_sha, new_sha = "a" * 40, "b" * 40
    old = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    new = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert old.poll() is None and new.poll() is None
        # The recorded runtime still reports the pre-update code.
        status = get_hermes_home() / "gateway_state.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps({"pid": old.pid, "code_sha": old_sha,
                                      "gateway_state": "running"}), encoding="utf-8")
        update_cmd_fleet._write_fleet_restart_pending_marker(
            expected_sha=new_sha,
            runtimes=[{"kind": "gateway", "profile": "default", "pid": old.pid,
                       "code_sha": old_sha}],
        )
        marker = update_host_obligation.host_obligation_path()
        assert marker is not None and marker.exists()

        monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: new_sha)
        # A verified new socket identity can win discovery for this profile,
        # leaving the still-live old runtime outside the one-row matrix.
        current_row = update_receipt._fleet_row(
            "default", new.pid, new_sha, "test", new_sha,
        )
        monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda: [current_row])

        assert update_cmd_fleet._marker_only_restart_obsolete() is False, (
            "marker was discharged while its recorded old-SHA runtime was still alive"
        )
        assert marker.exists()
        old.terminate()
        old.wait(timeout=5)
        assert new.poll() is None
        assert update_cmd_fleet._marker_only_restart_obsolete() is True
        assert not marker.exists()
    finally:
        for process in (old, new):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


def test_recycled_recorded_pid_discharge_requires_new_incarnation(monkeypatch):
    old_sha, new_sha = "a" * 40, "b" * 40
    recorded_pid = 42
    update_cmd_fleet._write_fleet_restart_pending_marker(
        expected_sha=new_sha,
        runtimes=[{"kind": "gateway", "profile": "default", "pid": recorded_pid,
                   "code_sha": old_sha}],
    )
    marker = update_host_obligation.host_obligation_path()
    assert marker is not None and marker.exists()
    started = update_host_obligation.read_host_obligation()["started"]
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: new_sha)
    current_row = update_receipt._fleet_row(
        "default", recorded_pid, new_sha, "test", new_sha,
    )
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda: [current_row])

    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: started - 1))
    assert update_cmd_fleet._marker_only_restart_obsolete() is False, (
        "marker was discharged while the recorded PID still named its old incarnation"
    )
    assert marker.exists()

    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: started + 1))
    assert update_cmd_fleet._marker_only_restart_obsolete() is True
    assert not marker.exists()
