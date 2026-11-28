"""Owed-restart duplicate obligation fixtures (mega-PR Target 2).

Native ground: Ara's Nov 6 record carried four dead entries — PID 57643 recorded
twice, once mapped and once as stopped_unmapped — and the standing obligation
survived the repair that should have discharged it.

These fixtures encode the repair contract against the CURRENT base (which does
not deduplicate): new records must not acquire a duplicate unmapped row for an
already-mapped PID, and existing duplicate records must discharge once the
mapped successor is verified current. Behavior specified alongside (not
against) NousResearch/hermes-agent#147888, which proposes the same read-time
and write-time deduplication; if that PR lands first these fixtures stand as
its acceptance coverage.
"""

from __future__ import annotations

import json

from hermes_cli import update_cmd_fleet as fleet
from hermes_cli import update_cmd_fleet_verify as fleet_verify
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan

SHA = "bf0203a794a07aac3ff764972ab9087baa4e4f4e"


def _armed_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: SHA)
    fleet._write_fleet_restart_pending_marker(expected_sha=SHA)


def _inventory():
    return json.loads(fleet._obligation_fields()["inventory"])


def test_write_time_drops_unmapped_duplicate_of_mapped_pid(tmp_path, monkeypatch):
    _armed_lock(tmp_path, monkeypatch)
    plan = UpdatePlan(runtimes=[RuntimeRecord(kind="gateway", profile="default", pid=912)])
    fleet_verify._record_owed_gateway_inventory(plan, stopped_unmapped_pids={912, 999})
    runtimes = _inventory()["runtimes"]
    assert {"kind": "gateway", "profile": None, "pid": 912, "stopped_unmapped": True} not in runtimes
    assert {"kind": "gateway", "profile": None, "pid": 999, "stopped_unmapped": True} in runtimes


def test_read_time_ignores_unmapped_duplicate_of_mapped_pid(tmp_path, monkeypatch):
    from hermes_cli import update_cmd_fleet as fleet

    _armed_lock(tmp_path, monkeypatch)
    inventory = {
        "version": 1,
        "runtimes": [
            {"kind": "gateway", "profile": "default", "pid": 912},
            {"kind": "gateway", "profile": None, "pid": 912, "stopped_unmapped": True},
        ]
    }
    owed = fleet._marker_owed_gateways(inventory)
    assert owed is not None
    assert ("gateway", "default") in owed
    # The duplicate unmapped row for the already-mapped PID names no second
    # obligation: the ("gateway", "") marker must not appear for it.
    assert ("gateway", "") not in owed


def test_read_time_keeps_genuinely_unmapped_pid_owed(tmp_path, monkeypatch):
    from hermes_cli import update_cmd_fleet as fleet

    _armed_lock(tmp_path, monkeypatch)
    inventory = {
        "version": 1,
        "runtimes": [
            {"kind": "gateway", "profile": "default", "pid": 912},
            {"kind": "gateway", "profile": None, "pid": 999, "stopped_unmapped": True},
        ]
    }
    owed = fleet._marker_owed_gateways(inventory)
    assert owed is not None
    assert ("gateway", "") in owed


def _duplicate_marker(tmp_path, monkeypatch):
    from hermes_cli import update_cmd_fleet as fleet

    _armed_lock(tmp_path, monkeypatch)
    fleet._write_fleet_restart_pending_marker(
        expected_sha=SHA,
        runtimes=[
            {"kind": "gateway", "profile": "default", "pid": 733},
            {"kind": "gateway", "profile": None, "pid": 733, "stopped_unmapped": True},
        ],
    )


def test_duplicate_record_discharges_once_mapped_successor_current(tmp_path, monkeypatch):
    from hermes_cli import update_cmd_fleet as fleet

    _duplicate_marker(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda: [{"kind": "gateway", "profile": "default", "pid": 8101,
                  "state": "current", "code_sha": SHA}],
    )
    assert fleet._marker_only_restart_obsolete() is True
    assert not fleet._fleet_restart_obligation_armed()


def test_duplicate_record_stays_owed_when_successor_stale(tmp_path, monkeypatch):
    from hermes_cli import update_cmd_fleet as fleet

    _duplicate_marker(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda: [{"kind": "gateway", "profile": "default", "pid": 8101,
                  "state": "current", "code_sha": "deadbeef" * 5}],
    )
    assert fleet._marker_only_restart_obsolete() is False
    assert fleet._fleet_restart_obligation_armed()
