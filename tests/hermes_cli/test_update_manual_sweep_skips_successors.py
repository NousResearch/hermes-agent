"""The macOS-26 manual sweep must not murder the restart phase's own successors.

Every ``hermes update`` on macOS 26 (launchd unmanageable, exit 5) follows the
same sequence: the launchd path drains the pre-update gateway, bootstrap fails,
and the detached fallback spawns a FRESH gateway + stderr_timestamp wrapper —
then ``_restart_manual_gateways`` runs, sees the newborn PIDs as "manual", and
SIGTERMs them as unmapped with "Restart manually". The fleet probe then finds
zero rows (nothing publishes) while rows were expected (pre snapshot saw the
old gateway), so the update exits incomplete — deterministically, every time.

The invariant pinned here: only PIDs from the pre-restart snapshot may be
stopped. A live PID absent from that snapshot was born mid-phase (fallback
spawn, watcher respawn) and is a successor, whether or not it is already
profile-mapped. Snapshot None (probe failed) keeps the old stop-everything
behavior rather than risking stale pre-update code left running.

Counterfactual: ``test_successors_are_never_stopped`` FAILS on the pre-fix
``_restart_manual_gateways`` (the newborn mapped PID gets a watcher armed and
is killed; the newborn unmapped PID lands in ``stopped_unmapped_pids``).
"""

from __future__ import annotations

import os
import types

import pytest

from hermes_cli import gateway as _gw
from hermes_cli import update_cmd_fleet as _fleet


def _outcome(pre_pids):
    return _fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=pre_pids,
        restarted_services=[], failed_or_stale_units=[],
        relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
    )


def _proc(pid):
    return types.SimpleNamespace(pid=pid, profile="default")


@pytest.fixture
def sweep_harness(monkeypatch):
    """Patch every external touchpoint of the manual sweep; return call logs."""
    calls = {"killed": [], "armed": [], "drained": []}
    monkeypatch.setattr(_gw, "_get_service_pids", lambda all_profiles=False: set())
    monkeypatch.setattr(_gw, "_wait_for_gateway_exit", lambda **kw: None)
    # `self_restart_pending` was added to the drain helper's signature after this test was
    # written; accept it so the harness tracks the real call shape.
    monkeypatch.setattr(
        _fleet, "_drain_or_signal_gateway_for_update",
        lambda pid, budget, profile, self_restart_pending=None: calls["drained"].append(pid) or True,
    )

    def _fake_kill(pid, sig):
        calls["killed"].append(pid)

    monkeypatch.setattr(os, "kill", _fake_kill)
    return calls


def test_successors_are_never_stopped(monkeypatch, sweep_harness):
    # 111 = pre-update manual; 222 = newborn successor already profile-mapped
    # (fast boot); 333 = newborn successor not yet mapped (fallback wrapper).
    monkeypatch.setattr(_gw, "find_gateway_pids", lambda **kw: {111, 222, 333})
    monkeypatch.setattr(
        _gw, "find_profile_gateway_processes",
        lambda **kw: iter([_proc(111), _proc(222)]),
    )
    monkeypatch.setattr(
        _gw, "_prepare_profile_gateway_update_restart",
        lambda profile, pid: sweep_harness["armed"].append(pid) or "detached",
    )
    out = _outcome([111])
    _fleet._restart_manual_gateways(out, 45.0)
    assert out.killed_pids == {111}
    assert out.relaunched_profiles == ["default"]
    assert out.stopped_unmapped_pids == set()
    assert sweep_harness["armed"] == [111]
    assert sweep_harness["drained"] == [111]
    assert sweep_harness["killed"] == []


def test_unknown_snapshot_stops_everything(monkeypatch, sweep_harness):
    # pre_restart snapshot failed (None): fail closed toward stopping, the
    # pre-fix behavior — a successor cannot be proven, so nothing is spared.
    # Pin scope ownership: `_scoped_manual_gateway_pids` (added after this test) leaves a PID
    # whose home cannot be read alone, which is a separate safety rule from snapshot certainty.
    # It late-imports the partitioner from its own module, so patch it THERE.
    from hermes_cli import update_fleet_scope

    monkeypatch.setattr(
        update_fleet_scope, "partition_gateway_pids_by_scope", lambda pids: (list(pids), [])
    )
    monkeypatch.setattr(_gw, "find_gateway_pids", lambda **kw: {111, 222})
    monkeypatch.setattr(
        _gw, "find_profile_gateway_processes", lambda **kw: iter([_proc(111)])
    )
    monkeypatch.setattr(
        _gw, "_prepare_profile_gateway_update_restart", lambda profile, pid: None
    )
    out = _outcome(None)
    _fleet._restart_manual_gateways(out, 45.0)
    assert out.killed_pids == {111, 222}
    assert out.stopped_unmapped_pids == {111, 222}
