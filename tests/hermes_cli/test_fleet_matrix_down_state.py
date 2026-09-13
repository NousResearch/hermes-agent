"""Fleet matrix DOWN-state coverage (Phase-1 verification gap, #91277).

A gateway that was ALIVE at update start, got stopped by the restart phase,
and never came back used to produce NO matrix row at all — the check passed
silently on a fleet that lost messaging. Now it yields a `down` row and the
matrix returns True (escalate/exit 1). Rollout safety: `down` requires the
pid to be in the caller's pre-restart snapshot, so stale state files from
long-dead gateways never false-positive.
"""

import json
import os
from pathlib import Path

import hermes_cli.update_receipt as ur


def _setup(monkeypatch, tmp_path, record: dict):
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": "HEADSHA", "version": "1.0"},
    )
    monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    monkeypatch.setattr("gateway.control_socket.identify_gateway", lambda h, **k: None)
    (home / "gateway_state.json").write_text(json.dumps(record), encoding="utf-8")
    return home


_DEAD_PID = 999999899  # never a live pid


def test_killed_never_replaced_gateway_is_a_down_row(monkeypatch, tmp_path):
    _setup(
        monkeypatch,
        tmp_path,
        {"pid": _DEAD_PID, "gateway_state": "running", "kind": "hermes-gateway",
         "code_version": "0.20.5"},
    )
    fleet = ur.collect_fleet_versions(pre_restart_pids=[_DEAD_PID])
    assert len(fleet) == 1
    assert fleet[0]["state"] == "down"
    assert fleet[0]["pid"] == _DEAD_PID
    # and the matrix escalates on it
    assert ur.print_fleet_version_matrix(fleet) is True


def test_dead_pid_not_in_pre_restart_snapshot_keeps_no_row(monkeypatch, tmp_path):
    """Stale state file from a long-dead gateway: rollout-safe no-row."""
    _setup(
        monkeypatch,
        tmp_path,
        {"pid": _DEAD_PID, "gateway_state": "running", "kind": "hermes-gateway"},
    )
    assert ur.collect_fleet_versions(pre_restart_pids=[12345]) == []
    assert ur.collect_fleet_versions(pre_restart_pids=None) == []
    assert ur.collect_fleet_versions() == []


def test_cleanly_stopped_gateway_is_not_down(monkeypatch, tmp_path):
    for benign_state in ("stopped", "startup_failed"):
        _setup(
            monkeypatch,
            tmp_path,
            {"pid": _DEAD_PID, "gateway_state": benign_state, "kind": "hermes-gateway"},
        )
        assert ur.collect_fleet_versions(pre_restart_pids=[_DEAD_PID]) == []


def test_recycled_pid_is_not_reported_stale(monkeypatch, tmp_path):
    """A dead gateway's PID reused by an unrelated process (#93258) must not
    be reported STALE just because *some* process now answers to that PID.
    """
    from gateway.status import _get_process_start_time

    reused_pid = os.getpid()
    wrong_start_time = (_get_process_start_time(reused_pid) or 0) + 12345
    _setup(
        monkeypatch,
        tmp_path,
        {
            "pid": reused_pid,
            "start_time": wrong_start_time,
            "gateway_state": "running",
            "code_sha": "OLDSHA",
            "kind": "hermes-gateway",
        },
    )
    fleet = ur.collect_fleet_versions(pre_restart_pids=[reused_pid])
    assert len(fleet) == 1
    assert fleet[0]["state"] == "down"


def test_matching_start_time_is_still_live(monkeypatch, tmp_path):
    """A record whose start_time matches the live process is not recycled."""
    from gateway.status import _get_process_start_time

    pid = os.getpid()
    _setup(
        monkeypatch,
        tmp_path,
        {
            "pid": pid,
            "start_time": _get_process_start_time(pid),
            "gateway_state": "running",
            "code_sha": "HEADSHA",
            "kind": "hermes-gateway",
        },
    )
    fleet = ur.collect_fleet_versions(pre_restart_pids=[pid])
    assert len(fleet) == 1
    assert fleet[0]["state"] == "current"


def test_live_gateway_rows_unchanged(monkeypatch, tmp_path):
    """The live-pid path is untouched by the down-state addition."""
    _setup(
        monkeypatch,
        tmp_path,
        {"pid": os.getpid(), "gateway_state": "running", "code_sha": "HEADSHA",
         "kind": "hermes-gateway"},
    )
    fleet = ur.collect_fleet_versions(pre_restart_pids=[os.getpid()])
    assert len(fleet) == 1
    assert fleet[0]["state"] == "current"
    assert ur.print_fleet_version_matrix(fleet) is False


def test_down_and_stale_both_escalate_with_remediation(capsys):
    fleet = [
        {"profile": "default", "pid": 1, "code_sha": "OLD", "state": "stale"},
        {"profile": "coder", "pid": 2, "code_sha": None, "state": "down"},
    ]
    assert ur.print_fleet_version_matrix(fleet) is True
    out = capsys.readouterr().out
    assert "STALE" in out
    assert "DOWN" in out
    assert "hermes gateway restart" in out
