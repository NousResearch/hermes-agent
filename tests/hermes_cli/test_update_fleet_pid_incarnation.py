"""PID-incarnation regressions for the fleet update proof surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import hermes_cli.update_inventory as update_inventory
import hermes_cli.update_receipt as update_receipt


_EXPECTED_SHA = "a" * 40


def _write_gateway_state(home: Path, *, start_time: int) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "gateway_state.json").write_text(
        json.dumps(
            {
                "pid": 4242,
                "start_time": start_time,
                "code_sha": _EXPECTED_SHA,
                "code_version": "1.0",
            }
        ),
        encoding="utf-8",
    )


def _bind_one_profile(monkeypatch, tmp_path: Path, *, live_start_time: int) -> Path:
    home = tmp_path / "home"
    _write_gateway_state(home, start_time=111)

    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "profiles"
    )
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: pid == 4242)
    monkeypatch.setattr(
        "gateway.status._get_process_start_time", lambda pid: live_start_time
    )
    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {
            "sha": _EXPECTED_SHA,
            "short_sha": _EXPECTED_SHA[:8],
            "version": "1.0",
            "source": "git",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._get_service_pids", lambda all_profiles=False: set()
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.find_profile_gateway_processes", lambda: []
    )
    monkeypatch.setattr(
        "hermes_cli.config.detect_install_method", lambda *args, **kwargs: "git"
    )
    monkeypatch.setattr("hermes_cli.config.get_managed_system", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.config.recommended_update_command_for_method",
        lambda method: "hermes update",
    )
    return home


def test_recycled_pid_is_excluded_from_both_fleet_proof_surfaces(
    monkeypatch, tmp_path
):
    _bind_one_profile(monkeypatch, tmp_path, live_start_time=222)

    assert update_receipt.collect_fleet_versions() == []
    assert update_inventory.collect_runtime_inventory().runtimes == []


def test_matching_pid_incarnation_is_admitted_by_both_fleet_proof_surfaces(
    monkeypatch, tmp_path
):
    _bind_one_profile(monkeypatch, tmp_path, live_start_time=111)

    fleet = update_receipt.collect_fleet_versions()
    assert [(entry["pid"], entry["state"]) for entry in fleet] == [(4242, "current")]

    inventory = update_inventory.collect_runtime_inventory()
    assert [(runtime.pid, runtime.code_sha) for runtime in inventory.runtimes] == [
        (4242, _EXPECTED_SHA)
    ]
