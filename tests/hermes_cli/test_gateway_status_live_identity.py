"""Regression tests for live gateway identity reporting."""

from __future__ import annotations

from pathlib import Path

import hermes_cli.gateway as gateway_cli


def _patch_systemd_status(monkeypatch, tmp_path: Path, *, deep: bool):
    calls = []
    unit = tmp_path / "hermes-gateway.service"
    unit.write_text("[Unit]\n", encoding="utf-8")

    monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit)
    monkeypatch.setattr(gateway_cli, "has_conflicting_systemd_units", lambda: False)
    monkeypatch.setattr(gateway_cli, "has_legacy_hermes_units", lambda: False)
    monkeypatch.setattr(gateway_cli, "systemd_unit_is_current", lambda system=False: True)
    monkeypatch.setattr(gateway_cli, "get_service_name", lambda: "hermes-gateway")
    monkeypatch.setattr(
        gateway_cli,
        "_read_systemd_unit_properties",
        lambda system=False, properties=None: {
            "ActiveState": "active",
            "SubState": "running",
            "MainPID": "4242",
        },
    )
    monkeypatch.setattr(gateway_cli, "_profile_suffix", lambda: "atlas")
    monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: tmp_path / "profiles" / "atlas")
    monkeypatch.setattr(gateway_cli, "_runtime_health_lines", lambda: [])
    monkeypatch.setattr(gateway_cli, "get_systemd_linger_status", lambda: (True, ""))
    monkeypatch.setattr(
        gateway_cli,
        "_run_systemctl",
        lambda args, **kwargs: calls.append(args),
    )
    monkeypatch.setattr(gateway_cli.subprocess, "run", lambda *args, **kwargs: calls.append(args[0]))
    return calls


def test_status_uses_structured_state_and_live_identity_without_logs(
    monkeypatch, tmp_path, capsys
):
    calls = _patch_systemd_status(monkeypatch, tmp_path, deep=False)

    gateway_cli.systemd_status(deep=False)

    output = capsys.readouterr().out
    assert "Gateway runtime identity:" in output
    assert "Host:" in output
    assert "Profile: atlas" in output
    assert "Gateway PID: 4242" in output
    assert "Service state: active (running)" in output
    assert not any(command and command[0] == "status" for command in calls)


def test_deep_status_keeps_explicit_log_view(monkeypatch, tmp_path, capsys):
    calls = _patch_systemd_status(monkeypatch, tmp_path, deep=True)

    gateway_cli.systemd_status(deep=True)

    capsys.readouterr()
    assert any(command[:2] == ["status", "hermes-gateway"] for command in calls)
    assert any(command and command[0] == "journalctl" for command in calls)
