"""Tests for top-level gateway cleanup during uninstall."""

from pathlib import Path
from types import SimpleNamespace

import hermes_cli.gateway as gateway_cli
import hermes_cli.uninstall as uninstall_cli


def test_uninstall_gateway_service_removes_legacy_custom_root_systemd_unit(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(gateway_cli, "find_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway_cli, "kill_gateway_processes", lambda: 0)
    monkeypatch.setattr(gateway_cli, "_systemctl_cmd", lambda system=False: ["systemctl", "--user"])

    legacy_unit = (
        os_home
        / ".config"
        / "systemd"
        / "user"
        / "hermes-gateway.service"
    )
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(uninstall_cli.subprocess, "run", fake_run)

    assert uninstall_cli.uninstall_gateway_service() is True

    assert not legacy_unit.exists()
    assert ["systemctl", "--user", "stop", "hermes-gateway"] in calls
    assert ["systemctl", "--user", "disable", "hermes-gateway"] in calls
    assert ["systemctl", "--user", "daemon-reload"] in calls


def test_uninstall_gateway_service_removes_legacy_custom_root_launchd_plist(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(gateway_cli, "find_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway_cli, "kill_gateway_processes", lambda: 0)
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: os_home)
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")

    legacy_plist = (
        os_home
        / "Library"
        / "LaunchAgents"
        / "ai.hermes.gateway.plist"
    )
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<plist><dict>"
        "<key>EnvironmentVariables</key><dict>"
        "<key>HERMES_HOME</key>"
        f"<string>{hermes_home.resolve()}</string>"
        "</dict></dict></plist>\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(uninstall_cli.subprocess, "run", fake_run)

    assert uninstall_cli.uninstall_gateway_service() is True

    assert not legacy_plist.exists()
    assert ["launchctl", "bootout", "gui/501/ai.hermes.gateway"] in calls
