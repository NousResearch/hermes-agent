"""Tests for gateway service management helpers."""

from types import SimpleNamespace

import hermes_cli.gateway as gateway_cli


class TestSystemdServiceRefresh:
    def test_systemd_start_refreshes_outdated_unit(self, tmp_path, monkeypatch):
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("old unit\n", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda: unit_path)
        monkeypatch.setattr(gateway_cli, "generate_systemd_unit", lambda: "new unit\n")

        calls = []

        def fake_run(cmd, check=True, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.systemd_start()

        assert unit_path.read_text(encoding="utf-8") == "new unit\n"
        assert calls[:2] == [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "start", gateway_cli.SERVICE_NAME],
        ]

    def test_systemd_restart_refreshes_outdated_unit(self, tmp_path, monkeypatch):
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("old unit\n", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda: unit_path)
        monkeypatch.setattr(gateway_cli, "generate_systemd_unit", lambda: "new unit\n")

        calls = []

        def fake_run(cmd, check=True, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.systemd_restart()

        assert unit_path.read_text(encoding="utf-8") == "new unit\n"
        assert calls[:2] == [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "restart", gateway_cli.SERVICE_NAME],
        ]


class TestGatewayStopCleanup:
    def test_stop_sweeps_manual_gateway_processes_after_service_stop(self, tmp_path, monkeypatch):
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("unit\n", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "is_linux", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda: unit_path)

        service_calls = []
        kill_calls = []

        monkeypatch.setattr(gateway_cli, "systemd_stop", lambda: service_calls.append("stop"))
        monkeypatch.setattr(
            gateway_cli,
            "kill_gateway_processes",
            lambda force=False: kill_calls.append(force) or 2,
        )

        gateway_cli.gateway_command(SimpleNamespace(gateway_command="stop"))

        assert service_calls == ["stop"]
        assert kill_calls == [False]
