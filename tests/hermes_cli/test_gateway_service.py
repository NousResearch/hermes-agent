"""Tests for gateway service management helpers."""

from types import SimpleNamespace

import hermes_cli.gateway as gateway_cli


class TestSystemdServiceRefresh:
    def test_systemd_start_refreshes_outdated_unit(self, tmp_path, monkeypatch):
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("old unit\n", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)
        monkeypatch.setattr(gateway_cli, "generate_systemd_unit", lambda system=False, run_as_user=None: "new unit\n")

        calls = []

        def fake_run(cmd, check=True, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.systemd_start()

        assert unit_path.read_text(encoding="utf-8") == "new unit\n"
        assert calls[:2] == [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "start", gateway_cli.get_service_name()],
        ]

    def test_systemd_restart_refreshes_outdated_unit(self, tmp_path, monkeypatch):
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("old unit\n", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)
        monkeypatch.setattr(gateway_cli, "generate_systemd_unit", lambda system=False, run_as_user=None: "new unit\n")

        calls = []

        def fake_run(cmd, check=True, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.systemd_restart()

        assert unit_path.read_text(encoding="utf-8") == "new unit\n"
        assert calls[:2] == [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "restart", gateway_cli.get_service_name()],
        ]


class TestGeneratedSystemdUnits:
    def test_user_unit_avoids_recursive_execstop_and_uses_extended_stop_timeout(self):
        unit = gateway_cli.generate_systemd_unit(system=False)

        assert "ExecStart=" in unit
        assert "ExecStop=" not in unit
        assert "TimeoutStopSec=60" in unit

    def test_user_unit_includes_resolved_node_directory_in_path(self, monkeypatch):
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda cmd: "/home/test/.nvm/versions/node/v24.14.0/bin/node" if cmd == "node" else None)

        unit = gateway_cli.generate_systemd_unit(system=False)

        assert "/home/test/.nvm/versions/node/v24.14.0/bin" in unit

    def test_system_unit_avoids_recursive_execstop_and_uses_extended_stop_timeout(self):
        unit = gateway_cli.generate_systemd_unit(system=True)

        assert "ExecStart=" in unit
        assert "ExecStop=" not in unit
        assert "TimeoutStopSec=60" in unit
        assert "WantedBy=multi-user.target" in unit


class TestGatewayStopCleanup:
    def test_stop_sweeps_manual_gateway_processes_after_service_stop(self, tmp_path, monkeypatch):
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("unit\n", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "is_linux", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)

        service_calls = []
        kill_calls = []

        monkeypatch.setattr(gateway_cli, "systemd_stop", lambda system=False: service_calls.append("stop"))
        monkeypatch.setattr(
            gateway_cli,
            "kill_gateway_processes",
            lambda force=False: kill_calls.append(force) or 2,
        )

        gateway_cli.gateway_command(SimpleNamespace(gateway_command="stop"))

        assert service_calls == ["stop"]
        assert kill_calls == [False]


class TestGatewayServiceDetection:
    def test_is_service_running_checks_system_scope_when_user_scope_is_inactive(self, monkeypatch):
        user_unit = SimpleNamespace(exists=lambda: True)
        system_unit = SimpleNamespace(exists=lambda: True)

        monkeypatch.setattr(gateway_cli, "is_linux", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(
            gateway_cli,
            "get_systemd_unit_path",
            lambda system=False: system_unit if system else user_unit,
        )

        def fake_run(cmd, capture_output=True, text=True, **kwargs):
            if cmd == ["systemctl", "--user", "is-active", gateway_cli.get_service_name()]:
                return SimpleNamespace(returncode=0, stdout="inactive\n", stderr="")
            if cmd == ["systemctl", "is-active", gateway_cli.get_service_name()]:
                return SimpleNamespace(returncode=0, stdout="active\n", stderr="")
            raise AssertionError(f"Unexpected command: {cmd}")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        assert gateway_cli._is_service_running() is True


class TestGatewaySystemServiceRouting:
    def test_gateway_install_passes_system_flags(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_linux", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)

        calls = []
        monkeypatch.setattr(
            gateway_cli,
            "systemd_install",
            lambda force=False, system=False, run_as_user=None: calls.append((force, system, run_as_user)),
        )

        gateway_cli.gateway_command(
            SimpleNamespace(gateway_command="install", force=True, system=True, run_as_user="alice")
        )

        assert calls == [(True, True, "alice")]

    def test_gateway_status_prefers_system_service_when_only_system_unit_exists(self, monkeypatch):
        user_unit = SimpleNamespace(exists=lambda: False)
        system_unit = SimpleNamespace(exists=lambda: True)

        monkeypatch.setattr(gateway_cli, "is_linux", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(
            gateway_cli,
            "get_systemd_unit_path",
            lambda system=False: system_unit if system else user_unit,
        )

        calls = []
        monkeypatch.setattr(gateway_cli, "systemd_status", lambda deep=False, system=False: calls.append((deep, system)))

        gateway_cli.gateway_command(SimpleNamespace(gateway_command="status", deep=False, system=False))

        assert calls == [(False, False)]
