"""Tests for gateway service management helpers."""

import os
from types import SimpleNamespace

import hermes_cli.gateway as gateway_cli


class TestSystemdServiceRefresh:
    def test_systemd_install_repairs_outdated_unit_without_force(self, tmp_path, monkeypatch):
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("old unit\n", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)
        monkeypatch.setattr(gateway_cli, "generate_systemd_unit", lambda system=False, run_as_user=None: "new unit\n")

        calls = []

        def fake_run(cmd, check=True, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.systemd_install()

        assert unit_path.read_text(encoding="utf-8") == "new unit\n"
        assert calls[:2] == [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "enable", gateway_cli.get_service_name()],
        ]

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


class TestLaunchdServiceRecovery:
    def test_launchd_install_repairs_outdated_plist_without_force(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist>old content</plist>", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        calls = []

        def fake_run(cmd, check=False, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.launchd_install()

        assert "--replace" in plist_path.read_text(encoding="utf-8")
        assert calls[:2] == [
            ["launchctl", "unload", str(plist_path)],
            ["launchctl", "load", str(plist_path)],
        ]

    def test_launchd_start_reloads_unloaded_job_and_retries(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text(gateway_cli.generate_launchd_plist(), encoding="utf-8")

        calls = []

        def fake_run(cmd, check=False, **kwargs):
            calls.append(cmd)
            if cmd == ["launchctl", "start", "ai.hermes.gateway"] and calls.count(cmd) == 1:
                raise gateway_cli.subprocess.CalledProcessError(3, cmd, stderr="Could not find service")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.launchd_start()

        assert calls == [
            ["launchctl", "start", "ai.hermes.gateway"],
            ["launchctl", "load", str(plist_path)],
            ["launchctl", "start", "ai.hermes.gateway"],
        ]

    def test_launchd_status_reports_local_stale_plist_when_unloaded(self, tmp_path, monkeypatch, capsys):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist>old content</plist>", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=113, stdout="", stderr="Could not find service"),
        )

        gateway_cli.launchd_status()

        output = capsys.readouterr().out
        assert str(plist_path) in output
        assert "stale" in output.lower()
        assert "not loaded" in output.lower()


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

    def test_gateway_restart_does_not_fallback_to_foreground_when_launchd_restart_fails(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("plist\n", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "is_linux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(
            gateway_cli,
            "launchd_restart",
            lambda: (_ for _ in ()).throw(
                gateway_cli.subprocess.CalledProcessError(5, ["launchctl", "start", "ai.hermes.gateway"])
            ),
        )

        run_calls = []
        monkeypatch.setattr(gateway_cli, "run_gateway", lambda verbose=False, replace=False: run_calls.append((verbose, replace)))
        monkeypatch.setattr(gateway_cli, "kill_gateway_processes", lambda force=False: 0)

        try:
            gateway_cli.gateway_command(SimpleNamespace(gateway_command="restart", system=False))
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("Expected gateway_command to exit when service restart fails")

        assert run_calls == []


class TestEnsureUserSystemdEnv:
    """Tests for _ensure_user_systemd_env() D-Bus session bus auto-detection."""

    def test_sets_xdg_runtime_dir_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setattr(os, "getuid", lambda: 42)

        # Patch Path so /run/user/42 resolves to our tmp dir (which exists)
        from pathlib import Path as RealPath

        class FakePath(type(RealPath())):
            def __new__(cls, *args):
                p = str(args[0]) if args else ""
                if p == "/run/user/42":
                    return RealPath.__new__(cls, str(tmp_path))
                return RealPath.__new__(cls, *args)

        monkeypatch.setattr(gateway_cli, "Path", FakePath)

        gateway_cli._ensure_user_systemd_env()

        # Function sets the canonical string, not the fake path
        assert os.environ.get("XDG_RUNTIME_DIR") == "/run/user/42"

    def test_sets_dbus_address_when_bus_socket_exists(self, tmp_path, monkeypatch):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        bus_socket = runtime / "bus"
        bus_socket.touch()  # simulate the socket file

        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setattr(os, "getuid", lambda: 99)

        gateway_cli._ensure_user_systemd_env()

        assert os.environ["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={bus_socket}"

    def test_preserves_existing_env_vars(self, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/custom/runtime")
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/custom/bus")

        gateway_cli._ensure_user_systemd_env()

        assert os.environ["XDG_RUNTIME_DIR"] == "/custom/runtime"
        assert os.environ["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/custom/bus"

    def test_no_dbus_when_bus_socket_missing(self, tmp_path, monkeypatch):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        # no bus socket created

        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setattr(os, "getuid", lambda: 99)

        gateway_cli._ensure_user_systemd_env()

        assert "DBUS_SESSION_BUS_ADDRESS" not in os.environ

    def test_systemctl_cmd_calls_ensure_for_user_mode(self, monkeypatch):
        calls = []
        monkeypatch.setattr(gateway_cli, "_ensure_user_systemd_env", lambda: calls.append("called"))

        result = gateway_cli._systemctl_cmd(system=False)
        assert result == ["systemctl", "--user"]
        assert calls == ["called"]

    def test_systemctl_cmd_skips_ensure_for_system_mode(self, monkeypatch):
        calls = []
        monkeypatch.setattr(gateway_cli, "_ensure_user_systemd_env", lambda: calls.append("called"))

        result = gateway_cli._systemctl_cmd(system=True)
        assert result == ["systemctl"]
        assert calls == []
