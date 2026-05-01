"""Tests for gateway service management helpers."""

import os
from pathlib import Path
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
        label = gateway_cli.get_launchd_label()

        calls = []

        def fake_run(cmd, check=False, **kwargs):
            calls.append(cmd)
            if cmd == ["launchctl", "start", label] and calls.count(cmd) == 1:
                raise gateway_cli.subprocess.CalledProcessError(3, cmd, stderr="Could not find service")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.launchd_start()

        assert calls == [
            ["launchctl", "start", label],
            ["launchctl", "load", str(plist_path)],
            ["launchctl", "start", label],
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

    def test_find_gateway_pids_windows_detects_gateway_module_run(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_windows", lambda: True)
        monkeypatch.setattr(gateway_cli.os, "getpid", lambda: 99999)

        def fake_run(cmd, capture_output=True, text=True, **kwargs):
            assert cmd[:4] == ["wmic", "process", "get", "ProcessId,CommandLine"]
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "CommandLine=C:\\Python\\python.exe -m gateway.run\r\r\n"
                    "ProcessId=1234\r\r\n"
                    "\r\r\n"
                    "CommandLine=C:\\Python\\python.exe -m hermes_cli.main gateway status\r\r\n"
                    "ProcessId=4321\r\r\n"
                    "\r\r\n"
                    "CommandLine=C:\\Python\\python.exe -m not_gateway.run\r\r\n"
                    "ProcessId=5678\r\r\n"
                ),
                stderr="",
            )

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        assert gateway_cli.find_gateway_pids() == [1234]


class TestWindowsGatewayService:
    def test_powershell_literal_escapes_single_quotes(self):
        assert gateway_cli._powershell_literal("C:\\AI\\Jaime's\\Hermes") == "'C:\\AI\\Jaime''s\\Hermes'"

    def test_windows_supervisor_script_restarts_gateway_with_profile_env(self, tmp_path, monkeypatch):
        project_root = tmp_path / "project"
        hermes_home = tmp_path / "hermes-home"
        venv = project_root / "venv"
        python_path = venv / "Scripts" / "python.exe"
        project_root.mkdir()
        hermes_home.mkdir()
        python_path.parent.mkdir(parents=True)
        python_path.write_text("", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", project_root)
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: hermes_home)
        monkeypatch.setattr(gateway_cli, "_detect_venv_dir", lambda: venv)
        monkeypatch.setattr(gateway_cli, "get_python_path", lambda: str(python_path))
        monkeypatch.setenv("PATH", "C:\\Windows\\System32")

        script = gateway_cli.generate_windows_supervisor_script()

        assert "while ($true)" in script
        assert "-m hermes_cli.main gateway run --replace" in script
        assert f"$env:HERMES_HOME = {gateway_cli._powershell_literal(str(hermes_home.resolve()))}" in script
        assert f"$env:VIRTUAL_ENV = {gateway_cli._powershell_literal(str(venv.resolve()))}" in script
        assert "$env:PYTHONUTF8 = '1'" in script
        assert "Start-Sleep -Seconds 10" in script

    def test_windows_install_writes_supervisor_and_creates_task(self, tmp_path, monkeypatch):
        script_path = tmp_path / "gateway-supervisor.ps1"
        startup_path = tmp_path / "Startup" / "HermesGateway.cmd"
        calls = []

        monkeypatch.setattr(gateway_cli, "get_windows_supervisor_script_path", lambda: script_path)
        monkeypatch.setattr(gateway_cli, "get_windows_startup_command_path", lambda: startup_path)
        monkeypatch.setattr(gateway_cli, "windows_task_exists", lambda: False)
        monkeypatch.setattr(gateway_cli, "get_windows_task_name", lambda: "HermesGateway")
        monkeypatch.setattr(gateway_cli, "generate_windows_supervisor_script", lambda: "supervisor script\n")

        def fake_run(cmd, check=False, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.windows_install(force=False)

        assert script_path.read_text(encoding="utf-8") == "supervisor script\n"
        create_calls = [cmd for cmd in calls if cmd[:2] == ["schtasks", "/Create"]]
        assert len(create_calls) == 1
        assert "/SC" in create_calls[0]
        assert "ONLOGON" in create_calls[0]
        assert "powershell.exe" in " ".join(create_calls[0])

    def test_windows_install_falls_back_to_startup_command_on_task_access_denied(self, tmp_path, monkeypatch):
        script_path = tmp_path / "gateway-supervisor.ps1"
        startup_path = tmp_path / "Startup" / "HermesGateway.cmd"

        monkeypatch.setattr(gateway_cli, "get_windows_supervisor_script_path", lambda: script_path)
        monkeypatch.setattr(gateway_cli, "get_windows_startup_command_path", lambda: startup_path)
        monkeypatch.setattr(gateway_cli, "windows_task_exists", lambda: False)
        monkeypatch.setattr(gateway_cli, "get_windows_task_name", lambda: "HermesGateway")
        monkeypatch.setattr(gateway_cli, "generate_windows_supervisor_script", lambda: "supervisor script\n")

        def fake_run(cmd, check=False, **kwargs):
            raise gateway_cli.subprocess.CalledProcessError(1, cmd, stderr="Access is denied.")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.windows_install(force=False)

        assert script_path.read_text(encoding="utf-8") == "supervisor script\n"
        startup = startup_path.read_text(encoding="utf-8")
        assert "powershell.exe" in startup
        assert str(script_path) in startup

    def test_gateway_start_routes_to_windows_task(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_linux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_windows", lambda: True)

        calls = []
        monkeypatch.setattr(gateway_cli, "windows_start", lambda: calls.append("start"))

        gateway_cli.gateway_command(SimpleNamespace(gateway_command="start", system=False))

        assert calls == ["start"]

    def test_gateway_install_routes_to_windows_task(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_linux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_windows", lambda: True)

        calls = []
        monkeypatch.setattr(gateway_cli, "windows_install", lambda force=False: calls.append(force))

        gateway_cli.gateway_command(SimpleNamespace(gateway_command="install", force=True, system=False))

        assert calls == [True]


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
        monkeypatch.setattr(gateway_cli, "run_gateway", lambda verbose=0, quiet=False, replace=False: run_calls.append((verbose, quiet, replace)))
        monkeypatch.setattr(gateway_cli, "kill_gateway_processes", lambda force=False: 0)

        try:
            gateway_cli.gateway_command(SimpleNamespace(gateway_command="restart", system=False))
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("Expected gateway_command to exit when service restart fails")

        assert run_calls == []


class TestDetectVenvDir:
    """Tests for _detect_venv_dir() virtualenv detection."""

    def test_detects_active_virtualenv_via_sys_prefix(self, tmp_path, monkeypatch):
        venv_path = tmp_path / "my-custom-venv"
        venv_path.mkdir()
        monkeypatch.setattr("sys.prefix", str(venv_path))
        monkeypatch.setattr("sys.base_prefix", "/usr")

        result = gateway_cli._detect_venv_dir()
        assert result == venv_path

    def test_falls_back_to_dot_venv_directory(self, tmp_path, monkeypatch):
        # Not inside a virtualenv
        monkeypatch.setattr("sys.prefix", "/usr")
        monkeypatch.setattr("sys.base_prefix", "/usr")
        monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", tmp_path)

        dot_venv = tmp_path / ".venv"
        dot_venv.mkdir()

        result = gateway_cli._detect_venv_dir()
        assert result == dot_venv

    def test_falls_back_to_venv_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.prefix", "/usr")
        monkeypatch.setattr("sys.base_prefix", "/usr")
        monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", tmp_path)

        venv = tmp_path / "venv"
        venv.mkdir()

        result = gateway_cli._detect_venv_dir()
        assert result == venv

    def test_prefers_dot_venv_over_venv(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.prefix", "/usr")
        monkeypatch.setattr("sys.base_prefix", "/usr")
        monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", tmp_path)

        (tmp_path / ".venv").mkdir()
        (tmp_path / "venv").mkdir()

        result = gateway_cli._detect_venv_dir()
        assert result == tmp_path / ".venv"

    def test_returns_none_when_no_virtualenv(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.prefix", "/usr")
        monkeypatch.setattr("sys.base_prefix", "/usr")
        monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", tmp_path)

        result = gateway_cli._detect_venv_dir()
        assert result is None


class TestSystemUnitHermesHome:
    """HERMES_HOME in system units must reference the target user, not root."""

    def test_system_unit_uses_target_user_home_not_calling_user(self, monkeypatch):
        # Simulate sudo: Path.home() returns /root, target user is alice
        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/root")))
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            gateway_cli, "_system_service_identity",
            lambda run_as_user=None: ("alice", "alice", "/home/alice"),
        )
        monkeypatch.setattr(
            gateway_cli, "_build_user_local_paths",
            lambda home, existing: [],
        )

        unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")

        assert 'HERMES_HOME=/home/alice/.hermes' in unit
        assert '/root/.hermes' not in unit

    def test_system_unit_remaps_profile_to_target_user(self, monkeypatch):
        # Simulate sudo with a profile: HERMES_HOME was resolved under root
        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/root")))
        monkeypatch.setenv("HERMES_HOME", "/root/.hermes/profiles/coder")
        monkeypatch.setattr(
            gateway_cli, "_system_service_identity",
            lambda run_as_user=None: ("alice", "alice", "/home/alice"),
        )
        monkeypatch.setattr(
            gateway_cli, "_build_user_local_paths",
            lambda home, existing: [],
        )

        unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")

        assert 'HERMES_HOME=/home/alice/.hermes/profiles/coder' in unit
        assert '/root/' not in unit

    def test_system_unit_preserves_custom_hermes_home(self, monkeypatch):
        # Custom HERMES_HOME not under any user's home — keep as-is
        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/root")))
        monkeypatch.setenv("HERMES_HOME", "/opt/hermes-shared")
        monkeypatch.setattr(
            gateway_cli, "_system_service_identity",
            lambda run_as_user=None: ("alice", "alice", "/home/alice"),
        )
        monkeypatch.setattr(
            gateway_cli, "_build_user_local_paths",
            lambda home, existing: [],
        )

        unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")

        assert 'HERMES_HOME=/opt/hermes-shared' in unit

    def test_user_unit_unaffected_by_change(self):
        # User-scope units should still use the calling user's HERMES_HOME
        unit = gateway_cli.generate_systemd_unit(system=False)

        hermes_home = str(gateway_cli.get_hermes_home().resolve())
        assert f'HERMES_HOME={hermes_home}' in unit


class TestHermesHomeForTargetUser:
    """Unit tests for _hermes_home_for_target_user()."""

    def test_remaps_default_home(self, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/root")))
        monkeypatch.delenv("HERMES_HOME", raising=False)

        result = gateway_cli._hermes_home_for_target_user("/home/alice")
        assert result == "/home/alice/.hermes"

    def test_remaps_profile_path(self, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/root")))
        monkeypatch.setenv("HERMES_HOME", "/root/.hermes/profiles/coder")

        result = gateway_cli._hermes_home_for_target_user("/home/alice")
        assert result == "/home/alice/.hermes/profiles/coder"

    def test_keeps_custom_path(self, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/root")))
        monkeypatch.setenv("HERMES_HOME", "/opt/hermes")

        result = gateway_cli._hermes_home_for_target_user("/home/alice")
        assert result == "/opt/hermes"

    def test_noop_when_same_user(self, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/home/alice")))
        monkeypatch.delenv("HERMES_HOME", raising=False)

        result = gateway_cli._hermes_home_for_target_user("/home/alice")
        assert result == "/home/alice/.hermes"


class TestGeneratedUnitUsesDetectedVenv:
    def test_systemd_unit_uses_dot_venv_when_detected(self, tmp_path, monkeypatch):
        dot_venv = tmp_path / ".venv"
        dot_venv.mkdir()
        (dot_venv / "bin").mkdir()

        monkeypatch.setattr(gateway_cli, "_detect_venv_dir", lambda: dot_venv)
        monkeypatch.setattr(gateway_cli, "get_python_path", lambda: str(dot_venv / "bin" / "python"))

        unit = gateway_cli.generate_systemd_unit(system=False)

        assert f"VIRTUAL_ENV={dot_venv}" in unit
        assert f"{dot_venv}/bin" in unit
        # Must NOT contain a hardcoded /venv/ path
        assert "/venv/" not in unit or "/.venv/" in unit


class TestGeneratedUnitIncludesLocalBin:
    """~/.local/bin must be in PATH so uvx/pipx tools are discoverable."""

    def test_user_unit_includes_local_bin_in_path(self):
        unit = gateway_cli.generate_systemd_unit(system=False)
        home = str(Path.home())
        assert f"{home}/.local/bin" in unit

    def test_system_unit_includes_local_bin_in_path(self):
        unit = gateway_cli.generate_systemd_unit(system=True)
        # System unit uses the resolved home dir from _system_service_identity
        assert "/.local/bin" in unit


class TestEnsureUserSystemdEnv:
    """Tests for _ensure_user_systemd_env() D-Bus session bus auto-detection."""

    def test_sets_xdg_runtime_dir_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setattr(os, "getuid", lambda: 42)

        # Patch Path.exists so /run/user/42 appears to exist.
        # Using a FakePath subclass breaks on Python 3.12+ where
        # PosixPath.__new__ ignores the redirected path argument.
        _orig_exists = gateway_cli.Path.exists
        monkeypatch.setattr(
            gateway_cli.Path, "exists",
            lambda self: True if str(self) == "/run/user/42" else _orig_exists(self),
        )

        gateway_cli._ensure_user_systemd_env()

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
