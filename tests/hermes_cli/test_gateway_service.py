"""Tests for gateway service management helpers."""

import os
import pwd
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli
from gateway import status
from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
)


class TestUserSystemdPrivateSocketPreflight:
    def test_preflight_accepts_private_socket_without_dbus_bus(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "_ensure_user_systemd_env", lambda: None)
        monkeypatch.setattr(gateway_cli, "_user_dbus_socket_path", lambda: Path("/tmp/missing-bus"))
        monkeypatch.setattr(gateway_cli, "_user_systemd_private_socket_path", lambda: Path("/tmp/private-socket"))
        monkeypatch.setattr(Path, "exists", lambda self: str(self) == "/tmp/private-socket")

        gateway_cli._preflight_user_systemd(auto_enable_linger=False)

    def test_wait_for_user_dbus_socket_accepts_private_socket(self, monkeypatch):
        calls = []
        monkeypatch.setattr(gateway_cli, "_ensure_user_systemd_env", lambda: calls.append("env"))
        monkeypatch.setattr(gateway_cli, "_user_dbus_socket_path", lambda: Path("/tmp/missing-bus"))
        monkeypatch.setattr(gateway_cli, "_user_systemd_private_socket_path", lambda: Path("/tmp/private-socket"))
        monkeypatch.setattr(Path, "exists", lambda self: str(self) == "/tmp/private-socket")

        assert gateway_cli._wait_for_user_dbus_socket(timeout=0.1) is True
        assert calls == ["env"]


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
        assert calls[:4] == [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "show", gateway_cli.get_service_name(), "--no-pager", "--property", "ActiveState,SubState,Result,ExecMainStatus"],
            ["systemctl", "--user", "reset-failed", gateway_cli.get_service_name()],
            ["systemctl", "--user", "reload-or-restart", gateway_cli.get_service_name()],
        ]

    def test_systemd_stop_marks_running_gateway_as_planned_stop(self, monkeypatch):
        calls = []
        markers = []

        monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
        monkeypatch.setattr(gateway_cli, "_require_service_installed", lambda action, system=False: None)
        monkeypatch.setattr(status, "get_running_pid", lambda cleanup_stale=True: 321)
        monkeypatch.setattr(
            status,
            "write_planned_stop_marker",
            lambda pid: markers.append(pid) or True,
        )

        def fake_run_systemctl(args, **kwargs):
            calls.append(args)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli, "_run_systemctl", fake_run_systemctl)

        gateway_cli.systemd_stop()

        assert markers == [321]
        assert calls == [["stop", gateway_cli.get_service_name()]]


    def test_run_gateway_refreshes_outdated_unit_on_boot(self, tmp_path, monkeypatch):
        """run_gateway() should refresh the systemd unit on boot so that
        restart settings take effect even when the process was respawned
        via exit-code-75 (bypassing `hermes gateway restart`)."""
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("old unit\n", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)
        monkeypatch.setattr(gateway_cli, "generate_systemd_unit", lambda system=False, run_as_user=None: "new unit\n")
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)

        calls = []

        def fake_run(cmd, check=True, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        # Prevent run_gateway from actually starting the gateway
        async def fake_start_gateway(**kwargs):
            return True

        monkeypatch.setattr("gateway.run.start_gateway", fake_start_gateway)

        gateway_cli.run_gateway()

        assert unit_path.read_text(encoding="utf-8") == "new unit\n"
        assert ["systemctl", "--user", "daemon-reload"] in calls


class TestRequireServiceInstalled:
    def test_exits_with_install_hint_when_unit_missing(self, tmp_path, monkeypatch, capsys):
        unit_path = tmp_path / "hermes-gateway.service"
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)

        with pytest.raises(SystemExit) as exc_info:
            gateway_cli._require_service_installed("start")

        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "not installed" in out
        assert "hermes gateway install" in out

    def test_passes_when_unit_exists(self, tmp_path, monkeypatch):
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("[Unit]\n", encoding="utf-8")
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)

        gateway_cli._require_service_installed("start")


class TestGeneratedSystemdUnits:
    def _expected_timeout_stop_sec(self) -> str:
        timeout = int(max(60, DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT) + 30)
        return f"TimeoutStopSec={timeout}"

    def test_user_unit_avoids_recursive_execstop_and_uses_extended_stop_timeout(self, monkeypatch):
        monkeypatch.setattr(
            gateway_cli,
            "_get_restart_drain_timeout",
            lambda: DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
        )
        unit = gateway_cli.generate_systemd_unit(system=False)

        assert "ExecStart=" in unit
        assert "ExecStop=" not in unit
        assert "ExecReload=/bin/kill -USR1 $MAINPID" in unit
        assert f"RestartForceExitStatus={GATEWAY_SERVICE_RESTART_EXIT_CODE}" in unit
        # TimeoutStopSec must exceed the default drain_timeout (60s) so
        # systemd doesn't SIGKILL the cgroup before post-interrupt cleanup
        # (tool subprocess kill, adapter disconnect) runs — issue #8202.
        assert self._expected_timeout_stop_sec() in unit

    def test_user_unit_includes_resolved_node_directory_in_path(self, monkeypatch):
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda cmd: "/home/test/.nvm/versions/node/v24.14.0/bin/node" if cmd == "node" else None)

        unit = gateway_cli.generate_systemd_unit(system=False)

        assert "/home/test/.nvm/versions/node/v24.14.0/bin" in unit

    def test_user_unit_includes_wsl_windows_interop_paths(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_wsl", lambda: True)
        monkeypatch.setenv(
            "PATH",
            "/usr/local/bin:/mnt/c/WINDOWS/system32:/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/",
        )
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda cmd: None)

        unit = gateway_cli.generate_systemd_unit(system=False)

        assert "/mnt/c/WINDOWS/system32" in unit
        assert "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/" in unit

    def test_user_unit_omits_windows_interop_paths_outside_wsl(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_wsl", lambda: False)
        monkeypatch.setenv("PATH", "/usr/local/bin:/mnt/c/WINDOWS/system32")
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda cmd: None)

        unit = gateway_cli.generate_systemd_unit(system=False)

        assert "/mnt/c/WINDOWS/system32" not in unit

    def test_system_unit_includes_wsl_windows_interop_paths(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_wsl", lambda: True)
        monkeypatch.setattr(
            gateway_cli,
            "_system_service_identity",
            lambda run_as_user=None: ("alice", "alice", "/home/alice"),
        )
        monkeypatch.setattr(gateway_cli, "_hermes_home_for_target_user", lambda home: "/home/alice/.hermes")
        monkeypatch.setenv("PATH", "/usr/local/bin:/mnt/c/WINDOWS/system32")
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda cmd: None)

        unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")

        assert "/mnt/c/WINDOWS/system32" in unit

    def test_system_unit_avoids_recursive_execstop_and_uses_extended_stop_timeout(self, monkeypatch):
        monkeypatch.setattr(
            gateway_cli,
            "_get_restart_drain_timeout",
            lambda: DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
        )
        monkeypatch.setattr(
            gateway_cli,
            "_system_service_identity",
            lambda run_as_user=None: ("testuser", "testgroup", "/home/testuser"),
        )
        unit = gateway_cli.generate_systemd_unit(system=True)

        assert "ExecStart=" in unit
        assert "ExecStop=" not in unit
        assert "ExecReload=/bin/kill -USR1 $MAINPID" in unit
        assert f"RestartForceExitStatus={GATEWAY_SERVICE_RESTART_EXIT_CODE}" in unit
        # TimeoutStopSec must exceed the default drain_timeout (60s) so
        # systemd doesn't SIGKILL the cgroup before post-interrupt cleanup
        # (tool subprocess kill, adapter disconnect) runs — issue #8202.
        assert self._expected_timeout_stop_sec() in unit
        assert "WantedBy=multi-user.target" in unit


class TestGatewayStopCleanup:
    def test_stop_only_kills_current_profile_by_default(self, tmp_path, monkeypatch):
        """Without --all, stop uses systemd (if available) and does NOT call
        the global kill_gateway_processes()."""
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("unit\n", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)

        service_calls = []
        kill_calls = []

        monkeypatch.setattr(gateway_cli, "systemd_stop", lambda system=False: service_calls.append("stop"))
        monkeypatch.setattr(
            gateway_cli,
            "kill_gateway_processes",
            lambda force=False, all_profiles=False: kill_calls.append(force) or 2,
        )

        gateway_cli.gateway_command(SimpleNamespace(gateway_command="stop"))

        assert service_calls == ["stop"]
        # Global kill should NOT be called without --all
        assert kill_calls == []

    def test_stop_all_sweeps_all_gateway_processes(self, tmp_path, monkeypatch):
        """With --all, stop uses systemd AND calls the global kill_gateway_processes()."""
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("unit\n", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)

        service_calls = []
        kill_calls = []

        monkeypatch.setattr(gateway_cli, "systemd_stop", lambda system=False: service_calls.append("stop"))
        monkeypatch.setattr(
            gateway_cli,
            "kill_gateway_processes",
            lambda force=False, all_profiles=False: kill_calls.append(force) or 2,
        )

        gateway_cli.gateway_command(SimpleNamespace(gateway_command="stop", **{"all": True}))

        assert service_calls == ["stop"]
        assert kill_calls == [False]


class TestLaunchdServiceRecovery:
    def test_get_restart_drain_timeout_prefers_env_then_config_then_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_RESTART_DRAIN_TIMEOUT", raising=False)
        monkeypatch.setattr(gateway_cli, "read_raw_config", lambda: {})

        assert (
            gateway_cli._get_restart_drain_timeout()
            == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
        )

        monkeypatch.setattr(
            gateway_cli,
            "read_raw_config",
            lambda: {"agent": {"restart_drain_timeout": 14}},
        )
        assert gateway_cli._get_restart_drain_timeout() == 14.0

        monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "9")
        assert gateway_cli._get_restart_drain_timeout() == 9.0

        monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "invalid")
        assert (
            gateway_cli._get_restart_drain_timeout()
            == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
        )

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

        label = gateway_cli.get_launchd_label()
        domain = gateway_cli._launchd_domain()
        assert "--replace" in plist_path.read_text(encoding="utf-8")
        assert calls[:2] == [
            ["launchctl", "bootout", f"{domain}/{label}"],
            ["launchctl", "bootstrap", domain, str(plist_path)],
        ]

    def test_launchd_start_reloads_unloaded_job_and_retries(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text(gateway_cli.generate_launchd_plist(), encoding="utf-8")
        label = gateway_cli.get_launchd_label()

        calls = []
        domain = gateway_cli._launchd_domain()
        target = f"{domain}/{label}"

        def fake_run(cmd, check=False, **kwargs):
            if cmd and cmd[0] == "launchctl":
                calls.append(cmd)
            if cmd == ["launchctl", "kickstart", target] and calls.count(cmd) == 1:
                raise gateway_cli.subprocess.CalledProcessError(3, cmd, stderr="Could not find service")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.launchd_start()

        assert calls == [
            ["launchctl", "kickstart", target],
            ["launchctl", "bootstrap", domain, str(plist_path)],
            ["launchctl", "kickstart", target],
        ]

    def test_launchd_start_reloads_on_kickstart_exit_code_113(self, tmp_path, monkeypatch):
        """Exit code 113 (\"Could not find service\") should also trigger bootstrap recovery."""
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text(gateway_cli.generate_launchd_plist(), encoding="utf-8")
        label = gateway_cli.get_launchd_label()

        calls = []
        domain = gateway_cli._launchd_domain()
        target = f"{domain}/{label}"

        def fake_run(cmd, check=False, **kwargs):
            if cmd and cmd[0] == "launchctl":
                calls.append(cmd)
            if cmd == ["launchctl", "kickstart", target] and calls.count(cmd) == 1:
                raise gateway_cli.subprocess.CalledProcessError(113, cmd, stderr="Could not find service")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.launchd_start()

        assert calls == [
            ["launchctl", "kickstart", target],
            ["launchctl", "bootstrap", domain, str(plist_path)],
            ["launchctl", "kickstart", target],
        ]

    def test_launchd_restart_drains_running_gateway_before_kickstart(self, monkeypatch):
        calls = []
        target = f"{gateway_cli._launchd_domain()}/{gateway_cli.get_launchd_label()}"

        monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 12.0)
        monkeypatch.setattr(gateway_cli, "_request_gateway_self_restart", lambda pid: False)
        monkeypatch.setattr(gateway_cli, "_wait_for_gateway_exit", lambda timeout, force_after=None: True)
        monkeypatch.setattr(gateway_cli, "terminate_pid", lambda pid, force=False: calls.append(("term", pid, force)))
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda: 321,
        )

        def fake_run(cmd, check=False, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.launchd_restart()

        assert calls == [
            ("term", 321, False),
            ["launchctl", "kickstart", "-k", target],
        ]

    def test_launchd_restart_self_requests_graceful_restart_without_kickstart(self, monkeypatch, capsys):
        calls = []

        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda: 321,
        )
        monkeypatch.setattr(
            gateway_cli,
            "_request_gateway_self_restart",
            lambda pid: calls.append(("self", pid)) or True,
        )
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("launchctl should not run")),
        )

        gateway_cli.launchd_restart()

        assert calls == [("self", 321)]
        assert "restart requested" in capsys.readouterr().out.lower()

    def test_launchd_stop_uses_bootout_not_kill(self, monkeypatch):
        """launchd_stop must bootout the service so KeepAlive doesn't respawn it."""
        label = gateway_cli.get_launchd_label()
        domain = gateway_cli._launchd_domain()
        target = f"{domain}/{label}"

        calls = []

        def fake_run(cmd, check=False, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr(gateway_cli, "_wait_for_gateway_exit", lambda **kw: None)

        gateway_cli.launchd_stop()

        assert calls == [["launchctl", "bootout", target]]

    def test_launchd_stop_tolerates_already_unloaded(self, monkeypatch, capsys):
        """launchd_stop silently handles exit codes 3/113 (job not loaded)."""
        label = gateway_cli.get_launchd_label()
        domain = gateway_cli._launchd_domain()
        target = f"{domain}/{label}"

        def fake_run(cmd, check=False, **kwargs):
            if "bootout" in cmd:
                raise gateway_cli.subprocess.CalledProcessError(3, cmd, stderr="Could not find service")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr(gateway_cli, "_wait_for_gateway_exit", lambda **kw: None)

        # Should not raise — exit code 3 means already unloaded
        gateway_cli.launchd_stop()

        output = capsys.readouterr().out
        assert "stopped" in output.lower()

    def test_launchd_stop_waits_for_process_exit(self, monkeypatch):
        """launchd_stop calls _wait_for_gateway_exit after bootout."""
        wait_called = []

        def fake_run(cmd, check=False, **kwargs):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def fake_wait(**kwargs):
            wait_called.append(kwargs)

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr(gateway_cli, "_wait_for_gateway_exit", fake_wait)

        gateway_cli.launchd_stop()

        assert len(wait_called) == 1
        assert wait_called[0] == {"timeout": 10.0, "force_after": 5.0}

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
    def test_supports_systemd_services_requires_systemctl_binary(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_linux", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda name: None)

        assert gateway_cli.supports_systemd_services() is False

    def test_supports_systemd_services_returns_true_when_systemctl_present(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_linux", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_wsl", lambda: False)
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda name: "/usr/bin/systemctl")

        assert gateway_cli.supports_systemd_services() is True

    def test_is_service_running_checks_system_scope_when_user_scope_is_inactive(self, monkeypatch):
        user_unit = SimpleNamespace(exists=lambda: True)
        system_unit = SimpleNamespace(exists=lambda: True)

        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
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

    def test_is_service_running_returns_false_when_systemctl_missing(self, monkeypatch):
        unit = SimpleNamespace(exists=lambda: True)

        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(
            gateway_cli,
            "get_systemd_unit_path",
            lambda system=False: unit,
        )

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("systemctl")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        assert gateway_cli._is_service_running() is False

class TestGatewaySystemServiceRouting:
    def test_systemd_restart_self_requests_graceful_restart_and_waits(self, monkeypatch, capsys):
        calls = []

        monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
        monkeypatch.setattr(gateway_cli, "_require_service_installed", lambda action, system=False: None)
        monkeypatch.setattr(gateway_cli, "refresh_systemd_unit_if_needed", lambda system=False: calls.append(("refresh", system)))
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda: 654,
        )
        monkeypatch.setattr(
            gateway_cli,
            "_request_gateway_self_restart",
            lambda pid: calls.append(("self", pid)) or True,
        )

        # Simulate: old process dies immediately, new process becomes active
        kill_call_count = [0]
        def fake_kill(pid, sig):
            kill_call_count[0] += 1
            if kill_call_count[0] >= 2:  # first call checks, second = dead
                raise ProcessLookupError()
        monkeypatch.setattr(os, "kill", fake_kill)

        # Simulate systemctl reset-failed/start followed by an active unit
        new_pid = [None]
        def fake_subprocess_run(cmd, **kwargs):
            if "reset-failed" in cmd:
                calls.append(("reset-failed", cmd))
                return SimpleNamespace(stdout="", returncode=0)
            if "start" in cmd:
                calls.append(("start", cmd))
                return SimpleNamespace(stdout="", returncode=0)
            if "show" in cmd:
                new_pid[0] = 999
                return SimpleNamespace(
                    stdout="ActiveState=active\nSubState=running\nResult=success\nExecMainStatus=0\n",
                    returncode=0,
                )
            raise AssertionError(f"Unexpected systemctl call: {cmd}")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_subprocess_run)
        # get_running_pid returns new PID after restart
        pid_calls = [0]
        def fake_get_pid():
            pid_calls[0] += 1
            return 999 if pid_calls[0] > 1 else 654
        monkeypatch.setattr("gateway.status.get_running_pid", fake_get_pid)

        gateway_cli.systemd_restart()

        assert ("self", 654) in calls
        assert any(call[0] == "reset-failed" for call in calls)
        assert any(call[0] == "start" for call in calls)
        out = capsys.readouterr().out.lower()
        assert "restarted" in out

    def test_systemd_restart_recovers_failed_planned_restart(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
        monkeypatch.setattr(gateway_cli, "_require_service_installed", lambda action, system=False: None)
        monkeypatch.setattr(gateway_cli, "refresh_systemd_unit_if_needed", lambda system=False: None)
        monkeypatch.setattr(
            "gateway.status.read_runtime_status",
            lambda: {"restart_requested": True, "gateway_state": "stopped"},
        )
        monkeypatch.setattr(gateway_cli, "_request_gateway_self_restart", lambda pid: False)

        calls = []
        started = {"value": False}

        def fake_subprocess_run(cmd, **kwargs):
            if "show" in cmd:
                if not started["value"]:
                    return SimpleNamespace(
                        stdout=(
                            "ActiveState=failed\n"
                            "SubState=failed\n"
                            "Result=exit-code\n"
                            f"ExecMainStatus={GATEWAY_SERVICE_RESTART_EXIT_CODE}\n"
                        ),
                        returncode=0,
                    )
                return SimpleNamespace(
                    stdout="ActiveState=active\nSubState=running\nResult=success\nExecMainStatus=0\n",
                    returncode=0,
                )
            if "reset-failed" in cmd:
                calls.append(("reset-failed", cmd))
                return SimpleNamespace(stdout="", returncode=0)
            if "start" in cmd:
                started["value"] = True
                calls.append(("start", cmd))
                return SimpleNamespace(stdout="", returncode=0)
            raise AssertionError(f"Unexpected command: {cmd}")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_subprocess_run)
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda: 999 if started["value"] else None,
        )

        gateway_cli.systemd_restart()

        assert any(call[0] == "reset-failed" for call in calls)
        assert any(call[0] == "start" for call in calls)
        out = capsys.readouterr().out.lower()
        assert "restarted" in out

    def test_systemd_status_surfaces_planned_restart_failure(self, monkeypatch, capsys):
        unit = SimpleNamespace(exists=lambda: True)
        monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit)
        monkeypatch.setattr(gateway_cli, "has_conflicting_systemd_units", lambda: False)
        monkeypatch.setattr(gateway_cli, "has_legacy_hermes_units", lambda: False)
        monkeypatch.setattr(gateway_cli, "systemd_unit_is_current", lambda system=False: True)
        monkeypatch.setattr(gateway_cli, "_runtime_health_lines", lambda: ["⚠ Last shutdown reason: Gateway restart requested"])
        monkeypatch.setattr(gateway_cli, "get_systemd_linger_status", lambda: (True, ""))
        monkeypatch.setattr(gateway_cli, "_read_systemd_unit_properties", lambda system=False: {
            "ActiveState": "failed",
            "SubState": "failed",
            "Result": "exit-code",
            "ExecMainStatus": str(GATEWAY_SERVICE_RESTART_EXIT_CODE),
        })

        calls = []

        def fake_run_systemctl(args, **kwargs):
            calls.append(args)
            if args[:2] == ["status", gateway_cli.get_service_name()]:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if args[:2] == ["is-active", gateway_cli.get_service_name()]:
                return SimpleNamespace(returncode=3, stdout="failed\n", stderr="")
            raise AssertionError(f"Unexpected args: {args}")

        monkeypatch.setattr(gateway_cli, "_run_systemctl", fake_run_systemctl)

        gateway_cli.systemd_status()

        out = capsys.readouterr().out
        assert "Planned restart is stuck in systemd failed state" in out

    def test_gateway_status_dispatches_full_flag(self, monkeypatch):
        user_unit = SimpleNamespace(exists=lambda: True)
        system_unit = SimpleNamespace(exists=lambda: False)

        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(
            gateway_cli,
            "get_systemd_unit_path",
            lambda system=False: system_unit if system else user_unit,
        )
        monkeypatch.setattr(
            gateway_cli,
            "get_gateway_runtime_snapshot",
            lambda system=False: gateway_cli.GatewayRuntimeSnapshot(
                manager="systemd (user)",
                service_installed=True,
                service_running=False,
                gateway_pids=(),
                service_scope="user",
            ),
        )

        calls = []
        monkeypatch.setattr(
            gateway_cli,
            "systemd_status",
            lambda deep=False, system=False, full=False: calls.append((deep, system, full)),
        )

        gateway_cli.gateway_command(
            SimpleNamespace(gateway_command="status", deep=False, system=False, full=True)
        )

        assert calls == [(False, False, True)]

    def test_gateway_install_passes_system_flags(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
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

    def test_gateway_install_reports_termux_manual_mode(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: True)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)

        try:
            gateway_cli.gateway_command(
                SimpleNamespace(gateway_command="install", force=False, system=False, run_as_user=None)
            )
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("Expected gateway_command to exit on unsupported Termux service install")

        out = capsys.readouterr().out
        assert "not supported on Termux" in out
        assert "Run manually: hermes gateway" in out

    def test_gateway_status_prefers_system_service_when_only_system_unit_exists(self, monkeypatch):
        user_unit = SimpleNamespace(exists=lambda: False)
        system_unit = SimpleNamespace(exists=lambda: True)

        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(
            gateway_cli,
            "get_systemd_unit_path",
            lambda system=False: system_unit if system else user_unit,
        )

        calls = []
        monkeypatch.setattr(
            gateway_cli,
            "systemd_status",
            lambda deep=False, system=False, full=False: calls.append((deep, system, full)),
        )

        gateway_cli.gateway_command(SimpleNamespace(gateway_command="status", deep=False, system=False))

        assert calls == [(False, False, False)]

    def test_gateway_status_reports_manual_process_when_service_is_stopped(self, monkeypatch, capsys):
        user_unit = SimpleNamespace(exists=lambda: True)
        system_unit = SimpleNamespace(exists=lambda: False)

        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(
            gateway_cli,
            "get_systemd_unit_path",
            lambda system=False: system_unit if system else user_unit,
        )
        monkeypatch.setattr(
            gateway_cli,
            "systemd_status",
            lambda deep=False, system=False, full=False: print("service stopped"),
        )
        monkeypatch.setattr(
            gateway_cli,
            "get_gateway_runtime_snapshot",
            lambda system=False: gateway_cli.GatewayRuntimeSnapshot(
                manager="systemd (user)",
                service_installed=True,
                service_running=False,
                gateway_pids=(4321,),
                service_scope="user",
            ),
        )

        gateway_cli.gateway_command(SimpleNamespace(gateway_command="status", deep=False, system=False))

        out = capsys.readouterr().out
        assert "service stopped" in out
        assert "Gateway process is running for this profile" in out
        assert "PID(s): 4321" in out

    def test_gateway_status_on_termux_shows_manual_guidance(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "find_gateway_pids", lambda exclude_pids=None: [])
        monkeypatch.setattr(gateway_cli, "_runtime_health_lines", lambda: [])

        gateway_cli.gateway_command(SimpleNamespace(gateway_command="status", deep=False, system=False))

        out = capsys.readouterr().out
        assert "Gateway is not running" in out
        assert "nohup hermes gateway" in out
        assert "install as user service" not in out

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
                gateway_cli.subprocess.CalledProcessError(5, ["launchctl", "kickstart", "-k", "gui/501/ai.hermes.gateway"])
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
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", tmp_path)

        dot_venv = tmp_path / ".venv"
        dot_venv.mkdir()

        result = gateway_cli._detect_venv_dir()
        assert result == dot_venv

    def test_falls_back_to_venv_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.prefix", "/usr")
        monkeypatch.setattr("sys.base_prefix", "/usr")
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", tmp_path)

        venv = tmp_path / "venv"
        venv.mkdir()

        result = gateway_cli._detect_venv_dir()
        assert result == venv

    def test_prefers_dot_venv_over_venv(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.prefix", "/usr")
        monkeypatch.setattr("sys.base_prefix", "/usr")
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", tmp_path)

        (tmp_path / ".venv").mkdir()
        (tmp_path / "venv").mkdir()

        result = gateway_cli._detect_venv_dir()
        assert result == tmp_path / ".venv"

    def test_returns_none_when_no_virtualenv(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.prefix", "/usr")
        monkeypatch.setattr("sys.base_prefix", "/usr")
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
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

    def test_user_unit_includes_local_bin_in_path(self, monkeypatch):
        home = Path.home()
        monkeypatch.setattr(
            gateway_cli,
            "_build_user_local_paths",
            lambda home_path, existing: [str(home / ".local" / "bin")],
        )
        unit = gateway_cli.generate_systemd_unit(system=False)
        assert f"{home}/.local/bin" in unit

    def test_system_unit_includes_local_bin_in_path(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "_system_service_identity", lambda run_as_user=None: ("testuser", "testgroup", "/home/testuser"))
        monkeypatch.setattr(
            gateway_cli,
            "_build_user_local_paths",
            lambda home_path, existing: [str(home_path / ".local" / "bin")],
        )
        unit = gateway_cli.generate_systemd_unit(system=True)
        # System unit uses the resolved home dir from _system_service_identity
        assert "/.local/bin" in unit


class TestSystemServiceIdentityRootHandling:
    """Root user handling in _system_service_identity()."""

    def test_auto_detected_root_is_rejected(self, monkeypatch):
        """When root is auto-detected (not explicitly requested), raise."""
        import pwd
        import grp

        monkeypatch.delenv("SUDO_USER", raising=False)
        monkeypatch.setenv("USER", "root")
        monkeypatch.setenv("LOGNAME", "root")

        import pytest
        with pytest.raises(ValueError, match="pass --run-as-user root to override"):
            gateway_cli._system_service_identity(run_as_user=None)

    def test_explicit_root_is_allowed(self, monkeypatch):
        """When root is explicitly passed via --run-as-user root, allow it."""
        import pwd
        import grp

        root_info = pwd.getpwnam("root")
        root_group = grp.getgrgid(root_info.pw_gid).gr_name

        username, group, home = gateway_cli._system_service_identity(run_as_user="root")
        assert username == "root"
        assert home == root_info.pw_dir

    def test_non_root_user_passes_through(self, monkeypatch):
        """Normal non-root user works as before."""
        import pwd
        import grp

        monkeypatch.delenv("SUDO_USER", raising=False)
        monkeypatch.setenv("USER", "nobody")
        monkeypatch.setenv("LOGNAME", "nobody")

        try:
            username, group, home = gateway_cli._system_service_identity(run_as_user=None)
            assert username == "nobody"
        except ValueError as e:
            # "nobody" might not exist on all systems
            assert "Unknown user" in str(e)


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


class TestPreflightUserSystemd:
    """Tests for _preflight_user_systemd() — D-Bus reachability before systemctl --user.

    Covers issue #5130 / Rick's RHEL 9.6 SSH scenario: setup tries to start the
    gateway via ``systemctl --user start`` in a shell with no user D-Bus session,
    which previously failed with a raw ``CalledProcessError`` and no remediation.
    """

    def test_noop_when_bus_socket_exists(self, monkeypatch):
        """Socket already there (desktop / linger + prior login) → no-op."""
        monkeypatch.setattr(
            gateway_cli, "_user_dbus_socket_path",
            lambda: type("P", (), {"exists": lambda self: True})(),
        )
        monkeypatch.setattr(
            gateway_cli, "_user_systemd_private_socket_path",
            lambda: type("P", (), {"exists": lambda self: False})(),
        )
        # Should not raise, no subprocess calls needed.
        gateway_cli._preflight_user_systemd()

    def test_raises_when_linger_disabled_and_loginctl_denied(self, monkeypatch):
        """Rick's scenario: no D-Bus, no linger, non-root SSH → clear error."""
        monkeypatch.setattr(
            gateway_cli, "_user_dbus_socket_path",
            lambda: type("P", (), {"exists": lambda self: False})(),
        )
        monkeypatch.setattr(
            gateway_cli, "_user_systemd_private_socket_path",
            lambda: type("P", (), {"exists": lambda self: False})(),
        )
        monkeypatch.setattr(
            gateway_cli, "get_systemd_linger_status", lambda: (False, ""),
        )
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda _: "/usr/bin/loginctl")

        class _Result:
            returncode = 1
            stdout = ""
            stderr = "Interactive authentication required."

        monkeypatch.setattr(
            gateway_cli.subprocess, "run", lambda *a, **kw: _Result(),
        )

        with pytest.raises(gateway_cli.UserSystemdUnavailableError) as exc_info:
            gateway_cli._preflight_user_systemd()

        msg = str(exc_info.value)
        assert "sudo loginctl enable-linger" in msg
        assert "hermes gateway run" in msg  # foreground fallback mentioned
        assert "Interactive authentication required" in msg

    def test_raises_when_loginctl_missing(self, monkeypatch):
        """No loginctl binary at all → suggest sudo install + manual fix."""
        monkeypatch.setattr(
            gateway_cli, "_user_dbus_socket_path",
            lambda: type("P", (), {"exists": lambda self: False})(),
        )
        monkeypatch.setattr(
            gateway_cli, "_user_systemd_private_socket_path",
            lambda: type("P", (), {"exists": lambda self: False})(),
        )
        monkeypatch.setattr(
            gateway_cli, "get_systemd_linger_status",
            lambda: (None, "loginctl not found"),
        )
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda _: None)

        with pytest.raises(gateway_cli.UserSystemdUnavailableError) as exc_info:
            gateway_cli._preflight_user_systemd()

        assert "sudo loginctl enable-linger" in str(exc_info.value)

    def test_linger_enabled_but_socket_still_missing(self, monkeypatch):
        """Edge case: linger says yes but the bus socket never came up."""
        monkeypatch.setattr(
            gateway_cli, "_user_dbus_socket_path",
            lambda: type("P", (), {"exists": lambda self: False})(),
        )
        monkeypatch.setattr(
            gateway_cli, "_user_systemd_private_socket_path",
            lambda: type("P", (), {"exists": lambda self: False})(),
        )
        monkeypatch.setattr(
            gateway_cli, "get_systemd_linger_status", lambda: (True, ""),
        )
        monkeypatch.setattr(
            gateway_cli, "_wait_for_user_dbus_socket", lambda timeout=3.0: False,
        )

        with pytest.raises(gateway_cli.UserSystemdUnavailableError) as exc_info:
            gateway_cli._preflight_user_systemd()

        assert "linger is enabled" in str(exc_info.value)

    def test_enable_linger_succeeds_and_socket_appears(self, monkeypatch, capsys):
        """Happy remediation path: polkit allows enable-linger, socket spawns."""
        monkeypatch.setattr(
            gateway_cli, "_user_dbus_socket_path",
            lambda: type("P", (), {"exists": lambda self: False})(),
        )
        monkeypatch.setattr(
            gateway_cli, "_user_systemd_private_socket_path",
            lambda: type("P", (), {"exists": lambda self: False})(),
        )
        monkeypatch.setattr(
            gateway_cli, "get_systemd_linger_status", lambda: (False, ""),
        )
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda _: "/usr/bin/loginctl")

        class _OkResult:
            returncode = 0
            stdout = ""
            stderr = ""

        monkeypatch.setattr(
            gateway_cli.subprocess, "run", lambda *a, **kw: _OkResult(),
        )
        monkeypatch.setattr(
            gateway_cli, "_wait_for_user_dbus_socket",
            lambda timeout=5.0: True,
        )

        # Should not raise.
        gateway_cli._preflight_user_systemd()
        out = capsys.readouterr().out
        assert "Enabled linger" in out


class TestProfileArg:
    """Tests for _profile_arg — returns '--profile <name>' for named profiles."""

    def test_default_hermes_home_returns_empty(self, tmp_path, monkeypatch):
        """Default ~/.hermes should not produce a --profile flag."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = gateway_cli._profile_arg(str(hermes_home))
        assert result == ""

    def test_named_profile_returns_flag(self, tmp_path, monkeypatch):
        """~/.hermes/profiles/mybot should return '--profile mybot'."""
        profile_dir = tmp_path / ".hermes" / "profiles" / "mybot"
        profile_dir.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        result = gateway_cli._profile_arg(str(profile_dir))
        assert result == "--profile mybot"

    def test_hash_path_returns_empty(self, tmp_path, monkeypatch):
        """Arbitrary non-profile HERMES_HOME should return empty string."""
        custom_home = tmp_path / "custom" / "hermes"
        custom_home.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        result = gateway_cli._profile_arg(str(custom_home))
        assert result == ""

    def test_nested_profile_path_returns_empty(self, tmp_path, monkeypatch):
        """~/.hermes/profiles/mybot/subdir should NOT match — too deep."""
        nested = tmp_path / ".hermes" / "profiles" / "mybot" / "subdir"
        nested.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        result = gateway_cli._profile_arg(str(nested))
        assert result == ""

    def test_invalid_profile_name_returns_empty(self, tmp_path, monkeypatch):
        """Profile names with invalid chars should not match the regex."""
        bad_profile = tmp_path / ".hermes" / "profiles" / "My Bot!"
        bad_profile.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        result = gateway_cli._profile_arg(str(bad_profile))
        assert result == ""

    def test_systemd_unit_includes_profile(self, tmp_path, monkeypatch):
        """generate_systemd_unit should include --profile in ExecStart for named profiles."""
        profile_dir = tmp_path / ".hermes" / "profiles" / "mybot"
        profile_dir.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: profile_dir)
        unit = gateway_cli.generate_systemd_unit(system=False)
        assert "--profile mybot" in unit
        assert "gateway run --replace" in unit

    def test_launchd_plist_includes_profile(self, tmp_path, monkeypatch):
        """generate_launchd_plist should include --profile in ProgramArguments for named profiles."""
        profile_dir = tmp_path / ".hermes" / "profiles" / "mybot"
        profile_dir.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: profile_dir)
        plist = gateway_cli.generate_launchd_plist()
        assert "<string>--profile</string>" in plist
        assert "<string>mybot</string>" in plist

    def test_launchd_plist_path_uses_real_user_home_not_profile_home(self, tmp_path, monkeypatch):
        profile_dir = tmp_path / ".hermes" / "profiles" / "orcha"
        profile_dir.mkdir(parents=True)
        machine_home = tmp_path / "machine-home"
        machine_home.mkdir()
        profile_home = profile_dir / "home"
        profile_home.mkdir()

        monkeypatch.setattr(Path, "home", lambda: profile_home)
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: profile_dir)
        monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(machine_home)))

        plist_path = gateway_cli.get_launchd_plist_path()

        assert plist_path == machine_home / "Library" / "LaunchAgents" / "ai.hermes.gateway-orcha.plist"


class TestRemapPathForUser:
    """Unit tests for _remap_path_for_user()."""

    def test_remaps_path_under_current_home(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "root")
        (tmp_path / "root").mkdir()
        result = gateway_cli._remap_path_for_user(
            str(tmp_path / "root" / ".hermes" / "hermes-agent"),
            str(tmp_path / "alice"),
        )
        assert result == str(tmp_path / "alice" / ".hermes" / "hermes-agent")

    def test_keeps_system_path_unchanged(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "root")
        (tmp_path / "root").mkdir()
        result = gateway_cli._remap_path_for_user("/opt/hermes", str(tmp_path / "alice"))
        assert result == "/opt/hermes"

    def test_noop_when_same_user(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "alice")
        (tmp_path / "alice").mkdir()
        original = str(tmp_path / "alice" / ".hermes" / "hermes-agent")
        result = gateway_cli._remap_path_for_user(original, str(tmp_path / "alice"))
        assert result == original


class TestSystemUnitPathRemapping:
    """System units must remap ALL paths from the caller's home to the target user."""

    def test_system_unit_has_no_root_paths(self, monkeypatch, tmp_path):
        root_home = tmp_path / "root"
        root_home.mkdir()
        project = root_home / ".hermes" / "hermes-agent"
        project.mkdir(parents=True)
        venv_bin = project / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text("")

        target_home = "/home/alice"

        monkeypatch.setattr(Path, "home", lambda: root_home)
        monkeypatch.setenv("HERMES_HOME", str(root_home / ".hermes"))
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: root_home / ".hermes")
        monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", project)
        monkeypatch.setattr(gateway_cli, "_detect_venv_dir", lambda: project / "venv")
        monkeypatch.setattr(gateway_cli, "get_python_path", lambda: str(venv_bin / "python"))
        monkeypatch.setattr(
            gateway_cli, "_system_service_identity",
            lambda run_as_user=None: ("alice", "alice", target_home),
        )

        unit = gateway_cli.generate_systemd_unit(system=True)

        # No root paths should leak into the unit
        assert str(root_home) not in unit
        # Target user paths should be present
        assert "/home/alice" in unit
        assert "WorkingDirectory=/home/alice/.hermes/hermes-agent" in unit


class TestDockerAwareGateway:
    """Tests for Docker container awareness in gateway commands."""

    def test_run_systemctl_raises_runtimeerror_when_missing(self, monkeypatch):
        """_run_systemctl raises RuntimeError with container guidance when systemctl is absent."""
        import pytest

        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("systemctl")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="systemctl is not available"):
            gateway_cli._run_systemctl(["start", "hermes-gateway"])

    def test_run_systemctl_passes_through_on_success(self, monkeypatch):
        """_run_systemctl delegates to subprocess.run when systemctl exists."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        result = gateway_cli._run_systemctl(["status", "hermes-gateway"])
        assert result.returncode == 0
        assert len(calls) == 1
        assert "status" in calls[0]

    def test_install_in_container_prints_docker_guidance(self, monkeypatch, capsys):
        """'hermes gateway install' inside Docker exits 0 with container guidance."""
        import pytest

        monkeypatch.setattr(gateway_cli, "is_managed", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_wsl", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_container", lambda: True)

        args = SimpleNamespace(gateway_command="install", force=False, system=False, run_as_user=None)
        with pytest.raises(SystemExit) as exc_info:
            gateway_cli.gateway_command(args)

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Docker" in out or "docker" in out
        assert "restart" in out.lower()

    def test_uninstall_in_container_prints_docker_guidance(self, monkeypatch, capsys):
        """'hermes gateway uninstall' inside Docker exits 0 with container guidance."""
        import pytest

        monkeypatch.setattr(gateway_cli, "is_managed", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_container", lambda: True)

        args = SimpleNamespace(gateway_command="uninstall", system=False)
        with pytest.raises(SystemExit) as exc_info:
            gateway_cli.gateway_command(args)

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "docker" in out.lower()

    def test_start_in_container_prints_docker_guidance(self, monkeypatch, capsys):
        """'hermes gateway start' inside Docker exits 0 with container guidance."""
        import pytest

        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_wsl", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_container", lambda: True)

        args = SimpleNamespace(gateway_command="start", system=False)
        with pytest.raises(SystemExit) as exc_info:
            gateway_cli.gateway_command(args)

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "docker" in out.lower()
        assert "hermes gateway run" in out


class TestLegacyHermesUnitDetection:
    """Tests for _find_legacy_hermes_units / has_legacy_hermes_units.

    These guard against the scenario that tripped Luis in April 2026: an
    older install left a ``hermes.service`` unit behind when the service was
    renamed to ``hermes-gateway.service``. After PR #5646 (signal recovery
    via systemd), the two services began SIGTERM-flapping over the same
    Telegram bot token in a 30-second cycle.

    The detector must flag ``hermes.service`` ONLY when it actually runs our
    gateway, and must NEVER flag profile units
    (``hermes-gateway-<profile>.service``) or unrelated third-party services.
    """

    # Minimal ExecStart that looks like our gateway
    _OUR_UNIT_TEXT = (
        "[Unit]\nDescription=Hermes Gateway\n[Service]\n"
        "ExecStart=/usr/bin/python -m hermes_cli.main gateway run --replace\n"
    )

    @staticmethod
    def _setup_search_paths(tmp_path, monkeypatch):
        """Redirect the legacy search to user_dir + system_dir under tmp_path."""
        user_dir = tmp_path / "user"
        system_dir = tmp_path / "system"
        user_dir.mkdir()
        system_dir.mkdir()
        monkeypatch.setattr(
            gateway_cli,
            "_legacy_unit_search_paths",
            lambda: [(False, user_dir), (True, system_dir)],
        )
        return user_dir, system_dir

    def test_detects_legacy_hermes_service_in_user_scope(self, tmp_path, monkeypatch):
        user_dir, _ = self._setup_search_paths(tmp_path, monkeypatch)
        legacy = user_dir / "hermes.service"
        legacy.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        results = gateway_cli._find_legacy_hermes_units()

        assert len(results) == 1
        name, path, is_system = results[0]
        assert name == "hermes.service"
        assert path == legacy
        assert is_system is False
        assert gateway_cli.has_legacy_hermes_units() is True

    def test_detects_legacy_hermes_service_in_system_scope(self, tmp_path, monkeypatch):
        _, system_dir = self._setup_search_paths(tmp_path, monkeypatch)
        legacy = system_dir / "hermes.service"
        legacy.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        results = gateway_cli._find_legacy_hermes_units()

        assert len(results) == 1
        name, path, is_system = results[0]
        assert name == "hermes.service"
        assert path == legacy
        assert is_system is True

    def test_ignores_profile_unit_hermes_gateway_coder(self, tmp_path, monkeypatch):
        """CRITICAL: profile units must NOT be flagged as legacy.

        Teknium's concern — ``hermes-gateway-coder.service`` is our standard
        naming for the ``coder`` profile. The legacy detector is an explicit
        allowlist, not a glob, so profile units are safe.
        """
        user_dir, system_dir = self._setup_search_paths(tmp_path, monkeypatch)
        # Drop profile units in BOTH scopes with our ExecStart
        for base in (user_dir, system_dir):
            (base / "hermes-gateway-coder.service").write_text(
                self._OUR_UNIT_TEXT, encoding="utf-8"
            )
            (base / "hermes-gateway-orcha.service").write_text(
                self._OUR_UNIT_TEXT, encoding="utf-8"
            )
            (base / "hermes-gateway.service").write_text(
                self._OUR_UNIT_TEXT, encoding="utf-8"
            )

        results = gateway_cli._find_legacy_hermes_units()

        assert results == []
        assert gateway_cli.has_legacy_hermes_units() is False

    def test_ignores_unrelated_hermes_service(self, tmp_path, monkeypatch):
        """Third-party ``hermes.service`` that isn't ours stays untouched.

        If a user has some other package named ``hermes`` installed as a
        service, we must not flag it.
        """
        user_dir, _ = self._setup_search_paths(tmp_path, monkeypatch)
        (user_dir / "hermes.service").write_text(
            "[Unit]\nDescription=Some Other Hermes\n[Service]\n"
            "ExecStart=/opt/other-hermes/bin/daemon --foreground\n",
            encoding="utf-8",
        )

        results = gateway_cli._find_legacy_hermes_units()

        assert results == []
        assert gateway_cli.has_legacy_hermes_units() is False

    def test_returns_empty_when_no_legacy_files_exist(self, tmp_path, monkeypatch):
        self._setup_search_paths(tmp_path, monkeypatch)

        assert gateway_cli._find_legacy_hermes_units() == []
        assert gateway_cli.has_legacy_hermes_units() is False

    def test_detects_both_scopes_simultaneously(self, tmp_path, monkeypatch):
        """When a user has BOTH user-scope and system-scope legacy units,
        both are reported so the migration step can remove them together."""
        user_dir, system_dir = self._setup_search_paths(tmp_path, monkeypatch)
        (user_dir / "hermes.service").write_text(self._OUR_UNIT_TEXT, encoding="utf-8")
        (system_dir / "hermes.service").write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        results = gateway_cli._find_legacy_hermes_units()

        scopes = sorted(is_system for _, _, is_system in results)
        assert scopes == [False, True]

    def test_accepts_alternate_execstart_formats(self, tmp_path, monkeypatch):
        """Older installs may have used different python invocations.

        ExecStart variants we've seen in the wild:
          - python -m hermes_cli.main gateway run
          - python path/to/hermes_cli/main.py gateway run
          - hermes gateway run   (direct binary)
          - python path/to/gateway/run.py
        """
        user_dir, _ = self._setup_search_paths(tmp_path, monkeypatch)
        variants = [
            "ExecStart=/venv/bin/python -m hermes_cli.main gateway run --replace",
            "ExecStart=/venv/bin/python /opt/hermes/hermes_cli/main.py gateway run",
            "ExecStart=/usr/local/bin/hermes gateway run --replace",
            "ExecStart=/venv/bin/python /opt/hermes/gateway/run.py",
        ]
        for i, execstart in enumerate(variants):
            name = f"hermes.service" if i == 0 else f"hermes.service"  # same name
            # Test each variant fresh
            (user_dir / "hermes.service").write_text(
                f"[Unit]\nDescription=Old Hermes\n[Service]\n{execstart}\n",
                encoding="utf-8",
            )
            results = gateway_cli._find_legacy_hermes_units()
            assert len(results) == 1, f"Variant {i} not detected: {execstart!r}"

    def test_print_legacy_unit_warning_is_noop_when_empty(self, tmp_path, monkeypatch, capsys):
        self._setup_search_paths(tmp_path, monkeypatch)

        gateway_cli.print_legacy_unit_warning()
        out = capsys.readouterr().out

        assert out == ""

    def test_print_legacy_unit_warning_shows_migration_hint(self, tmp_path, monkeypatch, capsys):
        user_dir, _ = self._setup_search_paths(tmp_path, monkeypatch)
        (user_dir / "hermes.service").write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        gateway_cli.print_legacy_unit_warning()
        out = capsys.readouterr().out

        assert "Legacy" in out
        assert "hermes.service" in out
        assert "hermes gateway migrate-legacy" in out

    def test_handles_unreadable_unit_file_gracefully(self, tmp_path, monkeypatch):
        """A permission error reading a unit file must not crash detection."""
        user_dir, _ = self._setup_search_paths(tmp_path, monkeypatch)
        unreadable = user_dir / "hermes.service"
        unreadable.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")
        # Simulate a read failure — monkeypatch Path.read_text to raise
        original_read_text = gateway_cli.Path.read_text

        def raising_read_text(self, *args, **kwargs):
            if self == unreadable:
                raise PermissionError("simulated")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(gateway_cli.Path, "read_text", raising_read_text)

        # Should not raise
        results = gateway_cli._find_legacy_hermes_units()
        assert results == []


class TestRemoveLegacyHermesUnits:
    """Tests for remove_legacy_hermes_units (the migration action)."""

    _OUR_UNIT_TEXT = (
        "[Unit]\nDescription=Hermes Gateway\n[Service]\n"
        "ExecStart=/usr/bin/python -m hermes_cli.main gateway run --replace\n"
    )

    @staticmethod
    def _setup(tmp_path, monkeypatch, as_root=False):
        user_dir = tmp_path / "user"
        system_dir = tmp_path / "system"
        user_dir.mkdir()
        system_dir.mkdir()
        monkeypatch.setattr(
            gateway_cli,
            "_legacy_unit_search_paths",
            lambda: [(False, user_dir), (True, system_dir)],
        )
        # Mock systemctl — return success for everything
        systemctl_calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            systemctl_calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr(gateway_cli.os, "geteuid", lambda: 0 if as_root else 1000)
        return user_dir, system_dir, systemctl_calls

    def test_returns_zero_when_no_legacy_units(self, tmp_path, monkeypatch, capsys):
        self._setup(tmp_path, monkeypatch)

        removed, remaining = gateway_cli.remove_legacy_hermes_units(interactive=False)

        assert removed == 0
        assert remaining == []
        assert "No legacy" in capsys.readouterr().out

    def test_dry_run_lists_without_removing(self, tmp_path, monkeypatch, capsys):
        user_dir, _, calls = self._setup(tmp_path, monkeypatch)
        legacy = user_dir / "hermes.service"
        legacy.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        removed, remaining = gateway_cli.remove_legacy_hermes_units(
            interactive=False, dry_run=True
        )

        assert removed == 0
        assert remaining == [legacy]
        assert legacy.exists()  # Not removed
        assert calls == []  # No systemctl invocations
        out = capsys.readouterr().out
        assert "dry-run" in out

    def test_removes_user_scope_legacy_unit(self, tmp_path, monkeypatch, capsys):
        user_dir, _, calls = self._setup(tmp_path, monkeypatch)
        legacy = user_dir / "hermes.service"
        legacy.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        removed, remaining = gateway_cli.remove_legacy_hermes_units(interactive=False)

        assert removed == 1
        assert remaining == []
        assert not legacy.exists()
        # Must have invoked stop → disable → daemon-reload on user scope
        cmds_joined = [" ".join(c) for c in calls]
        assert any("--user stop hermes.service" in c for c in cmds_joined)
        assert any("--user disable hermes.service" in c for c in cmds_joined)
        assert any("--user daemon-reload" in c for c in cmds_joined)

    def test_system_scope_without_root_defers_removal(self, tmp_path, monkeypatch, capsys):
        _, system_dir, calls = self._setup(tmp_path, monkeypatch, as_root=False)
        legacy = system_dir / "hermes.service"
        legacy.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        removed, remaining = gateway_cli.remove_legacy_hermes_units(interactive=False)

        assert removed == 0
        assert remaining == [legacy]
        assert legacy.exists()  # Not removed — requires sudo
        out = capsys.readouterr().out
        assert "sudo hermes gateway migrate-legacy" in out

    def test_system_scope_with_root_removes(self, tmp_path, monkeypatch, capsys):
        _, system_dir, calls = self._setup(tmp_path, monkeypatch, as_root=True)
        legacy = system_dir / "hermes.service"
        legacy.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        removed, remaining = gateway_cli.remove_legacy_hermes_units(interactive=False)

        assert removed == 1
        assert remaining == []
        assert not legacy.exists()
        cmds_joined = [" ".join(c) for c in calls]
        # System-scope uses plain "systemctl" (no --user)
        assert any(
            c.startswith("systemctl stop hermes.service") for c in cmds_joined
        )
        assert any(
            c.startswith("systemctl disable hermes.service") for c in cmds_joined
        )

    def test_removes_both_scopes_with_root(self, tmp_path, monkeypatch, capsys):
        user_dir, system_dir, _ = self._setup(tmp_path, monkeypatch, as_root=True)
        user_legacy = user_dir / "hermes.service"
        system_legacy = system_dir / "hermes.service"
        user_legacy.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")
        system_legacy.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        removed, remaining = gateway_cli.remove_legacy_hermes_units(interactive=False)

        assert removed == 2
        assert remaining == []
        assert not user_legacy.exists()
        assert not system_legacy.exists()

    def test_does_not_touch_profile_units_during_migration(
        self, tmp_path, monkeypatch, capsys
    ):
        """Teknium's constraint: profile units (hermes-gateway-coder.service)
        must survive a migration call, even if we somehow include them in the
        search dir."""
        user_dir, _, _ = self._setup(tmp_path, monkeypatch, as_root=True)
        profile_unit = user_dir / "hermes-gateway-coder.service"
        profile_unit.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")
        default_unit = user_dir / "hermes-gateway.service"
        default_unit.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        removed, remaining = gateway_cli.remove_legacy_hermes_units(interactive=False)

        assert removed == 0
        assert remaining == []
        # Both the profile unit and the current default unit must survive
        assert profile_unit.exists()
        assert default_unit.exists()

    def test_interactive_prompt_no_skips_removal(self, tmp_path, monkeypatch, capsys):
        """When interactive=True and user answers no, no removal happens."""
        user_dir, _, _ = self._setup(tmp_path, monkeypatch)
        legacy = user_dir / "hermes.service"
        legacy.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "prompt_yes_no", lambda *a, **k: False)

        removed, remaining = gateway_cli.remove_legacy_hermes_units(interactive=True)

        assert removed == 0
        assert remaining == [legacy]
        assert legacy.exists()


class TestMigrateLegacyCommand:
    """Tests for the `hermes gateway migrate-legacy` subcommand dispatch."""

    def test_migrate_legacy_subparser_accepts_dry_run_and_yes(self):
        """Verify the argparse subparser is registered and parses flags."""
        import hermes_cli.main as cli_main

        parser = cli_main.build_parser() if hasattr(cli_main, "build_parser") else None
        # Fall back to calling main's setup helper if direct access isn't exposed
        # The key thing: the subparser must exist. We verify by constructing
        # a namespace through argparse directly — but if build_parser isn't
        # public, just confirm that `hermes gateway --help` shows it.
        import subprocess
        import sys

        project_root = cli_main.PROJECT_ROOT if hasattr(cli_main, "PROJECT_ROOT") else None
        if project_root is None:
            import hermes_cli.gateway as gw
            project_root = gw.PROJECT_ROOT

        result = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "gateway", "--help"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "migrate-legacy" in result.stdout

    def test_gateway_command_migrate_legacy_dispatches(
        self, tmp_path, monkeypatch, capsys
    ):
        """gateway_command(args) with subcmd='migrate-legacy' calls the helper."""
        called = {}

        def fake_remove(interactive=True, dry_run=False):
            called["interactive"] = interactive
            called["dry_run"] = dry_run
            return 0, []

        monkeypatch.setattr(gateway_cli, "remove_legacy_hermes_units", fake_remove)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)

        args = SimpleNamespace(
            gateway_command="migrate-legacy", dry_run=False, yes=True
        )
        gateway_cli.gateway_command(args)

        assert called == {"interactive": False, "dry_run": False}


class TestGatewayStatusParser:
    def test_gateway_status_subparser_accepts_full_flag(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "gateway", "status", "-l", "--help"],
            cwd=str(gateway_cli.PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert result.returncode == 0
        assert "unrecognized arguments" not in result.stderr

    def test_gateway_command_migrate_legacy_dry_run_passes_through(
        self, monkeypatch
    ):
        called = {}

        def fake_remove(interactive=True, dry_run=False):
            called["interactive"] = interactive
            called["dry_run"] = dry_run
            return 0, []

        monkeypatch.setattr(gateway_cli, "remove_legacy_hermes_units", fake_remove)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)

        args = SimpleNamespace(
            gateway_command="migrate-legacy", dry_run=True, yes=False
        )
        gateway_cli.gateway_command(args)

        assert called == {"interactive": True, "dry_run": True}

    def test_migrate_legacy_on_unsupported_platform_prints_message(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)

        args = SimpleNamespace(
            gateway_command="migrate-legacy", dry_run=False, yes=True
        )
        gateway_cli.gateway_command(args)

        out = capsys.readouterr().out
        assert "only applies to systemd" in out


class TestSystemdInstallOffersLegacyRemoval:
    """Verify that systemd_install prompts to remove legacy units first."""

    def test_install_offers_removal_when_legacy_detected(
        self, tmp_path, monkeypatch, capsys
    ):
        """When legacy units exist, install flow should call the removal
        helper before writing the new unit."""
        remove_called = {}

        def fake_remove(interactive=True, dry_run=False):
            remove_called["invoked"] = True
            remove_called["interactive"] = interactive
            return 1, []

        # has_legacy_hermes_units must return True
        monkeypatch.setattr(gateway_cli, "has_legacy_hermes_units", lambda: True)
        monkeypatch.setattr(gateway_cli, "remove_legacy_hermes_units", fake_remove)
        monkeypatch.setattr(gateway_cli, "print_legacy_unit_warning", lambda: None)
        # Answer "yes" to the legacy-removal prompt
        monkeypatch.setattr(gateway_cli, "prompt_yes_no", lambda *a, **k: True)

        # Mock the rest of the install flow
        unit_path = tmp_path / "hermes-gateway.service"
        monkeypatch.setattr(
            gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path
        )
        monkeypatch.setattr(
            gateway_cli,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: "unit text\n",
        )
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        monkeypatch.setattr(gateway_cli, "_ensure_linger_enabled", lambda: None)

        gateway_cli.systemd_install()

        assert remove_called.get("invoked") is True
        assert remove_called.get("interactive") is False  # prompted elsewhere

    def test_install_declines_legacy_removal_when_user_says_no(
        self, tmp_path, monkeypatch
    ):
        """When legacy units exist and user declines, install still proceeds
        but doesn't touch them."""
        remove_called = {"invoked": False}

        def fake_remove(interactive=True, dry_run=False):
            remove_called["invoked"] = True
            return 0, []

        monkeypatch.setattr(gateway_cli, "has_legacy_hermes_units", lambda: True)
        monkeypatch.setattr(gateway_cli, "remove_legacy_hermes_units", fake_remove)
        monkeypatch.setattr(gateway_cli, "print_legacy_unit_warning", lambda: None)
        monkeypatch.setattr(gateway_cli, "prompt_yes_no", lambda *a, **k: False)

        unit_path = tmp_path / "hermes-gateway.service"
        monkeypatch.setattr(
            gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path
        )
        monkeypatch.setattr(
            gateway_cli,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: "unit text\n",
        )
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        monkeypatch.setattr(gateway_cli, "_ensure_linger_enabled", lambda: None)

        gateway_cli.systemd_install()

        # Helper must NOT have been called
        assert remove_called["invoked"] is False
        # New unit should still have been written
        assert unit_path.exists()
        assert unit_path.read_text() == "unit text\n"

    def test_install_skips_legacy_check_when_none_present(
        self, tmp_path, monkeypatch
    ):
        """No legacy → no prompt, no helper call."""
        prompt_called = {"count": 0}

        def counting_prompt(*a, **k):
            prompt_called["count"] += 1
            return True

        remove_called = {"invoked": False}

        def fake_remove(interactive=True, dry_run=False):
            remove_called["invoked"] = True
            return 0, []

        monkeypatch.setattr(gateway_cli, "has_legacy_hermes_units", lambda: False)
        monkeypatch.setattr(gateway_cli, "remove_legacy_hermes_units", fake_remove)
        monkeypatch.setattr(gateway_cli, "prompt_yes_no", counting_prompt)

        unit_path = tmp_path / "hermes-gateway.service"
        monkeypatch.setattr(
            gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path
        )
        monkeypatch.setattr(
            gateway_cli,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: "unit text\n",
        )
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        monkeypatch.setattr(gateway_cli, "_ensure_linger_enabled", lambda: None)

        gateway_cli.systemd_install()

        assert prompt_called["count"] == 0
        assert remove_called["invoked"] is False
