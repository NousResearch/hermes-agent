"""Tests for gateway service management helpers."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pwd = pytest.importorskip("pwd")
grp = pytest.importorskip("grp")

import hermes_cli.gateway as gateway_cli
from gateway import status
from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    GATEWAY_FATAL_CONFIG_EXIT_CODE,
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




    def test_systemd_restart_timeout_prints_status_guidance(self, monkeypatch, capsys):
        """`hermes gateway restart` must not surface a raw TimeoutExpired traceback.

        The dashboard spawns `hermes gateway restart` in the background; when a
        wedged adapter websocket pushes drain past the 90s CLI timeout, the
        dashboard would previously show a Python traceback (issue #19937
        follow-up: the same failure mode applies to restart, not just stop).
        """
        monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
        monkeypatch.setattr(gateway_cli, "_require_service_installed", lambda action, system=False: None)
        monkeypatch.setattr(gateway_cli, "_preflight_user_systemd", lambda: None)
        monkeypatch.setattr(gateway_cli, "refresh_systemd_unit_if_needed", lambda system=False: None)
        monkeypatch.setattr(status, "get_running_pid", lambda cleanup_stale=True: None)
        monkeypatch.setattr(gateway_cli, "_systemd_main_pid", lambda system=False: None)
        monkeypatch.setattr(
            gateway_cli,
            "_recover_pending_systemd_restart",
            lambda system=False, previous_pid=None: False,
        )
        monkeypatch.setattr(
            gateway_cli,
            "_systemd_service_is_start_limited",
            lambda system=False: False,
        )

        def fake_run_systemctl(args, **kwargs):
            # reset-failed is a pre-step (check=False, 30s) — let it pass.
            if args and args[0] == "reset-failed":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout"))

        monkeypatch.setattr(gateway_cli, "_run_systemctl", fake_run_systemctl)

        gateway_cli.systemd_restart()

        output = capsys.readouterr().out
        assert "still restarting after 90s" in output
        assert "hermes gateway status" in output


    def test_refresh_refuses_to_bake_pytest_tmpdir_into_real_user_unit(
        self, tmp_path, monkeypatch
    ):
        """Defense in depth: ``refresh_systemd_unit_if_needed()`` runs every
        time ``run_gateway()`` starts. The user-scope unit path resolves
        under ``Path.home()`` (NOT sandboxed by conftest), and
        ``generate_systemd_unit()`` bakes ``HERMES_HOME`` into the unit's
        ``Environment=`` line. Without this guard, any test that drives
        ``run_gateway()`` end-to-end on a real Linux dev box silently
        rewrites the developer's installed gateway unit with a
        ``/tmp/pytest-of-.../hermes_test`` HERMES_HOME — silently breaking
        their gateway on the next boot. The guard sniffs the generated
        unit body for tmpdir markers and refuses the write. Tests that
        legitimately exercise the refresh flow patch
        ``generate_systemd_unit`` to return synthetic content that doesn't
        carry those markers.
        """
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("old unit\n", encoding="utf-8")

        monkeypatch.setattr(
            gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path
        )
        # Realistic generated unit referencing a pytest tmpdir HERMES_HOME
        polluted_unit = (
            "[Service]\n"
            'Environment="HERMES_HOME=/tmp/pytest-of-alice/pytest-42/'
            'popen-gw0/test_x/hermes_test"\n'
        )
        monkeypatch.setattr(
            gateway_cli,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: polluted_unit,
        )

        # If the guard fails, daemon-reload would be called — record it.
        ran = []

        def fake_run(cmd, check=True, **kwargs):
            ran.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        result = gateway_cli.refresh_systemd_unit_if_needed(system=False)

        assert result is False, "refresh should refuse to write a polluted unit"
        assert (
            unit_path.read_text(encoding="utf-8") == "old unit\n"
        ), "installed unit must be left untouched"
        assert not any(
            "daemon-reload" in str(c) for c in ran
        ), "daemon-reload must not run when write was refused"



class TestTempHomeServiceDefinitionGuard:
    """_temp_home_in_service_definition() — structural temp-dir detection."""

    def test_detects_tmp_home_in_systemd_unit(self):
        unit = '[Service]\nEnvironment="HERMES_HOME=/tmp/hermes-e2e-41264"\n'
        assert (
            gateway_cli._temp_home_in_service_definition(unit)
            == "/tmp/hermes-e2e-41264"
        )


    def test_detects_tempdir_env_home(self, monkeypatch, tmp_path):
        import tempfile as _tempfile

        monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(tmp_path))
        unit = f'[Service]\nEnvironment="HERMES_HOME={tmp_path}/hermes-home"\n'
        assert gateway_cli._temp_home_in_service_definition(unit) is not None





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
        timeout = int(max(60, DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT + 30))
        return f"TimeoutStopSec={timeout}"



    def test_user_unit_does_not_leak_profile_node_symlink_target(self, tmp_path, monkeypatch):
        # Regression for the multi-profile gateway restart-loop flap (#48700):
        # ~/.local/bin/node is often a symlink into a *specific* profile's node
        # install. The generated unit's PATH must contain the symlink's own
        # directory (~/.local/bin), NOT the resolved profile target — otherwise
        # one profile's node path leaks into every profile's unit, making
        # systemd_unit_is_current() perpetually false and forcing a
        # daemon-reload restart loop on every boot.
        local_bin = tmp_path / ".local" / "bin"
        profile_node_bin = tmp_path / ".hermes" / "profiles" / "jarvis" / "node" / "bin"
        local_bin.mkdir(parents=True)
        profile_node_bin.mkdir(parents=True)
        real_node = profile_node_bin / "node"
        real_node.write_text("#!/bin/sh\n")
        link_node = local_bin / "node"
        link_node.symlink_to(real_node)

        monkeypatch.setattr(gateway_cli.shutil, "which", lambda cmd: str(link_node) if cmd == "node" else None)

        unit = gateway_cli.generate_systemd_unit(system=False)

        assert str(local_bin) in unit
        assert str(profile_node_bin) not in unit

    def test_launchd_plist_does_not_leak_profile_node_symlink_target(self, tmp_path, monkeypatch):
        # Same #48700 regression for the macOS twin generate_launchd_plist().
        local_bin = tmp_path / ".local" / "bin"
        profile_node_bin = tmp_path / ".hermes" / "profiles" / "jarvis" / "node" / "bin"
        local_bin.mkdir(parents=True)
        profile_node_bin.mkdir(parents=True)
        real_node = profile_node_bin / "node"
        real_node.write_text("#!/bin/sh\n")
        link_node = local_bin / "node"
        link_node.symlink_to(real_node)

        monkeypatch.setattr(gateway_cli.shutil, "which", lambda cmd: str(link_node) if cmd == "node" else None)

        plist = gateway_cli.generate_launchd_plist()

        assert str(local_bin) in plist
        assert str(profile_node_bin) not in plist



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


class TestLaunchdServiceRecovery:


    def test_refresh_defers_reload_when_running_inside_gateway_tree(self, tmp_path, monkeypatch):
        """#43842: when the refresh runs inside the gateway's own process tree,
        a direct bootout would kill this CLI before bootstrap. The reload must
        be delegated to a detached helper instead."""
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist>old content</plist>", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gateway_cli, "launchd_plist_is_current", lambda: False)
        monkeypatch.setattr(
            gateway_cli,
            "generate_launchd_plist",
            lambda: (
                "<plist>--replace\n<key>HERMES_HOME</key>"
                "<string>/Users/alice/.hermes</string></plist>"
            ),
        )
        # Pretend the gateway is running and that we ARE inside its tree.
        monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: 4242)
        monkeypatch.setattr(
            gateway_cli, "_is_pid_ancestor_of_current_process", lambda pid: pid == 4242
        )

        run_calls = []

        def fake_run(cmd, check=False, **kwargs):
            run_calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        popen_calls = []

        def fake_popen(cmd, **kwargs):
            popen_calls.append((cmd, kwargs))
            return SimpleNamespace(pid=9999)

        monkeypatch.setattr(gateway_cli.subprocess, "Popen", fake_popen)

        result = gateway_cli.refresh_launchd_plist_if_needed()

        assert result is True
        # The new plist was written.
        assert "--replace" in plist_path.read_text(encoding="utf-8")
        # No DIRECT bootout/bootstrap ran (those would kill us mid-sequence).
        assert not [c for c in run_calls if "bootout" in c or "bootstrap" in c]
        # Exactly one Popen call was made for the transient launchd job.
        assert len(popen_calls) == 1
        cmd, kwargs = popen_calls[0]
        # Must use `launchctl submit` (not `start_new_session=True`) so the
        # helper runs as a transient launchd job outside the gateway's process
        # coalition, surviving bootout (#69098).
        assert cmd[:3] == ["launchctl", "submit", "-l"]
        assert kwargs.get("start_new_session") is not True
        assert "-o" in cmd
        assert "-e" in cmd
        # The script is passed via -- /bin/bash -c ...
        bash_idx = cmd.index("--") + 1
        assert cmd[bash_idx] == "/bin/bash"
        assert cmd[bash_idx + 1] == "-c"
        script = cmd[bash_idx + 2]
        assert "bootout" in script and "bootstrap" in script
        assert str(plist_path) in script
        # The one-shot job must deregister its own transient label at the end,
        # otherwise every reload leaks a dead label in launchd.
        submit_label = cmd[cmd.index("-l") + 1]
        assert f"launchctl remove {submit_label}" in script


    def test_launchd_domain_uses_user_domain(self, monkeypatch):
        # The user/<uid> domain (not gui/<uid>) is the one reachable from
        # non-Aqua/background sessions on macOS 26+ (issue #23387).
        # When gui/<uid> fails to probe and user/<uid> succeeds,
        # _launchd_domain() must return user/<uid>.
        gateway_cli._resolved_launchd_domain = None
        monkeypatch.setattr(os, "getuid", lambda: 501)
        label = gateway_cli.get_launchd_label()

        def fake_run(cmd, check=False, **kwargs):
            if "print" in cmd and "gui/" in " ".join(cmd):
                raise subprocess.CalledProcessError(1, cmd, stderr="Domain error")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        assert gateway_cli._launchd_domain() == "user/501"



    # ── PID parsing ──────────────────────────────────────────────────────




    # ── Probe requires PID ───────────────────────────────────────────────


    # ── Unsupport marker lifecycle ───────────────────────────────────────



    # ── launchd_status with active supervision ───────────────────────────


    def test_launchd_status_reports_fallback_when_unsupported_and_pid_running(self, tmp_path, monkeypatch, capsys):
        """When the unsupported marker exists and a fallback PID is running."""
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text(gateway_cli.generate_launchd_plist(), encoding="utf-8")
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        def fake_run(cmd, capture_output=False, text=False, timeout=None, check=False, **kwargs):
            if isinstance(cmd, list) and cmd[:2] == ["launchctl", "list"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout='{\n    "Label" = "ai.hermes.gateway";\n    "OnDemand" = true;\n}',
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda cleanup_stale=False: 88888)
        # Pre-seed the unsupported marker
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: tmp_path)
        gateway_cli._write_launchd_unsupported_marker()

        gateway_cli.launchd_status()

        out = capsys.readouterr().out
        assert "cannot manage the gateway on this macos version" in out.lower()
        assert "Detached fallback process is running" in out
        assert "PID 88888" in out
        assert "NOT available" in out


class TestLaunchdDomainDetection:
    """Regression tests for _launchd_domain() probing (#40831).

    The function must detect which launchd domain actually contains (or can
    manage) the service, rather than hardcoding ``user/<uid>`` or ``gui/<uid>``.
    """

    def _reset_domain_cache(self):
        """Clear any cached domain result between tests."""
        gateway_cli._resolved_launchd_domain = None

    def test_prefers_gui_domain_when_service_loaded_there(self, monkeypatch):
        """In an Aqua session where the service is loaded under gui/<uid>,
        _launchd_domain() must return ``gui/<uid>`` — not ``user/<uid>``."""
        self._reset_domain_cache()
        monkeypatch.setattr(os, "getuid", lambda: 501)
        label = gateway_cli.get_launchd_label()

        run_calls = []

        def fake_run(cmd, check=False, **kwargs):
            run_calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        domain = gateway_cli._launchd_domain()
        assert domain == "gui/501"
        # Should have probed gui first
        assert run_calls[0] == ["launchctl", "print", f"gui/501/{label}"]


    def test_managername_background_selects_user_domain(self, monkeypatch):
        """When managername is Background (non-Aqua), use user/<uid>."""
        self._reset_domain_cache()
        monkeypatch.setattr(os, "getuid", lambda: 501)

        def fake_run(cmd, check=False, **kwargs):
            if "print" in cmd:
                raise subprocess.CalledProcessError(1, cmd, stderr="not found")
            if "managername" in cmd:
                return SimpleNamespace(returncode=0, stdout="Background\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        domain = gateway_cli._launchd_domain()
        assert domain == "user/501"


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


class TestGatewaySystemServiceRouting:
    def test_systemd_restart_gracefully_restarts_running_service_and_waits(self, monkeypatch, capsys):
        calls = []

        monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
        monkeypatch.setattr(gateway_cli, "_require_service_installed", lambda action, system=False: None)
        monkeypatch.setattr(gateway_cli, "refresh_systemd_unit_if_needed", lambda system=False: calls.append(("refresh", system)))
        monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 12.0)
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda: 654,
        )
        monkeypatch.setattr(
            gateway_cli,
            "_graceful_restart_via_sigusr1",
            lambda pid, timeout: calls.append(("graceful", pid, timeout)) or True,
        )

        # Simulate systemctl reset-failed/restart followed by an active unit.
        # A plain start does not break systemd's auto-restart timer once the
        # old gateway has exited with the planned restart code.
        def fake_subprocess_run(cmd, **kwargs):
            if "reset-failed" in cmd:
                calls.append(("reset-failed", cmd))
                return SimpleNamespace(stdout="", returncode=0)
            if "restart" in cmd:
                calls.append(("restart", cmd))
                return SimpleNamespace(stdout="", returncode=0)
            raise AssertionError(f"Unexpected systemctl call: {cmd}")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_subprocess_run)
        monkeypatch.setattr(
            gateway_cli,
            "_wait_for_systemd_service_restart",
            lambda system=False, previous_pid=None: calls.append(("wait", system, previous_pid)) or True,
        )

        gateway_cli.systemd_restart()

        assert ("graceful", 654, 17.0) in calls
        assert any(call[0] == "reset-failed" for call in calls)
        assert any(call[0] == "restart" for call in calls)
        assert ("wait", False, 654) in calls
        out = capsys.readouterr().out.lower()
        assert "restarting gracefully" in out









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


    def test_user_unit_unaffected_by_change(self):
        # User-scope units should still use the calling user's HERMES_HOME
        unit = gateway_cli.generate_systemd_unit(system=False)

        hermes_home = str(gateway_cli.get_hermes_home().resolve())
        assert f'HERMES_HOME={hermes_home}' in unit


class TestSystemUnitRefreshSyncsHermesHome:
    """sudo system refresh must not flip TimeoutStopSec via /root/.hermes."""

    def test_refresh_adopts_unit_hermes_home_before_rewriting(self, tmp_path, monkeypatch):
        root_home = tmp_path / "root"
        alice_home = tmp_path / "alice"
        root_hermes = root_home / ".hermes"
        alice_hermes = alice_home / ".hermes"
        root_hermes.mkdir(parents=True)
        alice_hermes.mkdir(parents=True)
        (root_hermes / "config.yaml").write_text(
            "agent:\n  restart_drain_timeout: 60\n", encoding="utf-8"
        )
        (alice_hermes / "config.yaml").write_text(
            "agent:\n  restart_drain_timeout: 180\n", encoding="utf-8"
        )

        unit_path = tmp_path / "hermes-gateway.service"
        monkeypatch.setattr(Path, "home", staticmethod(lambda: root_home))
        monkeypatch.setattr(
            gateway_cli,
            "_system_service_identity",
            lambda run_as_user=None: ("alice", "alice", str(alice_home)),
        )
        monkeypatch.setattr(
            gateway_cli, "_build_user_local_paths", lambda home, existing: []
        )
        monkeypatch.setattr(gateway_cli.shutil, "which", lambda cmd: None)
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)
        monkeypatch.setattr(gateway_cli, "_run_systemctl", lambda *a, **k: None)
        monkeypatch.delenv("HERMES_RESTART_DRAIN_TIMEOUT", raising=False)

        # Correct installed unit (operator's HERMES_HOME + drain timeout).
        monkeypatch.setenv("HERMES_HOME", str(alice_hermes))
        good_unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")
        assert "TimeoutStopSec=210" in good_unit
        unit_path.write_text(good_unit, encoding="utf-8")

        # Simulate sudo without inherited HERMES_HOME (falls back to root).
        monkeypatch.setenv("HERMES_HOME", str(root_hermes))
        assert gateway_cli.refresh_systemd_unit_if_needed(system=True) is False
        assert unit_path.read_text(encoding="utf-8") == good_unit
        assert os.environ["HERMES_HOME"] == str(alice_hermes)
        assert gateway_cli.systemd_unit_is_current(system=True) is True

    def test_is_current_syncs_before_reading_unit(self, tmp_path, monkeypatch):
        """CHOKEPOINT INVARIANT: systemd_unit_is_current() must adopt the
        unit's pinned HERMES_HOME *before* it reads/compares the unit.

        This is the single site that enforces sync-before-compare for every
        path (refresh gates on it; status/install call it). If a future edit
        moves the sync after the read (or drops it), this test fails.
        """
        order = []
        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("[Unit]\n", encoding="utf-8")

        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)

        real_read_text = Path.read_text

        def tracking_sync(system):
            order.append("sync")

        def tracking_read_text(self, *a, **k):
            if self == unit_path:
                order.append("read")
            return real_read_text(self, *a, **k)

        monkeypatch.setattr(gateway_cli, "_sync_hermes_home_from_systemd_unit", tracking_sync)
        monkeypatch.setattr(Path, "read_text", tracking_read_text)
        # Avoid a real generate/compare — we only assert sync precedes read.
        monkeypatch.setattr(gateway_cli, "generate_systemd_unit", lambda **k: "[Unit]\n")
        monkeypatch.setattr(gateway_cli, "_read_systemd_user_from_unit", lambda p: None)

        gateway_cli.systemd_unit_is_current(system=True)

        assert order, "systemd_unit_is_current did not run sync or read"
        assert order[0] == "sync", f"sync must precede unit read; got {order}"
        assert "read" in order and order.index("sync") < order.index("read")

    def test_start_and_restart_delegate_sync_to_chokepoint(self, monkeypatch):
        """start/restart must NOT pre-sync at the callsite — the sync is owned
        by the systemd_unit_is_current chokepoint that refresh gates on. This
        pins the single-chokepoint design so a future edit can't reintroduce a
        redundant (or, worse, out-of-order) callsite sync.
        """
        for entry in ("systemd_start", "systemd_restart"):
            calls = []
            monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: True)
            monkeypatch.setattr(gateway_cli, "_require_root_for_system_service", lambda action: None)
            monkeypatch.setattr(
                gateway_cli, "_require_service_installed", lambda action, system=False: None
            )
            monkeypatch.setattr(
                gateway_cli,
                "_sync_hermes_home_from_systemd_unit",
                lambda system: calls.append("sync"),
            )
            monkeypatch.setattr(
                gateway_cli,
                "refresh_systemd_unit_if_needed",
                lambda system=False: calls.append("refresh"),
            )
            monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
            monkeypatch.setattr(gateway_cli, "_systemd_main_pid", lambda system=False: None)
            monkeypatch.setattr(
                gateway_cli,
                "_run_systemctl",
                lambda args, **kwargs: calls.append("systemctl")
                or SimpleNamespace(returncode=0, stdout="", stderr=""),
            )
            monkeypatch.setattr(
                gateway_cli,
                "_wait_for_systemd_service_restart",
                lambda system=False, previous_pid=None: True,
            )

            getattr(gateway_cli, entry)(system=True)

            # refresh runs; the callsite adds NO separate sync before it (the
            # chokepoint inside refresh->is_current owns the sync). Here refresh
            # is mocked out, so no "sync" should appear at all for the refresh
            # phase — proving the callsite pre-sync was removed.
            assert "refresh" in calls, f"{entry} must call refresh_systemd_unit_if_needed"
            assert calls.count("sync") == 0, (
                f"{entry} should delegate sync to the chokepoint, not pre-sync "
                f"at the callsite; got {calls}"
            )


class TestHermesHomeForTargetUser:
    """Unit tests for _hermes_home_for_target_user()."""

    def test_remaps_default_home(self, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/root")))
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


    def test_system_unit_includes_local_bin_in_path(self, monkeypatch):
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

        monkeypatch.delenv("SUDO_USER", raising=False)
        monkeypatch.setenv("USER", "root")
        monkeypatch.setenv("LOGNAME", "root")

        with pytest.raises(ValueError, match="pass --run-as-user root to override"):
            gateway_cli._system_service_identity(run_as_user=None)

    def test_explicit_root_is_allowed(self, monkeypatch):
        """When root is explicitly passed via --run-as-user root, allow it."""

        root_info = pwd.getpwnam("root")
        root_group = grp.getgrgid(root_info.pw_gid).gr_name

        username, group, home = gateway_cli._system_service_identity(run_as_user="root")
        assert username == "root"
        assert home == root_info.pw_dir

    def test_non_root_user_passes_through(self, monkeypatch):
        """Normal non-root user works as before."""

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



    def test_systemctl_cmd_calls_ensure_for_user_mode(self, monkeypatch):
        calls = []
        monkeypatch.setattr(gateway_cli, "_ensure_user_systemd_env", lambda: calls.append("called"))

        result = gateway_cli._systemctl_cmd(system=False)
        assert result == ["systemctl", "--user"]
        assert calls == ["called"]


class TestPreflightUserSystemd:
    """Tests for _preflight_user_systemd() — D-Bus reachability before systemctl --user.

    Covers issue #5130 / Rick's RHEL 9.6 SSH scenario: setup tries to start the
    gateway via ``systemctl --user start`` in a shell with no user D-Bus session,
    which previously failed with a raw ``CalledProcessError`` and no remediation.
    """


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







    def test_systemd_unit_for_target_user_includes_named_profile(self, tmp_path, monkeypatch):
        """sudo system install must keep the target user's named profile in ExecStart."""
        root_home = tmp_path / "root"
        target_home = tmp_path / "home" / "alice"
        root_profile = root_home / ".hermes" / "profiles" / "mybot"
        root_profile.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", lambda: root_home)
        monkeypatch.setenv("HERMES_HOME", str(root_profile))
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: root_profile)
        monkeypatch.setattr(
            gateway_cli,
            "_system_service_identity",
            lambda run_as_user=None: ("alice", "alice", str(target_home)),
        )

        unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")

        assert "ExecStart=" in unit
        assert "--profile mybot gateway run" in unit
        assert f'HERMES_HOME={target_home / ".hermes" / "profiles" / "mybot"}' in unit



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
        # WorkingDirectory is anchored at the target user's HERMES_HOME (stable,
        # always exists) — NOT the source checkout under it. Pinning cwd to the
        # checkout is the rot bug fixed alongside this: a relocated/removed
        # checkout would crash-loop the unit on CHDIR (status=200).
        assert "WorkingDirectory=/home/alice/.hermes" in unit
        assert "WorkingDirectory=/home/alice/.hermes/hermes-agent" not in unit


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
            name = "hermes.service" if i == 0 else "hermes.service"  # same name
            # Test each variant fresh
            (user_dir / "hermes.service").write_text(
                f"[Unit]\nDescription=Old Hermes\n[Service]\n{execstart}\n",
                encoding="utf-8",
            )
            results = gateway_cli._find_legacy_hermes_units()
            assert len(results) == 1, f"Variant {i} not detected: {execstart!r}"


    def test_print_legacy_unit_warning_shows_migration_hint(self, tmp_path, monkeypatch, capsys):
        user_dir, _ = self._setup_search_paths(tmp_path, monkeypatch)
        (user_dir / "hermes.service").write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        gateway_cli.print_legacy_unit_warning()
        out = capsys.readouterr().out

        assert "Legacy" in out
        assert "hermes.service" in out
        assert "hermes gateway migrate-legacy" in out



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


class TestSystemScopeRequiresRootError:
    """Tests for the SystemScopeRequiresRootError replacement of sys.exit(1).

    Before this change, ``_require_root_for_system_service`` called
    ``sys.exit(1)`` when non-root code tried a system-scope systemd
    operation. The wizard's ``except Exception`` guards don't catch
    ``SystemExit`` (it's a ``BaseException`` subclass), so the user was
    dumped at a bare shell prompt mid-setup. The fix raises a typed
    exception instead, which the wizard intercepts and handles with
    actionable remediation.
    """

    def test_require_root_raises_when_non_root(self, monkeypatch):
        monkeypatch.setattr(gateway_cli.os, "geteuid", lambda: 1000)

        with pytest.raises(gateway_cli.SystemScopeRequiresRootError) as excinfo:
            gateway_cli._require_root_for_system_service("start")

        assert excinfo.value.args[0] == "System gateway start requires root. Re-run with sudo."
        assert excinfo.value.args[1] == "start"
        # str(e) renders only the message, not the tuple repr, so that
        # wizard format strings like f"Failed: {e}" print cleanly.
        assert str(excinfo.value) == "System gateway start requires root. Re-run with sudo."
        assert f"Failed: {excinfo.value}" == "Failed: System gateway start requires root. Re-run with sudo."


    def test_error_is_runtime_error_subclass(self):
        """Wizards use ``except Exception`` guards — the error must be a
        ``RuntimeError`` (catchable by ``Exception``), NOT a ``SystemExit``
        (``BaseException``), so the wizard can recover from it.
        """
        err = gateway_cli.SystemScopeRequiresRootError("msg", "start")
        assert isinstance(err, RuntimeError)
        assert isinstance(err, Exception)
        assert not isinstance(err, SystemExit)


class TestSystemScopeWizardPreCheck:
    """Tests for _system_scope_wizard_would_need_root — the guard the
    wizard uses to detect the dead-end BEFORE prompting the user to start
    a service that will fail without sudo.
    """

    @staticmethod
    def _setup_units(tmp_path, monkeypatch, system_present: bool, user_present: bool):
        sys_dir = tmp_path / "sys"
        usr_dir = tmp_path / "usr"
        sys_dir.mkdir()
        usr_dir.mkdir()
        if system_present:
            (sys_dir / "hermes-gateway.service").write_text("[Unit]\n")
        if user_present:
            (usr_dir / "hermes-gateway.service").write_text("[Unit]\n")
        monkeypatch.setattr(
            gateway_cli,
            "get_systemd_unit_path",
            lambda system=False: (sys_dir if system else usr_dir) / "hermes-gateway.service",
        )

    def test_non_root_with_only_system_unit_returns_true(self, tmp_path, monkeypatch):
        self._setup_units(tmp_path, monkeypatch, system_present=True, user_present=False)
        monkeypatch.setattr(gateway_cli.os, "geteuid", lambda: 1000)

        assert gateway_cli._system_scope_wizard_would_need_root() is True


    def test_non_root_with_explicit_system_arg_returns_true(self, tmp_path, monkeypatch):
        # Caller passed system=True explicitly (e.g. ``hermes gateway start --system``).
        self._setup_units(tmp_path, monkeypatch, system_present=False, user_present=False)
        monkeypatch.setattr(gateway_cli.os, "geteuid", lambda: 1000)

        assert gateway_cli._system_scope_wizard_would_need_root(system=True) is True


class TestSystemScopeRemediationOutput:
    """Tests for _print_system_scope_remediation — the actionable guidance
    shown when the wizard detects a system-scope-only setup as non-root.
    """

    def test_start_remediation_mentions_sudo_systemctl_and_uninstall(self, capsys, monkeypatch):
        monkeypatch.setattr(gateway_cli, "get_service_name", lambda: "hermes-gateway")

        gateway_cli._print_system_scope_remediation("start")
        out = capsys.readouterr().out

        assert "system-wide service" in out
        assert "start requires root" in out
        assert "sudo systemctl start hermes-gateway" in out
        assert "sudo hermes gateway uninstall --system" in out
        assert "hermes gateway install" in out


class TestGatewayCommandCatchesSystemScopeError:
    """The direct CLI path (``hermes gateway start --system`` etc.) must
    still exit 1 with a clean message when non-root. The top-level
    ``gateway_command`` catches ``SystemScopeRequiresRootError`` and
    converts it back to ``sys.exit(1)``, preserving existing CLI behavior.
    """

    def test_non_root_system_start_exits_one_with_clean_message(self, tmp_path, monkeypatch, capsys):
        sys_dir = tmp_path / "sys"
        usr_dir = tmp_path / "usr"
        sys_dir.mkdir()
        usr_dir.mkdir()
        (sys_dir / "hermes-gateway.service").write_text("[Unit]\n")
        monkeypatch.setattr(
            gateway_cli,
            "get_systemd_unit_path",
            lambda system=False: (sys_dir if system else usr_dir) / "hermes-gateway.service",
        )
        monkeypatch.setattr(gateway_cli.os, "geteuid", lambda: 1000)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli, "kill_gateway_processes", lambda **kw: 0)

        args = SimpleNamespace(gateway_command="start", system=True, all=False)

        with pytest.raises(SystemExit) as excinfo:
            gateway_cli.gateway_command(args)

        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        # Renders the message, NOT the ``('msg', 'action')`` tuple repr
        assert "System gateway start requires root. Re-run with sudo." in out
        assert "('" not in out  # no tuple repr leaking through


class TestServiceWorkingDirIsStable:
    """The gateway service must anchor WorkingDirectory at a stable path
    (HERMES_HOME), never the source checkout / worktree, so a relocated or
    deleted checkout can't crash-loop the unit on CHDIR (status=200).
    """



    def test_user_unit_workingdirectory_is_hermes_home_not_checkout(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: home)
        unit = gateway_cli.generate_systemd_unit(system=False)
        wd = [l for l in unit.splitlines() if l.startswith("WorkingDirectory=")]
        assert wd, "unit has no WorkingDirectory line"
        value = wd[0].split("=", 1)[1]
        assert Path(value).resolve() == home.resolve()
        # The bug class: never pin cwd inside a transient worktree checkout.
        assert "/.worktrees/" not in value

    def test_launchd_workingdirectory_is_hermes_home(self, tmp_path, monkeypatch):
        import re

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: home)
        plist = gateway_cli.generate_launchd_plist()
        m = re.search(r"<key>WorkingDirectory</key>\s*<string>(.*?)</string>", plist)
        assert m, "plist has no WorkingDirectory entry"
        assert Path(m.group(1)).resolve() == home.resolve()
        assert "/.worktrees/" not in m.group(1)


class TestLaunchctlBootstrapEioRetry:
    """`_launchctl_bootstrap` must recover from a stale already-loaded label.

    On macOS, ``launchctl bootstrap`` of a label that is still registered in
    the domain fails with ``5: Input/output error`` (EIO). That is the *already
    loaded* case — recoverable by booting the leftover out and retrying — not a
    sign the domain is unmanageable. The regression this guards against
    misclassified a stale registration as "launchd cannot manage this macOS
    version" and needlessly degraded the gateway to a detached process.
    """

    PLIST = "/tmp/ai.hermes.gateway.plist"
    DOMAIN = "gui/501"
    LABEL = "ai.hermes.gateway"


    def test_eio_triggers_bootout_then_retry(self, monkeypatch):
        calls = []

        def fake_run(cmd, check=True, **kwargs):
            calls.append(cmd)
            bootstrap_calls = [c for c in calls if c[1] == "bootstrap"]
            # First bootstrap hits EIO; bootout clears it; retry succeeds.
            if cmd[1] == "bootstrap" and len(bootstrap_calls) == 1:
                raise subprocess.CalledProcessError(5, cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli._launchctl_bootstrap(self.DOMAIN, self.PLIST, self.LABEL)

        assert calls == [
            ["launchctl", "bootstrap", self.DOMAIN, self.PLIST],
            ["launchctl", "bootout", f"{self.DOMAIN}/{self.LABEL}"],
            ["launchctl", "bootstrap", self.DOMAIN, self.PLIST],
        ]

    def test_persistent_eio_reraises_for_domain_fallback(self, monkeypatch):
        # When the retry also fails, the error must propagate so callers apply
        # their _launchctl_domain_unsupported fallback (degrade to detached).
        def fake_run(cmd, check=True, **kwargs):
            if cmd[1] == "bootstrap":
                raise subprocess.CalledProcessError(5, cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            gateway_cli._launchctl_bootstrap(self.DOMAIN, self.PLIST, self.LABEL)
        assert excinfo.value.returncode == 5


class TestRetryLaunchctlBootstrapUntilRegistered:
    """`_retry_launchctl_bootstrap_until_registered` — salvage of #53277.

    Covers the three review findings the salvage hardens: retry until the
    label is actually LISTED (not just a zero bootstrap exit), TimeoutExpired
    is retried (not escaped leaving the service unloaded), and the retry is
    bounded by a wall-clock deadline rather than a fixed short window.
    """

    DOMAIN = "gui/501"
    PLIST = "/tmp/ai.hermes.gateway.plist"
    LABEL = "ai.hermes.gateway"

    def test_returns_true_once_label_is_registered(self, monkeypatch):
        """Success requires launchctl list to confirm registration, not just
        a zero bootstrap exit."""
        list_results = iter([1, 0])  # first check: not registered, second: registered

        def fake_run(cmd, check=False, **kwargs):
            if cmd[:2] == ["launchctl", "list"]:
                return SimpleNamespace(returncode=next(list_results))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr(gateway_cli.time, "sleep", lambda *_a, **_k: None)

        ok = gateway_cli._retry_launchctl_bootstrap_until_registered(
            self.DOMAIN, self.PLIST, self.LABEL,
            deadline=gateway_cli.time.monotonic() + 60,
        )
        assert ok is True

    def test_timeout_expired_is_retried_not_escaped(self, monkeypatch):
        """A bootstrap that times out must be retried — it leaves the service
        unloaded, so it must not escape the retry/log path (finding #2)."""
        attempts = {"bootstrap": 0}

        def fake_run(cmd, check=False, **kwargs):
            if cmd[1] == "bootstrap":
                attempts["bootstrap"] += 1
                if attempts["bootstrap"] == 1:
                    raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 30))
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if cmd[:2] == ["launchctl", "list"]:
                # registered only after the second (successful) bootstrap
                return SimpleNamespace(returncode=0 if attempts["bootstrap"] >= 2 else 1)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr(gateway_cli.time, "sleep", lambda *_a, **_k: None)

        ok = gateway_cli._retry_launchctl_bootstrap_until_registered(
            self.DOMAIN, self.PLIST, self.LABEL,
            deadline=gateway_cli.time.monotonic() + 60,
        )
        assert ok is True
        assert attempts["bootstrap"] >= 2  # the timeout was retried, not raised

