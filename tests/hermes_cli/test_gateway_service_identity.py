"""Tests for gateway service identity, routing, and unit homes."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pwd = pytest.importorskip("pwd")
grp = pytest.importorskip("grp")

import hermes_cli.gateway as gateway_cli


class TestGatewaySystemServiceRouting:
    def test_systemd_restart_gracefully_restarts_running_service_and_waits(self, monkeypatch, capsys):
        calls = []

        monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
        monkeypatch.setattr(gateway_cli, "_require_service_installed", lambda action, system=False: None)
        monkeypatch.setattr(gateway_cli, "_preflight_user_systemd", lambda **kwargs: None)
        monkeypatch.setattr(gateway_cli, "refresh_systemd_unit_if_needed", lambda system=False: calls.append(("refresh", system)))
        # Wait budget covers after-turn deferral + drain + headroom (#77184).
        monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 27.0)
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda: 654,
        )
        monkeypatch.setattr(
            gateway_cli,
            "_graceful_restart_via_sigusr1",
            lambda pid, timeout: calls.append(("graceful", pid, timeout)) or True,
        )

        # Once SIGUSR1 makes the gateway exit with the planned restart code,
        # systemd is the only restart owner.  The CLI must only observe the
        # replacement instead of issuing a second stop/start transition.
        def fake_subprocess_run(cmd, **kwargs):
            raise AssertionError(f"Unexpected systemctl call: {cmd}")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_subprocess_run)
        monkeypatch.setattr(
            gateway_cli,
            "_wait_for_systemd_service_restart",
            lambda system=False, previous_pid=None, replacement_observed=None: calls.append(
                ("wait", system, previous_pid)
            )
            or True,
        )

        gateway_cli.systemd_restart()

        assert ("graceful", 654, 27.0) in calls
        assert ("wait", False, 654) in calls
        out = capsys.readouterr().out.lower()
        assert "restarting gracefully" in out
        assert "21627" not in out  # must use the mocked budget, not live defaults
        assert "27" in out

    def test_systemd_restart_forces_recovery_only_when_handoff_has_no_replacement(
        self, monkeypatch, capsys
    ):
        calls = []

        monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
        monkeypatch.setattr(gateway_cli, "_require_service_installed", lambda action, system=False: None)
        monkeypatch.setattr(gateway_cli, "_preflight_user_systemd", lambda **kwargs: None)
        monkeypatch.setattr(gateway_cli, "refresh_systemd_unit_if_needed", lambda system=False: None)
        monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 27.0)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: 654)
        monkeypatch.setattr(gateway_cli, "_graceful_restart_via_sigusr1", lambda pid, timeout: True)
        waits = iter((False, True))
        monkeypatch.setattr(
            gateway_cli,
            "_wait_for_systemd_service_restart",
            lambda system=False, previous_pid=None, replacement_observed=None: next(waits),
        )
        monkeypatch.setattr(gateway_cli, "_systemd_service_is_start_limited", lambda system=False: False)
        monkeypatch.setattr(
            gateway_cli,
            "_read_systemd_unit_properties",
            lambda system=False, properties=None: {"ActiveState": "inactive", "MainPID": "0"},
        )
        monkeypatch.setattr(
            gateway_cli,
            "_run_systemctl",
            lambda args, **kwargs: calls.append((args, kwargs))
            or SimpleNamespace(returncode=0, stdout="", stderr=""),
        )

        gateway_cli.systemd_restart()

        assert [call[0][0] for call in calls] == ["reset-failed", "start"]
        assert "did not relaunch" in capsys.readouterr().out

    def test_systemd_restart_does_not_force_an_unready_replacement(self, monkeypatch):
        calls = []

        monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
        monkeypatch.setattr(gateway_cli, "_require_service_installed", lambda action, system=False: None)
        monkeypatch.setattr(gateway_cli, "_preflight_user_systemd", lambda **kwargs: None)
        monkeypatch.setattr(gateway_cli, "refresh_systemd_unit_if_needed", lambda system=False: None)
        monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 27.0)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: 654)
        monkeypatch.setattr(gateway_cli, "_graceful_restart_via_sigusr1", lambda pid, timeout: True)
        monkeypatch.setattr(
            gateway_cli,
            "_wait_for_systemd_service_restart",
            lambda system=False, previous_pid=None, replacement_observed=None: False,
        )
        monkeypatch.setattr(gateway_cli, "_systemd_service_is_start_limited", lambda system=False: False)
        monkeypatch.setattr(
            gateway_cli,
            "_read_systemd_unit_properties",
            lambda system=False, properties=None: {"ActiveState": "active", "MainPID": "777"},
        )
        monkeypatch.setattr(gateway_cli, "_run_systemctl", lambda args, **kwargs: calls.append(args))

        gateway_cli.systemd_restart()

        assert calls == []

    def test_systemd_restart_does_not_recover_a_failed_replacement(self, monkeypatch):
        calls = []

        monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
        monkeypatch.setattr(gateway_cli, "_require_service_installed", lambda action, system=False: None)
        monkeypatch.setattr(gateway_cli, "_preflight_user_systemd", lambda **kwargs: None)
        monkeypatch.setattr(gateway_cli, "refresh_systemd_unit_if_needed", lambda system=False: None)
        monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 27.0)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: 654)
        monkeypatch.setattr(gateway_cli, "_graceful_restart_via_sigusr1", lambda pid, timeout: True)

        def failed_replacement_wait(
            system=False, previous_pid=None, replacement_observed=None
        ):
            replacement_observed.append(True)
            return False

        monkeypatch.setattr(
            gateway_cli,
            "_wait_for_systemd_service_restart",
            failed_replacement_wait,
        )
        monkeypatch.setattr(gateway_cli, "_run_systemctl", lambda args, **kwargs: calls.append(args))

        gateway_cli.systemd_restart()

        assert calls == []

    def test_systemd_restart_does_not_recover_when_handoff_state_is_unknown(
        self, monkeypatch
    ):
        calls = []

        monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
        monkeypatch.setattr(gateway_cli, "_require_service_installed", lambda action, system=False: None)
        monkeypatch.setattr(gateway_cli, "_preflight_user_systemd", lambda **kwargs: None)
        monkeypatch.setattr(gateway_cli, "refresh_systemd_unit_if_needed", lambda system=False: None)
        monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 27.0)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: 654)
        monkeypatch.setattr(gateway_cli, "_graceful_restart_via_sigusr1", lambda pid, timeout: True)
        monkeypatch.setattr(
            gateway_cli,
            "_wait_for_systemd_service_restart",
            lambda system=False, previous_pid=None, replacement_observed=None: False,
        )
        monkeypatch.setattr(gateway_cli, "_systemd_service_is_start_limited", lambda system=False: False)
        monkeypatch.setattr(
            gateway_cli,
            "_read_systemd_unit_properties",
            lambda system=False, properties=None: {},
        )
        monkeypatch.setattr(gateway_cli, "_run_systemctl", lambda args, **kwargs: calls.append(args))

        gateway_cli.systemd_restart()

        assert calls == []

    def test_systemd_restart_wait_timeout_includes_supervisor_budgets(self, monkeypatch):
        monkeypatch.setattr(
            gateway_cli,
            "_read_systemd_unit_properties",
            lambda system=False, properties=None: {
                "RestartUSec": "5s",
                "TimeoutStartUSec": "1min 30s",
            },
        )

        assert gateway_cli._systemd_restart_wait_timeout() == 155.0

    def test_wait_records_a_short_lived_failed_replacement(self, monkeypatch):
        ticks = iter((0.0, 0.0, 2.0))
        monkeypatch.setattr(gateway_cli.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(gateway_cli.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(
            gateway_cli,
            "_read_systemd_unit_properties",
            lambda system=False, properties=None: {
                "ActiveState": "failed",
                "MainPID": "0",
            },
        )
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
        monkeypatch.setattr(
            gateway_cli,
            "_read_gateway_runtime_status",
            lambda: {"pid": 777, "gateway_state": "startup_failed"},
        )
        replacement_observed = []

        result = gateway_cli._wait_for_systemd_service_restart(
            previous_pid=654,
            timeout=1.0,
            replacement_observed=replacement_observed,
        )

        assert result is False
        assert replacement_observed == [True]

    def test_launchd_restart_uses_sigusr1_and_exit_wait_budget(self, monkeypatch, capsys):
        """launchd_restart must take the same graceful path as systemd_restart.

        Regression: it previously sent a bare SIGTERM and waited
        ``_get_restart_drain_timeout()`` (default 0), so the wait could never
        succeed and every restart fell through to ``kickstart -k``. A bare
        SIGTERM leaves ``restart_requested`` False, so the gateway exits 1
        instead of 75 and announces itself as "shutting down" rather than
        "restarting", dropping the resume_pending handoff.
        """
        calls = []

        monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
        monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
        monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: 654)
        monkeypatch.setattr(gateway_cli, "_request_gateway_self_restart", lambda pid: False)
        monkeypatch.setattr(
            gateway_cli,
            "probe_gateway_loop_liveness",
            lambda pid, **kw: gateway_cli.GATEWAY_LOOP_ALIVE,
        )
        # Wait budget covers after-turn deferral + drain + headroom (#77184);
        # the raw drain timeout (0 by default) must not be used here.
        monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 0.0)
        monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 27.0)
        monkeypatch.setattr(
            gateway_cli,
            "_graceful_restart_via_sigusr1",
            lambda pid, timeout: calls.append(("graceful", pid, timeout)) or True,
        )
        monkeypatch.setattr(
            gateway_cli,
            "terminate_pid",
            lambda pid, force=False: calls.append(("sigterm", pid)),
        )
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda *a, **k: calls.append(("kickstart", a[0])) or SimpleNamespace(
                returncode=0, stdout="", stderr=""
            ),
        )
        monkeypatch.setattr(gateway_cli, "_clear_launchd_unsupported_marker", lambda: None)
        # KeepAlive revives the label on a fresh PID — replacement observed.
        monkeypatch.setattr(
            gateway_cli,
            "_wait_for_launchd_service_pid",
            lambda label, old_pid, timeout=10.0, *, domain: calls.append(
                ("observe", label, old_pid, domain)
            )
            or True,
        )

        gateway_cli.launchd_restart()

        assert ("graceful", 654, 27.0) in calls
        # A bare SIGTERM would strand the gateway on the unplanned-shutdown path.
        assert not any(call[0] == "sigterm" for call in calls)
        # ``-k`` after a successful graceful exit would kill the replacement.
        assert not any(call[0] == "kickstart" for call in calls)
        # The success message must follow an observed replacement PID.
        assert ("observe", "ai.hermes.gateway", 654, "gui/501") in calls
        out = capsys.readouterr().out
        assert "up to 27s" in out
        assert "up to 0s" not in out

    def test_launchd_restart_forces_kickstart_when_no_replacement_appears(
        self, monkeypatch, capsys
    ):
        """A graceful exit with no KeepAlive revival must not report success.

        Detached-fallback gateways (macOS 26 unsupported-domain marker) and
        unloaded jobs also exit cleanly on SIGUSR1, but nobody revives them —
        and ``_graceful_restart_via_sigusr1`` returns True for an already-gone
        PID. Without replacement observation the CLI would print
        \"✓ Service restart requested\" while the gateway stays down.
        """
        calls = []

        monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
        monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
        monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: 654)
        monkeypatch.setattr(gateway_cli, "_request_gateway_self_restart", lambda pid: False)
        monkeypatch.setattr(
            gateway_cli,
            "probe_gateway_loop_liveness",
            lambda pid, **kw: gateway_cli.GATEWAY_LOOP_ALIVE,
        )
        monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 27.0)
        monkeypatch.setattr(
            gateway_cli, "_graceful_restart_via_sigusr1", lambda pid, timeout: True
        )
        monkeypatch.setattr(
            gateway_cli,
            "_wait_for_launchd_service_pid",
            lambda label, old_pid, timeout=10.0, *, domain: False,
        )
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda *a, **k: calls.append(("kickstart", a[0])) or SimpleNamespace(
                returncode=0, stdout="", stderr=""
            ),
        )
        monkeypatch.setattr(gateway_cli, "_clear_launchd_unsupported_marker", lambda: None)

        gateway_cli.launchd_restart()

        # No replacement observed → must escalate to kickstart -k.
        assert any(call[0] == "kickstart" for call in calls)
        out = capsys.readouterr().out
        assert "did not revive" in out
        assert "✓ Service restarted" in out






    @pytest.mark.macos_only
    def test_gateway_restart_does_not_fallback_to_foreground_when_launchd_restart_fails(self, tmp_path, monkeypatch):
        """macOS-gated: the branch under test is ``elif is_macos() and
        get_launchd_plist_path().exists()``. Faking the platform flags on Linux
        left ``supports_systemd_services()`` / ``launchctl`` semantics untested;
        on a real macOS host only ``launchd_restart`` is stubbed (it would touch
        the user's real launchd domain).
        """
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("plist\n", encoding="utf-8")

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

    def test_empty_managed_node_dir_uses_only_ambient_fallback(
        self, monkeypatch, tmp_path
    ):
        managed_bin = tmp_path / ".hermes" / "node" / "bin"
        managed_bin.mkdir(parents=True)
        monkeypatch.setattr(
            gateway_cli.shutil, "which", lambda name: "/opt/external-node/bin/node"
        )
        entries: list[str] = []

        gateway_cli._append_node_dir_for_service(entries, tmp_path / ".hermes")

        assert entries == ["/opt/external-node/bin"]

    def test_non_executable_managed_node_uses_only_ambient_fallback(
        self, monkeypatch, tmp_path
    ):
        managed_bin = tmp_path / ".hermes" / "node" / "bin"
        managed_bin.mkdir(parents=True)
        node = managed_bin / "node"
        node.write_text("#!/bin/sh\n")
        node.chmod(0o644)
        monkeypatch.setattr(
            gateway_cli.shutil, "which", lambda name: "/opt/external-node/bin/node"
        )
        entries: list[str] = []

        gateway_cli._append_node_dir_for_service(entries, tmp_path / ".hermes")

        assert entries == ["/opt/external-node/bin"]

    def test_managed_node_makes_system_unit_independent_of_callers_path(
        self, monkeypatch, tmp_path
    ):
        """A target-managed Node must suppress caller-specific PATH fallbacks."""
        target_home = tmp_path / "home" / "alice"
        target_hermes = target_home / ".hermes"
        root_home = tmp_path / "root"
        root_hermes = root_home / ".hermes"
        managed_bin = target_hermes / "node" / "bin"
        managed_bin.mkdir(parents=True)
        node = managed_bin / "node"
        node.write_text("#!/bin/sh\n")
        node.chmod(0o755)
        root_hermes.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", staticmethod(lambda: root_home))
        monkeypatch.setenv("HERMES_HOME", str(root_hermes))
        monkeypatch.setattr(
            gateway_cli,
            "_system_service_identity",
            lambda run_as_user=None: ("alice", "alice", str(target_home), 1001),
        )
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: root_hermes)
        monkeypatch.setattr(gateway_cli, "_build_service_path_dirs", lambda: [])

        monkeypatch.setattr(gateway_cli.shutil, "which", lambda name: "/root/bin/node")
        root_unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")

        monkeypatch.setattr(gateway_cli.shutil, "which", lambda name: "/home/alice/.local/bin/node")
        user_unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")

        assert root_unit == user_unit
        assert str(managed_bin) in root_unit
        assert "/root/bin" not in root_unit

    def test_node_path_lookup_remains_fallback_without_managed_node(
        self, monkeypatch, tmp_path
    ):
        """External Node installs still work when the managed tree is absent."""
        monkeypatch.setattr(
            "hermes_constants.iter_hermes_node_dirs", lambda root=None: []
        )
        monkeypatch.setattr(
            "hermes_constants.hermes_managed_node_tree_present",
            lambda root=None: False,
        )
        monkeypatch.setattr(
            gateway_cli.shutil, "which", lambda name: "/opt/external-node/bin/node"
        )
        entries: list[str] = []

        gateway_cli._append_node_dir_for_service(entries, tmp_path / ".hermes")

        assert entries == ["/opt/external-node/bin"]

    def test_system_unit_orders_after_target_user_manager(self, monkeypatch, tmp_path):
        """#104893: restart-safe workers need user@<uid>.service; the system unit must not race it at boot."""
        root_home = tmp_path / "root"
        root_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: root_home))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setattr(
            gateway_cli, "_system_service_identity",
            lambda run_as_user=None: ("alice", "alice", str(tmp_path), 1001),
        )
        monkeypatch.setattr(gateway_cli, "_build_service_path_dirs", lambda: [])

        system_unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")
        user_unit = gateway_cli.generate_systemd_unit(system=False)

        unit_section = system_unit.split("[Service]")[0]
        assert "After=user@1001.service" in unit_section
        assert "Wants=user@1001.service" in unit_section
        assert "user@" not in user_unit

    def test_system_unit_uses_target_user_home_not_calling_user(self, monkeypatch):
        # Simulate sudo: Path.home() returns /root, target user is alice
        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/root")))
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            gateway_cli, "_system_service_identity",
            lambda run_as_user=None: ("alice", "alice", "/home/alice", 1001),
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
            lambda run_as_user=None: ("alice", "alice", str(alice_home), 1001),
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

        username, group, home, _uid = gateway_cli._system_service_identity(run_as_user="root")
        assert username == "root"
        assert home == root_info.pw_dir

    def test_non_root_user_passes_through(self, monkeypatch):
        """Normal non-root user works as before."""

        monkeypatch.delenv("SUDO_USER", raising=False)
        monkeypatch.setenv("USER", "nobody")
        monkeypatch.setenv("LOGNAME", "nobody")

        try:
            username, group, home, _uid = gateway_cli._system_service_identity(run_as_user=None)
            assert username == "nobody"
        except ValueError as e:
            # "nobody" might not exist on all systems
            assert "Unknown user" in str(e)
