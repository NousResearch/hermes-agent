"""Tests for cmd_update gateway auto-restart — systemd + launchd coverage.

Ensures ``hermes update`` correctly detects running gateways managed by
systemd (Linux) or launchd (macOS) and restarts/informs the user properly,
rather than leaving zombie processes or telling users to manually restart
when launchd will auto-respawn.
"""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

import hermes_cli.gateway as gateway_cli
import hermes_cli.main as cli_main
from hermes_cli.main import cmd_update


# ---------------------------------------------------------------------------
# Skip the real-time sleeps inside cmd_update's restart-verification path
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_restart_verify_sleep(monkeypatch):
    """hermes_cli/main.py uses time.sleep(3) after systemctl restart to
    verify the service survived. Tests mock subprocess.run — nothing
    actually restarts — so the 3s wait is dead time.

    main.py does ``import time as _time`` at both module level (line 167)
    and inside functions (lines 3281, 4384, 4401). Patching the global
    ``time.sleep`` affects only the duration of this test.
    """
    import time as _real_time
    monkeypatch.setattr(_real_time, "sleep", lambda *_a, **_k: None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_side_effect(
    branch="main",
    verify_ok=True,
    commit_count="3",
    systemd_active=False,
    system_service_active=False,
    system_restart_rc=0,
    launchctl_loaded=False,
):
    """Build a subprocess.run side_effect that simulates git + service commands."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        # git rev-parse --abbrev-ref HEAD
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")

        # git rev-parse --verify origin/{branch}
        if "rev-parse" in joined and "--verify" in joined:
            rc = 0 if verify_ok else 128
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

        # git rev-list HEAD..origin/{branch} --count
        if "rev-list" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

        # systemctl list-units hermes-gateway* — discover all gateway services
        if "systemctl" in joined and "list-units" in joined:
            if "--user" in joined and systemd_active:
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout="hermes-gateway.service loaded active running Hermes Gateway\n",
                    stderr="",
                )
            elif "--user" not in joined and system_service_active:
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout="hermes-gateway.service loaded active running Hermes Gateway\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        # systemctl is-active — distinguish --user from system scope
        if "systemctl" in joined and "is-active" in joined:
            if "--user" in joined:
                if systemd_active:
                    return subprocess.CompletedProcess(cmd, 0, stdout="active\n", stderr="")
                return subprocess.CompletedProcess(cmd, 3, stdout="inactive\n", stderr="")
            else:
                # System-level check (no --user)
                if system_service_active:
                    return subprocess.CompletedProcess(cmd, 0, stdout="active\n", stderr="")
                return subprocess.CompletedProcess(cmd, 3, stdout="inactive\n", stderr="")

        # systemctl restart — distinguish --user from system scope
        if "systemctl" in joined and "restart" in joined:
            if "--user" not in joined and system_service_active:
                stderr = "" if system_restart_rc == 0 else "Failed to restart: Permission denied"
                return subprocess.CompletedProcess(cmd, system_restart_rc, stdout="", stderr=stderr)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        # launchctl list ai.hermes.gateway
        if "launchctl" in joined and "list" in joined:
            if launchctl_loaded:
                return subprocess.CompletedProcess(cmd, 0, stdout="PID\tStatus\tLabel\n123\t0\tai.hermes.gateway\n", stderr="")
            return subprocess.CompletedProcess(cmd, 113, stdout="", stderr="Could not find service")

        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


@pytest.fixture
def mock_args():
    return SimpleNamespace()


# ---------------------------------------------------------------------------
# Launchd plist includes --replace
# ---------------------------------------------------------------------------


class TestLaunchdPlistReplace:
    """The generated launchd plist must include --replace so respawned
    gateways kill stale instances."""

    def test_plist_contains_replace_flag(self):
        plist = gateway_cli.generate_launchd_plist()
        assert "--replace" in plist

    def test_plist_program_arguments_order(self):
        """--replace comes after 'run' in the ProgramArguments."""
        plist = gateway_cli.generate_launchd_plist()
        lines = [line.strip() for line in plist.splitlines()]
        # Find 'run' and '--replace' in the string entries
        string_values = [
            line.replace("<string>", "").replace("</string>", "")
            for line in lines
            if "<string>" in line and "</string>" in line
        ]
        assert "run" in string_values
        assert "--replace" in string_values
        run_idx = string_values.index("run")
        replace_idx = string_values.index("--replace")
        assert replace_idx == run_idx + 1


class TestLaunchdPlistPath:
    def test_plist_contains_environment_variables(self):
        plist = gateway_cli.generate_launchd_plist()
        assert "<key>EnvironmentVariables</key>" in plist
        assert "<key>PATH</key>" in plist
        assert "<key>VIRTUAL_ENV</key>" in plist
        assert "<key>HERMES_HOME</key>" in plist

    def test_plist_path_includes_venv_bin(self):
        plist = gateway_cli.generate_launchd_plist()
        detected = gateway_cli._detect_venv_dir()
        venv_bin = str(detected / "bin") if detected else str(gateway_cli.PROJECT_ROOT / "venv" / "bin")
        assert venv_bin in plist

    def test_plist_path_starts_with_venv_bin(self):
        plist = gateway_cli.generate_launchd_plist()
        lines = plist.splitlines()
        for i, line in enumerate(lines):
            if "<key>PATH</key>" in line.strip():
                path_value = lines[i + 1].strip()
                path_value = path_value.replace("<string>", "").replace("</string>", "")
                detected = gateway_cli._detect_venv_dir()
                venv_bin = str(detected / "bin") if detected else str(gateway_cli.PROJECT_ROOT / "venv" / "bin")
                assert path_value.startswith(venv_bin + ":")
                break
        else:
            raise AssertionError("PATH key not found in plist")

    def test_plist_path_includes_node_modules_bin(self):
        plist = gateway_cli.generate_launchd_plist()
        node_bin = str(gateway_cli.PROJECT_ROOT / "node_modules" / ".bin")
        lines = plist.splitlines()
        for i, line in enumerate(lines):
            if "<key>PATH</key>" in line.strip():
                path_value = lines[i + 1].strip()
                path_value = path_value.replace("<string>", "").replace("</string>", "")
                assert node_bin in path_value.split(":")
                break
        else:
            raise AssertionError("PATH key not found in plist")

    def test_plist_path_includes_current_env_path(self, monkeypatch):
        monkeypatch.setenv("PATH", "/custom/bin:/usr/bin:/bin")
        plist = gateway_cli.generate_launchd_plist()
        assert "/custom/bin" in plist

    def test_plist_path_deduplicates_venv_bin_when_already_in_path(self, monkeypatch):
        detected = gateway_cli._detect_venv_dir()
        venv_bin = str(detected / "bin") if detected else str(gateway_cli.PROJECT_ROOT / "venv" / "bin")
        monkeypatch.setenv("PATH", f"{venv_bin}:/usr/bin:/bin")
        plist = gateway_cli.generate_launchd_plist()
        lines = plist.splitlines()
        for i, line in enumerate(lines):
            if "<key>PATH</key>" in line.strip():
                path_value = lines[i + 1].strip()
                path_value = path_value.replace("<string>", "").replace("</string>", "")
                parts = path_value.split(":")
                assert parts.count(venv_bin) == 1
                break
        else:
            raise AssertionError("PATH key not found in plist")


class TestLaunchdPlistCurrentness:
    def test_launchd_plist_is_current_ignores_path_drift(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        monkeypatch.setenv("PATH", "/custom/bin:/usr/bin:/bin")
        plist_path.write_text(gateway_cli.generate_launchd_plist(), encoding="utf-8")

        monkeypatch.setenv("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")

        assert gateway_cli.launchd_plist_is_current() is True


# ---------------------------------------------------------------------------
# cmd_update — macOS launchd detection
# ---------------------------------------------------------------------------


class TestLaunchdPlistRefresh:
    """refresh_launchd_plist_if_needed rewrites stale plists (like systemd's
    refresh_systemd_unit_if_needed)."""

    def test_refresh_rewrites_stale_plist(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist>old content</plist>")

        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        calls = []
        def fake_run(cmd, check=False, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        result = gateway_cli.refresh_launchd_plist_if_needed()

        assert result is True
        # Plist should now contain the generated content (which includes --replace)
        assert "--replace" in plist_path.read_text()
        # Should have booted out then bootstrapped
        assert any("bootout" in str(c) for c in calls)
        assert any("bootstrap" in str(c) for c in calls)

    def test_refresh_skips_when_current(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        # Write the current expected content
        plist_path.write_text(gateway_cli.generate_launchd_plist())

        calls = []
        monkeypatch.setattr(
            gateway_cli.subprocess, "run",
            lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0),
        )

        result = gateway_cli.refresh_launchd_plist_if_needed()

        assert result is False
        assert len(calls) == 0  # No launchctl calls needed

    def test_refresh_skips_when_no_plist(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "nonexistent.plist"
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        result = gateway_cli.refresh_launchd_plist_if_needed()
        assert result is False

    def test_launchd_start_calls_refresh(self, tmp_path, monkeypatch):
        """launchd_start refreshes the plist before starting."""
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist>old</plist>")
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        calls = []
        def fake_run(cmd, check=False, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.launchd_start()

        # First calls should be refresh (bootout/bootstrap), then kickstart
        cmd_strs = [" ".join(c) for c in calls]
        assert any("bootout" in s for s in cmd_strs)
        assert any("kickstart" in s for s in cmd_strs)

    def test_launchd_start_recreates_missing_plist_and_loads_service(self, tmp_path, monkeypatch):
        """launchd_start self-heals when the plist file is missing entirely."""
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        assert not plist_path.exists()

        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        calls = []
        def fake_run(cmd, check=False, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.launchd_start()

        # Should have created the plist
        assert plist_path.exists()
        assert "--replace" in plist_path.read_text()

        cmd_strs = [" ".join(c) for c in calls]
        # Should bootstrap the new plist, then kickstart
        assert any("bootstrap" in s for s in cmd_strs)
        assert any("kickstart" in s for s in cmd_strs)
        # Should NOT call bootout (nothing to bootout)
        assert not any("bootout" in s for s in cmd_strs)


class TestCmdUpdateLaunchdRestart:
    """cmd_update correctly detects and handles launchd on macOS."""

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_detects_launchd_and_skips_manual_restart_message(
        self, mock_run, _mock_which, mock_args, capsys, tmp_path, monkeypatch,
    ):
        """When launchd is running the gateway, update should print
        'auto-restart via launchd' instead of 'Restart it with: hermes gateway run'."""
        # Create a fake launchd plist so is_macos + plist.exists() passes
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist/>")

        monkeypatch.setattr(
            gateway_cli, "is_macos", lambda: True,
        )
        monkeypatch.setattr(
            gateway_cli, "get_launchd_plist_path", lambda: plist_path,
        )

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            launchctl_loaded=True,
        )

        # Mock launchd_restart + find_gateway_pids (new code discovers all gateways)
        with patch.object(gateway_cli, "launchd_restart") as mock_launchd_restart, \
             patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Restarted" in captured
        assert "Restart manually: hermes gateway run" not in captured
        mock_launchd_restart.assert_called_once_with()

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_without_launchd_shows_manual_restart(
        self, mock_run, _mock_which, mock_args, capsys, tmp_path, monkeypatch,
    ):
        """When no service manager is running but manual gateway is found, show manual restart hint."""
        monkeypatch.setattr(
            gateway_cli, "is_macos", lambda: True,
        )
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        # plist does NOT exist — no launchd service
        monkeypatch.setattr(
            gateway_cli, "get_launchd_plist_path", lambda: plist_path,
        )

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            launchctl_loaded=False,
        )

        # Simulate a manual gateway process found by find_gateway_pids
        with patch.object(gateway_cli, "find_gateway_pids", return_value=[12345]), \
             patch("os.kill"):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Restart manually: hermes gateway run" in captured

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_restarts_profile_manual_gateways(
        self, mock_run, _mock_which, mock_args, capsys, tmp_path, monkeypatch,
    ):
        """Profile-mapped manual gateways are relaunched automatically after update."""
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)
        monkeypatch.setattr(
            gateway_cli,
            "get_launchd_plist_path",
            lambda: tmp_path / "ai.hermes.gateway.plist",
        )

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            launchctl_loaded=False,
        )
        process = gateway_cli.ProfileGatewayProcess(
            profile="coder",
            path=tmp_path / ".hermes" / "profiles" / "coder",
            pid=12345,
        )

        # ``find_gateway_pids`` is invoked twice: once to enumerate manual
         # PIDs to restart, then again ~3s later by the post-restart survivor
         # sweep (#17648). Return the live PID first, then an empty list to
         # simulate the process actually exiting after the graceful restart
         # — otherwise the sweep would SIGKILL pid 12345 even though graceful
         # drain succeeded, and ``kill.assert_not_called()`` would fire.
        with patch.object(gateway_cli, "find_gateway_pids", side_effect=[[12345], []]), \
             patch.object(gateway_cli, "find_profile_gateway_processes", return_value=[process]), \
             patch.object(gateway_cli, "launch_detached_profile_gateway_restart", return_value=True) as restart, \
             patch.object(gateway_cli, "_graceful_restart_via_sigusr1", return_value=True) as graceful, \
             patch("os.kill") as kill:
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        restart.assert_called_once_with("coder", 12345)
        graceful.assert_called_once()
        # Graceful drain succeeded — no SIGTERM fallback needed.
        kill.assert_not_called()
        assert "Restarting manual gateway profile(s): coder" in captured
        assert "Restart manually: hermes gateway run" not in captured

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_profile_manual_gateway_falls_back_to_sigterm(
        self, mock_run, _mock_which, mock_args, capsys, tmp_path, monkeypatch,
    ):
        """When graceful SIGUSR1 drain fails, manual profile restart falls back to SIGTERM."""
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)
        monkeypatch.setattr(
            gateway_cli,
            "get_launchd_plist_path",
            lambda: tmp_path / "ai.hermes.gateway.plist",
        )

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            launchctl_loaded=False,
        )
        process = gateway_cli.ProfileGatewayProcess(
            profile="coder",
            path=tmp_path / ".hermes" / "profiles" / "coder",
            pid=12345,
        )

        # See note in ``test_update_restarts_profile_manual_gateways``: the
        # post-restart survivor sweep (#17648) re-queries ``find_gateway_pids``
        # ~3s after the restart attempt. Return ``[]`` on the second call so
        # the SIGTERM fallback isn't escalated to SIGKILL by the sweep.
        with patch.object(gateway_cli, "find_gateway_pids", side_effect=[[12345], []]), \
             patch.object(gateway_cli, "find_profile_gateway_processes", return_value=[process]), \
             patch.object(gateway_cli, "launch_detached_profile_gateway_restart", return_value=True) as restart, \
             patch.object(gateway_cli, "_graceful_restart_via_sigusr1", return_value=False) as graceful, \
             patch("os.kill") as kill:
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        restart.assert_called_once_with("coder", 12345)
        graceful.assert_called_once()
        # Graceful drain returned False → SIGTERM fallback.
        kill.assert_called_once()
        assert "Restarting manual gateway profile(s): coder" in captured

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_with_systemd_still_restarts_via_systemd(
        self, mock_run, _mock_which, mock_args, capsys, monkeypatch,
    ):
        """On Linux with systemd active, update should restart via systemctl."""
        monkeypatch.setattr(
            gateway_cli, "is_macos", lambda: False,
        )
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            systemd_active=True,
        )

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Restarted hermes-gateway" in captured
        # Verify systemctl restart was called
        restart_calls = [
            c for c in mock_run.call_args_list
            if "restart" in " ".join(str(a) for a in c.args[0])
            and "systemctl" in " ".join(str(a) for a in c.args[0])
        ]
        assert len(restart_calls) == 1

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_prefers_sigusr1_over_systemctl_restart_when_mainpid_known(
        self, mock_run, _mock_which, mock_args, capsys, monkeypatch,
    ):
        """Drain-aware update: when systemctl show reports a MainPID, the
        update path sends SIGUSR1 and waits for graceful exit + respawn,
        instead of ``systemctl restart`` (which SIGKILLs in-flight agents).
        """
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        # Track state: before kill → "active" (old PID),
        # after kill + exit → briefly inactive, then "active" again (new PID).
        state = {"killed": False}

        def side_effect(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)

            if "rev-parse" in joined and "--abbrev-ref" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
            if "rev-parse" in joined and "--verify" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if "rev-list" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="3\n", stderr="")

            # Only expose a user-scope service.
            if "systemctl" in joined and "list-units" in joined:
                if "--user" in joined:
                    return subprocess.CompletedProcess(
                        cmd, 0,
                        stdout="hermes-gateway.service loaded active running\n",
                        stderr="",
                    )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            if "systemctl" in joined and "is-active" in joined:
                # Pre-kill: active.  Post-kill: active again (respawned by
                # Restart=on-failure).  The drain loop verifies liveness
                # separately via os.kill(pid, 0).
                return subprocess.CompletedProcess(cmd, 0, stdout="active\n", stderr="")

            # The new code path.
            if "systemctl" in joined and "show" in joined and "MainPID" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="4242\n", stderr="")

            # If systemctl restart is called, this test fails its intent —
            # but still let it succeed so we can assert it was NOT called.
            if "systemctl" in joined and "restart" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        # Track SIGUSR1 delivery and simulate the gateway draining + exiting.
        sigusr1_sent = {"value": False}

        def fake_kill(pid, sig):
            import signal as _s
            if pid == 4242 and sig == _s.SIGUSR1:
                sigusr1_sent["value"] = True
                state["killed"] = True
                return
            if pid == 4242 and sig == 0:
                # Liveness probe — report dead once SIGUSR1 has been sent.
                if state["killed"]:
                    raise ProcessLookupError()
                return
            # For any other PID/sig combination, succeed silently.
            return

        monkeypatch.setattr("os.kill", fake_kill)

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(mock_args)

        # SIGUSR1 must have been delivered to the gateway MainPID.
        assert sigusr1_sent["value"], "Expected SIGUSR1 to be sent to MainPID"

        # And `systemctl restart` must NOT have been used (that's the
        # non-draining kill-everything path we're moving away from).
        restart_calls = [
            c for c in mock_run.call_args_list
            if "systemctl" in " ".join(str(a) for a in c.args[0])
            and "restart" in " ".join(str(a) for a in c.args[0])
        ]
        assert restart_calls == [], (
            "Graceful SIGUSR1 succeeded; `systemctl restart` should not "
            f"have been called. Got: {restart_calls}"
        )

        captured = capsys.readouterr().out
        assert "draining" in captured.lower()
        assert "Restarted hermes-gateway" in captured

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_falls_back_to_systemctl_restart_when_sigusr1_times_out(
        self, mock_run, _mock_which, mock_args, capsys, monkeypatch,
    ):
        """If the gateway doesn't exit within the drain budget (e.g. old unit
        missing ``Restart=on-failure`` or an agent ignoring SIGUSR1), the
        update path falls back to ``systemctl restart``.
        """
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            systemd_active=True,
        )

        # Patch systemctl show to report MainPID=4242 so cmd_update attempts
        # the graceful path.
        orig = mock_run.side_effect
        def wrapped(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)
            if "systemctl" in joined and "show" in joined and "MainPID" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="4242\n", stderr="")
            return orig(cmd, **kwargs)
        mock_run.side_effect = wrapped

        # Simulate the drain helper failing to confirm a clean exit — either
        # because the gateway ignored SIGUSR1 or the drain budget was
        # exceeded.  cmd_update() should detect this and escalate.
        monkeypatch.setattr(
            "hermes_cli.gateway._graceful_restart_via_sigusr1",
            lambda pid, drain_timeout: False,
        )

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(mock_args)

        # Fallback kicked in → systemctl restart was called.
        restart_calls = [
            c for c in mock_run.call_args_list
            if "systemctl" in " ".join(str(a) for a in c.args[0])
            and "restart" in " ".join(str(a) for a in c.args[0])
        ]
        assert len(restart_calls) >= 1, (
            "Drain path failed; expected fallback `systemctl restart`."
        )

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_no_gateway_running_skips_restart(
        self, mock_run, _mock_which, mock_args, capsys, monkeypatch,
    ):
        """When no gateway is running, update should skip the restart section entirely."""
        monkeypatch.setattr(
            gateway_cli, "is_macos", lambda: False,
        )

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            systemd_active=False,
        )

        with patch("gateway.status.get_running_pid", return_value=None):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Stopped gateway" not in captured
        assert "Gateway restarted" not in captured
        assert "Gateway restarted via launchd" not in captured


# ---------------------------------------------------------------------------
# cmd_update — system-level systemd service detection
# ---------------------------------------------------------------------------


class TestCmdUpdateSystemService:
    """cmd_update detects system-level gateway services where --user fails."""

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_detects_system_service_and_restarts(
        self, mock_run, _mock_which, mock_args, capsys, monkeypatch,
    ):
        """When user systemd is inactive but a system service exists, restart via system scope."""
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            systemd_active=False,
            system_service_active=True,
        )

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Restarted hermes-gateway" in captured
        # Verify systemctl restart (no --user) was called
        restart_calls = [
            c for c in mock_run.call_args_list
            if "restart" in " ".join(str(a) for a in c.args[0])
            and "systemctl" in " ".join(str(a) for a in c.args[0])
            and "--user" not in " ".join(str(a) for a in c.args[0])
        ]
        assert len(restart_calls) == 1

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_system_service_restart_failure_shows_error(
        self, mock_run, _mock_which, mock_args, capsys, monkeypatch,
    ):
        """When system service restart fails, show the failure message."""
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            systemd_active=False,
            system_service_active=True,
            system_restart_rc=1,
        )

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Failed to restart" in captured

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_user_service_takes_priority_over_system(
        self, mock_run, _mock_which, mock_args, capsys, monkeypatch,
    ):
        """When both user and system services are active, both are restarted."""
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            systemd_active=True,
            system_service_active=True,
        )

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        # Both scopes are discovered and restarted
        assert "Restarted hermes-gateway" in captured


# ---------------------------------------------------------------------------
# Service PID exclusion — the core bug fix
# ---------------------------------------------------------------------------


class TestServicePidExclusion:
    """After restarting a service, the stale-process sweep must NOT kill
    the freshly-spawned service PID.  This was the root cause of the bug
    where ``hermes update`` would restart the gateway and immediately kill it.
    """

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_launchd_does_not_kill_service_pid(
        self, mock_run, _mock_which, mock_args, capsys, monkeypatch, tmp_path,
    ):
        """After launchd restart, the sweep must exclude the service PID."""
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist/>")

        monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_linux", lambda: False)
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        # The service PID that launchd manages after restart
        SERVICE_PID = 42000

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            launchctl_loaded=True,
        )

        # Simulate find_gateway_pids returning the service PID (the bug scenario)
        # and _get_service_pids returning the same PID to exclude it
        with patch.object(
            gateway_cli, "_get_service_pids", return_value={SERVICE_PID}
        ), patch.object(
            gateway_cli, "find_gateway_pids",
            side_effect=lambda exclude_pids=None, all_profiles=False: (
                [SERVICE_PID] if not exclude_pids else
                [p for p in [SERVICE_PID] if p not in exclude_pids]
            ),
        ), patch("os.kill") as mock_kill:
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        # Service was restarted
        assert "Restarted" in captured
        # The service PID should NOT have been killed by the manual sweep
        kill_calls = [
            c for c in mock_kill.call_args_list
            if c.args[0] == SERVICE_PID
        ]
        assert len(kill_calls) == 0, (
            f"Service PID {SERVICE_PID} was killed by the manual sweep — "
            f"this is the bug where update restarts then immediately kills the gateway"
        )
        # Should NOT show manual restart message
        assert "Restart manually" not in captured

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_systemd_does_not_kill_service_pid(
        self, mock_run, _mock_which, mock_args, capsys, monkeypatch,
    ):
        """After systemd restart, the sweep must exclude the service PID."""
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        SERVICE_PID = 55000

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            systemd_active=True,
        )

        with patch.object(
            gateway_cli, "_get_service_pids", return_value={SERVICE_PID}
        ), patch.object(
            gateway_cli, "find_gateway_pids",
            side_effect=lambda exclude_pids=None, all_profiles=False: (
                [SERVICE_PID] if not exclude_pids else
                [p for p in [SERVICE_PID] if p not in exclude_pids]
            ),
        ), patch("os.kill") as mock_kill:
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Restarted hermes-gateway" in captured
        # Service PID must not be killed
        kill_calls = [
            c for c in mock_kill.call_args_list
            if c.args[0] == SERVICE_PID
        ]
        assert len(kill_calls) == 0
        assert "Restart manually" not in captured

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_kills_manual_pid_but_not_service_pid(
        self, mock_run, _mock_which, mock_args, capsys, monkeypatch, tmp_path,
    ):
        """When both a service PID and a manual PID exist, only the manual one
        is killed."""
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist/>")

        monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_linux", lambda: False)
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        SERVICE_PID = 42000
        MANUAL_PID = 42999

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            launchctl_loaded=True,
        )

        # Survivor sweep (#17648) re-queries ``find_gateway_pids`` after
         # SIGTERM. ``os.kill`` is mocked, so the PID never "dies" — track
         # the killed-via-SIGTERM PIDs ourselves and exclude them on later
         # calls to simulate the OS reaping the process. Without this the
         # sweep escalates with SIGKILL and ``manual_kills == 2`` instead of 1.
        _killed_pids: set[int] = set()

        def fake_find(exclude_pids=None, all_profiles=False):
            _exclude = (exclude_pids or set()) | _killed_pids
            return [p for p in [SERVICE_PID, MANUAL_PID] if p not in _exclude]

        def fake_kill(pid, _sig):
            _killed_pids.add(pid)

        with patch.object(
            gateway_cli, "_get_service_pids", return_value={SERVICE_PID}
        ), patch.object(
            gateway_cli, "find_gateway_pids", side_effect=fake_find,
        ), patch("os.kill", side_effect=fake_kill) as mock_kill:
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Restarted" in captured
        # Manual PID should be killed
        manual_kills = [c for c in mock_kill.call_args_list if c.args[0] == MANUAL_PID]
        assert len(manual_kills) == 1
        # Service PID should NOT be killed
        service_kills = [c for c in mock_kill.call_args_list if c.args[0] == SERVICE_PID]
        assert len(service_kills) == 0
        # Should show manual stop message since manual PID was killed
        assert "Stopped 1 manual gateway" in captured


class TestGetServicePids:
    """Unit tests for _get_service_pids()."""

    def test_returns_systemd_main_pid(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)

        def fake_run(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)
            if "list-units" in joined:
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout="hermes-gateway.service loaded active running Hermes Gateway\n",
                    stderr="",
                )
            if "show" in joined and "MainPID" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="12345\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        pids = gateway_cli._get_service_pids()
        assert 12345 in pids

    def test_returns_launchd_pid(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_linux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)
        monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")

        def fake_run(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)
            if "launchctl" in joined and "list" in joined:
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout="PID\tStatus\tLabel\n67890\t0\tai.hermes.gateway\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        pids = gateway_cli._get_service_pids()
        assert 67890 in pids

    def test_returns_empty_when_no_services(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_linux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)

        pids = gateway_cli._get_service_pids()
        assert pids == set()

    def test_excludes_zero_pid(self, monkeypatch):
        """systemd returns MainPID=0 for stopped services; skip those."""
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)

        def fake_run(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)
            if "list-units" in joined:
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout="hermes-gateway.service loaded inactive dead Hermes Gateway\n",
                    stderr="",
                )
            if "show" in joined and "MainPID" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="0\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        pids = gateway_cli._get_service_pids()
        assert 0 not in pids
        assert pids == set()


class TestFindGatewayPidsExclude:
    """find_gateway_pids respects exclude_pids."""

    def test_excludes_specified_pids(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_windows", lambda: False)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0,
                stdout=(
                    "user  100  0.0  0.0  0  0  ?  S  00:00  0:00  python gateway/run.py\n"
                    "user  200  0.0  0.0  0  0  ?  S  00:00  0:00  python gateway/run.py\n"
                ),
                stderr="",
            )

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr("os.getpid", lambda: 999)

        pids = gateway_cli.find_gateway_pids(exclude_pids={100})
        assert 100 not in pids
        assert 200 in pids

    def test_no_exclude_returns_all(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "is_windows", lambda: False)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0,
                stdout=(
                    "user  100  0.0  0.0  0  0  ?  S  00:00  0:00  python gateway/run.py\n"
                    "user  200  0.0  0.0  0  0  ?  S  00:00  0:00  python gateway/run.py\n"
                ),
                stderr="",
            )

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr("os.getpid", lambda: 999)

        pids = gateway_cli.find_gateway_pids()
        assert 100 in pids
        assert 200 in pids

    def test_filters_to_current_profile(self, monkeypatch, tmp_path):
        profile_dir = tmp_path / ".hermes" / "profiles" / "orcha"
        profile_dir.mkdir(parents=True)
        monkeypatch.setattr(gateway_cli, "is_windows", lambda: False)
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: profile_dir)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0,
                stdout=(
                    "100 /Users/dgrieco/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main --profile orcha gateway run --replace\n"
                    "200 /Users/dgrieco/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main --profile other gateway run --replace\n"
                ),
                stderr="",
            )

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr("os.getpid", lambda: 999)
        monkeypatch.setattr(gateway_cli, "_get_service_pids", lambda: set())
        monkeypatch.setattr(gateway_cli, "_profile_arg", lambda hermes_home=None: "--profile orcha")

        pids = gateway_cli.find_gateway_pids()

        assert pids == [100]


# ---------------------------------------------------------------------------
# Gateway mode writes exit code before restart (#8300)
# ---------------------------------------------------------------------------


class TestGatewayModeWritesExitCodeEarly:
    """When running as ``hermes update --gateway``, the exit code marker must be
    written *before* the gateway restart attempt.  Without this, systemd's
    ``KillMode=mixed`` kills the update process (and its wrapping shell) during
    the cgroup teardown, so the shell epilogue that normally writes the exit
    code never executes.  The new gateway's update watcher then polls for 30
    minutes and sends a spurious timeout message.
    """

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_exit_code_written_in_gateway_mode(
        self, mock_run, _mock_which, capsys, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        # Point HERMES_HOME at a temp dir so the marker file lands there
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        import hermes_cli.config as _cfg
        monkeypatch.setattr(_cfg, "get_hermes_home", lambda: hermes_home)
        # Also patch the module-level ref used by cmd_update
        import hermes_cli.main as _main_mod
        monkeypatch.setattr(_main_mod, "get_hermes_home", lambda: hermes_home)

        mock_run.side_effect = _make_run_side_effect(commit_count="1")

        args = SimpleNamespace(gateway=True)

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(args)

        exit_code_path = hermes_home / ".update_exit_code"
        assert exit_code_path.exists(), ".update_exit_code not written in gateway mode"
        assert exit_code_path.read_text() == "0"

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_exit_code_not_written_in_normal_mode(
        self, mock_run, _mock_which, capsys, tmp_path, monkeypatch,
    ):
        """Non-gateway mode should NOT write the exit code (the shell does it)."""
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        import hermes_cli.config as _cfg
        monkeypatch.setattr(_cfg, "get_hermes_home", lambda: hermes_home)
        import hermes_cli.main as _main_mod
        monkeypatch.setattr(_main_mod, "get_hermes_home", lambda: hermes_home)

        mock_run.side_effect = _make_run_side_effect(commit_count="1")

        args = SimpleNamespace(gateway=False)

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(args)

        exit_code_path = hermes_home / ".update_exit_code"
        assert not exit_code_path.exists(), ".update_exit_code should not be written outside gateway mode"

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_exit_code_written_before_restart_call(
        self, mock_run, _mock_which, capsys, tmp_path, monkeypatch,
    ):
        """Exit code must exist BEFORE systemctl restart is called."""
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        import hermes_cli.config as _cfg
        monkeypatch.setattr(_cfg, "get_hermes_home", lambda: hermes_home)
        import hermes_cli.main as _main_mod
        monkeypatch.setattr(_main_mod, "get_hermes_home", lambda: hermes_home)

        exit_code_path = hermes_home / ".update_exit_code"

        # Track whether exit code exists when systemctl restart is called
        exit_code_existed_at_restart = []

        original_side_effect = _make_run_side_effect(
            commit_count="1", systemd_active=True,
        )

        def tracking_side_effect(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)
            if "systemctl" in joined and "restart" in joined:
                exit_code_existed_at_restart.append(exit_code_path.exists())
            return original_side_effect(cmd, **kwargs)

        mock_run.side_effect = tracking_side_effect

        args = SimpleNamespace(gateway=True)

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(args)

        assert exit_code_existed_at_restart, "systemctl restart was never called"
        assert exit_code_existed_at_restart[0] is True, \
            ".update_exit_code must exist BEFORE systemctl restart (cgroup kill race)"


class TestCmdUpdateLegacyGatewayWarning:
    """Tests for the legacy hermes.service warning printed by `hermes update`.

    Users who installed Hermes before the service rename often have a
    dormant ``hermes.service`` that starts flap-fighting the current
    ``hermes-gateway.service`` after PR #5646. Every ``hermes update``
    should remind them to run ``hermes gateway migrate-legacy`` until
    they do.
    """

    _OUR_UNIT_TEXT = (
        "[Unit]\nDescription=Hermes Gateway\n[Service]\n"
        "ExecStart=/usr/bin/python -m hermes_cli.main gateway run --replace\n"
    )

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_prints_legacy_warning_when_detected(
        self, mock_run, _mock_which, mock_args, capsys, tmp_path, monkeypatch,
    ):
        """Legacy units present → warning in update output with migrate command."""
        user_dir = tmp_path / "user"
        system_dir = tmp_path / "system"
        user_dir.mkdir()
        system_dir.mkdir()
        legacy_path = user_dir / "hermes.service"
        legacy_path.write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        monkeypatch.setattr(
            gateway_cli,
            "_legacy_unit_search_paths",
            lambda: [(False, user_dir), (True, system_dir)],
        )
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        mock_run.side_effect = _make_run_side_effect(commit_count="3")

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Legacy Hermes gateway unit(s) detected" in captured
        assert "hermes.service" in captured
        assert "hermes gateway migrate-legacy" in captured
        assert "(user scope)" in captured

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_silent_when_no_legacy_units(
        self, mock_run, _mock_which, mock_args, capsys, tmp_path, monkeypatch,
    ):
        """No legacy units → no warning printed."""
        user_dir = tmp_path / "user"
        system_dir = tmp_path / "system"
        user_dir.mkdir()
        system_dir.mkdir()

        monkeypatch.setattr(
            gateway_cli,
            "_legacy_unit_search_paths",
            lambda: [(False, user_dir), (True, system_dir)],
        )
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        mock_run.side_effect = _make_run_side_effect(commit_count="3")

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Legacy Hermes gateway" not in captured
        assert "migrate-legacy" not in captured

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_does_not_flag_profile_units(
        self, mock_run, _mock_which, mock_args, capsys, tmp_path, monkeypatch,
    ):
        """Profile units (hermes-gateway-coder.service) must not trigger the warning.

        This is the core safety invariant: the legacy allowlist is
        ``hermes.service`` only, no globs.
        """
        user_dir = tmp_path / "user"
        system_dir = tmp_path / "system"
        user_dir.mkdir()
        system_dir.mkdir()
        # Drop a profile unit that an over-eager glob would match
        (user_dir / "hermes-gateway-coder.service").write_text(
            self._OUR_UNIT_TEXT, encoding="utf-8"
        )
        (user_dir / "hermes-gateway.service").write_text(
            self._OUR_UNIT_TEXT, encoding="utf-8"
        )

        monkeypatch.setattr(
            gateway_cli,
            "_legacy_unit_search_paths",
            lambda: [(False, user_dir), (True, system_dir)],
        )
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        mock_run.side_effect = _make_run_side_effect(commit_count="3")

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Legacy Hermes gateway" not in captured
        assert "hermes-gateway-coder.service" not in captured  # not flagged

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_skips_legacy_check_on_non_systemd_platforms(
        self, mock_run, _mock_which, mock_args, capsys, tmp_path, monkeypatch,
    ):
        """macOS / Windows / Termux — skip check entirely since the rename
        is systemd-specific."""
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        # Put a file that WOULD match if the check ran
        (user_dir / "hermes.service").write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        monkeypatch.setattr(
            gateway_cli,
            "_legacy_unit_search_paths",
            lambda: [(False, user_dir), (True, tmp_path / "system")],
        )
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3", launchctl_loaded=False,
        )

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        # Must not print the warning on non-systemd platforms
        assert "Legacy Hermes gateway" not in captured

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_lists_system_scope_unit_with_sudo_hint(
        self, mock_run, _mock_which, mock_args, capsys, tmp_path, monkeypatch,
    ):
        """System-scope legacy units need sudo — the warning must point that out."""
        user_dir = tmp_path / "user"
        system_dir = tmp_path / "system"
        user_dir.mkdir()
        system_dir.mkdir()
        (system_dir / "hermes.service").write_text(self._OUR_UNIT_TEXT, encoding="utf-8")

        monkeypatch.setattr(
            gateway_cli,
            "_legacy_unit_search_paths",
            lambda: [(False, user_dir), (True, system_dir)],
        )
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
        monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)

        mock_run.side_effect = _make_run_side_effect(commit_count="3")

        with patch.object(gateway_cli, "find_gateway_pids", return_value=[]):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Legacy Hermes gateway" in captured
        assert "(system scope)" in captured
        assert "sudo" in captured
