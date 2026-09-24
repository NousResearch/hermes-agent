"""Tests for hermes_cli.gateway."""

import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway


_BREAKAWAY_MARKER = "_HERMES_GATEWAY_BREAKAWAY"


@pytest.fixture(autouse=True)
def inert_task_scheduler_probe():
    """Tests that fake ``is_windows()`` send the reaper through ``_windows_scheduled_task_state``,
    which spawns ``pwsh`` whenever one is on PATH (GitHub's ubuntu runners ship it). On a loaded
    runner that spawn outlives its 10 s timeout and ``subprocess.run`` kills it through the test's
    globally patched ``os.kill`` — a foreign PID lands in ``killed_pids``. Tests of the probe itself
    call ``.undo()`` on this fixture to reach the real function."""
    mp = pytest.MonkeyPatch()
    mp.setattr(gateway, "_windows_scheduled_task_state", lambda name: None)
    yield mp
    mp.undo()


def _run_native_windows_gateway_start_diag(
    tmp_path, breakaway_marker: str | None
):
    script = textwrap.dedent(
        """
        import ctypes
        import json
        import os
        import pathlib
        import sys
        import types

        import hermes_cli.gateway as gateway_cli

        async def start_gateway(**kwargs):
            assert "_HERMES_GATEWAY_BREAKAWAY" not in os.environ
            return True

        fake_run = types.ModuleType("gateway.run")
        fake_run.start_gateway = start_gateway
        fake_run._exit_after_graceful_shutdown = lambda code: None
        sys.modules["gateway.run"] = fake_run

        gateway_cli._guard_official_docker_root_gateway = lambda: None
        gateway_cli._guard_named_profile_under_multiplexer = lambda force=False: None
        gateway_cli._attach_to_host_gateway_or_guard = lambda **kwargs: None
        gateway_cli._guard_supervised_gateway_conflict = lambda force=False: None
        gateway_cli._guard_existing_gateway_process_conflict = lambda replace=False: None
        gateway_cli._guard_fragile_foreground_gateway = lambda replace=False, force=False: None
        gateway_cli.supports_systemd_services = lambda: False
        gateway_cli.run_gateway(quiet=True)

        diag_path = pathlib.Path(os.environ["HERMES_HOME"]) / "logs" / "gateway-exit-diag.log"
        rows = [json.loads(line) for line in diag_path.read_text(encoding="utf-8").splitlines()]
        start = next(row for row in rows if row["tag"] == "gateway.start")
        payload = {
            "diag": start,
            "get_console_window": bool(ctypes.windll.kernel32.GetConsoleWindow()),
        }
        print("DIAG_JSON=" + json.dumps(payload))
        """
    )
    env: dict[str, str] = dict(os.environ)
    env.update(
        {
            "HERMES_HOME": str(tmp_path),
            "HERMES_GATEWAY_DETACHED": "1",
            "HERMES_GATEWAY_EXIT_DIAG": "1",
            "HERMES_GATEWAY_MAX_STARTS": "0",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    if breakaway_marker is None:
        env.pop(_BREAKAWAY_MARKER, None)
    else:
        env[_BREAKAWAY_MARKER] = breakaway_marker

    from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway

    completed = subprocess.run(
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=windows_detach_flags_without_breakaway(),
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    line = next(
        line for line in completed.stdout.splitlines() if line.startswith("DIAG_JSON=")
    )
    return json.loads(line.removeprefix("DIAG_JSON="))


@pytest.mark.windows_only
@pytest.mark.parametrize(
    ("marker", "expected_breakaway"),
    [("1", True), ("0", False), (None, None)],
)
def test_windows_gateway_start_diag_reports_detach_state(
    tmp_path, marker, expected_breakaway
):
    """DEVNULL is a Windows TTY but must not masquerade as a console window."""
    payload = _run_native_windows_gateway_start_diag(tmp_path, marker)
    diag = payload["diag"]

    assert payload["get_console_window"] is False
    assert diag["stdin_is_tty"] is True
    assert diag["console_window_attached"] is False
    assert diag["detached"] is True
    assert diag["breakaway"] is expected_breakaway


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX PTY coverage")
@pytest.mark.parametrize(
    ("stdin_is_tty", "outcome", "expected_exit"),
    [
        (True, "systemexit:75", 75),
        (False, "systemexit:75", 75),
        (False, "systemexit:78", 78),
        (False, "failure", 1),
    ],
)
def test_gateway_run_subprocess_preserves_daemon_exit_codes(
    tmp_path, stdin_is_tty, outcome, expected_exit
):
    """TTY state must not rewrite the gateway's process-level exit contract.

    Exit 75 is the intentional systemd/launchd restart handoff, exit 78 is a
    fatal configuration error, and a false startup result is a generic failure.
    In particular, a non-TTY daemon launch must not blanket-catch SystemExit,
    because doing so would hide genuine startup/configuration failures.
    """
    script = textwrap.dedent(
        """
        import os
        import sys
        import types

        import hermes_cli.gateway as gateway_cli

        outcome = os.environ["HERMES_TEST_GATEWAY_OUTCOME"]

        async def start_gateway(**kwargs):
            if outcome == "failure":
                return False
            raise SystemExit(int(outcome.split(":", 1)[1]))

        fake_run = types.ModuleType("gateway.run")
        fake_run.start_gateway = start_gateway
        setattr(fake_run, "_exit_after_graceful_shutdown", sys.exit)
        sys.modules["gateway.run"] = fake_run

        gateway_cli._guard_official_docker_root_gateway = lambda: None
        gateway_cli._guard_named_profile_under_multiplexer = lambda force=False: None
        gateway_cli._attach_to_host_gateway_or_guard = lambda **kwargs: None
        gateway_cli._guard_supervised_gateway_conflict = lambda force=False: None
        gateway_cli._guard_existing_gateway_process_conflict = lambda replace=False: None
        gateway_cli._guard_fragile_foreground_gateway = lambda replace=False, force=False: None
        gateway_cli.supports_systemd_services = lambda: False
        gateway_cli.run_gateway()
        """
    )
    env = {
        **os.environ,
        "HERMES_HOME": str(tmp_path),
        "HERMES_GATEWAY_EXIT_DIAG": "0",
        "HERMES_TEST_GATEWAY_OUTCOME": outcome,
        "INVOCATION_ID": "systemd-test",
    }

    master_fd = slave_fd = None
    try:
        if stdin_is_tty:
            # Imported here, not at module scope: ``pty`` pulls in ``termios``,
            # which does not exist on Windows, so a top-level import raises
            # ModuleNotFoundError during *collection* — before the skipif above
            # can take effect — and takes the whole module's Windows-viable
            # tests down with it.
            import pty

            master_fd, slave_fd = pty.openpty()
            stdin = slave_fd
        else:
            stdin = subprocess.DEVNULL
        completed = subprocess.run(
            [sys.executable, "-c", script],
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
    finally:
        if slave_fd is not None:
            os.close(slave_fd)
        if master_fd is not None:
            os.close(master_fd)

    assert completed.returncode == expected_exit, completed.stderr


def test_s6_runtime_snapshot_reports_supervised_service(monkeypatch, tmp_path):
    service_dir = tmp_path / "gateway-default"
    service_dir.mkdir()

    class FakeS6Manager:
        scandir = tmp_path

        def is_running(self, name):
            assert name == "gateway-default"
            return True

    monkeypatch.setattr(gateway, "is_linux", lambda: True)
    monkeypatch.setattr("hermes_constants.is_container", lambda: True)
    monkeypatch.setattr("hermes_cli.service_manager.detect_service_manager", lambda: "s6")
    monkeypatch.setattr("hermes_cli.service_manager.get_service_manager", lambda: FakeS6Manager())
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda: [123])
    monkeypatch.setattr(gateway, "_profile_suffix", lambda: "")

    snapshot = gateway.get_gateway_runtime_snapshot()

    assert snapshot.manager == "s6 (container supervisor)"
    assert snapshot.service_installed is True
    assert snapshot.service_running is True
    assert snapshot.service_scope == "s6"
    assert snapshot.gateway_pids == (123,)


class TestSystemdLingerStatus:
    def test_reports_enabled(self, monkeypatch):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setenv("USER", "alice")
        monkeypatch.setattr(
            gateway.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="yes\n", stderr=""),
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/loginctl")

        assert gateway.get_systemd_linger_status() == (True, "")


    def test_reports_termux_as_not_supported(self, monkeypatch):
        monkeypatch.setattr(gateway, "is_termux", lambda: True)

        assert gateway.get_systemd_linger_status() == (None, "not supported in Termux")


class TestContainerSystemdSupport:
    def test_supports_systemd_services_in_container_with_user_manager(self, monkeypatch):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "is_wsl", lambda: False)
        monkeypatch.setattr(gateway, "is_container", lambda: True)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemctl")
        monkeypatch.setattr(gateway, "_systemd_operational", lambda system=False: not system)

        assert gateway.supports_systemd_services() is True


def test_spawn_detached_gateway_timestamps_stderr(monkeypatch, tmp_path):
    calls = []
    child_cmd = [
        "/usr/bin/python3",
        "-m",
        "hermes_cli.main",
        "gateway",
        "run",
        "--replace",
    ]

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(gateway, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(gateway, "get_python_path", lambda: "/usr/bin/python3")
    monkeypatch.setattr(gateway, "_gateway_run_command", lambda: child_cmd)
    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)

    assert gateway._spawn_detached_gateway() is True

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd == [
        "/usr/bin/python3",
        "-m",
        "hermes_cli.stderr_timestamp",
        "--error-log",
        str(tmp_path / "logs" / "gateway.error.log"),
        "--",
        *child_cmd,
    ]
    assert kwargs["stdin"] is gateway.subprocess.DEVNULL
    assert kwargs["stderr"] is gateway.subprocess.DEVNULL
    assert kwargs["stdout"].name == str(tmp_path / "logs" / "gateway.log")


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="systemd user-linger is Linux-only (drives os.getuid())",
)
def test_systemd_install_checks_linger_status(monkeypatch, tmp_path):
    unit_path = tmp_path / "systemd" / "user" / "hermes-gateway.service"

    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: unit_path)
    # Synthetic unit with a non-temp home: the real generator bakes the
    # hermetic test HERMES_HOME (a tmp dir), which the temp-home write
    # guard correctly refuses.
    monkeypatch.setattr(
        gateway,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: (
            '[Service]\nEnvironment="HERMES_HOME=/home/alice/.hermes"\n'
        ),
    )

    calls = []
    helper_calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append((cmd, check))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway.subprocess, "run", fake_run)
    monkeypatch.setattr(gateway, "_ensure_linger_enabled", lambda: helper_calls.append(True))

    gateway.systemd_install(force=False)

    assert unit_path.exists()
    assert [cmd for cmd, _ in calls] == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", gateway.get_service_name()],
    ]
    assert helper_calls == [True]


def test_gateway_install_noninteractive_skips_legacy_unit_prompt(monkeypatch, tmp_path):
    """In non-TTY, the legacy-unit removal prompt in systemd_install is skipped.

    Covers the second hidden prompt that --start-now/--start-on-login do not
    guard. Originally contributed via PR #42124 (kyssta-exe).
    """
    monkeypatch.setattr(gateway, "has_legacy_hermes_units", lambda: True)

    calls = []
    monkeypatch.setattr(
        gateway,
        "prompt_yes_no",
        lambda question, default=True: calls.append(("prompt", question)) or True,
    )
    monkeypatch.setattr(gateway, "remove_legacy_hermes_units", lambda interactive=False: calls.append(("remove_legacy",)))
    monkeypatch.setattr(gateway, "print_legacy_unit_warning", lambda: None)

    fake_path = tmp_path / "hermes-gateway.service"
    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: fake_path)
    monkeypatch.setattr(gateway, "generate_systemd_unit", lambda system=False, run_as_user=None: "[Service]")
    monkeypatch.setattr(gateway, "_run_systemctl", lambda *a, **kw: None)
    monkeypatch.setattr(gateway, "_ensure_linger_enabled", lambda: None)
    monkeypatch.setattr(gateway, "print_systemd_scope_conflict_warning", lambda: None)
    monkeypatch.setattr(gateway, "_service_scope_label", lambda system=False: "user")

    gateway.systemd_install(non_interactive=True)

    # Legacy units removed without prompting.
    assert ("remove_legacy",) in calls
    assert all(c[0] != "prompt" for c in calls)


# ---------------------------------------------------------------------------
# _wait_for_gateway_exit
# ---------------------------------------------------------------------------


class TestWaitForGatewayExit:
    """PID-based wait with force-kill on timeout."""


    def test_force_kills_after_grace_period(self, monkeypatch):
        """When the process doesn't exit, force-kill the saved PID."""

        # Simulate monotonic time advancing past force_after
        call_num = 0
        def fake_monotonic():
            nonlocal call_num
            call_num += 1
            # First two calls: initial deadline + force_deadline setup (time 0)
            # Then each loop iteration advances time
            return call_num * 2.0  # 2, 4, 6, 8, ...

        kills = []
        def mock_terminate(pid, force=False, **kwargs):
            kills.append((pid, force))

        # get_running_pid returns the PID until kill is sent, then None
        def mock_get_running_pid():
            return None if kills else 42

        monkeypatch.setattr("time.monotonic", fake_monotonic)
        monkeypatch.setattr("time.sleep", lambda _: None)
        monkeypatch.setattr("gateway.status.get_running_pid", mock_get_running_pid)
        monkeypatch.setattr(gateway, "terminate_pid", mock_terminate)

        gateway._wait_for_gateway_exit(timeout=10.0, force_after=5.0)
        assert (42, True) in kills


    def test_kill_gateway_processes_force_uses_helper(self, monkeypatch):
        calls = []

        monkeypatch.setattr(gateway, "find_gateway_pids", lambda exclude_pids=None, all_profiles=False: [11, 22])
        # Kill-time re-verification: force-kills only proceed when the LIVE
        # cmdline still looks like a gateway.
        monkeypatch.setattr(
            gateway, "_capture_gateway_argv", lambda pid: ["python", "-m", "hermes_cli.main", "gateway", "run"]
        )
        monkeypatch.setattr(
            gateway,
            "terminate_pid",
            lambda pid, force=False, **kwargs: calls.append((pid, force)),
        )

        killed = gateway.kill_gateway_processes(force=True)

        assert killed == 2
        assert calls == [(11, True), (22, True)]

    def test_kill_gateway_processes_force_refuses_recycled_pid(self, monkeypatch):
        """A scanned PID whose live argv no longer looks like a gateway is skipped."""
        calls = []

        monkeypatch.setattr(gateway, "find_gateway_pids", lambda exclude_pids=None, all_profiles=False: [11, 22])
        monkeypatch.setattr(
            gateway,
            "_capture_gateway_argv",
            lambda pid: None if pid == 11 else ["python", "-m", "hermes_cli.main", "gateway", "run"],
        )
        monkeypatch.setattr(
            gateway,
            "terminate_pid",
            lambda pid, force=False, **kwargs: calls.append((pid, force)),
        )

        killed = gateway.kill_gateway_processes(force=True)

        assert killed == 1
        assert calls == [(22, True)]


class TestRestartWaitsForApiServerPort:
    """Regression for #91547: ``hermes gateway restart`` waited only for the old PID; on macOS the
    replacement then hit EADDRINUSE and ran with no API server."""

    def test_port_is_reported_free_once_the_old_listener_closes(self):
        import socket
        import threading

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        port = listener.getsockname()[1]
        threading.Timer(0.3, listener.close).start()

        assert gateway._wait_for_tcp_port_free("127.0.0.1", port, timeout=5.0) is True

    def test_wait_targets_the_configured_api_server_port_only_when_enabled(self, monkeypatch):
        import socket

        from gateway.config import GatewayConfig, Platform, PlatformConfig

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        port = listener.getsockname()[1]
        cfg = GatewayConfig()
        cfg.platforms[Platform.API_SERVER] = PlatformConfig(enabled=True, extra={"port": port})
        monkeypatch.setattr(gateway, "load_gateway_config", lambda: cfg)
        monkeypatch.delenv("API_SERVER_PORT", raising=False)
        try:
            # config.yaml port wins over the env default: the busy configured port is what we wait on
            assert gateway._wait_for_api_server_port_free(timeout=0.3) is False
            cfg.platforms[Platform.API_SERVER].enabled = False
            assert gateway._wait_for_api_server_port_free(timeout=0.3) is True
        finally:
            listener.close()


class TestStopProfileGateway:
    @pytest.mark.windows_only
    def test_windows_stop_drains_marker_before_force_termination(self, monkeypatch):
        """Windows must let the marker watcher run before escalating (#112750)."""
        import hermes_cli.gateway_windows as gateway_windows

        pid = 12345
        calls = []
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: pid)
        monkeypatch.setattr(gateway_windows, "_windows_stop_drain_timeout", lambda: 7.0)
        monkeypatch.setattr(
            gateway_windows,
            "_drain_gateway_pid",
            lambda target, timeout: calls.append(("drain", target, timeout)) or True,
        )
        monkeypatch.setattr(
            gateway_windows,
            "_force_terminate_known_gateway_pids",
            lambda pids: calls.append(("force", pids)),
        )
        monkeypatch.setattr(gateway.os, "kill", lambda *_: calls.append(("kill",)))
        monkeypatch.setattr("gateway.status._pid_exists", lambda target: False)
        monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
        monkeypatch.setattr(gateway, "_reap_unsupervised_gateway_orphans", lambda **_: False)

        assert gateway.stop_profile_gateway() is True
        assert calls == [("drain", pid, 7.0)]

    @pytest.mark.windows_only
    def test_windows_stop_force_terminates_only_after_drain_timeout(self, monkeypatch):
        """A wedged Windows gateway still has a bounded force-stop fallback (#112750)."""
        import hermes_cli.gateway_windows as gateway_windows

        pid = 12345
        calls = []
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: pid)
        monkeypatch.setattr(gateway_windows, "_windows_stop_drain_timeout", lambda: 7.0)
        monkeypatch.setattr("gateway.status.get_process_start_time", lambda target: 100)
        monkeypatch.setattr(
            gateway_windows,
            "_drain_gateway_pid",
            lambda target, timeout: calls.append(("drain", target, timeout)) or False,
        )
        monkeypatch.setattr(
            gateway_windows,
            "_force_terminate_known_gateway_pids",
            lambda pids: calls.append(("force", pids)),
        )
        monkeypatch.setattr(gateway.os, "kill", lambda *_: calls.append(("kill",)))
        monkeypatch.setattr("gateway.status._pid_exists", lambda target: False)
        monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
        monkeypatch.setattr(gateway, "_reap_unsupervised_gateway_orphans", lambda **_: False)

        assert gateway.stop_profile_gateway() is True
        assert calls == [("drain", pid, 7.0), ("force", {pid: 100})]

    @pytest.mark.windows_only
    def test_windows_stop_force_kill_carries_pre_drain_identity(self, monkeypatch):
        """The post-drain taskkill must be guarded by the start time captured BEFORE the <=30 s drain:
        a PID recycled during the wait shows a different start time, so ``terminate_pid``'s mismatch
        refusal fires instead of killing an unrelated process. Reading it at kill time is a vacuous
        self-comparison."""
        import gateway.status as status
        import hermes_cli.gateway_windows as gateway_windows

        pid = 4242
        calls = []
        clock = {"now": 0.0}
        alive = {"value": True}

        monkeypatch.setattr(gateway_windows.time, "monotonic", lambda: clock["now"])
        monkeypatch.setattr(
            gateway_windows.time, "sleep", lambda _s: clock.__setitem__("now", clock["now"] + 10.0)
        )
        monkeypatch.setattr(gateway_windows, "_windows_stop_drain_timeout", lambda: 7.0)
        monkeypatch.setattr(status, "get_running_pid", lambda: pid if alive["value"] else None)
        monkeypatch.setattr(status, "write_planned_stop_marker", lambda target: None)
        monkeypatch.setattr(status, "_pid_exists", lambda target: alive["value"])
        # The original gateway (start time 100) exits during the drain and its PID is recycled (999).
        monkeypatch.setattr(
            status, "get_process_start_time", lambda target: 100 if clock["now"] < 7.0 else 999
        )

        def _terminate(target, force=False, expected_start_time=None):
            calls.append((target, force, expected_start_time))
            alive["value"] = False

        monkeypatch.setattr(status, "terminate_pid", _terminate)
        monkeypatch.setattr(status, "remove_pid_file", lambda: None)
        monkeypatch.setattr(gateway, "_reap_unsupervised_gateway_orphans", lambda **_: False)

        assert gateway.stop_profile_gateway() is True
        assert calls == [(pid, True, 100)]

    def test_stop_profile_gateway_keeps_pid_file_when_process_still_running(self, monkeypatch):
        calls = {"kill": 0, "alive_probes": 0, "remove": 0, "reap_calls": 0}

        monkeypatch.setattr("gateway.status.get_running_pid", lambda: 12345)
        # Post-#21561: the stop loop sends one SIGTERM via ``os.kill`` then
        # polls liveness via ``gateway.status._pid_exists`` (safe on
        # Windows — bpo-14484). Instrument both seams separately.
        monkeypatch.setattr(
            gateway.os,
            "kill",
            lambda pid, sig: calls.__setitem__("kill", calls["kill"] + 1),
        )
        monkeypatch.setattr(
            "gateway.status._pid_exists",
            lambda pid: calls.__setitem__("alive_probes", calls["alive_probes"] + 1) or True,
        )
        monkeypatch.setattr("time.sleep", lambda _: None)
        monkeypatch.setattr(
            "gateway.status.remove_pid_file",
            lambda: calls.__setitem__("remove", calls["remove"] + 1),
        )
        # Mock the orphan reap so it doesn't scan for real gateway processes
        # (#75936 — stop_profile_gateway now calls _reap_unsupervised_gateway_orphans
        # after killing the pid-file PID).
        monkeypatch.setattr(
            gateway,
            "_reap_unsupervised_gateway_orphans",
            lambda extra_exclude=None: calls.__setitem__("reap_calls", calls["reap_calls"] + 1) or False,
        )

        assert gateway.stop_profile_gateway() is True
        assert calls["kill"] == 1          # one SIGTERM
        assert calls["remove"] == 0
        assert calls["reap_calls"] == 1    # orphan sweep ran after kill

    def test_stop_profile_gateway_excludes_killed_pid_from_orphan_reap(self, monkeypatch):
        """The PID we killed must be excluded from the orphan sweep (#75936)."""
        killed_pid = 99999
        reap_extra_excludes = []

        monkeypatch.setattr("gateway.status.get_running_pid", lambda: killed_pid)
        monkeypatch.setattr(gateway.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
        monkeypatch.setattr("time.sleep", lambda _: None)
        monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)

        def fake_reap(extra_exclude=None):
            if extra_exclude:
                reap_extra_excludes.append(extra_exclude)
            return False

        monkeypatch.setattr(gateway, "_reap_unsupervised_gateway_orphans", fake_reap)

        assert gateway.stop_profile_gateway() is True
        assert len(reap_extra_excludes) == 1
        assert killed_pid in reap_extra_excludes[0]


@pytest.mark.macos_only
class TestReapUnsupervisedGatewayOrphansMacOS:
    """Tests that the orphan reaper excludes launchd-managed PIDs on macOS.

    Regression guard: without the ``is_macos()`` exclusion of
    ``_get_service_pids()``, the reaper would SIGTERM the launchd-supervised
    gateway every time Hermes Desktop opens (``hermes serve`` calls
    ``_reap_unsupervised_gateway_orphans`` during startup).
    """

    def test_macos_excludes_launchd_pid_from_kill(self, monkeypatch):
        """A launchd-managed PID must not appear in the orphan kill list."""
        launchd_pid = 52615

        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)

        # _get_service_pids returns the launchd-managed gateway PID.
        # (accepts all_profiles: the reaper asks for the whole fleet, #74075)
        monkeypatch.setattr(
            gateway, "_get_service_pids", lambda all_profiles=False: {launchd_pid}
        )
        # No pidfile-recorded gateway in this scenario.
        monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)

        # find_gateway_pids returns the launchd PID plus a real orphan.
        # The reaper should only kill the orphan, not the launchd PID.
        orphan_pid = 99998
        monkeypatch.setattr(
            gateway,
            "find_gateway_pids",
            lambda exclude_pids=None: [p for p in [launchd_pid, orphan_pid] if p not in (exclude_pids or set())],
        )

        killed_pids = []
        monkeypatch.setattr(gateway.os, "kill", lambda pid, sig: killed_pids.append((pid, sig)))
        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
        monkeypatch.setattr("gateway.status.write_planned_stop_marker", lambda pid: None)
        monkeypatch.setattr("time.sleep", lambda _: None)
        monkeypatch.setattr("time.monotonic", lambda: 1.0)

        result = gateway._reap_unsupervised_gateway_orphans()

        assert result is True  # at least one orphan was reaped
        killed = [pid for pid, _ in killed_pids]
        assert orphan_pid in killed       # the real orphan was killed
        assert launchd_pid not in killed  # the launchd PID was NOT killed


@pytest.mark.windows_only
class TestReapUnsupervisedGatewayOrphansWindows:
    """Tests that the orphan reaper spares the recorded gateway PID and its
    supervision chain on Windows.

    Regression guard: without the Windows exemption of the recorded healthy
    gateway PID (and its parent chain), the reaper would SIGTERM/SIGKILL a
    Scheduled-Task-supervised gateway every time Hermes Desktop opens
    (``hermes serve`` calls ``_reap_unsupervised_gateway_orphans`` during
    startup). The Scheduled-Task bootstrap's argv matches the gateway scan,
    so it is reaped as an "orphan" — and when the bootstrap dies, the
    detached gateway it spawned exits with it (#86098).
    """

    @staticmethod
    def _install_fake_psutil(monkeypatch, chain):
        """Install a fake psutil module exposing the given process chain."""
        by_pid = {proc.pid: proc for proc in chain}
        fake_psutil = SimpleNamespace(Process=lambda pid: by_pid[pid])
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    def test_windows_excludes_recorded_pid_and_bootstrap_from_kill(self, monkeypatch):
        """The recorded gateway PID and its bootstrap parent must not be killed."""
        recorded_pid = 52615   # detached gateway recorded in gateway.pid
        bootstrap_pid = 52616  # Scheduled-Task bootstrap (argv matches scan)
        orphan_pid = 99998     # a real orphan that should still be reaped

        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)

        # gateway.pid records the detached gateway; its parent is the
        # Scheduled-Task bootstrap whose argv matches the gateway scan.
        bootstrap = SimpleNamespace(pid=bootstrap_pid, parent=lambda: None)
        recorded = SimpleNamespace(pid=recorded_pid, parent=lambda: bootstrap)
        self._install_fake_psutil(monkeypatch, [recorded, bootstrap])

        # get_running_pid() returns the recorded healthy gateway PID.
        monkeypatch.setattr(
            "gateway.status.get_running_pid", lambda cleanup_stale=True: recorded_pid
        )

        # find_gateway_pids returns the recorded PID, its bootstrap parent
        # and a real orphan. The reaper should only kill the orphan.
        monkeypatch.setattr(
            gateway,
            "find_gateway_pids",
            lambda exclude_pids=None: [
                p
                for p in [recorded_pid, bootstrap_pid, orphan_pid]
                if p not in (exclude_pids or set())
            ],
        )

        killed_pids = []
        marked_pids = []
        monkeypatch.setattr(gateway.os, "kill", lambda pid, sig: killed_pids.append((pid, sig)))
        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
        monkeypatch.setattr("gateway.status.write_planned_stop_marker", marked_pids.append)
        monkeypatch.setattr("time.sleep", lambda _: None)
        monkeypatch.setattr("time.monotonic", lambda: 1.0)

        result = gateway._reap_unsupervised_gateway_orphans()

        assert result is True  # at least one orphan was reaped
        assert marked_pids == [orphan_pid]  # the real orphan was asked to drain
        assert killed_pids == []            # Windows SIGTERM must not TerminateProcess

    def test_windows_raw_record_supplies_exclusion_when_validation_fails(
        self, monkeypatch
    ):
        """A registration that fails liveness VALIDATION must still shield
        the recorded PID from the sweep.

        get_running_pid returns None whenever the record fails validation
        (start-time mismatch, argv drift, lock hiccup) — regardless of
        cleanup_stale, which only controls unlinking. The exclusion set is
        therefore built from the RAW pidfile/lock records: a stale recorded
        PID at worst spares one process, while a validation false-negative
        would TerminateProcess a healthy standalone gateway (#87158).
        """
        recorded_pid = 52615

        monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)

        recorded = SimpleNamespace(pid=recorded_pid, parent=lambda: None)
        self._install_fake_psutil(monkeypatch, [recorded])

        # The raw record is present on disk; the validated probe rejects it.
        monkeypatch.setattr(
            "gateway.status._read_pid_record", lambda *a, **k: {"pid": recorded_pid}
        )
        monkeypatch.setattr(
            "gateway.status._read_gateway_lock_record", lambda *a, **k: None
        )
        probe_kwargs = []

        def failing_validation(*a, cleanup_stale=True, **k):
            probe_kwargs.append(cleanup_stale)
            return None

        monkeypatch.setattr(
            "gateway.status.get_running_pid", failing_validation
        )
        monkeypatch.setattr(
            gateway,
            "find_gateway_pids",
            lambda exclude_pids=None: [
                p for p in [recorded_pid] if p not in (exclude_pids or set())
            ],
        )

        killed_pids = []
        monkeypatch.setattr(
            gateway.os, "kill", lambda pid, sig: killed_pids.append((pid, sig))
        )

        result = gateway._reap_unsupervised_gateway_orphans()

        assert probe_kwargs == [False], (
            "the fallback probe must still be non-destructive so the sweep "
            f"never unlinks the record it just read, got {probe_kwargs}"
        )
        assert result is False
        assert killed_pids == []  # the standalone gateway survived


class TestReaperCandidateIsSupervisorOwned:
    """Regression for the Windows pidfile-less supervisor-owned case (#83683).

    On Windows ``_get_service_pids()`` is empty and a Scheduled-Task gateway
    that lost ``gateway.pid`` is invisible to both the service-PID and
    recorded-PID exclusions — the backstop spares it via services.exe
    ancestry. On POSIX the backstop must be inert: every process (and
    especially a genuine orphan, which is reparented to PID 1) has
    launchd/init in its ancestry, so ancestry carries no supervision signal
    there (#51325, #75936).
    """

    def test_backstop_is_inert_on_posix(self, monkeypatch):
        """Direct unit guard: on non-Windows the backstop returns False without
        touching psutil, even for a launchd/init-ancestored process."""
        monkeypatch.setattr(gateway, "is_windows", lambda: False)

        def _boom(_pid):
            raise AssertionError("psutil must not be consulted on POSIX")

        monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=_boom))
        assert gateway._reaper_candidate_is_supervisor_owned(12345) is False


class TestWindowsScheduledTaskSupervisorGuard:
    """The reaper must skip when the profile's scheduled task is still a
    supervisor — Running *or* Ready.

    Regression guard: ``_reaper_candidate_is_supervisor_owned`` walks the
    parent chain up to ``services.exe`` and fails open when the Task-launched
    bootstrap has already exited (Windows does not reparent, so the chain
    breaks). After that exit the task is typically Ready, not Running. A
    Running-only check then treats the detached gateway as an orphan: the
    reaper writes the planned-stop marker, the gateway exits cleanly with
    code 0, and the scheduler never restarts it — silently killing
    A2A/messaging on every desktop-app launch (#86098, #87001).
    """


    def test_windows_scheduled_task_running_returns_false_off_windows(
        self, monkeypatch, inert_task_scheduler_probe
    ):
        """The state helper is inert on POSIX (no subprocess spawned)."""
        inert_task_scheduler_probe.undo()  # exercise the real probe, not the module-wide stand-in
        monkeypatch.setattr(gateway, "is_windows", lambda: False)

        def _boom_run(*_a, **_k):
            raise AssertionError("subprocess must not run off Windows")

        monkeypatch.setattr(gateway.subprocess, "run", _boom_run)
        assert gateway._windows_scheduled_task_supervises("HermesGateway") is False
        assert gateway._windows_scheduled_task_state("HermesGateway") is None

    def test_supervises_ready_and_queued_but_not_disabled(self, monkeypatch):
        states = {"Running": True, "Ready": True, "Queued": True, "Disabled": False, "MISSING": False}

        for state, expected in states.items():
            monkeypatch.setattr(gateway, "_windows_scheduled_task_state", lambda name, s=state: s)
            assert gateway._windows_scheduled_task_supervises("Hermes_Gateway") is expected, state


@pytest.mark.windows_only
def test_find_windows_gateway_services_ignores_task_scheduler_ancestor(monkeypatch):
    """gateway <- cmd.exe <- svchost.exe(Schedule) <- services.exe: the Task Scheduler host is not the
    gateway's supervisor, so a task-launched gateway is a plain process (#97208); the same tree under a
    Hermes-owned service (by binary path) stays SCM-supervised."""
    import psutil
    import hermes_cli.gateway_windows as gateway_windows

    monkeypatch.setattr(gateway_windows, "hermes_service_roots", lambda: (r"C:\hermes\hermes-agent",))
    profile = SimpleNamespace(profile="default", pid=18480, create_time=18480.0)

    class FakeService:
        def __init__(self, name, binpath):
            self._name, self._binpath = name, binpath

        def as_dict(self):
            return {"name": self._name, "binpath": self._binpath, "pid": 2360, "status": "running"}

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def parents(self):
            return [FakeProcess(12296), FakeProcess(2360), FakeProcess(4)]

        def children(self, recursive=False):
            assert self.pid == 2360 and recursive is True
            return [FakeProcess(12296), FakeProcess(18480)]

        def create_time(self):
            return float(self.pid)

    def run(service):
        return gateway.find_windows_gateway_services(
            psutil_module=SimpleNamespace(
                win_service_iter=lambda: [service], Process=FakeProcess, AccessDenied=psutil.AccessDenied),
            profile_processes=[profile],
        )

    assert run(FakeService("Schedule", r"C:\Windows\system32\svchost.exe -k netsvcs -p -s Schedule")) == []
    owned = run(FakeService("gw", r'"C:\hermes\hermes-agent\venv\Scripts\hermes.exe" gateway run'))
    assert [(s.name, s.service_pid, s.gateway_pid) for s in owned] == [("gw", 2360, 18480)]

    # QueryServiceConfig unreadable to this user (hardened or malformed third-party service): not
    # Hermes's, and never a reason to abort; a Hermes-NAMED service is settled without asking binpath.
    class UnreadableConfigService(FakeService):
        def __init__(self, name, error):
            super().__init__(name, "")
            self._error = error

        def binpath(self):
            raise self._error

    assert run(UnreadableConfigService("Hardened", psutil.AccessDenied(2360, "Hardened"))) == []
    assert run(UnreadableConfigService("BrokenMui", OSError(15100, "MUI file missing"))) == []
    named = run(UnreadableConfigService("HermesGateway", OSError(15100, "MUI file missing")))
    assert [(s.name, s.service_pid, s.gateway_pid) for s in named] == [("HermesGateway", 2360, 18480)]


def test_find_profile_gateway_processes_strict_propagates_profile_listing_failure(
    monkeypatch,
):
    import hermes_cli.profiles as profiles_mod

    monkeypatch.setattr(
        profiles_mod,
        "list_profiles",
        lambda: (_ for _ in ()).throw(RuntimeError("profile listing failed")),
    )

    with pytest.raises(RuntimeError, match="profile listing failed"):
        gateway.find_profile_gateway_processes(strict=True)
