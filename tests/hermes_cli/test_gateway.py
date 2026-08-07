"""Tests for hermes_cli.gateway."""

import argparse
import os
import signal
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest

import hermes_cli.gateway as gateway


def _install_fake_gateway_run(monkeypatch, start_gateway):
    module = ModuleType("gateway.run")
    module.start_gateway = start_gateway

    def _exit_after_graceful_shutdown(code):
        if code:
            raise SystemExit(code)

    setattr(module, "_exit_after_graceful_shutdown", _exit_after_graceful_shutdown)
    monkeypatch.setitem(sys.modules, "gateway.run", module)
    # ``run_gateway()`` calls ``refresh_systemd_unit_if_needed()`` on every
    # invocation so that restart settings stay current after exit-code-75
    # respawns. That helper writes to ``Path.home() / ".config/systemd/user
    # /hermes-gateway.service"`` and runs ``systemctl --user daemon-reload``
    # — both target the *real* user environment because the conftest only
    # sandboxes ``HERMES_HOME``, not ``HOME``. Tests that drive
    # ``run_gateway()`` end-to-end with a fake ``start_gateway`` MUST stub
    # the refresh call too, or every run rewrites the developer's installed
    # unit (baking in the test's pytest-tmp ``HERMES_HOME`` value, which
    # systemd then uses on the next boot — silently breaking the gateway
    # for the developer).
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(
        gateway, "refresh_systemd_unit_if_needed", lambda system=False: False
    )
    # Neutralize the supervised-gateway conflict guard by default so these
    # end-to-end tests don't trip over a launchd/systemd gateway that happens
    # to be installed+running on the developer's machine. Conflict-guard tests
    # override this snapshot after calling the helper.
    monkeypatch.setattr(
        gateway,
        "get_gateway_runtime_snapshot",
        lambda *a, **k: gateway.GatewayRuntimeSnapshot(manager="manual process"),
    )




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
    # POSIX-only; Windows has no PTY (no termios) so the import would fail at
    # collection time. Lazy-import here so the skipif on the parametrize covers
    # Windows before this line is reached.
    import pty
    script = textwrap.dedent(
        """
        import os
        import sys
        import types

        import hermes_cli.gateway as gateway_cli

        outcome = os.environ["HERMES_TEST_GATEWAY_OUTCOME"]

        async def start_gateway(*, replace, verbosity):
            if outcome == "failure":
                return False
            raise SystemExit(int(outcome.split(":", 1)[1]))

        fake_run = types.ModuleType("gateway.run")
        fake_run.start_gateway = start_gateway
        setattr(fake_run, "_exit_after_graceful_shutdown", sys.exit)
        sys.modules["gateway.run"] = fake_run

        gateway_cli._guard_official_docker_root_gateway = lambda: None
        gateway_cli._guard_named_profile_under_multiplexer = lambda force=False: None
        gateway_cli._guard_supervised_gateway_conflict = lambda force=False: None
        gateway_cli._guard_existing_gateway_process_conflict = lambda replace=False: None
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


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only test",
)
def test_run_gateway_refuses_root_in_official_docker(monkeypatch, tmp_path, capsys):
    project_root = tmp_path / "opt" / "hermes"
    (project_root / "docker").mkdir(parents=True)
    (project_root / "docker" / "entrypoint.sh").write_text("#!/bin/sh\n")

    monkeypatch.setattr(gateway, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(gateway.os, "geteuid", lambda: 0)
    monkeypatch.delenv("HERMES_ALLOW_ROOT_GATEWAY", raising=False)
    monkeypatch.setattr(gateway, "_is_official_docker_checkout", lambda: True)

    with pytest.raises(SystemExit) as exc_info:
        gateway.run_gateway()

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "Refusing to run the Hermes gateway as root" in out
    assert "/opt/hermes/docker/entrypoint.sh" in out


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only test",
)
def test_run_gateway_root_guard_has_escape_hatch(monkeypatch):
    calls = []

    def fake_start_gateway(*, replace, verbosity):
        calls.append((replace, verbosity))
        return object()

    _install_fake_gateway_run(monkeypatch, fake_start_gateway)
    monkeypatch.setattr(gateway.asyncio, "run", lambda coro: True)
    monkeypatch.setattr(gateway.os, "geteuid", lambda: 0)
    monkeypatch.setattr(gateway, "_is_official_docker_checkout", lambda: True)
    monkeypatch.setenv("HERMES_ALLOW_ROOT_GATEWAY", "1")

    gateway.run_gateway(verbose=2, replace=True)

    assert calls == [(True, 2)]


def _clear_supervisor_markers(monkeypatch):
    """Make ``_running_under_gateway_supervisor()`` report a plain shell."""
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    # Interactive macOS shells inherit XPC_SERVICE_NAME="0"; launchd jobs get
    # the real label. Default to the shell sentinel so the guard can fire.
    monkeypatch.setenv("XPC_SERVICE_NAME", "0")


def _running_snapshot(manager="systemd (user)"):
    return gateway.GatewayRuntimeSnapshot(
        manager=manager, service_installed=True, service_running=True
    )


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


def test_gateway_install_in_container_with_operational_systemd_uses_systemd(monkeypatch):
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "is_wsl", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "is_managed", lambda: False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    calls = []
    monkeypatch.setattr(gateway, "prompt_yes_no", lambda question, default=True: calls.append(("prompt", question, default)) or True)
    monkeypatch.setattr(
        gateway,
        "systemd_install",
        lambda force=False, system=False, run_as_user=None, enable_on_startup=True, **kw: calls.append(("install", force, system, run_as_user, enable_on_startup)),
    )
    monkeypatch.setattr(gateway, "systemd_start", lambda system=False: calls.append(("start", system)))

    args = SimpleNamespace(
        gateway_command="install",
        force=False,
        system=False,
        run_as_user=None,
    )
    gateway.gateway_command(args)

    assert calls == [
        ("prompt", "Start the gateway now after installing the service?", True),
        ("prompt", "Start the gateway automatically on login/boot with systemd?", True),
        ("install", False, False, None, True),
        ("start", False),
    ]


def test_gateway_start_in_container_with_operational_systemd_uses_systemd(monkeypatch):
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "is_wsl", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)

    calls = []
    monkeypatch.setattr(gateway, "systemd_start", lambda system=False: calls.append(system))

    args = SimpleNamespace(gateway_command="start", system=False, all=False)
    gateway.gateway_command(args)

    assert calls == [False]


def test_gateway_start_ignores_legacy_platform_selector(monkeypatch):
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "is_wsl", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)

    calls = []
    monkeypatch.setattr(gateway, "systemd_start", lambda system=False: calls.append(system))

    args = SimpleNamespace(gateway_command="start", system=False, all=False, platform="photon")
    gateway.gateway_command(args)

    assert calls == [False]


def test_gateway_restart_on_windows_without_service_uses_detached_backend(monkeypatch):
    """Windows manual restart must not fall back to foreground run_gateway().

    A Telegram-hosted agent may run `hermes gateway restart` via the terminal
    tool. The generic manual fallback stops the gateway and then calls
    run_gateway() in the same foreground subprocess; on Windows that subprocess
    can be reaped when its gateway parent is terminated, leaving the gateway
    down. The Windows backend restarts via detached pythonw.exe even when no
    Scheduled Task / Startup item is installed.
    """
    import hermes_cli.gateway_windows as gateway_windows

    calls = []

    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "is_windows", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: False)
    monkeypatch.setattr(gateway_windows, "restart", lambda: calls.append("restart"))
    monkeypatch.setattr(
        gateway,
        "run_gateway",
        lambda *args, **kwargs: pytest.fail("Windows restart must not use foreground run_gateway()"),
    )
    monkeypatch.setattr(
        gateway,
        "stop_profile_gateway",
        lambda: pytest.fail("Windows restart must not use generic manual stop fallback"),
    )

    args = SimpleNamespace(gateway_command="restart", system=False, all=False)
    gateway.gateway_command(args)

    assert calls == ["restart"]


def test_gateway_restart_on_windows_preserves_failure_fallback(monkeypatch):
    """If the Windows backend cannot launch, keep the existing fallback."""
    import hermes_cli.gateway_windows as gateway_windows

    calls = []

    def fail_restart():
        calls.append("restart")
        raise OSError("simulated detached backend failure")

    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "is_windows", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: False)
    monkeypatch.setattr(gateway_windows, "restart", fail_restart)
    monkeypatch.setattr(gateway, "stop_profile_gateway", lambda: calls.append("stop") or False)
    monkeypatch.setattr(gateway, "_wait_for_gateway_exit", lambda *args, **kwargs: calls.append("wait"))
    monkeypatch.setattr(gateway, "run_gateway", lambda *args, **kwargs: calls.append("run"))

    args = SimpleNamespace(gateway_command="restart", system=False, all=False)
    gateway.gateway_command(args)

    assert calls == ["restart", "stop", "wait", "run"]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only test",
)
def test_systemd_status_warns_when_linger_disabled(monkeypatch, tmp_path, capsys):
    unit_path = tmp_path / "hermes-gateway.service"
    unit_path.write_text("[Unit]\n")

    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: unit_path)
    monkeypatch.setattr(gateway, "get_systemd_linger_status", lambda: (False, ""))

    def fake_run(cmd, capture_output=False, text=False, check=False, **kwargs):
        if cmd[:4] == ["systemctl", "--user", "status", gateway.get_service_name()]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["systemctl", "--user", "is-active"]:
            return SimpleNamespace(returncode=0, stdout="active\n", stderr="")
        if cmd[:3] == ["systemctl", "--user", "show"]:
            return SimpleNamespace(
                returncode=0,
                stdout="ActiveState=active\nSubState=running\nResult=success\nExecMainStatus=0\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(gateway.subprocess, "run", fake_run)

    gateway.systemd_status(deep=False)

    out = capsys.readouterr().out
    assert "gateway service is running" in out
    assert "Systemd linger is disabled" in out
    assert "loginctl enable-linger" in out


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="systemd user-linger is Linux-only (drives os.getuid())",
)
def test_systemd_install_checks_linger_status(monkeypatch, tmp_path, capsys):
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

    out = capsys.readouterr().out
    assert unit_path.exists()
    assert [cmd for cmd, _ in calls] == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", gateway.get_service_name()],
    ]
    assert helper_calls == [True]
    assert "User service installed and enabled" in out








@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only test",
)
def test_install_linux_gateway_from_setup_non_root_never_offers_system(monkeypatch, capsys):
    # Non-root sessions must not be offered system scope, and must never be
    # handed a `sudo hermes …` self-elevation recipe.
    captured = {}

    def fake_prompt_choice(_msg, options, default=0):
        captured["options"] = options
        return 0  # pick "user"

    monkeypatch.setattr(gateway.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(gateway, "prompt_choice", fake_prompt_choice)
    monkeypatch.setattr(gateway, "systemd_install", lambda *a, **k: None)

    scope = gateway.prompt_linux_gateway_install_scope()
    out = capsys.readouterr().out

    assert scope == "user"
    assert not any("System service" in opt for opt in captured["options"])
    assert "sudo hermes" not in out


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only test",
)
def test_install_linux_gateway_from_setup_system_choice_without_root_no_sudo_recipe(monkeypatch, capsys):
    # Defensive guard: if "system" is forced non-root (not reachable via wizard),
    # we refuse and do NOT print a self-elevation recipe.
    monkeypatch.setattr(gateway, "prompt_linux_gateway_install_scope", lambda: "system")
    monkeypatch.setattr(gateway.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(gateway, "_default_system_service_user", lambda: "alice")
    monkeypatch.setattr(gateway, "systemd_install", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not install")))

    scope, did_install = gateway.install_linux_gateway_from_setup(force=False)

    out = capsys.readouterr().out
    assert (scope, did_install) == ("system", False)
    assert "sudo hermes" not in out
    assert "requires root" in out


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only test",
)
def test_install_linux_gateway_from_setup_system_choice_as_root_installs(monkeypatch):
    monkeypatch.setattr(gateway, "prompt_linux_gateway_install_scope", lambda: "system")
    monkeypatch.setattr(gateway.os, "geteuid", lambda: 0)
    monkeypatch.setattr(gateway, "_default_system_service_user", lambda: "alice")

    calls = []
    monkeypatch.setattr(
        gateway,
        "systemd_install",
        lambda force=False, system=False, run_as_user=None, enable_on_startup=True, **kw: calls.append((force, system, run_as_user, enable_on_startup)),
    )

    scope, did_install = gateway.install_linux_gateway_from_setup(force=True)

    assert (scope, did_install) == ("system", True)
    assert calls == [(True, True, "alice", True)]


def test_install_linux_gateway_from_setup_passes_startup_choice(monkeypatch):
    monkeypatch.setattr(gateway, "prompt_linux_gateway_install_scope", lambda: "user")

    calls = []
    monkeypatch.setattr(
        gateway,
        "systemd_install",
        lambda force=False, system=False, run_as_user=None, enable_on_startup=True, **kw: calls.append((force, system, run_as_user, enable_on_startup)),
    )

    scope, did_install = gateway.install_linux_gateway_from_setup(force=False, enable_on_startup=False)

    assert (scope, did_install) == ("user", True)
    assert calls == [(False, False, None, False)]


def test_gateway_install_can_decline_start_now_and_startup(monkeypatch):
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "is_wsl", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "is_managed", lambda: False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    answers = iter([False, False])
    calls = []
    monkeypatch.setattr(gateway, "prompt_yes_no", lambda question, default=True: calls.append(("prompt", question, default)) or next(answers))
    monkeypatch.setattr(
        gateway,
        "systemd_install",
        lambda force=False, system=False, run_as_user=None, enable_on_startup=True, **kw: calls.append(("install", force, system, run_as_user, enable_on_startup)),
    )
    monkeypatch.setattr(gateway, "systemd_start", lambda system=False: calls.append(("start", system)))

    args = SimpleNamespace(gateway_command="install", force=True, system=False, run_as_user=None)
    gateway.gateway_command(args)

    assert calls == [
        ("prompt", "Start the gateway now after installing the service?", True),
        ("prompt", "Start the gateway automatically on login/boot with systemd?", True),
        ("install", True, False, None, False),
    ]


def test_gateway_install_systemd_honors_start_now_flag(monkeypatch):
    """--start-now / --no-start-now should bypass the interactive prompt."""
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "is_wsl", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "is_managed", lambda: False)

    calls = []
    monkeypatch.setattr(gateway, "prompt_yes_no", lambda question, default=True: calls.append(("prompt", question)))
    monkeypatch.setattr(
        gateway,
        "systemd_install",
        lambda force=False, system=False, run_as_user=None, enable_on_startup=True, **kw: calls.append(("install", enable_on_startup)),
    )
    monkeypatch.setattr(gateway, "systemd_start", lambda system=False: calls.append(("start",)))

    args = SimpleNamespace(
        gateway_command="install", force=False, system=False,
        run_as_user=None, start_now=True, start_on_login=False,
    )
    gateway.gateway_command(args)

    assert ("prompt", "Start the gateway now after installing the service?") not in calls
    assert ("start",) in calls
    assert ("install", False) in calls


def test_gateway_install_systemd_non_tty_uses_defaults(monkeypatch):
    """Non-TTY stdin (headless/CI) should use True defaults without prompting."""
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "is_wsl", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "is_managed", lambda: False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    calls = []
    monkeypatch.setattr(gateway, "prompt_yes_no", lambda question, default=True: calls.append(("prompt", question)))
    monkeypatch.setattr(
        gateway,
        "systemd_install",
        lambda force=False, system=False, run_as_user=None, enable_on_startup=True, **kw: calls.append(("install", enable_on_startup)),
    )
    monkeypatch.setattr(gateway, "systemd_start", lambda system=False: calls.append(("start",)))

    args = SimpleNamespace(gateway_command="install", force=False, system=False, run_as_user=None)
    gateway.gateway_command(args)

    # No prompts — defaults used (start_now=True, start_on_login=True)
    assert all(c[0] != "prompt" for c in calls)
    assert ("install", True) in calls
    assert ("start",) in calls


def test_gateway_install_systemd_no_start_now_flag_non_tty(monkeypatch):
    """--no-start-now in non-TTY should skip starting the service."""
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "is_wsl", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "is_managed", lambda: False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    calls = []
    monkeypatch.setattr(gateway, "prompt_yes_no", lambda question, default=True: calls.append(("prompt", question)))
    monkeypatch.setattr(
        gateway,
        "systemd_install",
        lambda force=False, system=False, run_as_user=None, enable_on_startup=True, **kw: calls.append(("install", enable_on_startup)),
    )
    monkeypatch.setattr(gateway, "systemd_start", lambda system=False: calls.append(("start",)))

    args = SimpleNamespace(
        gateway_command="install", force=False, system=False,
        run_as_user=None, start_now=False, start_on_login=True,
    )
    gateway.gateway_command(args)

    assert all(c[0] != "prompt" for c in calls)
    assert ("install", True) in calls
    assert ("start",) not in calls


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








@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only test",
)
def test_reap_unsupervised_orphans_sigterms_then_sigkills_survivor(monkeypatch):
    """No-systemd: orphan gets SIGTERM, and a survivor is force-killed."""
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda exclude_pids=None: [708])
    monkeypatch.setattr("gateway.status.write_planned_stop_marker", lambda pid: True)
    # Orphan ignores SIGTERM (matches the field report) and stays alive, so the
    # follow-up SIGKILL must fire.
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)

    sent = []
    monkeypatch.setattr(gateway.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    # Collapse the drain window: no real sleeping, and jump past the deadline
    # after the first check so the loop exits immediately.
    monkeypatch.setattr(gateway.time, "sleep", lambda _s: None)
    ticks = iter([0.0, 100.0, 200.0])
    monkeypatch.setattr(gateway.time, "monotonic", lambda: next(ticks, 200.0))

    assert gateway._reap_unsupervised_gateway_orphans() is True
    assert (708, signal.SIGTERM) in sent
    assert (708, signal.SIGKILL) in sent




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
        def mock_terminate(pid, force=False):
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
        monkeypatch.setattr(gateway, "terminate_pid", lambda pid, force=False: calls.append((pid, force)))

        killed = gateway.kill_gateway_processes(force=True)

        assert killed == 2
        assert calls == [(11, True), (22, True)]


class TestStopProfileGateway:
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
        assert calls["alive_probes"] == 20 # 20 liveness polls over the 2s window
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


def test_module_has_logger():
    """Verify module has a logger instance (regression guard for #27154)."""
    assert hasattr(gateway, "logger")
    assert gateway.logger.name == "hermes_cli.gateway"
