"""Tests for hermes_cli.gateway_windows."""

from pathlib import Path

import pytest

import hermes_cli.gateway as gateway
import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.setup as setup


@pytest.mark.parametrize(
    "detail",
    [
        "ERROR: Access is denied.",
        "ERROR: Acceso denegado.",
        "ERROR: Přístup byl odepřen.",
        "schtasks timed out after 15s",
        "schtasks produced no output",
    ],
)
def test_schtasks_fallback_patterns_cover_localized_access_denied(detail):
    """Localized schtasks access-denied errors should use Startup fallback."""

    assert gateway_windows._should_fall_back(1, detail) is True


def test_schtasks_fallback_does_not_hide_unknown_errors():
    assert gateway_windows._should_fall_back(1, "ERROR: The system cannot find the file specified.") is False


def test_build_gateway_argv_uses_base_pythonw_for_uv_venv_launcher(monkeypatch, tmp_path):
    """Avoid uv's venv pythonw launcher because it respawns console python.exe."""

    project = tmp_path / "project"
    scripts = project / "venv" / "Scripts"
    site_packages = project / "venv" / "Lib" / "site-packages"
    base = tmp_path / "uv" / "python" / "cpython-3.11-windows-x86_64-none"
    scripts.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    base.mkdir(parents=True)

    venv_python = scripts / "python.exe"
    venv_pythonw = scripts / "pythonw.exe"
    base_pythonw = base / "pythonw.exe"
    for exe in (venv_python, venv_pythonw, base_pythonw):
        exe.write_text("", encoding="utf-8")
    (project / "venv" / "pyvenv.cfg").write_text(
        f"home = {base}\nimplementation = CPython\nuv = 0.11.14\nversion_info = 3.11.15\n",
        encoding="utf-8",
    )

    import hermes_cli.gateway as gateway

    monkeypatch.setattr(gateway_windows.sys, "platform", "win32")
    monkeypatch.setattr(gateway, "PROJECT_ROOT", project)
    monkeypatch.setattr(gateway, "get_python_path", lambda: str(venv_python))
    monkeypatch.setattr(gateway, "_profile_arg", lambda hermes_home: "")
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path / "hermes-home"))

    argv, cwd, env_overlay = gateway_windows._build_gateway_argv()

    assert argv[:3] == [str(base_pythonw), "-m", "hermes_cli.main"]
    assert cwd == str(project)
    assert env_overlay["VIRTUAL_ENV"] == str(project / "venv")
    assert str(project) in env_overlay["PYTHONPATH"].split(gateway_windows.os.pathsep)
    assert str(site_packages) in env_overlay["PYTHONPATH"].split(gateway_windows.os.pathsep)


def _arrange_startup_fallback(monkeypatch, tmp_path, running_pids):
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"
    startup_entry = tmp_path / "Startup" / "Hermes_Gateway_alice.cmd"
    calls = []

    monkeypatch.setattr(gateway_windows, "_prompt_install_choices", lambda *args, **kwargs: (False, True))
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice")
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script_path)
    monkeypatch.setattr(
        gateway_windows,
        "_install_scheduled_task",
        lambda task_name, script_path: (
            False,
            "schtasks /Create failed (code 1): ERROR: Access is denied.",
        ),
    )
    monkeypatch.setattr(gateway_windows, "_should_fall_back", lambda code, detail: True)
    monkeypatch.setattr(gateway_windows, "_is_running_as_admin", lambda: True)
    monkeypatch.setattr(
        gateway_windows,
        "_launch_elevated_install",
        lambda force=False, start_now=None, start_on_login=None: calls.append(("elevate", force, start_now, start_on_login)) or True,
    )

    def fake_install_startup_entry(path: Path) -> Path:
        calls.append(("install_startup", path))
        return startup_entry

    monkeypatch.setattr(gateway_windows, "_install_startup_entry", fake_install_startup_entry)
    monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda path: calls.append(("spawn", path)) or 12345)
    monkeypatch.setattr(gateway_windows, "_report_gateway_start", lambda via: calls.append(("report_start", via)))
    monkeypatch.setattr(gateway_windows, "_print_next_steps", lambda: calls.append(("next_steps", None)))
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda: running_pids)
    monkeypatch.setattr(gateway, "_profile_arg", lambda: "--profile alice")
    return script_path, calls


def test_gateway_cmd_script_uses_pythonw_without_replace_or_start_churn(monkeypatch):
    """Scheduled Task wrapper should launch pythonw once and avoid replace loops."""
    monkeypatch.setattr(gateway_windows, "_derive_venv_pythonw", lambda exe: exe.replace("python.exe", "pythonw.exe"))

    content = gateway_windows._build_gateway_cmd_script(
        r"C:\\Hermes\\hermes-agent\\venv\\Scripts\\python.exe",
        r"C:\\Hermes\\hermes-agent",
        r"C:\\HermesHome\\profiles\\alice",
        "--profile alice",
    )

    assert "pythonw.exe" in content
    assert "gateway run" in content
    assert "--replace" not in content
    assert "start \"\"" not in content
    assert "exit /b 0" in content


def test_elevated_gateway_command_uses_pythonw_hidden_console(monkeypatch):
    """UAC handoff should not leave a second elevated cmd.exe window open."""
    calls = []

    class FakeShell32:
        def ShellExecuteW(self, hwnd, verb, executable, params, cwd, show):
            calls.append((hwnd, verb, executable, params, cwd, show))
            return 33

    class FakeWindll:
        shell32 = FakeShell32()

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "_current_profile_cli_args", lambda: ["--profile", "alice"])
    monkeypatch.setattr(gateway_windows, "_derive_venv_pythonw", lambda exe: exe.replace("python.exe", "pythonw.exe"))
    monkeypatch.setattr(gateway_windows.sys, "executable", r"C:\Hermes\venv\Scripts\python.exe")
    monkeypatch.setattr(gateway_windows.ctypes, "windll", FakeWindll(), raising=False)

    assert gateway_windows._launch_elevated_gateway_command("install", ["--start-now", "--elevated-handoff"])

    assert len(calls) == 1
    _hwnd, verb, executable, params, cwd, show = calls[0]
    assert verb == "runas"
    assert executable.endswith("pythonw.exe")
    assert "--profile alice gateway install --start-now --elevated-handoff" in params
    assert show == 0
    assert cwd


def test_install_scheduled_task_recreates_instead_of_change(monkeypatch, tmp_path):
    """Install must delete+create so stale minute-repeat task settings are not preserved."""
    calls = []
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)

    def fake_schtasks(args):
        calls.append(tuple(args))
        if args[0] == "/Delete":
            return (0, "SUCCESS", "")
        if args[0] == "/Create":
            return (0, "SUCCESS", "")
        raise AssertionError(f"unexpected schtasks args: {args}")

    monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)
    ok, detail = gateway_windows._install_scheduled_task("Hermes_Gateway_alice", script_path)

    assert ok is True
    assert "/Change" not in [arg for call in calls for arg in call]
    assert calls[0][:4] == ("/Delete", "/F", "/TN", "Hermes_Gateway_alice")
    assert calls[1][0] == "/Create"
    assert "/SC" in calls[1]
    assert "ONLOGON" in calls[1]


def test_install_scheduled_task_success_start_now_uses_direct_spawn_not_task_run(monkeypatch, tmp_path, capsys):
    """Install start-now should not /Run the task; that preserved old restart loops."""
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"
    calls = []

    monkeypatch.setattr(gateway_windows, "_prompt_install_choices", lambda *args, **kwargs: (True, True))
    monkeypatch.setattr(gateway_windows, "_is_running_as_admin", lambda: True)
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice")
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script_path)
    monkeypatch.setattr(
        gateway_windows,
        "_install_scheduled_task",
        lambda task_name, script_path: (True, "Created Scheduled Task 'Hermes_Gateway_alice'"),
    )
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda args: calls.append(("schtasks", tuple(args))) or (0, "", ""))
    monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda path=None: calls.append(("spawn", path)) or 12345)
    monkeypatch.setattr(gateway_windows, "_report_gateway_start", lambda via: calls.append(("report_start", via)))
    monkeypatch.setattr(gateway_windows, "_print_next_steps", lambda: calls.append(("next_steps", None)))

    gateway_windows.install(force=False)

    assert not any(call[0] == "schtasks" and "/Run" in call[1] for call in calls)
    assert ("spawn", None) in calls
    assert any(call[0] == "report_start" for call in calls)
    out = capsys.readouterr().out
    assert "auto-start installed for Windows login" in out


def test_install_scheduled_task_success_does_not_auto_start(monkeypatch, tmp_path, capsys):
    """Install should register/update the task only; start is explicit."""
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"
    calls = []

    monkeypatch.setattr(gateway_windows, "_prompt_install_choices", lambda *args, **kwargs: (False, True))
    monkeypatch.setattr(gateway_windows, "_is_running_as_admin", lambda: True)
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice")
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script_path)
    monkeypatch.setattr(
        gateway_windows,
        "_install_scheduled_task",
        lambda task_name, script_path: (True, "Created Scheduled Task 'Hermes_Gateway_alice'"),
    )
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda args: calls.append(("schtasks", tuple(args))) or (0, "", ""))
    monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda path=None: calls.append(("spawn", path)) or 12345)
    monkeypatch.setattr(gateway_windows, "_report_gateway_start", lambda via: calls.append(("report_start", via)))
    monkeypatch.setattr(gateway_windows, "_print_next_steps", lambda: calls.append(("next_steps", None)))

    gateway_windows.install(force=False)

    assert not any(call[0] == "schtasks" and "/Run" in call[1] for call in calls)
    assert not any(call[0] == "spawn" for call in calls)
    assert not any(call[0] == "report_start" for call in calls)
    assert ("next_steps", None) in calls
    out = capsys.readouterr().out
    assert "auto-start installed for Windows login" in out


def test_install_access_denied_launches_elevated_install_before_startup_fallback(monkeypatch, tmp_path, capsys):
    """Non-admin Scheduled Task access denied should hand off to UAC elevation."""
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"
    calls = []

    monkeypatch.setattr(gateway_windows, "_prompt_install_choices", lambda *args, **kwargs: (False, True))
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice")
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script_path)
    monkeypatch.setattr(
        gateway_windows,
        "_install_scheduled_task",
        lambda task_name, script_path: (
            False,
            "schtasks /Create failed (code 1): ERROR: Access is denied.",
        ),
    )
    monkeypatch.setattr(gateway_windows, "_is_running_as_admin", lambda: False)
    monkeypatch.setattr(
        gateway_windows,
        "_launch_elevated_install",
        lambda force=False, start_now=None, start_on_login=None: calls.append(("elevate", force, start_now, start_on_login)) or True,
    )
    monkeypatch.setattr(setup, "prompt_yes_no", lambda prompt, default=True: calls.append(("prompt", prompt, default)) or True)
    monkeypatch.setattr(gateway_windows, "_install_startup_entry", lambda path: calls.append(("install_startup", path)) or path)
    monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda path=None: calls.append(("spawn", path)) or 12345)

    gateway_windows.install(force=True)

    assert calls == [("prompt", "  Open the UAC prompt now?", False), ("elevate", True, False, True)]
    out = capsys.readouterr().out
    assert "administrator approval" in out
    assert "UAC is Windows' admin approval prompt" in out
    assert "Launched elevated Hermes gateway install prompt" in out


def test_install_prompts_start_choices_before_uac(monkeypatch, tmp_path, capsys):
    """Windows install asks start-now and auto-start before any UAC handoff."""
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"
    calls = []
    answers = iter([True, True, True])

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice")
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script_path)
    monkeypatch.setattr(
        gateway_windows,
        "_install_scheduled_task",
        lambda task_name, script_path: (
            False,
            "schtasks /Create failed (code 1): ERROR: Access is denied.",
        ),
    )
    monkeypatch.setattr(gateway_windows, "_is_running_as_admin", lambda: False)
    monkeypatch.setattr(setup, "prompt_yes_no", lambda prompt, default=True: calls.append(("prompt", prompt, default)) or next(answers))
    monkeypatch.setattr(
        gateway_windows,
        "_launch_elevated_install",
        lambda force=False, start_now=None, start_on_login=None: calls.append(("elevate", force, start_now, start_on_login)) or True,
    )

    gateway_windows.install(force=False)

    assert calls == [
        ("prompt", "Start the gateway now after install?", True),
        ("prompt", "Start the gateway automatically on Windows login with a Scheduled Task?", True),
        ("prompt", "  Open the UAC prompt now?", False),
        ("elevate", False, True, True),
    ]
    out = capsys.readouterr().out
    assert "elevated install will start the gateway afterwards" in out


def test_install_start_now_without_login_autostart_never_escalates(monkeypatch, capsys):
    """If auto-start is declined, install can start directly without touching schtasks/UAC."""
    calls = []
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "_prompt_install_choices", lambda *args, **kwargs: (True, False))
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda path=None: calls.append(("spawn", path)) or 12345)
    monkeypatch.setattr(gateway_windows, "_report_gateway_start", lambda via: calls.append(("report_start", via)))
    monkeypatch.setattr(gateway_windows, "_install_scheduled_task", lambda *args, **kwargs: calls.append(("install_task", args)) or (True, "should not happen"))
    monkeypatch.setattr(gateway_windows, "_launch_elevated_install", lambda *args, **kwargs: calls.append(("elevate", args, kwargs)) or True)

    gateway_windows.install(force=False)

    assert not any(call[0] in {"install_task", "elevate"} for call in calls)
    assert ("spawn", None) in calls
    assert any(call[0] == "report_start" for call in calls)
    out = capsys.readouterr().out
    assert "Skipped Windows login auto-start install" in out


def test_start_noops_when_gateway_already_running(monkeypatch, capsys):
    """Repeated start should not invoke schtasks /Run or spawn another process."""
    calls = []
    monkeypatch.setattr(gateway_windows, "_prompt_install_choices", lambda *args, **kwargs: (False, True))
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda: [27128])
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: calls.append("task_check") or True)
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda args: calls.append(("schtasks", tuple(args))) or (0, "", ""))
    monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda path=None: calls.append(("spawn", path)) or 12345)

    gateway_windows.start()

    assert calls == []
    out = capsys.readouterr().out
    assert "already running" in out
    assert "27128" in out


def test_install_startup_fallback_does_not_spawn_when_gateway_already_running(monkeypatch, tmp_path, capsys):
    """Repeated Windows fallback installs should not spawn duplicate gateways."""
    script_path, calls = _arrange_startup_fallback(monkeypatch, tmp_path, [24476])

    gateway_windows.install(force=False)

    assert ("install_startup", script_path) in calls
    assert not any(call[0] == "spawn" for call in calls)
    assert not any(call[0] == "report_start" for call in calls)
    assert ("next_steps", None) in calls
    out = capsys.readouterr().out
    assert "already running" in out
    assert "24476" in out


def test_install_startup_fallback_does_not_auto_spawn_when_gateway_stopped(monkeypatch, tmp_path, capsys):
    """Startup fallback install should only install login item, not launch pythonw."""
    script_path, calls = _arrange_startup_fallback(monkeypatch, tmp_path, [])

    gateway_windows.install(force=False)

    assert ("install_startup", script_path) in calls
    assert not any(call[0] == "spawn" for call in calls)
    assert not any(call[0] == "report_start" for call in calls)
    assert ("next_steps", None) in calls
    out = capsys.readouterr().out
    assert "gateway not started now" in out
    assert "hermes --profile alice gateway start" in out


def test_install_access_denied_declined_elevation_uses_startup_fallback(monkeypatch, tmp_path, capsys):
    """Install should ask before UAC; declining keeps the non-jarring fallback path."""
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"
    calls = []

    monkeypatch.setattr(gateway_windows, "_prompt_install_choices", lambda *args, **kwargs: (False, True))
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice")
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script_path)
    monkeypatch.setattr(
        gateway_windows,
        "_install_scheduled_task",
        lambda task_name, script_path: (
            False,
            "schtasks /Create failed (code 1): ERROR: Access is denied.",
        ),
    )
    monkeypatch.setattr(gateway_windows, "_is_running_as_admin", lambda: False)
    monkeypatch.setattr(setup, "prompt_yes_no", lambda prompt, default=True: calls.append(("prompt", prompt, default)) or False)
    monkeypatch.setattr(
        gateway_windows,
        "_launch_elevated_install",
        lambda force=False, start_now=None, start_on_login=None: calls.append(("elevate", force, start_now, start_on_login)) or True,
    )
    monkeypatch.setattr(gateway_windows, "_install_startup_entry", lambda path: calls.append(("install_startup", path)) or path)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway, "_profile_arg", lambda: "--profile alice")
    monkeypatch.setattr(gateway_windows, "_print_next_steps", lambda: calls.append(("next_steps", None)))

    gateway_windows.install(force=False)

    assert ("prompt", "  Open the UAC prompt now?", False) in calls
    assert not any(call[0] == "elevate" for call in calls)
    assert ("install_startup", script_path) in calls
    out = capsys.readouterr().out
    assert "Skipped elevation" in out
    assert "UAC is Windows' admin approval prompt" in out


def test_uninstall_access_denied_prompts_before_elevating(monkeypatch, tmp_path, capsys):
    """Uninstall should hand off to an elevated uninstall only after user consent."""
    calls = []
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"
    startup_entry = tmp_path / "Startup" / "Hermes_Gateway_alice.cmd"

    monkeypatch.setattr(gateway_windows, "_prompt_install_choices", lambda *args, **kwargs: (False, True))
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice")
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: script_path)
    monkeypatch.setattr(gateway_windows, "get_startup_entry_path", lambda: startup_entry)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(
        gateway_windows,
        "_exec_schtasks",
        lambda args: calls.append(("schtasks", tuple(args))) or (1, "", "ERROR: Access is denied."),
    )
    monkeypatch.setattr(gateway_windows, "_is_running_as_admin", lambda: False)
    monkeypatch.setattr(setup, "prompt_yes_no", lambda prompt, default=True: calls.append(("prompt", prompt, default)) or True)
    monkeypatch.setattr(gateway_windows, "_launch_elevated_uninstall", lambda: calls.append(("elevate_uninstall", None)) or True)

    gateway_windows.uninstall()

    assert ("prompt", "  Open the UAC prompt now?", False) in calls
    assert ("elevate_uninstall", None) in calls
    out = capsys.readouterr().out
    assert "uninstall needs administrator approval" in out
    assert "UAC is Windows' admin approval prompt" in out
    assert "Launched elevated Hermes gateway uninstall prompt" in out


def test_uninstall_access_denied_declined_keeps_task_and_cleans_files(monkeypatch, tmp_path, capsys):
    """Declining UAC should not surprise the user, but should still remove user-writable artifacts."""
    calls = []
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"
    startup_entry = tmp_path / "Startup" / "Hermes_Gateway_alice.cmd"
    startup_entry.parent.mkdir(parents=True)
    script_path.write_text("task", encoding="utf-8")
    startup_entry.write_text("startup", encoding="utf-8")

    monkeypatch.setattr(gateway_windows, "_prompt_install_choices", lambda *args, **kwargs: (False, True))
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice")
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: script_path)
    monkeypatch.setattr(gateway_windows, "get_startup_entry_path", lambda: startup_entry)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(
        gateway_windows,
        "_exec_schtasks",
        lambda args: calls.append(("schtasks", tuple(args))) or (1, "", "ERROR: Access is denied."),
    )
    monkeypatch.setattr(gateway_windows, "_is_running_as_admin", lambda: False)
    monkeypatch.setattr(setup, "prompt_yes_no", lambda prompt, default=True: calls.append(("prompt", prompt, default)) or False)
    monkeypatch.setattr(gateway_windows, "_launch_elevated_uninstall", lambda: calls.append(("elevate_uninstall", None)) or True)

    gateway_windows.uninstall()

    assert not any(call[0] == "elevate_uninstall" for call in calls)
    assert not script_path.exists()
    assert not startup_entry.exists()
    out = capsys.readouterr().out
    assert "Skipped elevation" in out
    assert "UAC is Windows' admin approval prompt" in out
    assert "Scheduled Task still registered" in out


# ---------------------------------------------------------------------------
# stop() drain semantics — issue #33778
#
# Background: on Windows, asyncio.add_signal_handler raises NotImplementedError,
# so the gateway's SIGTERM handler (which drains in-flight agents and writes
# resume_pending=True) never fires when `hermes gateway stop` kills the
# process. The fix: stop() writes the planned_stop_marker first, waits for
# the gateway's marker-watcher thread to drain + exit cleanly, then escalates
# to taskkill if drain times out.
# ---------------------------------------------------------------------------


def test_stop_writes_planned_stop_marker_before_killing(monkeypatch):
    """stop() must write the planned-stop marker BEFORE any kill signal.

    Without this, the gateway's drain loop never runs on Windows and
    sessions silently lose context across restarts.
    """
    pid = 99999
    events = []

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)

    # Stub the marker write so we can record the order of operations.
    from gateway import status as status_mod

    def fake_write_marker(target_pid):
        events.append(("write_marker", target_pid))
        return True

    def fake_pid_exists(check_pid):
        # Drain succeeds: pid "exits" right after the marker write.
        return ("write_marker", pid) not in events

    monkeypatch.setattr(status_mod, "write_planned_stop_marker", fake_write_marker)
    monkeypatch.setattr(status_mod, "_pid_exists", fake_pid_exists)
    monkeypatch.setattr(status_mod, "get_running_pid", lambda: pid)

    def fake_kill(**kwargs):
        events.append(("kill", kwargs.get("force", False)))
        return 0

    monkeypatch.setattr("hermes_cli.gateway.kill_gateway_processes", fake_kill)
    monkeypatch.setattr("hermes_cli.gateway._get_restart_drain_timeout", lambda: 5.0)

    gateway_windows.stop()

    # Marker MUST be written before any kill.
    kinds = [e[0] for e in events]
    assert "write_marker" in kinds, "stop() never wrote the planned-stop marker"
    marker_idx = kinds.index("write_marker")
    kill_idx = kinds.index("kill") if "kill" in kinds else len(kinds)
    assert marker_idx < kill_idx, (
        f"stop() killed before writing the marker (events={events})"
    )


def test_stop_waits_for_graceful_drain_before_force_kill(monkeypatch):
    """When drain succeeds, stop() should NOT force-kill the gateway.

    drained=True means the gateway exited cleanly after seeing the
    marker — escalating to taskkill /F afterwards would be wasted
    work and may emit confusing "killed N processes" output.
    """
    pid = 88888
    events = []

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)

    from gateway import status as status_mod
    monkeypatch.setattr(status_mod, "write_planned_stop_marker", lambda p: True)

    # Simulate the gateway exiting cleanly after one poll tick.
    poll_count = [0]
    def fake_pid_exists(check_pid):
        poll_count[0] += 1
        return poll_count[0] < 2  # alive on first poll, gone on second
    monkeypatch.setattr(status_mod, "_pid_exists", fake_pid_exists)
    monkeypatch.setattr(status_mod, "get_running_pid", lambda: pid)

    def fake_kill(**kwargs):
        events.append(("kill", kwargs.get("force", False)))
        return 0
    monkeypatch.setattr("hermes_cli.gateway.kill_gateway_processes", fake_kill)
    monkeypatch.setattr("hermes_cli.gateway._get_restart_drain_timeout", lambda: 5.0)

    gateway_windows.stop()

    # kill_gateway_processes is still called as the no-op sweep, but
    # NOT with force=True — drain succeeded, gateway is already gone.
    assert events == [("kill", False)], (
        f"After clean drain, force kill should be disabled (events={events})"
    )


def test_stop_escalates_to_force_kill_when_drain_times_out(monkeypatch):
    """When drain times out, stop() MUST escalate to force=True.

    Drain timeout = gateway is stuck or unresponsive. Without the
    taskkill /T /F escalation, the gateway stays alive and the next
    `hermes gateway start` fails with "another instance is running".
    """
    pid = 77777
    events = []

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)

    from gateway import status as status_mod
    monkeypatch.setattr(status_mod, "write_planned_stop_marker", lambda p: True)
    # PID never exits — drain times out.
    monkeypatch.setattr(status_mod, "_pid_exists", lambda check_pid: True)
    monkeypatch.setattr(status_mod, "get_running_pid", lambda: pid)

    def fake_kill(**kwargs):
        events.append(("kill", kwargs.get("force", False)))
        return 1
    monkeypatch.setattr("hermes_cli.gateway.kill_gateway_processes", fake_kill)
    # Tiny drain timeout to keep the test fast.
    monkeypatch.setattr("hermes_cli.gateway._get_restart_drain_timeout", lambda: 1.0)

    gateway_windows.stop()

    # When drain times out, kill is invoked with force=True so taskkill /T /F
    # walks the process tree.
    assert events == [("kill", True)], (
        f"After drain timeout, kill must use force=True (events={events})"
    )


def test_stop_no_running_gateway_skips_drain(monkeypatch):
    """When no gateway is running, skip the drain wait entirely."""
    events = []

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)

    from gateway import status as status_mod
    monkeypatch.setattr(status_mod, "get_running_pid", lambda: None)

    def fake_write_marker(target_pid):
        events.append(("write_marker", target_pid))
        return True
    monkeypatch.setattr(status_mod, "write_planned_stop_marker", fake_write_marker)
    monkeypatch.setattr(status_mod, "_pid_exists", lambda check_pid: False)

    def fake_kill(**kwargs):
        events.append(("kill", kwargs.get("force", False)))
        return 0
    monkeypatch.setattr("hermes_cli.gateway.kill_gateway_processes", fake_kill)
    monkeypatch.setattr("hermes_cli.gateway._get_restart_drain_timeout", lambda: 5.0)

    gateway_windows.stop()

    # With no PID to drain, no marker is written.  Kill sweep still runs
    # (defensive — covers the case where a stray gateway is alive without
    # a PID file).  force=True because drained=False.
    assert ("write_marker", None) not in events
    assert all(e[0] != "write_marker" for e in events), (
        f"Should not write marker when no PID is running (events={events})"
    )
    assert events == [("kill", True)]


def test_drain_helper_handles_invalid_pid(monkeypatch):
    """_drain_gateway_pid returns False for invalid PIDs without crashing."""
    assert gateway_windows._drain_gateway_pid(0, 5.0) is False
    assert gateway_windows._drain_gateway_pid(-1, 5.0) is False


def test_drain_helper_returns_true_when_pid_exits_quickly(monkeypatch):
    """_drain_gateway_pid polls _pid_exists until it returns False."""
    pid = 66666
    poll_count = [0]

    def fake_pid_exists(check_pid):
        poll_count[0] += 1
        return poll_count[0] < 3  # alive twice, then gone

    from gateway import status as status_mod
    monkeypatch.setattr(status_mod, "write_planned_stop_marker", lambda p: True)
    monkeypatch.setattr(status_mod, "_pid_exists", fake_pid_exists)

    assert gateway_windows._drain_gateway_pid(pid, drain_timeout=5.0) is True


def test_drain_helper_returns_false_on_timeout(monkeypatch):
    """_drain_gateway_pid returns False when the PID never exits."""
    from gateway import status as status_mod
    monkeypatch.setattr(status_mod, "write_planned_stop_marker", lambda p: True)
    monkeypatch.setattr(status_mod, "_pid_exists", lambda check_pid: True)

    assert gateway_windows._drain_gateway_pid(55555, drain_timeout=1.0) is False


def test_drain_helper_still_waits_if_marker_write_fails(monkeypatch):
    """Marker-write failures are swallowed; drain still polls for PID exit.

    If the marker can't be written (disk full, permission error), the
    gateway can't drain — but the wait still happens so a slow-shutdown
    gateway from a different code path (e.g. SIGTERM working on this
    platform after all) still gets observed cleanly.
    """
    pid = 44444
    def fake_write(target_pid):
        raise OSError("disk full")

    from gateway import status as status_mod
    monkeypatch.setattr(status_mod, "write_planned_stop_marker", fake_write)
    monkeypatch.setattr(status_mod, "_pid_exists", lambda check_pid: False)

    # Returns True because _pid_exists immediately says "gone".
    assert gateway_windows._drain_gateway_pid(pid, drain_timeout=5.0) is True


# ---------------------------------------------------------------------------
# Coverage for pure helpers, path resolution, and command flows (issue #36401)
#
# gateway_windows is win32-gated and several helpers call into ctypes.windll,
# which only exists on Windows. To keep these tests cross-platform (they run in
# CI on Linux) we no-op _assert_windows and inject a fake windll where needed.
# ---------------------------------------------------------------------------


def _patch_windows(monkeypatch):
    """Make the Windows-only guard a no-op so tests run on non-Windows CI too."""
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)


class _FakeShell32:
    def __init__(self, is_admin=1, shellexec=33):
        self._is_admin = is_admin
        self._shellexec = shellexec
        self.shellexec_calls = []

    def IsUserAnAdmin(self):
        return self._is_admin

    def ShellExecuteW(self, *args):
        self.shellexec_calls.append(args)
        return self._shellexec


class _FakeWindll:
    def __init__(self, shell32):
        self.shell32 = shell32


def _patch_windll(monkeypatch, *, is_admin=1, shellexec=33):
    shell32 = _FakeShell32(is_admin=is_admin, shellexec=shellexec)
    # raising=False: ctypes has no `windll` attribute on non-Windows platforms.
    monkeypatch.setattr(
        gateway_windows.ctypes, "windll", _FakeWindll(shell32), raising=False
    )
    return shell32


# --- quoting helpers -------------------------------------------------------


def test_quote_cmd_script_arg_variants():
    assert gateway_windows._quote_cmd_script_arg("") == '""'
    assert gateway_windows._quote_cmd_script_arg("plain") == "plain"
    assert gateway_windows._quote_cmd_script_arg("a b") == '"a b"'
    assert gateway_windows._quote_cmd_script_arg('a"b') == '"a""b"'
    with pytest.raises(ValueError):
        gateway_windows._quote_cmd_script_arg("line\nbreak")


def test_quote_schtasks_arg_backslash_escapes_quotes():
    assert gateway_windows._quote_schtasks_arg("plain") == "plain"
    assert gateway_windows._quote_schtasks_arg("a b") == '"a b"'
    assert gateway_windows._quote_schtasks_arg('a"b') == '"a\\"b"'


# --- schtasks wrapper ------------------------------------------------------


def test_exec_schtasks_reports_missing_binary(monkeypatch):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(gateway_windows.shutil, "which", lambda name: None)
    code, out, err = gateway_windows._exec_schtasks(["/Query"])
    assert code == 1
    assert "not found" in err


def test_exec_schtasks_success_passes_args_and_no_window_flag(monkeypatch):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(
        gateway_windows.shutil, "which", lambda name: "C:/Windows/schtasks.exe"
    )
    captured = {}

    class _Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _Result()

    monkeypatch.setattr(gateway_windows.subprocess, "run", fake_run)
    code, out, err = gateway_windows._exec_schtasks(["/Query", "/TN", "X"])
    assert (code, out, err) == (0, "ok", "")
    assert captured["argv"][0].endswith("schtasks.exe")
    assert captured["argv"][1:] == ["/Query", "/TN", "X"]
    assert captured["kwargs"]["creationflags"] == 0x08000000


def test_exec_schtasks_timeout_returns_124(monkeypatch):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(gateway_windows.shutil, "which", lambda name: "schtasks")

    def fake_run(*args, **kwargs):
        raise gateway_windows.subprocess.TimeoutExpired(cmd="schtasks", timeout=15)

    monkeypatch.setattr(gateway_windows.subprocess, "run", fake_run)
    code, out, err = gateway_windows._exec_schtasks(["/Query"])
    assert code == 124
    assert "timed out" in err


def test_exec_schtasks_oserror_returns_1(monkeypatch):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(gateway_windows.shutil, "which", lambda name: "schtasks")

    def fake_run(*args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(gateway_windows.subprocess, "run", fake_run)
    code, out, err = gateway_windows._exec_schtasks(["/Query"])
    assert code == 1
    assert "invocation failed" in err


def test_is_access_denied_matches_localized_phrases():
    assert gateway_windows._is_access_denied("ERROR: Access is denied.") is True
    assert gateway_windows._is_access_denied("acceso denegado") is True
    assert gateway_windows._is_access_denied("file not found") is False
    assert gateway_windows._is_access_denied("") is False


def test_is_running_as_admin_reads_shell32(monkeypatch):
    _patch_windows(monkeypatch)
    _patch_windll(monkeypatch, is_admin=1)
    assert gateway_windows._is_running_as_admin() is True
    _patch_windll(monkeypatch, is_admin=0)
    assert gateway_windows._is_running_as_admin() is False


def test_is_running_as_admin_false_on_error(monkeypatch):
    _patch_windows(monkeypatch)

    class _Boom:
        @property
        def shell32(self):
            raise OSError("no shell32 here")

    monkeypatch.setattr(gateway_windows.ctypes, "windll", _Boom(), raising=False)
    assert gateway_windows._is_running_as_admin() is False


def test_current_profile_cli_args_splits_profile_arg(monkeypatch):
    monkeypatch.setattr("hermes_cli.gateway._profile_arg", lambda: "--profile alice")
    assert gateway_windows._current_profile_cli_args() == ["--profile", "alice"]
    monkeypatch.setattr("hermes_cli.gateway._profile_arg", lambda: "")
    assert gateway_windows._current_profile_cli_args() == []


# --- elevated launchers ----------------------------------------------------


def test_launch_elevated_gateway_command_success(monkeypatch):
    _patch_windows(monkeypatch)
    monkeypatch.setattr("hermes_cli.gateway._profile_arg", lambda: "")
    monkeypatch.setattr(
        gateway_windows, "_derive_venv_pythonw", lambda exe: "C:/py/pythonw.exe"
    )
    shell32 = _patch_windll(monkeypatch, shellexec=42)
    assert (
        gateway_windows._launch_elevated_gateway_command("status", ["--deep"]) is True
    )
    verb, params = shell32.shellexec_calls[0][1], shell32.shellexec_calls[0][3]
    assert verb == "runas"
    assert "status" in params and "--deep" in params


def test_launch_elevated_gateway_command_failure_code(monkeypatch, capsys):
    _patch_windows(monkeypatch)
    monkeypatch.setattr("hermes_cli.gateway._profile_arg", lambda: "")
    monkeypatch.setattr(gateway_windows, "_derive_venv_pythonw", lambda exe: "pythonw")
    _patch_windll(monkeypatch, shellexec=5)  # <= 32 means ShellExecuteW failed
    assert gateway_windows._launch_elevated_gateway_command("install") is False
    assert "not started" in capsys.readouterr().out


def test_launch_elevated_gateway_command_exception(monkeypatch, capsys):
    _patch_windows(monkeypatch)
    monkeypatch.setattr("hermes_cli.gateway._profile_arg", lambda: "")
    monkeypatch.setattr(gateway_windows, "_derive_venv_pythonw", lambda exe: "pythonw")

    class _Shell32Raise:
        def ShellExecuteW(self, *args):
            raise OSError("shell exec blew up")

    monkeypatch.setattr(
        gateway_windows.ctypes, "windll", _FakeWindll(_Shell32Raise()), raising=False
    )
    assert gateway_windows._launch_elevated_gateway_command("install") is False
    assert "Could not launch" in capsys.readouterr().out


def test_launch_elevated_install_sets_then_restores_env(monkeypatch):
    _patch_windows(monkeypatch)
    # START_NOW is unset (exercises the unset->removed restore branch); HANDOFF
    # has a pre-existing value (exercises the restore-to-previous-value branch).
    monkeypatch.delenv("HERMES_GATEWAY_INSTALL_START_NOW", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_ELEVATED_HANDOFF", "OLD")
    captured = {}

    def fake_cmd(command, extra_args=None):
        captured["command"] = command
        captured["extra_args"] = list(extra_args or [])
        captured["start_now_during"] = gateway_windows.os.environ.get(
            "HERMES_GATEWAY_INSTALL_START_NOW"
        )
        captured["handoff_during"] = gateway_windows.os.environ.get(
            "HERMES_GATEWAY_ELEVATED_HANDOFF"
        )
        return True

    monkeypatch.setattr(gateway_windows, "_launch_elevated_gateway_command", fake_cmd)

    result = gateway_windows._launch_elevated_install(
        force=True, start_now=True, start_on_login=False
    )
    assert result is True
    assert captured["command"] == "install"
    assert "--elevated-handoff" in captured["extra_args"]
    assert "--force" in captured["extra_args"]
    assert "--start-now" in captured["extra_args"]
    assert "--no-start-on-login" in captured["extra_args"]
    # env vars are set for the duration of the elevated handoff...
    assert captured["start_now_during"] == "1"
    assert captured["handoff_during"] == "1"
    # ...and restored afterwards: the previously-unset var is removed, and the
    # pre-existing one is put back to its original value.
    assert "HERMES_GATEWAY_INSTALL_START_NOW" not in gateway_windows.os.environ
    assert gateway_windows.os.environ["HERMES_GATEWAY_ELEVATED_HANDOFF"] == "OLD"


def test_launch_elevated_uninstall_delegates(monkeypatch):
    monkeypatch.setattr(
        gateway_windows,
        "_launch_elevated_gateway_command",
        lambda command: command == "uninstall",
    )
    assert gateway_windows._launch_elevated_uninstall() is True


# --- names and paths -------------------------------------------------------


def test_get_task_name_default_and_profiled(monkeypatch):
    _patch_windows(monkeypatch)
    monkeypatch.setattr("hermes_cli.gateway._profile_suffix", lambda: "")
    assert gateway_windows.get_task_name() == gateway_windows._TASK_NAME_DEFAULT
    monkeypatch.setattr("hermes_cli.gateway._profile_suffix", lambda: "alice")
    assert (
        gateway_windows.get_task_name() == f"{gateway_windows._TASK_NAME_DEFAULT}_alice"
    )


def test_sanitize_filename_replaces_illegal_chars():
    assert gateway_windows._sanitize_filename('a:b/c\\d*e?f"g') == "a_b_c_d_e_f_g"
    assert gateway_windows._sanitize_filename("clean_name") == "clean_name"


def test_get_task_script_path_under_hermes_home(monkeypatch, tmp_path):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(
        gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice"
    )
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path))
    path = gateway_windows.get_task_script_path()
    assert path == tmp_path / "gateway-service" / "Hermes_Gateway_alice.cmd"
    assert (tmp_path / "gateway-service").is_dir()


def test_startup_dir_prefers_appdata(monkeypatch):
    monkeypatch.setenv("APPDATA", "/roamingdir/Roaming")
    result = gateway_windows._startup_dir()
    assert result.name == "Startup"
    assert "Roaming" in str(result)


def test_startup_dir_falls_back_to_userprofile(monkeypatch):
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setenv("USERPROFILE", "/home/user")
    result = gateway_windows._startup_dir()
    assert result.name == "Startup"
    assert "AppData" in result.parts and "Roaming" in result.parts


def test_startup_dir_raises_without_env(monkeypatch):
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("USERPROFILE", raising=False)
    monkeypatch.delenv("HOME", raising=False)
    with pytest.raises(RuntimeError):
        gateway_windows._startup_dir()


def test_get_startup_entry_path(monkeypatch, tmp_path):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    monkeypatch.setattr(gateway_windows, "_startup_dir", lambda: tmp_path)
    assert gateway_windows.get_startup_entry_path() == tmp_path / "Hermes_Gateway.cmd"


# --- venv / python resolution ----------------------------------------------


def test_read_pyvenv_cfg_parses_and_handles_missing(tmp_path):
    (tmp_path / "pyvenv.cfg").write_text(
        "home = C:/py\nuv = 0.1\nno_equals_line\n", encoding="utf-8"
    )
    cfg = gateway_windows._read_pyvenv_cfg(tmp_path)
    assert cfg["home"] == "C:/py"
    assert cfg["uv"] == "0.1"
    assert "no_equals_line" not in cfg
    assert gateway_windows._read_pyvenv_cfg(tmp_path / "absent") == {}


def test_derive_venv_pythonw_prefers_w_variant(tmp_path):
    py = tmp_path / "python.exe"
    pyw = tmp_path / "pythonw.exe"
    py.write_text("", encoding="utf-8")
    pyw.write_text("", encoding="utf-8")
    assert gateway_windows._derive_venv_pythonw(str(py)) == str(pyw)

    lonely = tmp_path / "sub" / "python.exe"
    lonely.parent.mkdir()
    lonely.write_text("", encoding="utf-8")
    assert gateway_windows._derive_venv_pythonw(str(lonely)) == str(lonely)


def test_resolve_detached_python_uv_uses_base_pythonw(tmp_path):
    venv = tmp_path / "venv"
    scripts = venv / "Scripts"
    site = venv / "Lib" / "site-packages"
    base = tmp_path / "base"
    for directory in (scripts, site, base):
        directory.mkdir(parents=True)
    (scripts / "python.exe").write_text("", encoding="utf-8")
    (base / "pythonw.exe").write_text("", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text(f"home = {base}\nuv = 0.1\n", encoding="utf-8")

    windowed, venv_dir, extra = gateway_windows._resolve_detached_python(
        str(scripts / "python.exe")
    )
    assert windowed == str(base / "pythonw.exe")
    assert venv_dir == venv
    assert extra == [str(site)]


def test_resolve_detached_python_non_uv_returns_windowed(tmp_path):
    venv = tmp_path / "venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_text("", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text("home = C:/py\n", encoding="utf-8")  # no uv key

    windowed, venv_dir, extra = gateway_windows._resolve_detached_python(
        str(scripts / "python.exe")
    )
    assert extra == []
    assert venv_dir == venv
    # No pythonw.exe sibling and no uv launcher: fall back to the given python.
    assert windowed == str(scripts / "python.exe")


def test_prepend_pythonpath_joins_and_skips_empty(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    overlay = {}
    gateway_windows._prepend_pythonpath(overlay, ["", "a", "b"])
    assert overlay["PYTHONPATH"] == gateway_windows.os.pathsep.join(["a", "b"])


def test_prepend_pythonpath_appends_existing(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "EXISTING")
    overlay = {}
    gateway_windows._prepend_pythonpath(overlay, ["a"])
    assert overlay["PYTHONPATH"] == "a" + gateway_windows.os.pathsep + "EXISTING"


def test_prepend_pythonpath_noop_when_all_empty():
    overlay = {}
    gateway_windows._prepend_pythonpath(overlay, ["", ""])
    assert "PYTHONPATH" not in overlay


# --- install-choice env parsing --------------------------------------------


def test_install_choice_from_env(monkeypatch):
    monkeypatch.delenv("HERMES_TEST_CHOICE", raising=False)
    assert gateway_windows._install_choice_from_env("HERMES_TEST_CHOICE") is None
    monkeypatch.setenv("HERMES_TEST_CHOICE", "Yes")
    assert gateway_windows._install_choice_from_env("HERMES_TEST_CHOICE") is True
    monkeypatch.setenv("HERMES_TEST_CHOICE", "off")
    assert gateway_windows._install_choice_from_env("HERMES_TEST_CHOICE") is False
    monkeypatch.setenv("HERMES_TEST_CHOICE", "maybe")
    assert gateway_windows._install_choice_from_env("HERMES_TEST_CHOICE") is None


# --- detached spawn --------------------------------------------------------


def test_spawn_detached_returns_pid_and_writes_stray_log(monkeypatch, tmp_path):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(
        gateway_windows,
        "_build_gateway_argv",
        lambda: (["pythonw", "-m", "hermes_cli.main"], str(tmp_path), {}),
    )
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path))
    captured = {}

    class _Proc:
        pid = 4321

    def fake_popen(argv, **kwargs):
        captured["flags"] = kwargs["creationflags"]
        return _Proc()

    monkeypatch.setattr(gateway_windows.subprocess, "Popen", fake_popen)
    pid = gateway_windows._spawn_detached()
    assert pid == 4321
    assert (tmp_path / "logs" / "gateway-stdio.log").exists()
    # First attempt includes CREATE_BREAKAWAY_FROM_JOB.
    assert captured["flags"] & 0x01000000


def test_spawn_detached_retries_without_breakaway_on_oserror(monkeypatch, tmp_path):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(
        gateway_windows, "_build_gateway_argv", lambda: (["pythonw"], str(tmp_path), {})
    )
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path))
    attempts = []

    class _Proc:
        pid = 999

    def fake_popen(argv, **kwargs):
        attempts.append(kwargs["creationflags"])
        if len(attempts) == 1:
            raise OSError("breakaway not permitted")
        return _Proc()

    monkeypatch.setattr(gateway_windows.subprocess, "Popen", fake_popen)
    pid = gateway_windows._spawn_detached()
    assert pid == 999
    assert len(attempts) == 2
    # Retry drops the CREATE_BREAKAWAY_FROM_JOB bit.
    assert not (attempts[1] & 0x01000000)


# --- status query / installed checks ---------------------------------------


def test_query_task_status_empty_on_nonzero(monkeypatch):
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "T")
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda args: (1, "", "err"))
    assert gateway_windows.query_task_status() == {}


def test_query_task_status_parses_list_output(monkeypatch):
    out = (
        "Folder: \\\n"
        "Status:                  Ready\n"
        "Last Run Time:           6/3/2026 1:00:00 PM\n"
        "Last Result:             0\n"
        "Irrelevant line without colon\n"
    )
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "T")
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda args: (0, out, ""))
    info = gateway_windows.query_task_status()
    assert info["status"] == "Ready"
    assert info["last run time"].startswith("6/3/2026")
    # "Last Result" (some locales) aliases onto "last run result".
    assert info["last run result"] == "0"


def test_is_task_registered_uses_query_exit_code(monkeypatch):
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "T")
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda args: (0, "", ""))
    assert gateway_windows.is_task_registered() is True
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda args: (1, "", ""))
    assert gateway_windows.is_task_registered() is False


def test_is_installed_true_when_either_present(monkeypatch):
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
    monkeypatch.setattr(gateway_windows, "is_startup_entry_installed", lambda: True)
    assert gateway_windows.is_installed() is True


def test_gateway_pids_delegates_to_finder(monkeypatch):
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [1, 2])
    assert gateway_windows._gateway_pids() == [1, 2]


def test_wait_for_gateway_ready_returns_pids(monkeypatch):
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [7])
    assert gateway_windows._wait_for_gateway_ready(timeout_s=1, interval_s=0.01) == [7]


def test_wait_for_gateway_ready_times_out(monkeypatch):
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway_windows.time, "sleep", lambda s: None)
    assert (
        gateway_windows._wait_for_gateway_ready(timeout_s=0.05, interval_s=0.01) == []
    )


def test_report_gateway_start_success_and_failure(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda: [11])
    gateway_windows._report_gateway_start("scheduled task")
    assert "11" in capsys.readouterr().out

    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda: [])
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path))
    gateway_windows._report_gateway_start("scheduled task")
    assert "no process detected" in capsys.readouterr().out


def test_print_next_steps(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path))
    gateway_windows._print_next_steps()
    assert "Next steps" in capsys.readouterr().out


# --- status / start / restart ----------------------------------------------


def test_status_reports_task_and_running_process(monkeypatch, capsys):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_startup_entry_installed", lambda: False)
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda: [42])
    monkeypatch.setattr(
        gateway_windows,
        "query_task_status",
        lambda: {"status": "Ready", "last run result": "0"},
    )
    gateway_windows.status()
    out = capsys.readouterr().out
    assert "Scheduled Task registered" in out
    assert "Status: Ready" in out
    assert "42" in out


def test_status_not_installed_shows_install_hint(monkeypatch, capsys):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
    monkeypatch.setattr(gateway_windows, "is_startup_entry_installed", lambda: False)
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda: [])
    gateway_windows.status()
    out = capsys.readouterr().out
    assert "not installed" in out
    assert "hermes gateway install" in out


def test_status_deep_prints_paths(monkeypatch, capsys, tmp_path):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
    monkeypatch.setattr(gateway_windows, "is_startup_entry_installed", lambda: True)
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda: [])
    monkeypatch.setattr(
        gateway_windows, "get_task_script_path", lambda: tmp_path / "t.cmd"
    )
    monkeypatch.setattr(
        gateway_windows, "get_startup_entry_path", lambda: tmp_path / "s.cmd"
    )
    gateway_windows.status(deep=True)
    out = capsys.readouterr().out
    assert "login item installed" in out
    assert "Task name:" in out


def test_start_runs_scheduled_task_when_registered(monkeypatch):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_startup_entry_installed", lambda: False)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda args: (0, "", ""))
    reported = []
    monkeypatch.setattr(
        gateway_windows, "_report_gateway_start", lambda via: reported.append(via)
    )
    gateway_windows.start()
    assert reported and "Scheduled Task" in reported[0]


def test_start_falls_back_to_spawn_when_run_fails(monkeypatch, capsys):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_startup_entry_installed", lambda: False)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda args: (1, "", "boom"))
    spawned = []
    monkeypatch.setattr(
        gateway_windows, "_spawn_detached", lambda: spawned.append(True) or 7
    )
    monkeypatch.setattr(gateway_windows, "_report_gateway_start", lambda via: None)
    gateway_windows.start()
    assert spawned == [True]


def test_start_declined_install_is_noop(monkeypatch, capsys):
    _patch_windows(monkeypatch)
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
    monkeypatch.setattr(gateway_windows, "is_startup_entry_installed", lambda: False)
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *a, **k: False)
    gateway_windows.start()
    assert "hermes gateway install" in capsys.readouterr().out


def test_restart_stops_then_starts(monkeypatch):
    _patch_windows(monkeypatch)
    order = []
    monkeypatch.setattr(gateway_windows, "stop", lambda: order.append("stop"))
    monkeypatch.setattr(gateway_windows, "start", lambda: order.append("start"))
    monkeypatch.setattr(gateway_windows.time, "sleep", lambda s: None)
    gateway_windows.restart()
    assert order == ["stop", "start"]
