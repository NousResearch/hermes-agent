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
    monkeypatch.setattr(gateway_windows.ctypes, "windll", FakeWindll())

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