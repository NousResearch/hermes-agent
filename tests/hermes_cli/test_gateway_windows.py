"""Tests for hermes_cli.gateway_windows."""

from pathlib import Path

import pytest

import hermes_cli.gateway as gateway
import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.setup as setup




def test_schtasks_encoding_falls_back_to_utf8(monkeypatch):
    """A broken/empty locale must not leave us without a decoder (issue #38172)."""

    monkeypatch.setattr(gateway_windows.locale, "getpreferredencoding", lambda *a, **k: "")
    assert gateway_windows._schtasks_encoding() == "utf-8"

    def _boom(*args, **kwargs):
        raise RuntimeError("locale exploded")

    monkeypatch.setattr(gateway_windows.locale, "getpreferredencoding", _boom)
    assert gateway_windows._schtasks_encoding() == "utf-8"




def test_build_gateway_argv_keeps_venv_console_python_for_uv_venv(monkeypatch, tmp_path):
    """No pythonw / base-interpreter detour: the venv console python.exe is
    launched hidden (CREATE_NO_WINDOW) so descendants inherit its hidden
    console instead of flashing their own (#54220/#56747)."""

    project = tmp_path / "project"
    scripts = project / "venv" / "Scripts"
    site_packages = project / "venv" / "Lib" / "site-packages"
    hermes_home = tmp_path / "hermes-home"
    base = tmp_path / "uv" / "python" / "cpython-3.11-windows-x86_64-none"
    scripts.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    hermes_home.mkdir()
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
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(hermes_home))

    argv, cwd, env_overlay = gateway_windows._build_gateway_argv()

    assert argv[:3] == [str(venv_python), "-m", "hermes_cli.main"]
    assert cwd == str(hermes_home.resolve())
    assert env_overlay["VIRTUAL_ENV"] == str(project / "venv")
    assert str(project) in env_overlay["PYTHONPATH"].split(gateway_windows.os.pathsep)


class TestStableWindowsGatewayWorkingDir:
    def test_stable_gateway_working_dir_uses_hermes_home(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home)
        assert gateway_windows._stable_gateway_working_dir(tmp_path / "checkout") == str(home.resolve())

    def test_stable_gateway_working_dir_falls_back_to_project_root(self, tmp_path, monkeypatch):
        missing = tmp_path / "missing" / ".hermes"
        project = tmp_path / "checkout"
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: missing)
        assert gateway_windows._stable_gateway_working_dir(project) == str(project)




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




def test_elevated_gateway_command_uses_hidden_console_python(monkeypatch):
    """UAC handoff launches console python with SW_HIDE — a single hidden
    console, not console-less pythonw (#54220/#56747), and no visible
    elevated cmd.exe window left open."""
    calls = []

    class FakeShell32:
        def ShellExecuteW(self, hwnd, verb, executable, params, cwd, show):
            calls.append((hwnd, verb, executable, params, cwd, show))
            return 33

    class FakeWindll:
        shell32 = FakeShell32()

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "_current_profile_cli_args", lambda: ["--profile", "alice"])
    monkeypatch.setattr(gateway_windows.sys, "executable", r"C:\Hermes\venv\Scripts\python.exe")
    monkeypatch.setattr(gateway_windows.ctypes, "windll", FakeWindll(), raising=False)

    assert gateway_windows._launch_elevated_gateway_command("install", ["--start-now", "--elevated-handoff"])

    assert len(calls) == 1
    _hwnd, verb, executable, params, cwd, show = calls[0]
    assert verb == "runas"
    assert executable == r"C:\Hermes\venv\Scripts\python.exe"
    assert "--profile alice gateway install --start-now --elevated-handoff" in params
    assert show == 0
    assert cwd


def test_install_scheduled_task_recreates_instead_of_change(monkeypatch, tmp_path):
    """Install must delete+create so stale minute-repeat task settings are not preserved."""
    calls = []
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"
    xml_seen = {}

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: r"DOMAIN\\alice")

    def fake_schtasks(args):
        calls.append(tuple(args))
        if args[0] == "/Delete":
            return (0, "SUCCESS", "")
        if args[0] == "/Create":
            xml_path = Path(args[args.index("/XML") + 1])
            xml_seen["text"] = xml_path.read_text(encoding="utf-16")
            return (0, "SUCCESS", "")
        raise AssertionError(f"unexpected schtasks args: {args}")

    monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)
    ok, detail = gateway_windows._install_scheduled_task("Hermes_Gateway_alice", script_path)

    assert ok is True
    assert "/Change" not in [arg for call in calls for arg in call]
    assert calls[0][:4] == ("/Delete", "/F", "/TN", "Hermes_Gateway_alice")
    assert calls[1][0] == "/Create"
    assert "/XML" in calls[1]
    assert "/SC" not in calls[1]
    assert "<Delay>PT30S</Delay>" in xml_seen["text"]
    assert "<StartWhenAvailable>true</StartWhenAvailable>" in xml_seen["text"]
    assert "<StopOnIdleEnd>false</StopOnIdleEnd>" in xml_seen["text"]
    assert "<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>" in xml_seen["text"]
    assert "<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>" in xml_seen["text"]
    assert "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>" in xml_seen["text"]
    assert "<RestartOnFailure>" in xml_seen["text"]
    assert "<Count>999</Count>" in xml_seen["text"]
    # Scheduled Task launches the console-less .vbs via wscript.exe, never cmd.exe
    # (issue #45599 fix A: no console -> no logon CTRL_CLOSE_EVENT / 0xC000013A).
    assert "<Command>wscript.exe</Command>" in xml_seen["text"]
    assert "//B //Nologo" in xml_seen["text"]
    assert "Hermes_Gateway_alice.vbs" in xml_seen["text"]
    assert "cmd.exe" not in xml_seen["text"]


def test_gateway_vbs_script_is_console_less(monkeypatch):
    """The .vbs launcher must avoid cmd.exe entirely and Run pythonw hidden
    (issue #45599 fix A: no console -> no logon CTRL_CLOSE_EVENT / 0xC000013A)."""
    monkeypatch.setattr(
        gateway_windows,
        "_resolve_detached_python",
        lambda exe: (r"C:\venv\Scripts\pythonw.exe", Path(r"C:\venv"), []),
    )
    content = gateway_windows._build_gateway_vbs_script(
        r"C:\venv\Scripts\python.exe",
        r"C:\Hermes",
        r"C:\Hermes",
        "--profile work",
    )
    assert "cmd.exe" not in content.lower()
    assert 'CreateObject("WScript.Shell")' in content
    assert "pythonw.exe" in content
    assert "hermes_cli.main" in content
    assert "gateway run" in content
    assert ", 0, False" in content  # hidden window, detached/async
    for var in ("HERMES_HOME", "PYTHONIOENCODING", "HERMES_GATEWAY_DETACHED", "VIRTUAL_ENV", "PYTHONPATH"):
        assert var in content
    assert "--profile" in content and "work" in content
    assert content.endswith("\r\n")


def test_gateway_vbs_script_sets_strict_auth_home(monkeypatch, tmp_path):
    auth_home = tmp_path / 'auth "residence"'
    monkeypatch.setenv("HERMES_AUTH_HOME", str(auth_home))
    monkeypatch.setattr(
        gateway_windows,
        "_resolve_detached_python",
        lambda exe: (r"C:\venv\Scripts\python.exe", Path(r"C:\venv"), []),
    )
    monkeypatch.setattr(
        gateway_windows,
        "_preserve_hermes_home_path",
        lambda path: str(path),
    )

    content = gateway_windows._build_gateway_vbs_script(
        r"C:\venv\Scripts\python.exe",
        r"C:\Hermes",
        r"C:\Hermes",
        "",
    )

    escaped = str(auth_home.resolve()).replace('"', '""')
    assert f'env.Item("HERMES_AUTH_HOME") = "{escaped}"' in content


def test_persistent_windows_launchers_run_auth_aware_vbs(
    monkeypatch, tmp_path
):
    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth-residence"
    runtime_home.mkdir()
    script_path = tmp_path / "Hermes_Gateway.cmd"
    startup_path = tmp_path / "Startup" / "Hermes_Gateway.vbs"
    startup_path.parent.mkdir()

    monkeypatch.setenv("HERMES_AUTH_HOME", str(auth_home))
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(
        gateway_windows,
        "get_task_script_path",
        lambda: script_path,
    )
    monkeypatch.setattr(
        gateway_windows,
        "get_startup_entry_path",
        lambda: startup_path,
    )
    monkeypatch.setattr(
        gateway_windows,
        "_legacy_startup_entry_path",
        lambda: startup_path.with_suffix(".cmd"),
    )
    monkeypatch.setattr(gateway, "get_python_path", lambda: r"C:\venv\python.exe")
    monkeypatch.setattr(gateway, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(gateway, "_profile_arg", lambda home: "")
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home",
        lambda: runtime_home,
    )
    monkeypatch.setattr(
        gateway_windows,
        "_stable_gateway_working_dir",
        lambda project_root: str(runtime_home),
    )
    monkeypatch.setattr(
        gateway_windows,
        "_resolve_detached_python",
        lambda exe: (exe, Path(r"C:\venv"), []),
    )
    monkeypatch.setattr(
        gateway_windows,
        "_preserve_hermes_home_path",
        lambda path: str(path),
    )

    written_script = gateway_windows._write_task_script()
    vbs_path = written_script.with_suffix(".vbs")
    vbs = vbs_path.read_text(encoding="utf-8")
    assert (
        f'env.Item("HERMES_AUTH_HOME") = "{auth_home.resolve()}"'
        in vbs
    )

    task_xml = gateway_windows._build_scheduled_task_xml(
        "Hermes Gateway",
        vbs_path,
        r"DOMAIN\alice",
    )
    assert "<Command>wscript.exe</Command>" in task_xml
    assert str(vbs_path) in task_xml

    startup_entry = gateway_windows._install_startup_entry(written_script)
    startup = startup_entry.read_text(encoding="utf-8")
    assert str(vbs_path) in startup
    assert "wscript.exe" in startup


@pytest.mark.parametrize("operation", ("start", "restart"))
@pytest.mark.parametrize("explicit_override", (False, True))
def test_windows_manual_service_actions_preserve_auth_home(
    monkeypatch,
    tmp_path,
    operation,
    explicit_override,
):
    runtime_home = tmp_path / "runtime"
    installed_auth_home = tmp_path / "installed-auth"
    caller_auth_home = tmp_path / "caller-auth"
    runtime_home.mkdir()
    script_path = tmp_path / "Hermes_Gateway.cmd"
    spawned = []

    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(installed_auth_home))
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(
        gateway_windows,
        "get_task_script_path",
        lambda: script_path,
    )
    monkeypatch.setattr(gateway, "get_python_path", lambda: str(tmp_path / "python.exe"))
    monkeypatch.setattr(gateway, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(gateway, "_profile_arg", lambda home: "")
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home",
        lambda: runtime_home,
    )
    monkeypatch.setattr(
        gateway_windows,
        "_stable_gateway_working_dir",
        lambda project_root: str(runtime_home),
    )
    monkeypatch.setattr(
        gateway_windows,
        "_resolve_detached_python",
        lambda exe: (exe, tmp_path / "venv", []),
    )
    monkeypatch.setattr(
        gateway_windows,
        "_preserve_hermes_home_path",
        lambda path: str(path),
    )
    gateway_windows._write_task_script()

    if explicit_override:
        monkeypatch.setenv("HERMES_AUTH_HOME", str(caller_auth_home))
        expected_auth_home = caller_auth_home.resolve()
    else:
        monkeypatch.delenv("HERMES_AUTH_HOME")
        expected_auth_home = installed_auth_home.resolve()

    class SpawnedProcess:
        pid = 4321

    def fake_popen(argv, **kwargs):
        spawned.append((argv, kwargs))
        return SpawnedProcess()

    monkeypatch.setattr(gateway_windows.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gateway_windows, "windows_detach_flags", lambda: 0)
    monkeypatch.setattr(
        gateway_windows,
        "windows_detach_flags_without_breakaway",
        lambda: 0,
    )
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(
        gateway_windows,
        "is_startup_entry_installed",
        lambda: False,
    )
    monkeypatch.setattr(
        gateway_windows,
        "_report_gateway_start",
        lambda via: None,
    )

    if operation == "restart":
        monkeypatch.setattr(gateway_windows, "stop", lambda: None)
        monkeypatch.setattr(
            gateway_windows,
            "_wait_for_gateway_absent",
            lambda timeout_s: True,
        )
        monkeypatch.setattr(
            gateway_windows,
            "_wait_for_gateway_ready",
            lambda timeout_s: True,
        )
        monkeypatch.setattr(gateway_windows.time, "sleep", lambda seconds: None)
        gateway_windows.restart()
    else:
        gateway_windows.start()

    assert len(spawned) == 1
    assert (
        spawned[0][1]["env"]["HERMES_AUTH_HOME"]
        == str(expected_auth_home)
    )


def test_windows_direct_spawn_falls_back_to_installed_cmd_pin(
    monkeypatch,
    tmp_path,
):
    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth%residence"
    runtime_home.mkdir()
    script_path = tmp_path / "Hermes_Gateway.cmd"
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(auth_home))
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(
        gateway_windows,
        "get_task_script_path",
        lambda: script_path,
    )
    monkeypatch.setattr(gateway, "get_python_path", lambda: str(tmp_path / "python.exe"))
    monkeypatch.setattr(gateway, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(gateway, "_profile_arg", lambda home: "")
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home",
        lambda: runtime_home,
    )
    monkeypatch.setattr(
        gateway_windows,
        "_stable_gateway_working_dir",
        lambda project_root: str(runtime_home),
    )
    monkeypatch.setattr(
        gateway_windows,
        "_resolve_detached_python",
        lambda exe: (exe, tmp_path / "venv", []),
    )
    monkeypatch.setattr(
        gateway_windows,
        "_preserve_hermes_home_path",
        lambda path: str(path),
    )
    gateway_windows._write_task_script()
    vbs_path = script_path.with_suffix(".vbs")
    vbs_without_pin = "\r\n".join(
        line
        for line in vbs_path.read_text(encoding="utf-8").splitlines()
        if "HERMES_AUTH_HOME" not in line
    )
    vbs_path.write_text(vbs_without_pin, encoding="utf-8")
    monkeypatch.delenv("HERMES_AUTH_HOME")

    assert gateway_windows._installed_auth_home_for_service() == str(
        auth_home.resolve()
    )













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








