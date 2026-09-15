"""Tests for hermes_cli.gateway_windows."""

import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway
import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.setup as setup


_BREAKAWAY_MARKER = "_HERMES_GATEWAY_BREAKAWAY"




def test_schtasks_encoding_falls_back_to_utf8(monkeypatch):
    """A broken/empty locale must not leave us without a decoder (issue #38172)."""

    monkeypatch.setattr(gateway_windows.locale, "getpreferredencoding", lambda *a, **k: "")
    assert gateway_windows._schtasks_encoding() == "utf-8"

    def _boom(*args, **kwargs):
        raise RuntimeError("locale exploded")

    monkeypatch.setattr(gateway_windows.locale, "getpreferredencoding", _boom)
    assert gateway_windows._schtasks_encoding() == "utf-8"




@pytest.mark.windows_only
def test_build_gateway_argv_keeps_venv_console_python_for_uv_venv(monkeypatch, tmp_path):
    """No pythonw / base-interpreter detour: the venv console python.exe is
    launched hidden (CREATE_NO_WINDOW) so descendants inherit its hidden
    console instead of flashing their own (#54220/#56747).

    Windows-only: ``_build_gateway_argv()`` asserts the host is Windows and the
    argv/env overlay it returns is built from real Windows path separators and
    ``Scripts/python.exe`` layout — a patched ``sys.platform`` covered the
    branch but not any of that.
    """

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

    monkeypatch.setattr(gateway, "PROJECT_ROOT", project)
    monkeypatch.setattr(gateway, "get_python_path", lambda: str(venv_python))
    monkeypatch.setattr(gateway, "_profile_arg", lambda hermes_home: "")
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(hermes_home))

    argv, cwd, env_overlay = gateway_windows._build_gateway_argv()

    assert argv[:3] == [str(venv_python), "-m", "hermes_cli.main"]
    assert cwd == str(hermes_home.resolve())
    assert env_overlay["VIRTUAL_ENV"] == str(project / "venv")
    assert str(project) in env_overlay["PYTHONPATH"].split(gateway_windows.os.pathsep)


@pytest.mark.windows_only
def test_spawn_detached_marks_primary_breakaway_success(monkeypatch, tmp_path, caplog):
    """A successful breakaway spawn reports true without a warning."""
    argv = ["python.exe", "-m", "hermes_cli.main", "gateway", "run"]
    cwd = str(tmp_path)
    calls = []

    def fake_popen(call_argv, **kwargs):
        calls.append((call_argv, kwargs))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(
        gateway_windows,
        "_build_gateway_argv",
        lambda: (argv, cwd, {"HERMES_GATEWAY_DETACHED": "1"}),
    )
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(gateway_windows.subprocess, "Popen", fake_popen)
    caplog.set_level(logging.WARNING, logger=gateway_windows.__name__)

    assert gateway_windows._spawn_detached() == 12345
    assert len(calls) == 1
    actual_argv, kwargs = calls[0]
    assert actual_argv == argv
    assert kwargs["cwd"] == cwd
    assert kwargs["creationflags"] == gateway_windows.windows_detach_flags()
    assert kwargs["env"][_BREAKAWAY_MARKER] == "1"
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is kwargs["stderr"]
    assert not caplog.records


@pytest.mark.windows_only
def test_spawn_detached_warns_and_marks_no_breakaway_fallback(
    monkeypatch, tmp_path, caplog
):
    """A denied breakaway retries once with private false metadata."""
    argv = ["python.exe", "-m", "hermes_cli.main", "gateway", "run"]
    cwd = str(tmp_path)
    calls = []

    def fake_popen(call_argv, **kwargs):
        calls.append((call_argv, kwargs))
        if len(calls) == 1:
            error = OSError(13, "Access is denied")
            error.winerror = 5
            raise error
        return SimpleNamespace(pid=23456)

    monkeypatch.setattr(
        gateway_windows,
        "_build_gateway_argv",
        lambda: (
            argv,
            cwd,
            {"HERMES_GATEWAY_DETACHED": "1", "SECRET_SENTINEL": "do-not-log"},
        ),
    )
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(gateway_windows.subprocess, "Popen", fake_popen)
    caplog.set_level(logging.WARNING, logger=gateway_windows.__name__)

    assert gateway_windows._spawn_detached() == 23456
    assert len(calls) == 2
    (argv_primary, primary), (argv_fallback, fallback) = calls
    assert argv_primary == argv_fallback == argv
    assert primary["cwd"] == fallback["cwd"] == cwd
    assert primary["creationflags"] == gateway_windows.windows_detach_flags()
    assert (
        fallback["creationflags"]
        == gateway_windows.windows_detach_flags_without_breakaway()
    )
    assert primary["stdin"] is fallback["stdin"] is subprocess.DEVNULL
    assert primary["stdout"] is primary["stderr"]
    assert fallback["stdout"] is fallback["stderr"]
    assert Path(primary["stdout"].name) == Path(fallback["stdout"].name)
    assert primary["close_fds"] is fallback["close_fds"] is True
    assert primary["env"] is not fallback["env"]
    assert primary["env"][_BREAKAWAY_MARKER] == "1"
    assert fallback["env"][_BREAKAWAY_MARKER] == "0"
    assert {
        key: value for key, value in primary["env"].items() if key != _BREAKAWAY_MARKER
    } == {
        key: value for key, value in fallback["env"].items() if key != _BREAKAWAY_MARKER
    }

    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "5" in warnings[0].getMessage()
    assert "do-not-log" not in warnings[0].getMessage()
    assert str(tmp_path) not in warnings[0].getMessage()


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




@pytest.mark.windows_only
def test_elevated_gateway_command_uses_hidden_console_python(monkeypatch):
    """UAC handoff launches console python with SW_HIDE — a single hidden
    console, not console-less pythonw (#54220/#56747), and no visible
    elevated cmd.exe window left open.

    Windows-only: the code path runs behind ``_assert_windows()`` and goes
    through ``ctypes.windll.shell32``, neither of which exists on a faked
    host. ShellExecuteW itself stays mocked — it would raise a real UAC
    prompt — but the host identity is genuine.
    """
    calls = []

    class FakeShell32:
        def ShellExecuteW(self, hwnd, verb, executable, params, cwd, show):
            calls.append((hwnd, verb, executable, params, cwd, show))
            return 33

    class FakeWindll:
        shell32 = FakeShell32()

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
    """Install must delete+create so stale minute-repeat task settings are not preserved.

    Host-agnostic on purpose: ``_install_scheduled_task`` only renders the task
    XML and shells out through ``_exec_schtasks`` (mocked here as the genuine
    external dependency), so no platform fake is needed.
    """
    calls = []
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"
    xml_seen = {}

    monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: r"DOMAIN\\alice")
    # Pin the .vbs launcher kind: on a host without the VBScript engine the builder legitimately
    # writes a .ps1 instead, which is covered by test_scheduled_task_xml_targets_powershell_launcher.
    monkeypatch.setattr(gateway_windows, "_prefer_powershell_launcher", lambda: False)

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


# ---------------------------------------------------------------------------
# VBScript-free Windows hosts — the launcher must be a .ps1 instead of a .vbs
#
# Background: VBScript is no longer installed by default on current Windows 11
# builds. `wscript.exe` there exits 1 with "There is no script engine for file
# extension .vbs", so a Scheduled Task / Startup entry pointing at a .vbs
# SILENTLY never starts the gateway (schtasks still reports SUCCESS), while
# `hermes gateway start` keeps working because it spawns directly. The launcher
# kind therefore follows the host.
# ---------------------------------------------------------------------------

def test_vbscript_engine_availability_follows_system_root(tmp_path):
    """Engine detection is file-driven, so a missing/present DLL is decided by DATA, not the host OS."""
    present = tmp_path / "with-vbs"
    absent = tmp_path / "without-vbs"
    for root in (present, absent):
        (root / "System32").mkdir(parents=True)
    (present / "System32" / "vbscript.dll").write_bytes(b"MZ")

    assert gateway_windows._vbscript_engine_available(present) is True
    assert gateway_windows._vbscript_engine_available(absent) is False


def test_prefer_powershell_launcher_when_vbscript_missing(monkeypatch):
    """No VBScript engine + a usable powershell.exe => the .ps1 launcher is required."""
    monkeypatch.setattr(gateway_windows, "_vbscript_engine_available", lambda *a, **k: False)
    monkeypatch.setattr(gateway_windows, "_powershell_exe", lambda: r"C:\Windows\powershell.exe")
    assert gateway_windows._prefer_powershell_launcher() is True


def test_no_powershell_fallback_keeps_vbs_launcher(monkeypatch):
    """Never switch kinds without a working interpreter — a .ps1 with no powershell.exe is worse."""
    monkeypatch.setattr(gateway_windows, "_vbscript_engine_available", lambda *a, **k: False)
    monkeypatch.setattr(gateway_windows, "_powershell_exe", lambda: None)
    assert gateway_windows._prefer_powershell_launcher() is False


def test_scheduled_task_xml_targets_powershell_launcher(monkeypatch):
    """The task action must run the .ps1 through powershell.exe, never wscript.exe or cmd.exe."""
    monkeypatch.setattr(gateway_windows, "_powershell_exe", lambda: r"C:\Windows\powershell.exe")
    launcher = Path(r"C:\Hermes\gateway-service\Hermes_Gateway.ps1")

    xml = gateway_windows._build_scheduled_task_xml("Hermes_Gateway", launcher, r"DOMAIN\alice")

    assert r"<Command>C:\Windows\powershell.exe</Command>" in xml
    assert "-NoProfile -ExecutionPolicy Bypass -File" in xml
    assert "Hermes_Gateway.ps1" in xml
    assert "wscript.exe" not in xml
    assert "cmd.exe" not in xml


def test_gateway_ps1_script_is_console_less(monkeypatch):
    """The .ps1 launcher must hide the window and detach, matching the .vbs contract."""
    monkeypatch.setattr(
        gateway_windows,
        "_resolve_detached_python",
        lambda exe: (r"C:\venv\Scripts\python.exe", Path(r"C:\venv"), []),
    )
    content = gateway_windows._build_gateway_ps1_script(
        r"C:\venv\Scripts\python.exe",
        r"C:\Hermes",
        r"C:\Hermes",
        "--profile work",
    )
    assert "cmd.exe" not in content.lower()
    assert "Start-Process" in content
    assert "-WindowStyle Hidden" in content           # window style 0 equivalent
    assert "python.exe" in content
    assert "'hermes_cli.main'" in content
    assert "'gateway', 'run'" in content
    for var in ("HERMES_HOME", "PYTHONIOENCODING", "HERMES_GATEWAY_DETACHED", "VIRTUAL_ENV", "PYTHONPATH"):
        assert var in content
    assert "--profile" in content and "work" in content
    assert content.endswith("\r\n")


def test_write_task_script_switches_kind_and_removes_the_stale_one(monkeypatch, tmp_path):
    """Switching launcher kind must delete the other file so no dead launcher is left behind."""
    monkeypatch.setattr(gateway_windows, "_is_windows", lambda: True, raising=False)
    monkeypatch.setattr(gateway_windows, "_prefer_powershell_launcher", lambda: True)
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: tmp_path / "Hermes_Gateway.cmd")
    monkeypatch.setattr(
        gateway_windows, "_launcher_settings",
        lambda: (r"C:\venv\Scripts\python.exe", r"C:\Hermes", r"C:\Hermes", ""),
    )
    monkeypatch.setattr(
        gateway_windows, "_resolve_detached_python",
        lambda exe: (r"C:\venv\Scripts\python.exe", Path(r"C:\venv"), []),
    )
    monkeypatch.setattr(gateway_windows, "_powershell_exe", lambda: r"C:\Windows\powershell.exe")
    stale_vbs = tmp_path / "Hermes_Gateway.vbs"
    stale_vbs.write_text("' stale")

    gateway_windows._write_task_script()

    assert (tmp_path / "Hermes_Gateway.ps1").is_file()
    assert not stale_vbs.exists()


def test_repair_login_persistence_repoints_a_registered_task(monkeypatch):
    """A registered task keeps the action it was CREATED with, so refreshing the launcher file
    alone never retargets it — the repair must recreate the task."""
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: Path(r"C:\Hermes\x.cmd"))
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    calls = []
    monkeypatch.setattr(
        gateway_windows, "_install_scheduled_task",
        lambda name, script: (calls.append(name), (True, "Created"))[1],
    )

    assert gateway_windows.repair_login_persistence() is True
    assert calls == ["Hermes_Gateway"]


def test_repair_login_persistence_falls_back_to_startup_without_elevation(monkeypatch):
    """Recreating a task needs elevation; an access-denied refusal must still leave login
    persistence working via the (user-level) Startup entry."""
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: Path(r"C:\Hermes\x.cmd"))
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    monkeypatch.setattr(
        gateway_windows, "_install_scheduled_task",
        lambda name, script: (False, "schtasks /Delete failed (code 1): ERROR: Access is denied."),
    )
    installed = []
    monkeypatch.setattr(gateway_windows, "_install_startup_entry", lambda script: installed.append(script))

    assert gateway_windows.repair_login_persistence() is True
    assert installed, "expected the Startup-folder fallback to be installed"


def test_repair_login_persistence_is_a_noop_when_not_installed(monkeypatch):
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: False)
    called = []
    monkeypatch.setattr(gateway_windows, "_install_scheduled_task", lambda *a: called.append(a))

    assert gateway_windows.repair_login_persistence() is False
    assert called == []














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








