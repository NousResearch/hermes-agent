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


# ---------------------------------------------------------------------------
# Dual autostart persistence convergence — issue #80569
# ---------------------------------------------------------------------------


def _arrange_startup_files(monkeypatch, tmp_path):
    """Create a Startup folder with both the .vbs fallback and legacy .cmd."""
    vbs = tmp_path / "Startup" / "Hermes_Gateway_alice.vbs"
    cmd = tmp_path / "Startup" / "Hermes_Gateway_alice.cmd"
    vbs.parent.mkdir(parents=True, exist_ok=True)
    vbs.write_text("' fake vbs fallback", encoding="utf-8")
    cmd.write_text("@echo off", encoding="utf-8")
    monkeypatch.setattr(gateway_windows, "get_startup_entry_path", lambda: vbs)
    monkeypatch.setattr(gateway_windows, "_legacy_startup_entry_path", lambda: cmd)
    return vbs, cmd


def test_reconcile_removes_both_entries_when_task_registered(monkeypatch, tmp_path):
    """Task + Startup entries coexisting -> both Startup entries are removed."""
    vbs, cmd = _arrange_startup_files(monkeypatch, tmp_path)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)

    actions = gateway_windows.reconcile_autostart_launchers()

    assert not vbs.exists()
    assert not cmd.exists()
    assert len(actions) == 2
    assert all("Removed redundant Windows login item" in a for a in actions)


def test_reconcile_migrates_legacy_cmd_to_vbs_when_no_task(monkeypatch, tmp_path):
    """No task + legacy .cmd -> .vbs fallback installed, legacy .cmd removed."""
    cmd = tmp_path / "Startup" / "Hermes_Gateway_alice.cmd"
    vbs = tmp_path / "Startup" / "Hermes_Gateway_alice.vbs"
    cmd.parent.mkdir(parents=True, exist_ok=True)
    cmd.write_text("@echo off", encoding="utf-8")
    script_path = tmp_path / "gateway-service" / "Hermes_Gateway_alice.cmd"
    monkeypatch.setattr(gateway_windows, "get_startup_entry_path", lambda: vbs)
    monkeypatch.setattr(gateway_windows, "_legacy_startup_entry_path", lambda: cmd)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script_path)

    actions = gateway_windows.reconcile_autostart_launchers()

    assert not cmd.exists()
    assert vbs.exists()
    assert len(actions) == 1
    assert "Migrated legacy Windows login item" in actions[0]


def test_reconcile_noop_when_converged(monkeypatch, tmp_path):
    """Task-only or .vbs-fallback-only installs are already converged."""
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(
        gateway_windows, "get_startup_entry_path", lambda: tmp_path / "missing.vbs"
    )
    monkeypatch.setattr(
        gateway_windows, "_legacy_startup_entry_path", lambda: tmp_path / "missing.cmd"
    )
    assert gateway_windows.reconcile_autostart_launchers() == []

    vbs = tmp_path / "Startup" / "Hermes_Gateway_alice.vbs"
    vbs.parent.mkdir(parents=True, exist_ok=True)
    vbs.write_text("' fake vbs fallback", encoding="utf-8")
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
    monkeypatch.setattr(gateway_windows, "get_startup_entry_path", lambda: vbs)
    assert gateway_windows.reconcile_autostart_launchers() == []


def test_install_success_removes_redundant_startup_entries(monkeypatch, tmp_path):
    """install() with a successful Scheduled Task converges to task-only."""
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"
    removed_calls = []

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(
        gateway_windows,
        "_prompt_install_choices",
        lambda *args, **kwargs: (False, True),
    )
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice")
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script_path)
    monkeypatch.setattr(gateway_windows, "_is_running_as_admin", lambda: True)
    monkeypatch.setattr(
        gateway_windows,
        "_install_scheduled_task",
        lambda task_name, script_path: (True, "Created Scheduled Task 'Hermes_Gateway_alice'"),
    )

    def fake_remove_startup_entries():
        removed_calls.append(True)
        return ([tmp_path / "Startup" / "Hermes_Gateway_alice.vbs"], [])

    monkeypatch.setattr(gateway_windows, "_remove_startup_entries", fake_remove_startup_entries)
    monkeypatch.setattr(gateway_windows, "_print_next_steps", lambda: None)

    gateway_windows.install()

    assert removed_calls == [True]


def test_reconcile_reports_unlink_failure(monkeypatch, tmp_path):
    """A locked Startup entry must be reported, not silently claimed removed."""
    vbs, cmd = _arrange_startup_files(monkeypatch, tmp_path)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)

    real_unlink = Path.unlink

    def flaky_unlink(self):
        if self.name.endswith(".vbs"):
            raise PermissionError("file is locked")
        return real_unlink(self)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    actions = gateway_windows.reconcile_autostart_launchers()

    assert vbs.exists()
    assert not cmd.exists()
    assert any("Removed redundant Windows login item" in a for a in actions)
    assert any("Could not remove redundant Windows login item" in a for a in actions)


def test_reconcile_warns_when_schtasks_wedges(monkeypatch, tmp_path):
    """A schtasks query failure must warn, not crash or remove anything."""
    vbs, cmd = _arrange_startup_files(monkeypatch, tmp_path)

    def raise_wedge():
        raise RuntimeError("schtasks timed out")

    monkeypatch.setattr(gateway_windows, "is_task_registered", raise_wedge)

    actions = gateway_windows.reconcile_autostart_launchers()

    assert len(actions) == 1
    assert "Could not verify Scheduled Task state" in actions[0]
    assert vbs.exists()
    assert cmd.exists()


def test_install_fallback_skips_entry_when_task_still_registered(monkeypatch, tmp_path):
    """Startup fallback must not be installed while the Scheduled Task exists."""
    script_path = tmp_path / "Hermes_Gateway_alice.cmd"
    calls = []
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(
        gateway_windows, "_install_startup_entry", lambda p: calls.append(("entry", p)) or p
    )
    monkeypatch.setattr(
        gateway_windows, "_spawn_detached", lambda: calls.append(("spawn", None)) or 12345
    )
    monkeypatch.setattr(gateway_windows, "_report_gateway_start", lambda via: calls.append(("report", via)))
    monkeypatch.setattr(gateway_windows, "_print_next_steps", lambda: calls.append(("next", None)))
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway, "_profile_arg", lambda: "--profile alice")

    gateway_windows._install_startup_fallback(script_path, start_now=True, detail="blocked")

    assert not any(c[0] == "entry" for c in calls)
    assert any(c[0] == "spawn" for c in calls)


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
# stop() drain semantics — issue #33778
#
# Background: on Windows, asyncio.add_signal_handler raises NotImplementedError,
# so the gateway's SIGTERM handler (which drains in-flight agents and writes
# resume_pending=True) never fires when `hermes gateway stop` kills the
# process. The fix: stop() writes the planned_stop_marker first, waits for
# the gateway's marker-watcher thread to drain + exit cleanly, then escalates
# to taskkill if drain times out.
# ---------------------------------------------------------------------------










