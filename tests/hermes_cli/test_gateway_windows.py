"""Tests for hermes_cli.gateway_windows."""

import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway
import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.setup as setup


_BREAKAWAY_MARKER = "_HERMES_GATEWAY_BREAKAWAY"


def test_exec_schtasks_decodes_ansi_output_under_utf8_mode(monkeypatch):
    """schtasks emits the ANSI code page even when Python runs in UTF-8 mode; a non-ASCII account
    path in the task XML must survive `_exec_schtasks` intact or `scheduled_task_drift` reports
    "launcher arguments differs" forever (#116193). Drives the production seam with the bytes the
    reporter captured (GBK for 方舟)."""
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows.shutil, "which", lambda name: "schtasks.exe")
    monkeypatch.setattr(gateway_windows.locale, "getpreferredencoding", lambda *a, **k: "utf-8")
    monkeypatch.setattr(gateway_windows, "_windows_console_encodings", lambda: ["cp936"], raising=False)
    xml = r'<Arguments>//B //Nologo "C:\Users\方舟\AppData\Local\hermes\gateway-service\Hermes_Gateway.vbs"</Arguments>'

    def fake_run(argv, **kwargs):
        # schtasks writes cp936 bytes; honour text/encoding like the real subprocess would.
        out, err = xml.encode("gbk"), "错误: 拒绝访问。".encode("gbk")
        if kwargs.get("text"):
            enc, errors = kwargs.get("encoding") or "utf-8", kwargs.get("errors") or "strict"
            out, err = out.decode(enc, errors), err.decode(enc, errors)
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr=err)

    monkeypatch.setattr(gateway_windows.subprocess, "run", fake_run)

    code, out, err = gateway_windows._exec_schtasks(["/Query", "/TN", "Hermes_Gateway", "/XML"])

    assert (code, out) == (0, xml)
    assert gateway_windows._is_access_denied(err) and gateway_windows._should_fall_back(1, err)


@pytest.mark.windows_only
def test_exec_schtasks_round_trips_non_ascii_task_argument_live(monkeypatch):
    """Real schtasks.exe on a real task whose argument carries a non-ASCII (ANSI-representable)
    character, queried from a UTF-8-mode interpreter: the template/live comparison in
    `scheduled_task_drift` needs the exact characters back (#116193)."""
    monkeypatch.setattr(gateway_windows.locale, "getpreferredencoding", lambda *a, **k: "utf-8")
    task = f"Hermes_Test_{os.getpid()}"
    marker = "Zo\u00eb"  # ë: one byte in every Western OEM/ANSI code page, invalid as a lone UTF-8 byte
    created = subprocess.run(
        ["schtasks", "/Create", "/F", "/TN", task, "/SC", "ONLOGON", "/TR", f'wscript.exe //B "C:\\{marker}\\x.vbs"'],
        capture_output=True, timeout=30,
    )
    assert created.returncode == 0, created.stderr
    try:
        code, out, _err = gateway_windows._exec_schtasks(["/Query", "/TN", task, "/XML"])
    finally:
        subprocess.run(["schtasks", "/Delete", "/F", "/TN", task], capture_output=True, timeout=30)
    assert code == 0
    assert marker in out, out


def test_localized_access_denied_uses_existing_fallback_paths():
    """Localized schtasks denial still reaches elevation/startup fallback handling."""
    detail = "错误: 拒绝访问。"

    assert gateway_windows._should_fall_back(1, detail)
    assert gateway_windows._is_access_denied(detail)


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
        lambda home=None: (argv, cwd, {"HERMES_GATEWAY_DETACHED": "1"}),
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
        lambda home=None: (
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


def test_atomic_write_leaves_no_staging_file_when_swap_fails(monkeypatch, tmp_path):
    """The Startup folder is the staging dir: a leftover .tmp there is opened by Windows at every login."""
    startup = tmp_path / "Startup"
    startup.mkdir()
    entry, staging = startup / "Hermes_Gateway.vbs", startup / "Hermes_Gateway.tmp"

    def _denied(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", _denied)
    with pytest.raises(PermissionError):
        gateway_windows._atomic_write(entry, "launcher\r\n", staging)

    assert sorted(p.name for p in startup.iterdir()) == []


def test_uninstall_and_reinstall_sweep_stale_startup_staging_file(monkeypatch, tmp_path):
    """Debris from a pre-fix failed swap is removed by uninstall() and by an install() that takes the
    Scheduled Task path (which never rewrites the Startup folder itself)."""
    startup = tmp_path / "Startup"
    startup.mkdir()
    entry, staging = startup / "Hermes_Gateway_alice.vbs", startup / "Hermes_Gateway_alice.tmp"
    script = tmp_path / "task" / "Hermes_Gateway_alice.cmd"

    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_alice")
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: script)
    monkeypatch.setattr(gateway_windows, "get_startup_entry_path", lambda: entry)
    monkeypatch.setattr(gateway_windows, "_legacy_startup_entry_path", lambda: startup / "Hermes_Gateway_alice.cmd")
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)

    staging.write_text("stale", encoding="utf-8")
    gateway_windows.uninstall()
    assert not staging.exists()

    staging.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(gateway_windows, "_prompt_install_choices", lambda *a, **k: (False, True))
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script)
    monkeypatch.setattr(gateway_windows, "_is_running_as_admin", lambda: True)
    monkeypatch.setattr(gateway_windows, "_install_scheduled_task", lambda name, path: (True, "created"))
    monkeypatch.setattr(gateway_windows, "_print_next_steps", lambda: None)
    gateway_windows.install()
    assert not staging.exists()


# Reporter's `Export-ScheduledTask` of a task registered before the hardened template (#113670).
_PRE_HARDENING_TASK_XML = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.3" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Settings>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
  </Settings>
  <Triggers>
    <LogonTrigger />
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>wscript.exe</Command>
      <Arguments>"C:\\Users\\me\\.hermes\\gateway-service\\Hermes_Gateway.vbs"</Arguments>
    </Exec>
  </Actions>
</Task>
"""


def test_scheduled_task_drift_names_missing_hardening_leaves(monkeypatch):
    """A pre-hardening registration is reported leaf by leaf, and the report is what
    ``hermes gateway status`` prints together with the ``hermes gateway install`` repair hint."""
    launcher = Path(r"C:\Users\me\.hermes\gateway-service\Hermes_Gateway.vbs")
    template = gateway_windows._build_scheduled_task_xml("Hermes_Gateway", launcher, r"PC\me")
    drift = gateway_windows.compare_scheduled_task_drift(_PRE_HARDENING_TASK_XML, template)
    assert drift == [
        "missing: RestartOnFailure, LogonTrigger Delay",
        "launcher arguments differs",
        "version 1.3 vs 1.4",
    ]

    printed: list[str] = []
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda args: (0, _PRE_HARDENING_TASK_XML if "/XML" in args else "", ""))
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: launcher.with_suffix(".cmd"))
    monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: r"PC\me")
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    gateway_windows._print_scheduled_task_drift("Hermes_Gateway")
    assert printed[0].startswith("⚠ Scheduled Task registration predates the current template (missing: RestartOnFailure")
    assert "hermes gateway install" in printed[1]


def test_scheduled_task_drift_is_silent_when_aligned_or_unqueryable(monkeypatch):
    """The template compared to itself (with the SID-style <UserId> schtasks exports) is not drift, and
    a failed ``schtasks /Query /XML`` prints nothing — status must never nag a healthy install."""
    launcher = Path(r"C:\Users\me\.hermes\gateway-service\Hermes_Gateway.vbs")
    template = gateway_windows._build_scheduled_task_xml("Hermes_Gateway", launcher, r"PC\me")
    exported = template.replace(r"<UserId>PC\me</UserId>", "<UserId>S-1-5-21-1-2-3-1001</UserId>")
    assert exported != template
    assert gateway_windows.compare_scheduled_task_drift(exported, template) == []
    assert gateway_windows.compare_scheduled_task_drift("not xml", template) == []

    printed: list[str] = []
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", lambda args: (1, "", "ERROR: The system cannot find the file specified."))
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    gateway_windows._print_scheduled_task_drift("Hermes_Gateway")
    assert printed == []


def test_reconcile_scheduled_task_reregisters_only_on_drift(monkeypatch, tmp_path):
    """The Windows sibling of ``refresh_systemd_unit_if_needed`` (#113670): a pre-hardening
    registration is deleted and re-created from the current template (so ``RestartOnFailure`` and the
    logon ``Delay`` reach existing installs), while an aligned one is left alone."""
    script_path = tmp_path / "gateway.cmd"
    launcher = script_path.with_suffix(".vbs")
    template = gateway_windows._build_scheduled_task_xml("Hermes_Gateway", launcher, r"PC\me")
    calls: list[list[str]] = []
    registered = {"xml": _PRE_HARDENING_TASK_XML}

    def fake_schtasks(args):
        calls.append(list(args))
        if "/XML" in args and "/Query" in args:
            return (0, registered["xml"], "")
        if "/Create" in args:
            registered["xml"] = Path(args[args.index("/XML") + 1]).read_text(encoding="utf-16")
        return (0, "", "")

    monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script_path)
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: script_path)
    monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: r"PC\me")
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    assert gateway_windows.reconcile_scheduled_task("Hermes_Gateway") is True
    assert [c[0] for c in calls if c[0] in ("/Delete", "/Create")] == ["/Delete", "/Create"]
    assert "<RestartOnFailure>" in registered["xml"]
    assert gateway_windows.compare_scheduled_task_drift(registered["xml"], template) == []

    calls.clear()
    assert gateway_windows.reconcile_scheduled_task("Hermes_Gateway") is False
    assert not any(c[0] in ("/Delete", "/Create") for c in calls)


def _arrange_uninstalled_start(monkeypatch):
    """start() with no Scheduled Task / Startup entry; returns (install_calls, spawn_count)."""
    installs, spawns = [], []
    monkeypatch.delenv("HERMES_GATEWAY_INSTALL_START_ON_LOGIN", raising=False)
    monkeypatch.delenv("HERMES_NONINTERACTIVE", raising=False)
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "_print_start_attestation_warning", lambda: None)
    monkeypatch.setattr(gateway_windows, "_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
    monkeypatch.setattr(gateway_windows, "is_startup_entry_installed", lambda: False)
    monkeypatch.setattr(gateway_windows, "install", lambda **kwargs: installs.append(kwargs))
    monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda: spawns.append(1) or 4242)
    monkeypatch.setattr(gateway_windows, "_report_gateway_start", lambda via: None)
    monkeypatch.setattr(gateway_windows, "_stdin_console_mode_ok", lambda: True)
    return installs, spawns


def test_stdin_interactive_only_when_isatty_and_a_console_answers_get_console_mode():
    """Windows CRT isatty() is True for the NUL device (`hermes gateway start < NUL`, stdin=DEVNULL), so
    isatty alone must not open the prompt; off Windows (no console-mode fact) isatty decides (#113977)."""
    assert gateway_windows._stdin_is_interactive(isatty=True, console_mode_ok=False) is False   # NUL
    assert gateway_windows._stdin_is_interactive(isatty=True, console_mode_ok=True) is True     # console
    assert gateway_windows._stdin_is_interactive(isatty=False, console_mode_ok=True) is False   # pipe
    assert gateway_windows._stdin_is_interactive(isatty=True, console_mode_ok=None) is True     # POSIX tty


def test_start_with_nul_stdin_starts_the_gateway_but_never_installs_login_persistence(monkeypatch, capsys):
    """isatty says TTY, GetConsoleMode says no console: `< NUL` gets the same treatment as a pipe."""
    installs, spawns = _arrange_uninstalled_start(monkeypatch)
    monkeypatch.setattr(setup, "is_interactive_stdin", lambda: True)
    monkeypatch.setattr(gateway_windows, "_stdin_console_mode_ok", lambda: False)
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *a, **k: pytest.fail("no prompt on a NUL stdin"))

    gateway_windows.start()

    assert installs == [] and spawns == [1]
    assert "hermes gateway install" in capsys.readouterr().out


def test_start_without_tty_starts_the_gateway_but_never_installs_login_persistence(monkeypatch, capsys):
    """`hermes gateway start < /dev/null` must not answer the persistence question with a default Yes
    (#113977); it starts the gateway once and points at the explicit install command."""
    installs, spawns = _arrange_uninstalled_start(monkeypatch)
    monkeypatch.setattr(setup, "is_interactive_stdin", lambda: False)
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *a, **k: pytest.fail("no prompt without a TTY"))

    gateway_windows.start()

    assert installs == [] and spawns == [1]
    out = capsys.readouterr().out
    assert "hermes gateway install" in out and "did not complete" not in out


def test_start_on_tty_hands_both_answers_to_install_and_honours_the_env_opt_out(monkeypatch):
    """Yes → one install() carrying start_now+start_on_login (install spawns; start() must not spawn
    again). HERMES_GATEWAY_INSTALL_START_ON_LOGIN=0 → no question, no install, a plain start."""
    installs, spawns = _arrange_uninstalled_start(monkeypatch)
    monkeypatch.setattr(setup, "is_interactive_stdin", lambda: True)
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *a, **k: True)

    gateway_windows.start()
    assert installs == [{"force": False, "start_now": True, "start_on_login": True}] and spawns == []

    installs.clear()
    monkeypatch.setenv("HERMES_GATEWAY_INSTALL_START_ON_LOGIN", "0")
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *a, **k: pytest.fail("env override must skip the prompt"))
    gateway_windows.start()
    assert installs == [] and spawns == [1]














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










def test_hermes_owns_windows_service_requires_name_or_binary_under_a_hermes_root():
    """Task Scheduler (``Schedule`` in svchost) above a task-launched gateway is never its supervisor;
    a service is Hermes-owned only by a ``hermes*`` name or a binary under the install (#97208)."""
    roots = (
        r"C:\Users\kaize\AppData\Local\hermes\hermes-agent",
        r"C:\Users\kaize\AppData\Local\hermes\hermes-agent\venv\Scripts",
        r"C:\Users\kaize\AppData\Local\hermes\gateway-service",
    )
    owns = gateway_windows.hermes_owns_windows_service

    assert not owns("Schedule", r"C:\Windows\system32\svchost.exe -k netsvcs -p -s Schedule", roots)
    assert not owns("BITS", r"C:\Windows\System32\svchost.exe -k netsvcs -p -s BITS", roots)
    assert not owns("Other", r"C:\Users\kaize\AppData\Local\hermes\hermes-agent-fork\run.exe", roots)

    assert owns("HermesGateway", r"C:\nssm\nssm.exe", roots)
    assert owns("Hermes_Gateway_derek", "", roots)
    assert owns("gw", r'"C:\Users\KAIZE\AppData\Local\hermes\hermes-agent\venv\Scripts\hermes.exe" gateway run', roots)
    assert owns("gw", r"C:\Users\kaize\AppData\Local\hermes\gateway-service\Hermes_Gateway.cmd", roots)


def test_wizard_install_service_asks_once_and_never_starts_after_windows_install(monkeypatch):
    """The wizard asks start-now/start-on-login once, forwards both answers, and returns
    without a second start: the Windows installer owns start and the elevated child
    starts itself, so a parent-side start would re-ask the install questions and re-offer
    UAC while the child is still waiting on consent (#116550)."""
    answers = iter([True, True])
    monkeypatch.setattr(gateway, "prompt_yes_no", lambda *a, **k: next(answers))
    monkeypatch.setattr(gateway, "is_wsl", lambda: False)
    installs, starts = [], []
    monkeypatch.setattr(gateway, "_gw_windows", lambda: SimpleNamespace(
        install=lambda **kw: installs.append(kw) or True,
    ))
    monkeypatch.setattr(gateway, "_setup_service_action", lambda *a, **k: starts.append((a, k)))

    gateway._wizard_install_service("windows")

    assert installs == [{"force": False, "start_now": True, "start_on_login": True}]
    assert starts == []
