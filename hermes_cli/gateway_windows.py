"""Windows gateway service backend (Scheduled Task + Startup-folder fallback).

This mirrors the contract exposed by ``launchd_install`` / ``launchd_start`` /
``launchd_status`` etc. on macOS and ``systemd_install`` / ``systemd_start`` on
Linux. It uses ``schtasks`` under the hood with ``/SC ONLOGON`` and restart-on-
failure XML settings, and falls back to a ``%APPDATA%\\...\\Startup\\<name>.vbs``
dropper when Scheduled Task creation is denied (locked-down corporate boxes).

Design notes
------------
* ``schtasks /Create /SC ONLOGON /RL LIMITED`` means the task runs at the
  CURRENT USER's next logon without any elevation prompt. Manual starts and
  install ``--start-now`` use the direct hidden-console launcher instead
  of ``schtasks /Run`` so start/restart behavior is consistent.
* We write a shared ``gateway.cmd`` wrapper plus a console-less ``gateway.vbs``
  launcher. Scheduled Task and Startup-folder persistence both route through
  VBS/wscript; immediate manual starts route through direct ``subprocess`` spawn.
* Status = merge of "is the schtasks entry registered?" + "is the startup
  login item present?" + "is there a gateway process running?" so the status
  command keeps working regardless of which install path was taken.
* Quoting is tricky: schtasks parses ``/TR`` itself and cmd.exe parses the
  generated ``gateway.cmd``. Those are DIFFERENT parsers. We keep two
  separate quote helpers (same pattern OpenClaw uses) and never cross them.
* All of this is Windows-only. ``import`` paths are still safe on POSIX but
  the functions raise if called on non-Windows.
"""

from __future__ import annotations

import ctypes
import locale
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape

from hermes_cli._subprocess_compat import (
    windows_detach_flags,
    windows_detach_flags_without_breakaway,
    windows_hide_flags,
)

# Short timeouts: schtasks occasionally wedges and we don't want to hang forever.
_SCHTASKS_TIMEOUT_S = 15
_SCHTASKS_NO_OUTPUT_TIMEOUT_S = 30
# Patterns in schtasks stderr that mean "fall back to the Startup folder".
_FALLBACK_PATTERNS = re.compile(
    r"(access is denied|acceso denegado|přístup byl odepřen|schtasks timed out|schtasks produced no output)",
    re.IGNORECASE,
)
_ACCESS_DENIED_PATTERN = re.compile(r"(access is denied|acceso denegado)", re.IGNORECASE)

_TASK_NAME_DEFAULT = "Hermes_Gateway"
_TASK_DESCRIPTION = "Hermes Agent Gateway - Messaging Platform Integration"
_TASK_LOGON_DELAY = "PT30S"
_TASK_RESTART_INTERVAL = "PT1M"
_TASK_RESTART_COUNT = 999


def _schtasks_encoding() -> str:
    """Best-effort console encoding for decoding ``schtasks.exe`` output.

    On localized Windows (e.g. Chinese), ``schtasks`` emits text in the OEM/ANSI
    code page rather than UTF-8. Decoding with the wrong codec raised
    ``UnicodeDecodeError`` inside ``subprocess``' reader threads. Prefer the
    locale's preferred encoding and fall back to UTF-8.
    """
    try:
        return locale.getpreferredencoding(False) or "utf-8"
    except Exception:
        return "utf-8"


# ---------------------------------------------------------------------------
# Platform guard
# ---------------------------------------------------------------------------

def _assert_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError("gateway_windows is Windows-only")


def _preserve_hermes_home_path(path: str | Path) -> str:
    """Render Hermes-owned paths under the configured HERMES_HOME spelling.

    Windows installs may keep ``%LOCALAPPDATA%\\hermes`` as a symlink/junction to
    another drive. Runtime state should still identify itself by the configured
    AppData path, so launcher files must not bake in the resolved target when a
    path lives under HERMES_HOME.
    """
    candidate = Path(path)
    try:
        from hermes_cli.config import get_hermes_home

        home = Path(get_hermes_home())
        resolved_home = home.resolve()
        resolved_candidate = candidate.resolve()
        home_key = os.path.normcase(str(resolved_home))
        candidate_key = os.path.normcase(str(resolved_candidate))
        if os.path.commonpath([home_key, candidate_key]) == home_key:
            rel = os.path.relpath(str(resolved_candidate), str(resolved_home))
            return str(home / rel)
    except Exception:
        pass
    return str(candidate)


# ---------------------------------------------------------------------------
# Quoting helpers (two DIFFERENT parsers — do not mix)
# ---------------------------------------------------------------------------

def _quote_cmd_script_arg(value: str) -> str:
    """Quote a single argument for use INSIDE a .cmd file, for cmd.exe parsing.

    cmd.exe splits on spaces/tabs outside of double quotes. Embedded quotes
    are doubled. We also refuse line breaks because they'd terminate the
    logical command line mid-script.
    """
    if "\r" in value or "\n" in value:
        raise ValueError(f"refusing to quote value containing newline: {value!r}")
    if not value:
        return '""'
    if not re.search(r'[ \t"]', value):
        return value
    return '"' + value.replace('"', '""') + '"'


def _quote_schtasks_arg(value: str) -> str:
    """Quote a single argument for schtasks.exe's /TR parser.

    Schtasks uses a different quoting convention than cmd.exe: embedded
    quotes are backslash-escaped, and the whole thing is wrapped in double
    quotes if it contains whitespace or quotes.
    """
    if not re.search(r'[ \t"]', value):
        return value
    return '"' + value.replace('"', '\\"') + '"'


# ---------------------------------------------------------------------------
# schtasks.exe wrapper
# ---------------------------------------------------------------------------

def _exec_schtasks(args: list[str]) -> tuple[int, str, str]:
    """Run ``schtasks.exe`` with a hard timeout. Return (code, stdout, stderr).

    If schtasks wedges, returns code=124 with a synthetic stderr string —
    same convention OpenClaw uses, so the fallback detection regex matches.
    """
    _assert_windows()
    schtasks = shutil.which("schtasks")
    if schtasks is None:
        return (1, "", "schtasks.exe not found on PATH")
    try:
        proc = subprocess.run(
            [schtasks, *args],
            capture_output=True,
            text=True,
            # Localized Windows emits schtasks output in the console code page,
            # not UTF-8. Decode with the locale encoding and replace undecodable
            # bytes so a non-UTF-8 status line never surfaces a UnicodeDecodeError
            # traceback from subprocess' reader threads (issue #38172).
            encoding=_schtasks_encoding(),
            errors="replace",
            timeout=_SCHTASKS_TIMEOUT_S,
            # CREATE_NO_WINDOW avoids a flashing console window when the CLI
            # is itself hosted in a TUI. See tools/browser_tool.py for the
            # same pattern and the windows-subprocess-sigint-storm.md ref.
            creationflags=windows_hide_flags(),
        )
        return (proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired:
        return (124, "", f"schtasks timed out after {_SCHTASKS_TIMEOUT_S}s")
    except OSError as e:
        return (1, "", f"schtasks invocation failed: {e}")


def _should_fall_back(code: int, detail: str) -> bool:
    return code == 124 or bool(_FALLBACK_PATTERNS.search(detail or ""))


def _is_access_denied(detail: str) -> bool:
    return bool(_ACCESS_DENIED_PATTERN.search(detail or ""))


def _is_running_as_admin() -> bool:
    """Return True when the current Windows process is elevated."""
    _assert_windows()
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _current_profile_cli_args() -> list[str]:
    """Return CLI args that preserve the current Hermes profile."""
    from hermes_cli.gateway import _profile_arg

    profile_arg = _profile_arg()
    return shlex.split(profile_arg) if profile_arg else []


def _launch_elevated_gateway_command(command: str, extra_args: list[str] | None = None) -> bool:
    """Launch an elevated gateway subcommand via UAC and return True on handoff.

    The elevated child is the console ``python.exe`` launched with
    ``SW_HIDE``: ShellExecuteW applies the show-command to a console app's
    console window, so the child owns a single *hidden* console that its own
    subprocess spawns (schtasks, taskkill, …) inherit — no visible window
    after the UAC approval, and no per-descendant conhost flashes (the
    console-less pythonw.exe alternative re-created #54220/#56747 for every
    console-subsystem child). All operator decisions are already collected in
    the parent shell before this point.
    """
    _assert_windows()
    args = ["-m", "hermes_cli.main", *_current_profile_cli_args(), "gateway", command]
    if extra_args:
        args.extend(extra_args)
    params = subprocess.list2cmdline(args)
    cwd = str(Path(__file__).resolve().parent.parent)
    elevated_python = sys.executable
    try:
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            elevated_python,
            params,
            cwd,
            0,  # SW_HIDE: the child's console exists but is never shown.
        )
    except Exception as exc:
        print(f"⚠ Could not launch elevated gateway {command} prompt: {exc}")
        return False
    if result <= 32:
        print(f"⚠ Elevated gateway {command} prompt was not started (ShellExecuteW={result})")
        return False
    return True


def _launch_elevated_install(
    force: bool = False,
    *,
    start_now: bool | None = None,
    start_on_login: bool | None = None,
) -> bool:
    """Launch an elevated gateway install via UAC and return True on handoff."""
    old_start_now = os.environ.get("HERMES_GATEWAY_INSTALL_START_NOW")
    old_start_on_login = os.environ.get("HERMES_GATEWAY_INSTALL_START_ON_LOGIN")
    old_handoff = os.environ.get("HERMES_GATEWAY_ELEVATED_HANDOFF")
    try:
        if start_now is not None:
            os.environ["HERMES_GATEWAY_INSTALL_START_NOW"] = "1" if start_now else "0"
        if start_on_login is not None:
            os.environ["HERMES_GATEWAY_INSTALL_START_ON_LOGIN"] = "1" if start_on_login else "0"
        os.environ["HERMES_GATEWAY_ELEVATED_HANDOFF"] = "1"
        extra_args = ["--elevated-handoff"]
        if force:
            extra_args.append("--force")
        if start_now is not None:
            extra_args.append("--start-now" if start_now else "--no-start-now")
        if start_on_login is not None:
            extra_args.append("--start-on-login" if start_on_login else "--no-start-on-login")
        return _launch_elevated_gateway_command("install", extra_args)
    finally:
        for key, old in (
            ("HERMES_GATEWAY_INSTALL_START_NOW", old_start_now),
            ("HERMES_GATEWAY_INSTALL_START_ON_LOGIN", old_start_on_login),
            ("HERMES_GATEWAY_ELEVATED_HANDOFF", old_handoff),
        ):
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _launch_elevated_uninstall() -> bool:
    """Launch an elevated gateway uninstall via UAC and return True on handoff."""
    return _launch_elevated_gateway_command("uninstall")


# ---------------------------------------------------------------------------
# Paths: where we stash our task script and where Startup lives
# ---------------------------------------------------------------------------

def get_task_name() -> str:
    """Scheduled Task name, scoped per profile.

    Default profile: ``Hermes_Gateway``
    Named profile X: ``Hermes_Gateway_<X>``
    """
    _assert_windows()
    # Local import to avoid circular module initialization during hermes_cli boot.
    from hermes_cli.gateway import _profile_suffix

    suffix = _profile_suffix()
    if not suffix:
        return _TASK_NAME_DEFAULT
    return f"{_TASK_NAME_DEFAULT}_{suffix}"


def _sanitize_filename(value: str) -> str:
    """Remove characters illegal in Windows filenames."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)


def get_task_script_path() -> Path:
    """The generated ``gateway.cmd`` wrapper kept beside the VBS launcher.

    Lives under ``%LOCALAPPDATA%\\hermes\\gateway-service\\<task_name>.cmd``
    (or ``<HERMES_HOME>/gateway-service/<task_name>.cmd`` so per-profile
    Hermes installs stay self-contained).
    """
    _assert_windows()
    from hermes_cli.config import get_hermes_home

    script_dir = Path(get_hermes_home()) / "gateway-service"
    script_dir.mkdir(parents=True, exist_ok=True)
    return script_dir / f"{_sanitize_filename(get_task_name())}.cmd"


def _startup_dir() -> Path:
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    userprofile = os.environ.get("USERPROFILE", "").strip() or os.environ.get("HOME", "").strip()
    if not userprofile:
        raise RuntimeError("neither APPDATA nor USERPROFILE is set — cannot resolve Startup folder")
    return (
        Path(userprofile)
        / "AppData"
        / "Roaming"
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )


def get_startup_entry_path() -> Path:
    _assert_windows()
    return _startup_dir() / f"{_sanitize_filename(get_task_name())}.vbs"


def _legacy_startup_entry_path() -> Path:
    _assert_windows()
    return _startup_dir() / f"{_sanitize_filename(get_task_name())}.cmd"


# ---------------------------------------------------------------------------
# Stable working directory
# ---------------------------------------------------------------------------

def _stable_gateway_working_dir(project_root: Path) -> str:
    """Return a stable cwd for detached/startup gateway runs.

    Mirror the POSIX service invariant: anchor at ``HERMES_HOME`` whenever it
    exists so Scheduled Task / Startup launches do not fail at the ``cd`` step
    after a transient checkout or worktree is moved away. Fall back to the
    source checkout only if ``HERMES_HOME`` cannot be used yet. Preserve the
    configured spelling instead of resolving symlinks so AppData installs backed
    by a junction/symlink still identify themselves as AppData.
    """
    from hermes_cli.config import get_hermes_home

    try:
        home = get_hermes_home()
        if home:
            home_path = Path(home)
            if home_path.is_dir():
                return str(home_path)
    except Exception:
        pass
    return str(project_root)


# ---------------------------------------------------------------------------
# Script rendering
# ---------------------------------------------------------------------------

def _build_gateway_cmd_script(
    python_path: str,
    working_dir: str,
    hermes_home: str,
    profile_arg: str,
) -> str:
    """Build the ``gateway.cmd`` wrapper content (CRLF-terminated).

    The script:
      - cd's into a stable working directory
      - exports HERMES_HOME, PYTHONIOENCODING, VIRTUAL_ENV
      - invokes ``python -m hermes_cli.main [--profile X] gateway run``

    The .cmd is a compatibility/manual-run artifact: service persistence
    (Scheduled Task, Startup folder) routes through the ``.vbs`` launcher,
    which runs this same command line hidden (window style 0).  Run by hand
    in a real terminal, the console interpreter keeps the gateway attached
    to that terminal like a normal foreground ``hermes gateway run``.

    We intentionally do NOT inline PATH overrides here — cmd.exe inherits
    the per-user PATH the Scheduled Task was created with, and forcibly
    rewriting PATH tends to break Homebrew/nvm-style installations.
    """
    lines = ["@echo off", f"rem {_TASK_DESCRIPTION}"]
    lines.append(f"cd /d {_quote_cmd_script_arg(working_dir)}")
    lines.append(f'set "HERMES_HOME={hermes_home}"')
    lines.append('set "PYTHONIOENCODING=utf-8"')
    lines.append('set "HERMES_GATEWAY_DETACHED=1"')
    python_exe_path, venv_dir, extra_pythonpath = _resolve_detached_python(python_path)
    # VIRTUAL_ENV lets the gateway's own python detection find the venv
    # if someone imports hermes_constants-based logic during startup.
    lines.append(f'set "VIRTUAL_ENV={_preserve_hermes_home_path(venv_dir)}"')
    pythonpath_entries = [
        _preserve_hermes_home_path(Path(__file__).resolve().parent.parent),
        *[_preserve_hermes_home_path(entry) for entry in extra_pythonpath],
    ]
    lines.append(f'set "PYTHONPATH={";".join([*pythonpath_entries, "%PYTHONPATH%"])}"')

    prog_args = [python_exe_path, "-m", "hermes_cli.main"]
    if profile_arg:
        prog_args.extend(profile_arg.split())
    prog_args.extend(["gateway", "run"])
    # Do NOT use `start` here; that creates an extra wrapper process and made
    # gateway lifecycle/status harder to reason about.
    # Do NOT use `--replace` for service-managed starts; repeated /Run calls
    # should be idempotent, not churn parent/child takeover loops.
    lines.append(" ".join(_quote_cmd_script_arg(a) for a in prog_args))
    lines.append("exit /b 0")
    return "\r\n".join(lines) + "\r\n"


def _quote_vbs_string(value: str) -> str:
    """Quote a value as a VBScript double-quoted string literal.

    VBScript escapes an embedded double-quote by doubling it. A newline cannot
    appear inside a literal, so refuse it (same guard as ``_quote_cmd_script_arg``).
    """
    if "\r" in value or "\n" in value:
        raise ValueError(f"refusing to quote VBScript value containing newline: {value!r}")
    return '"' + value.replace('"', '""') + '"'


def _build_gateway_vbs_script(
    python_path: str,
    working_dir: str,
    hermes_home: str,
    profile_arg: str,
) -> str:
    """Build a hidden-console ``gateway.vbs`` launcher (CRLF-terminated).

    The Scheduled Task runs this through ``wscript.exe`` instead of ``cmd.exe``.

    Why: issue #45599 root cause #1. Driving the gateway through ``cmd.exe``
    allocates a console, and during logon Windows broadcasts ``CTRL_CLOSE_EVENT``
    to console process groups — reaping cmd.exe and the half-initialized gateway
    with ``STATUS_CONTROL_C_EXIT`` (``0xC000013A``). Task Scheduler treats that
    code as a user cancel, so the ``RestartOnFailure`` policy never fires and the
    gateway silently disappears on every reboot.

    ``wscript.exe`` is a GUI-subsystem executable with no console, so this
    launcher receives no console control events. It ``Run``s the console
    ``python.exe`` with window style 0 (hidden): the gateway owns a single
    hidden console — never shown, never CTRL_CLOSE'd at logon, and inherited
    by every console-subsystem descendant (git, gh, node, …) so none of them
    allocate a visible flashing conhost (#54220/#56747; the previous
    console-less pythonw.exe gateway forced exactly that per-descendant
    flash). No cmd.exe anywhere in the chain. Mirrors
    ``_build_gateway_cmd_script`` (same env + argv via
    ``_resolve_detached_python``).
    """
    python_exe_path, venv_dir, extra_pythonpath = _resolve_detached_python(python_path)

    prog_args = [python_exe_path, "-m", "hermes_cli.main"]
    if profile_arg:
        prog_args.extend(profile_arg.split())
    prog_args.extend(["gateway", "run"])
    # list2cmdline gives CreateProcess-correct quoting for WScript.Shell.Run.
    command_line = subprocess.list2cmdline(prog_args)

    repo_root = _preserve_hermes_home_path(Path(__file__).resolve().parent.parent)
    static_pythonpath = os.pathsep.join(
        [repo_root, *[_preserve_hermes_home_path(entry) for entry in extra_pythonpath]]
    )

    lines = [
        f"' {_TASK_DESCRIPTION}",
        "Option Explicit",
        "Dim sh, env, existing_pp",
        'Set sh = CreateObject("WScript.Shell")',
        'Set env = sh.Environment("PROCESS")',
        f"env.Item({_quote_vbs_string('HERMES_HOME')}) = {_quote_vbs_string(hermes_home)}",
        f"env.Item({_quote_vbs_string('PYTHONIOENCODING')}) = {_quote_vbs_string('utf-8')}",
        f"env.Item({_quote_vbs_string('HERMES_GATEWAY_DETACHED')}) = {_quote_vbs_string('1')}",
        f"env.Item({_quote_vbs_string('VIRTUAL_ENV')}) = {_quote_vbs_string(_preserve_hermes_home_path(venv_dir))}",
        # Mirror the cmd wrapper's ``PYTHONPATH=<static>;%PYTHONPATH%``: chain onto
        # whatever PYTHONPATH the task environment already carries, at runtime.
        f"existing_pp = env.Item({_quote_vbs_string('PYTHONPATH')})",
        "If Len(existing_pp) > 0 Then",
        f"  env.Item({_quote_vbs_string('PYTHONPATH')}) = {_quote_vbs_string(static_pythonpath + os.pathsep)} & existing_pp",
        "Else",
        f"  env.Item({_quote_vbs_string('PYTHONPATH')}) = {_quote_vbs_string(static_pythonpath)}",
        "End If",
        f"sh.CurrentDirectory = {_quote_vbs_string(working_dir)}",
        # Window style 0 = hidden; bWaitOnReturn False = detached/async. The
        # console python's one console is created hidden and inherited by all
        # descendants, so nothing ever flashes.
        f"sh.Run {_quote_vbs_string(command_line)}, 0, False",
    ]
    return "\r\n".join(lines) + "\r\n"


def _build_startup_launcher(script_path: Path) -> str:
    """The tiny .vbs that goes in the Startup folder and chains hidden.

    Defense-in-depth: bail out silently if the target script is gone. Test
    fixtures historically wrote Startup entries pointing at pytest tmp_path
    directories that vanish after the test session. Without the existence
    guard, every subsequent Windows login could attempt a stale launcher. The
    check + ``WScript.Quit 0`` keeps that case silent.
    """
    target = str(script_path.with_suffix(".vbs"))
    command = subprocess.list2cmdline(["wscript.exe", target])
    lines = [
        f"' {_TASK_DESCRIPTION}",
        "Option Explicit",
        "Dim fso, sh, target",
        f"target = {_quote_vbs_string(target)}",
        'Set fso = CreateObject("Scripting.FileSystemObject")',
        "If Not fso.FileExists(target) Then WScript.Quit 0",
        'Set sh = CreateObject("WScript.Shell")',
        f"sh.Run {_quote_vbs_string(command)}, 0, False",
    ]
    return "\r\n".join(lines) + "\r\n"


def _write_task_script() -> Path:
    """Generate and write the gateway.cmd wrapper. Return its absolute path."""
    _assert_windows()
    # Local imports to avoid circular-init at module load time.
    from hermes_cli.config import get_hermes_home
    from hermes_cli.gateway import (
        PROJECT_ROOT,
        _profile_arg,
        get_python_path,
    )

    python_path = _preserve_hermes_home_path(get_python_path())
    working_dir = _stable_gateway_working_dir(PROJECT_ROOT)
    hermes_home = str(Path(get_hermes_home()))
    profile_arg = _profile_arg(hermes_home)

    content = _build_gateway_cmd_script(python_path, working_dir, hermes_home, profile_arg)
    script_path = get_task_script_path()
    tmp = script_path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8", newline="")
    tmp.replace(script_path)

    # Also render the console-less .vbs launcher used by Scheduled Task and the
    # Startup-folder fallback via wscript.exe (issue #45599 fix A). The .cmd
    # wrapper stays as a generated helper/compatibility artifact.
    vbs_content = _build_gateway_vbs_script(python_path, working_dir, hermes_home, profile_arg)
    vbs_path = script_path.with_suffix(".vbs")
    vbs_tmp = vbs_path.with_name(vbs_path.name + ".tmp")
    vbs_tmp.write_text(vbs_content, encoding="utf-8", newline="")
    vbs_tmp.replace(vbs_path)
    return script_path


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

def _resolve_task_user() -> str | None:
    """Return ``DOMAIN\\USER`` if available, else bare USERNAME, else None."""
    username = os.environ.get("USERNAME") or os.environ.get("USER") or os.environ.get("LOGNAME")
    if not username:
        return None
    if "\\" in username:
        return username
    domain = os.environ.get("USERDOMAIN")
    return f"{domain}\\{username}" if domain else username


def _build_scheduled_task_xml(task_name: str, launcher_path: Path, user: str | None) -> str:
    """Render a Task Scheduler XML definition with safe long-running defaults.

    ``launcher_path`` is the console-less ``.vbs`` the task runs via
    ``wscript.exe`` — not the ``.cmd`` (see ``_build_gateway_vbs_script`` /
    issue #45599 root cause #1).
    """
    user_principal = f"\n      <UserId>{escape(user)}</UserId>" if user else ""
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{escape(_TASK_DESCRIPTION)}</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <Delay>{_TASK_LOGON_DELAY}</Delay>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">{user_principal}
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>{_TASK_RESTART_INTERVAL}</Interval>
      <Count>{_TASK_RESTART_COUNT}</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>wscript.exe</Command>
      <Arguments>//B //Nologo "{escape(str(launcher_path))}"</Arguments>
    </Exec>
  </Actions>
</Task>
"""


def _write_scheduled_task_xml(task_name: str, launcher_path: Path, user: str | None) -> Path:
    xml_path = launcher_path.with_suffix(".task.xml")
    xml_path.write_text(
        _build_scheduled_task_xml(task_name, launcher_path, user),
        encoding="utf-16",
        newline="",
    )
    return xml_path


def _install_scheduled_task(task_name: str, script_path: Path) -> tuple[bool, str]:
    """Create or replace the Scheduled Task. Returns (success, detail).

    Always recreate instead of ``/Change``. Older Hermes builds and failed
    experiments may have left repeat/restart settings on the task; ``/Change``
    preserves those stale triggers and can make the gateway relaunch every
    minute. Delete+create gives us a clean ONLOGON task every install.
    """
    delete_code, delete_out, delete_err = _exec_schtasks(["/Delete", "/F", "/TN", task_name])
    delete_detail = (delete_err or delete_out or "").strip()
    if delete_code != 0 and delete_detail and "cannot find" not in delete_detail.lower():
        if _is_access_denied(delete_detail):
            return (False, f"schtasks /Delete failed (code {delete_code}): {delete_detail}")
        # Non-fatal: /Create /F below may still replace it. Keep the detail in
        # the final error if creation also fails.
    user = _resolve_task_user()
    # The Scheduled Task launches the console-less .vbs (issue #45599 fix A), not
    # the .cmd. Immediate manual starts use _spawn_detached().
    launcher_path = script_path.with_suffix(".vbs")
    xml_path = _write_scheduled_task_xml(task_name, launcher_path, user)
    base = ["/Create", "/F", "/TN", task_name, "/XML", str(xml_path)]
    variants = [[*base, "/RU", user, "/NP", "/IT"]] if user else []
    variants.append(base)

    last_code = 1
    last_err = ""
    try:
        for argv in variants:
            code, out, err = _exec_schtasks(argv)
            if code == 0:
                return (True, f"Created Scheduled Task {task_name!r}")
            last_code, last_err = code, (err or out or "")
    finally:
        try:
            xml_path.unlink(missing_ok=True)
        except OSError:
            pass
    if delete_detail and "cannot find" not in delete_detail.lower():
        last_err = f"{last_err.strip()} (delete detail: {delete_detail})"
    return (False, f"schtasks /Create failed (code {last_code}): {last_err.strip()}")




def _install_startup_entry(script_path: Path) -> Path:
    """Write the Startup-folder fallback launcher. Returns its path."""
    entry = get_startup_entry_path()
    entry.parent.mkdir(parents=True, exist_ok=True)
    tmp = entry.with_suffix(".tmp")
    tmp.write_text(_build_startup_launcher(script_path), encoding="utf-8", newline="")
    tmp.replace(entry)
    legacy_entry = _legacy_startup_entry_path()
    try:
        if legacy_entry.exists():
            legacy_entry.unlink()
    except OSError:
        pass
    return entry


def _resolve_detached_python(python_exe: str) -> tuple[str, Path, list[str]]:
    """Return (hidden_console_python, venv_dir, extra_pythonpath) for detached runs.

    Returns the venv's **console** ``python.exe`` — deliberately NOT
    ``pythonw.exe``.  Every detached launch path pairs this interpreter with a
    hidden-console mechanism (``CREATE_NO_WINDOW`` creationflags, or
    ``WScript.Shell.Run`` window style 0), so the daemon owns a single hidden
    console that all of its console-subsystem descendants (git, gh, cmd, node,
    wmic, powershell, …) inherit instead of each allocating a visible flashing
    one.  A GUI-subsystem ``pythonw.exe`` daemon has NO console, which is what
    made every descendant spawn flash (#54220/#56747) and forced the endless
    per-call-site CREATE_NO_WINDOW sweep.  Root cause isolated + A/B verified
    on Windows 11 by the desktop backend fix (commit aa2ae36c3f).

    Two historical premises behind the old pythonw selection were re-tested on
    current Windows in that fix and did not hold up:

    - uv venv launcher: ``venv\\Scripts\\python.exe`` under ``CREATE_NO_WINDOW``
      re-execs the base interpreter *windowless* — the child inherits the
      shim's hidden console, so no conhost flashes (the #52239 concern).  The
      historical "CREATE_NO_WINDOW cannot suppress the second window"
      observations were made while ``DETACHED_PROCESS`` was in the flag
      bundle, where MSDN specifies CREATE_NO_WINDOW is IGNORED — the hide bit
      was dead, not ineffective.  The base-interpreter + PYTHONPATH-overlay
      detour is therefore unnecessary; the venv shim resolves imports itself.
    - Console python restores stdout/stderr, so daemon logs flow normally.

    ``extra_pythonpath`` is always empty now; the tuple shape is kept so the
    call sites (argv builders, cmd/vbs renderers, restart-spec rewriter,
    gateway watcher) stay unchanged.
    """
    p = Path(python_exe)
    venv_dir = p.parent.parent
    return (python_exe, venv_dir, [])


def _prepend_pythonpath(env_overlay: dict[str, str], entries: list[str]) -> None:
    clean_entries = [entry for entry in entries if entry]
    if not clean_entries:
        return
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        clean_entries.append(existing)
    env_overlay["PYTHONPATH"] = os.pathsep.join(clean_entries)


def _build_gateway_argv() -> tuple[list[str], str, dict[str, str]]:
    """Build (argv, working_dir, env_overlay) for the gateway subprocess.

    Same logical command as what gateway.cmd runs, but assembled as a
    native argv for direct ``subprocess.Popen`` invocation — no cmd.exe
    layer in between.
    """
    _assert_windows()
    from hermes_cli.config import get_hermes_home
    from hermes_cli.gateway import (
        PROJECT_ROOT,
        _profile_arg,
        get_python_path,
    )

    python_exe, venv_dir, extra_pythonpath = _resolve_detached_python(
        _preserve_hermes_home_path(get_python_path())
    )
    project_root = _preserve_hermes_home_path(PROJECT_ROOT)
    working_dir = _stable_gateway_working_dir(PROJECT_ROOT)
    hermes_home = str(Path(get_hermes_home()))
    profile_arg = _profile_arg(hermes_home)

    argv = [python_exe, "-m", "hermes_cli.main"]
    if profile_arg:
        argv.extend(profile_arg.split())
    argv.extend(["gateway", "run"])

    env_overlay = {
        "HERMES_HOME": hermes_home,
        "PYTHONIOENCODING": "utf-8",
        "HERMES_GATEWAY_DETACHED": "1",
        "VIRTUAL_ENV": _preserve_hermes_home_path(venv_dir),
    }
    _prepend_pythonpath(
        env_overlay,
        [project_root, *[_preserve_hermes_home_path(entry) for entry in extra_pythonpath]]
        if extra_pythonpath
        else [project_root],
    )
    return argv, working_dir, env_overlay


def windowless_gateway_restart_spec(
    run_argv: list[str],
) -> tuple[list[str], str, dict[str, str]]:
    """Return the (argv, cwd, env overlay) for a hidden-console gateway respawn.

    The post-update restart paths build their respawn command from
    ``get_python_path()`` (the venv's console ``python.exe``).  That is the
    right interpreter: the watcher launches it with ``CREATE_NO_WINDOW``
    detach flags, so the respawned gateway owns a single hidden console that
    all of its descendants inherit — nothing flashes (#54220/#56747; the old
    pythonw.exe rewrite here produced a console-less gateway whose every
    console-subsystem child allocated a visible conhost).  This helper now
    only normalizes the interpreter via ``_resolve_detached_python`` and
    supplies the stable cwd + env overlay (HERMES_HOME, VIRTUAL_ENV,
    PYTHONPATH) so the respawn doesn't depend on the watcher's transient
    working directory.

    Returns ``(new_argv, working_dir, env_overlay)``.  ``new_argv``
    preserves every argument after the interpreter (``-m hermes_cli.main
    [--profile X] gateway run [--replace]``) verbatim.  On non-Windows, or
    if ``run_argv`` doesn't start with a resolvable python, the argv is
    returned unchanged with an empty overlay.
    """
    if not run_argv:
        return run_argv, "", {}
    if sys.platform != "win32":
        return run_argv, "", {}

    from hermes_cli.config import get_hermes_home
    from hermes_cli.gateway import PROJECT_ROOT

    python_exe = run_argv[0]
    rest = run_argv[1:]

    # Normalize the leading interpreter token and derive the venv layout.
    # If a caller passed something other than a python path (a non-python
    # launcher), leave the argv alone.
    try:
        hidden_console_python, venv_dir, extra_pythonpath = _resolve_detached_python(
            python_exe
        )
    except Exception:
        return run_argv, "", {}

    new_argv = [hidden_console_python, *rest]

    working_dir = _stable_gateway_working_dir(PROJECT_ROOT)
    project_root = str(PROJECT_ROOT)
    try:
        hermes_home = str(Path(get_hermes_home()).resolve())
    except Exception:
        hermes_home = ""

    env_overlay: dict[str, str] = {
        "PYTHONIOENCODING": "utf-8",
        "HERMES_GATEWAY_DETACHED": "1",
        "VIRTUAL_ENV": str(venv_dir),
    }
    if hermes_home:
        env_overlay["HERMES_HOME"] = hermes_home
    _prepend_pythonpath(
        env_overlay,
        [project_root, *extra_pythonpath] if extra_pythonpath else [project_root],
    )
    return new_argv, working_dir, env_overlay


def _spawn_detached(script_path: Path | None = None) -> int:
    """Launch the gateway as a fully detached background process.

    We spawn ``python.exe -m hermes_cli.main gateway run`` directly — NOT
    through a cmd.exe shim — because on Windows a cmd.exe child inherits the
    parent session's console handle and tends to get reaped when the spawning
    shell exits.  With ``CREATE_NO_WINDOW`` the gateway gets its OWN hidden
    console instead of inheriting ours, so it survives our shell closing, and
    every console-subsystem descendant it spawns inherits that hidden console
    instead of flashing a visible one (#54220/#56747 — this is why we don't
    use console-less pythonw.exe here). Combined with
    CREATE_NEW_PROCESS_GROUP + DEVNULL stdin + a fresh env, the resulting
    process is independent of whichever shell started it.

    Arg ``script_path`` is accepted for API symmetry with older callers
    but ignored — we don't need it now that we go direct.

    Returns the spawned PID so callers can verify the process actually
    came up.
    """
    _assert_windows()
    argv, working_dir, env_overlay = _build_gateway_argv()

    # Inherit PATH etc. from the current env, overlay our required vars.
    env = {**os.environ, **env_overlay}

    # CREATE_NEW_PROCESS_GROUP 0x00000200 — child gets its own group, won't
    #                                       receive Ctrl+C from our group
    # CREATE_NO_WINDOW         0x08000000 — child owns a hidden console:
    #                                       detached from our console's
    #                                       lifetime AND inheritable by its
    #                                       descendants (no conhost flashes)
    # CREATE_BREAKAWAY_FROM_JOB 0x01000000 — escape any job object the
    #                                       parent is in (prevents parent-
    #                                       job teardown from reaping us;
    #                                       some Windows Terminal versions
    #                                       wrap their children in a job).
    flags = windows_detach_flags()

    # Redirect any stray stdout/stderr output to a sidecar log. Python's
    # logging module writes to gateway.log through a FileHandler, so the
    # real gateway logs still land there — this just captures anything
    # that goes to print() or native stderr.
    from hermes_cli.config import get_hermes_home

    log_dir = Path(get_hermes_home()) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stray_log = log_dir / "gateway-stdio.log"

    try:
        with open(stray_log, "ab", buffering=0) as log_fh:
            proc = subprocess.Popen(
                argv,
                cwd=working_dir,
                env=env,
                creationflags=flags,
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=log_fh,
            )
    except OSError:
        # CREATE_BREAKAWAY_FROM_JOB can fail with "access denied" when the
        # parent's job object doesn't permit breakaway (some Windows
        # Terminal configs). Retry without the breakaway flag — in most
        # setups the hidden-console CREATE_NO_WINDOW spawn is enough on
        # its own.
        flags_no_breakaway = windows_detach_flags_without_breakaway()
        with open(stray_log, "ab", buffering=0) as log_fh:
            proc = subprocess.Popen(
                argv,
                cwd=working_dir,
                env=env,
                creationflags=flags_no_breakaway,
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=log_fh,
            )
    return proc.pid


def _install_choice_from_env(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _prompt_install_choices(
    start_now: bool | None = None,
    start_on_login: bool | None = None,
) -> tuple[bool, bool]:
    """Return (start_now, start_on_login), asking before any UAC escalation."""
    env_start_now = _install_choice_from_env("HERMES_GATEWAY_INSTALL_START_NOW")
    env_start_on_login = _install_choice_from_env("HERMES_GATEWAY_INSTALL_START_ON_LOGIN")
    if start_now is None:
        start_now = env_start_now
    if start_on_login is None:
        start_on_login = env_start_on_login
    if start_now is not None and start_on_login is not None:
        return start_now, start_on_login

    from hermes_cli.setup import prompt_yes_no

    if start_now is None:
        start_now = prompt_yes_no("Start the gateway now after install?", True)
    if start_on_login is None:
        start_on_login = prompt_yes_no(
            "Start the gateway automatically on Windows login with a Scheduled Task?",
            True,
        )
    return start_now, start_on_login


def _install_startup_fallback(script_path: Path, start_now: bool, detail: str) -> None:
    """Install the Startup-folder fallback and optionally start once."""
    print(f"↻ Scheduled Task install blocked ({detail.splitlines()[0]}) — using Startup folder fallback")
    entry = _install_startup_entry(script_path)
    print(f"✓ Installed Windows login item: {entry}")
    print(f"  Task script: {script_path}")

    # Re-running `hermes -p <profile> gateway install` must be safe.
    # Startup-folder fallback only installs login persistence. Starting is
    # controlled by the pre-UAC start_now answer so all user decisions happen
    # before any elevation prompt.
    from hermes_cli.gateway import find_gateway_pids, _profile_arg

    running_pids = list(find_gateway_pids())
    if running_pids:
        print(f"✓ Gateway already running (PID: {', '.join(map(str, running_pids))})")
    elif start_now:
        pid = _spawn_detached()
        _report_gateway_start(f"direct spawn (PID {pid})")
    else:
        profile_arg = _profile_arg()
        start_cmd = f"hermes {profile_arg} gateway start" if profile_arg else "hermes gateway start"
        print("ℹ Startup fallback installed; gateway not started now.")
        print(f"  Start manually with: {start_cmd}")
    _print_next_steps()


def install(
    force: bool = False,
    *,
    start_now: bool | None = None,
    start_on_login: bool | None = None,
    elevated_handoff: bool = False,
) -> None:
    """Install the gateway as a Windows Scheduled Task (with Startup fallback).

    Idempotent: re-running updates the task to point at the current python/
    project paths. ``force`` is accepted for API parity with ``launchd_install``
    / ``systemd_install`` but isn't needed — we always reconcile.
    """
    _assert_windows()
    start_now, start_on_login = _prompt_install_choices(start_now, start_on_login)

    if not start_on_login:
        print("ℹ Skipped Windows login auto-start install.")
        if start_now:
            running_pids = _gateway_pids()
            if running_pids:
                print(f"✓ Gateway already running (PID: {', '.join(map(str, running_pids))})")
            else:
                pid = _spawn_detached()
                _report_gateway_start(f"direct spawn (PID {pid})")
        else:
            print("ℹ Gateway not started and no auto-start service installed.")
            print("  Run later with: hermes gateway start")
        return

    task_name = get_task_name()
    script_path = _write_task_script()

    # On machines where the current user's scheduled-task ACL is locked down,
    # schtasks /Create or /Change can sit for the timeout before returning
    # Access Denied. We already collected all intent questions above, so avoid
    # a mysterious post-question pause: ask for UAC before touching schtasks.
    if not _is_running_as_admin() and not elevated_handoff:
        from hermes_cli.setup import prompt_yes_no

        print("↻ Scheduled Task install may need administrator approval on this Windows account.")
        print("  UAC is Windows' admin approval prompt; it is needed to create/update the Scheduled Task.")
        if prompt_yes_no("  Open the UAC prompt now?", False):
            if _launch_elevated_install(force=force, start_now=start_now, start_on_login=start_on_login):
                print("✓ Launched elevated Hermes gateway install prompt.")
                if start_now:
                    print("  Approve the Windows UAC prompt; the elevated install will start the gateway afterwards.")
                else:
                    print("  Approve the Windows UAC prompt, then run: hermes gateway status")
                return
            print("⚠ Falling back to Startup folder because elevation was unavailable or cancelled.")
        else:
            print("  Skipped elevation. Falling back to Startup folder.")
        _install_startup_fallback(script_path, start_now, "administrator approval was not used")
        return

    ok, detail = _install_scheduled_task(task_name, script_path)
    if ok:
        print(f"✓ {detail}")
        print(f"  Task script: {script_path}")
        print("ℹ Gateway auto-start installed for Windows login.")
        if start_now:
            running_pids = _gateway_pids()
            if running_pids:
                print(f"✓ Gateway already running (PID: {', '.join(map(str, running_pids))})")
            else:
                pid = _spawn_detached()
                _report_gateway_start(f"direct spawn (PID {pid})")
        else:
            print("ℹ Gateway not started now.")
            print("  Start manually with: hermes gateway start")
        _print_next_steps()
        return

    # schtasks create didn't work. Prefer a real Scheduled Task over the
    # Startup-folder fallback when the only blocker is elevation. This gives
    # users a UAC prompt instead of silently installing a less reliable login
    # item, and keeps the fallback for locked-down boxes / cancelled prompts.
    if _is_access_denied(detail) and not _is_running_as_admin():
        from hermes_cli.setup import prompt_yes_no

        print(f"↻ Scheduled Task install needs administrator approval ({detail.splitlines()[0]})")
        print("  UAC is Windows' admin approval prompt; it is needed to create/update the Scheduled Task.")
        if prompt_yes_no("  Open the UAC prompt now?", False):
            if _launch_elevated_install(force=force, start_now=start_now, start_on_login=start_on_login):
                print("✓ Launched elevated Hermes gateway install prompt.")
                if start_now:
                    print("  Approve the Windows UAC prompt; the elevated install will start the gateway afterwards.")
                else:
                    print("  Approve the Windows UAC prompt, then run: hermes gateway status")
                return
            print("⚠ Falling back to Startup folder because elevation was unavailable or cancelled.")
        else:
            print("  Skipped elevation. Falling back to Startup folder.")

    # schtasks create didn't work. See if it's a "fall back to startup" case.
    if _should_fall_back(1, detail):
        print(f"↻ Scheduled Task install blocked ({detail.splitlines()[0]}) — using Startup folder fallback")
        entry = _install_startup_entry(script_path)
        print(f"✓ Installed Windows login item: {entry}")
        print(f"  Task script: {script_path}")

        # Re-running `hermes -p <profile> gateway install` must be safe.
        # Startup-folder fallback only installs login persistence. Starting is
        # controlled by the pre-UAC start_now answer so all user decisions happen
        # before any elevation prompt.
        from hermes_cli.gateway import find_gateway_pids, _profile_arg

        running_pids = list(find_gateway_pids())
        if running_pids:
            print(f"✓ Gateway already running (PID: {', '.join(map(str, running_pids))})")
        elif start_now:
            pid = _spawn_detached()
            _report_gateway_start(f"direct spawn (PID {pid})")
        else:
            profile_arg = _profile_arg()
            start_cmd = f"hermes {profile_arg} gateway start" if profile_arg else "hermes gateway start"
            print("ℹ Startup fallback installed; gateway not started now.")
            print(f"  Start manually with: {start_cmd}")
        _print_next_steps()
        return

    # Unknown schtasks error — surface it and bail.
    raise RuntimeError(f"Windows gateway install failed: {detail}")


def _wait_for_gateway_ready(timeout_s: float = 6.0, interval_s: float = 0.4) -> list[int]:
    """Poll for a live gateway process for up to ``timeout_s`` seconds.

    Returns the list of PIDs found. Empty list means nothing came up in
    time — the caller should surface that to the user as a failed start.
    """
    from hermes_cli.gateway import find_gateway_pids

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pids = list(find_gateway_pids())
        if pids:
            return pids
        time.sleep(interval_s)
    return []


def _report_gateway_start(via: str) -> None:
    pids = _wait_for_gateway_ready()
    if pids:
        print(f"✓ Gateway started via {via} (PID: {', '.join(map(str, pids))})")
    else:
        print(f"⚠ Launched gateway via {via}, but no process detected after 6s.")
        print("  Check the log for startup errors:")
        from hermes_cli.config import get_hermes_home
        print(f"    type {Path(get_hermes_home())}\\logs\\gateway.log")
        print(f"    type {Path(get_hermes_home())}\\logs\\gateway-stdio.log")


def _print_next_steps() -> None:
    from hermes_cli.config import get_hermes_home

    hermes_home = Path(get_hermes_home())
    print()
    print("Next steps:")
    print("  hermes gateway status                      # Check status")
    print(f"  type {hermes_home}\\logs\\gateway.log       # View logs")


def uninstall() -> None:
    """Remove both the Scheduled Task and the Startup-folder fallback, if present."""
    _assert_windows()
    task_name = get_task_name()
    script_path = get_task_script_path()
    vbs_script_path = script_path.with_suffix(".vbs")
    startup_entry = get_startup_entry_path()
    legacy_startup_entry = _legacy_startup_entry_path()

    scheduled_task_removed = False
    if is_task_registered():
        code, _out, err = _exec_schtasks(["/Delete", "/F", "/TN", task_name])
        detail = err.strip()
        if code == 0:
            scheduled_task_removed = True
            print(f"✓ Removed Scheduled Task {task_name!r}")
        elif _is_access_denied(detail) and not _is_running_as_admin():
            from hermes_cli.setup import prompt_yes_no

            print(f"↻ Scheduled Task uninstall needs administrator approval ({detail or 'access denied'})")
            print("  UAC is Windows' admin approval prompt; it is needed to remove the Scheduled Task.")
            if prompt_yes_no("  Open the UAC prompt now?", False):
                if _launch_elevated_uninstall():
                    print("✓ Launched elevated Hermes gateway uninstall prompt.")
                    print("  Approve the Windows UAC prompt, then run: hermes gateway status")
                    return
                print("⚠ Elevated uninstall prompt was unavailable or cancelled.")
            else:
                print("  Skipped elevation. Scheduled Task was not removed.")
        else:
            print(f"⚠ schtasks /Delete returned code {code}: {detail}")

    for path, label in [
        (startup_entry, "Windows login item"),
        (legacy_startup_entry, "legacy Windows login item"),
        (script_path, "Task script"),
        (vbs_script_path, "Task launcher"),
    ]:
        try:
            path.unlink()
            print(f"✓ Removed {label}: {path}")
        except FileNotFoundError:
            pass

    if is_task_registered() and not scheduled_task_removed:
        print(f"⚠ Scheduled Task still registered: {task_name}")


# ---------------------------------------------------------------------------
# Status / start / stop / restart
# ---------------------------------------------------------------------------

def is_task_registered() -> bool:
    code, _out, _err = _exec_schtasks(["/Query", "/TN", get_task_name()])
    return code == 0


def is_startup_entry_installed() -> bool:
    return get_startup_entry_path().exists() or _legacy_startup_entry_path().exists()


def is_installed() -> bool:
    """True when either the schtasks entry or the Startup fallback is present."""
    return is_task_registered() or is_startup_entry_installed()


def query_task_status() -> dict[str, str]:
    """Parse ``schtasks /Query /V /FO LIST`` and pull the interesting keys."""
    code, out, err = _exec_schtasks(["/Query", "/TN", get_task_name(), "/V", "/FO", "LIST"])
    if code != 0:
        return {}
    info: dict[str, str] = {}
    for raw in out.splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        # Some Windows locales emit "Last Result" instead of "Last Run Result".
        if key in {"status", "last run time", "last run result", "last result"}:
            if key == "last result":
                info.setdefault("last run result", value)
            else:
                info[key] = value
    return info


def _gateway_pids() -> list[int]:
    """Reuse the cross-platform PID scanner in gateway.py."""
    from hermes_cli.gateway import find_gateway_pids

    return list(find_gateway_pids())


def _print_deep_probes() -> None:
    """Print PASS/FAIL per individual probe of gateway liveness.

    The default ``status`` output collapses several signals into one
    ✓ / ✗ line, which is great when they agree and confusing when they
    don't. The deep-probe block shows each underlying check independently
    so the user can see exactly which signal is wrong.

    Probes:
      [1] PID file present
      [2] Lock file present and held by some process
      [3] gateway.status.get_running_pid() returns a PID
      [4] _pid_exists(pid) — OS confirms the process is alive
      [5] gateway_state.json exists and parses (and is fresh-ish)
      [6] Last lifecycle event in gateway-exit-diag.log
    """
    import json
    from datetime import datetime, timezone

    from hermes_cli.config import get_hermes_home

    home = Path(get_hermes_home())
    pid_path = home / "gateway.pid"
    lock_path = home / "gateway.lock"
    state_path = home / "gateway_state.json"
    diag_path = home / "logs" / "gateway-exit-diag.log"

    print()
    print("Deep probes:")

    def _mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    # [1] PID file
    pid_exists = pid_path.exists()
    pid_value: int | None = None
    if pid_exists:
        try:
            data = json.loads(pid_path.read_text(encoding="utf-8"))
            pid_value = int(data.get("pid")) if data.get("pid") is not None else None
            print(f"  [1] {_mark(True):4s}  PID file present: {pid_path} (pid={pid_value})")
        except Exception as exc:
            print(f"  [1] {_mark(False):4s}  PID file present but unreadable: {exc}")
    else:
        print(f"  [1] {_mark(False):4s}  PID file missing: {pid_path}")

    # [2] Lock file present + held
    lock_held = False
    lock_present = lock_path.exists()
    if lock_present:
        try:
            from gateway.status import is_gateway_runtime_lock_active

            lock_held = is_gateway_runtime_lock_active(lock_path)
            print(f"  [2] {_mark(lock_held):4s}  Lock file held by a live process: {lock_path}")
        except Exception as exc:
            print(f"  [2] {_mark(False):4s}  Could not probe lock: {exc}")
    else:
        print(f"  [2] {_mark(False):4s}  Lock file missing: {lock_path}")

    # [3] get_running_pid()
    running_pid: int | None = None
    try:
        from gateway.status import get_running_pid

        running_pid = get_running_pid(cleanup_stale=False)
        print(f"  [3] {_mark(running_pid is not None):4s}  get_running_pid() => {running_pid}")
    except Exception as exc:
        print(f"  [3] {_mark(False):4s}  get_running_pid() raised: {exc!r}")

    # [4] _pid_exists() on the probed PID
    candidate_pid = running_pid if running_pid is not None else pid_value
    if candidate_pid is not None:
        try:
            from gateway.status import _pid_exists

            alive = bool(_pid_exists(candidate_pid))
            print(f"  [4] {_mark(alive):4s}  _pid_exists({candidate_pid}) => {alive}")
        except Exception as exc:
            print(f"  [4] {_mark(False):4s}  _pid_exists raised: {exc!r}")
    else:
        print(f"  [4] {_mark(False):4s}  No candidate PID to verify")

    # [5] runtime status file
    if state_path.exists():
        try:
            state_data = json.loads(state_path.read_text(encoding="utf-8"))
            gateway_state = state_data.get("gateway_state")
            updated_at = state_data.get("updated_at")
            age_str = ""
            if updated_at:
                try:
                    updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    age_seconds = int((now - updated_dt).total_seconds())
                    age_str = f" (updated {age_seconds}s ago)"
                except Exception:
                    pass
            ok = gateway_state == "running"
            print(f"  [5] {_mark(ok):4s}  gateway_state.json state={gateway_state!r}{age_str}")
        except Exception as exc:
            print(f"  [5] {_mark(False):4s}  gateway_state.json present but unreadable: {exc}")
    else:
        print(f"  [5] {_mark(False):4s}  gateway_state.json missing: {state_path}")

    # [6] Last lifecycle event from the exit-diag log
    if diag_path.exists():
        try:
            with open(diag_path, "rb") as fh:
                # Read last ~4KB; one event is well under 500 bytes.
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 4096))
                tail = fh.read().decode("utf-8", errors="replace").splitlines()
            last_event = next((ln for ln in reversed(tail) if ln.strip()), "")
            if last_event:
                try:
                    event = json.loads(last_event)
                    tag = event.get("tag", "?")
                    pid = event.get("pid", "?")
                    ts = event.get("ts", "?")
                    healthy = tag in ("gateway.start",)
                    print(f"  [6] {_mark(healthy):4s}  Last lifecycle event: tag={tag} pid={pid} ts={ts}")
                except Exception:
                    print(f"  [6] {_mark(False):4s}  Last lifecycle line not JSON: {last_event[:120]}")
            else:
                print(f"  [6] {_mark(False):4s}  exit-diag log empty: {diag_path}")
        except Exception as exc:
            print(f"  [6] {_mark(False):4s}  exit-diag log unreadable: {exc}")
    else:
        print(f"  [6] {_mark(False):4s}  exit-diag log missing: {diag_path}")


def status(deep: bool = False) -> None:
    """Print a status report for the Windows gateway service."""
    _assert_windows()
    task_name = get_task_name()
    task_installed = is_task_registered()
    startup_installed = is_startup_entry_installed()
    pids = _gateway_pids()

    if task_installed:
        print(f"✓ Scheduled Task registered: {task_name}")
        info = query_task_status()
        if info:
            for key in ("status", "last run time", "last run result"):
                if key in info:
                    print(f"  {key.title()}: {info[key]}")
    elif startup_installed:
        entry = get_startup_entry_path()
        if not entry.exists():
            entry = _legacy_startup_entry_path()
        print(f"✓ Windows login item installed: {entry}")
    else:
        print("✗ Gateway service not installed")

    if pids:
        print(f"✓ Gateway process running (PID: {', '.join(map(str, pids))})")
    else:
        print("✗ No gateway process detected")

    if deep:
        print()
        print(f"  Task name:        {task_name}")
        print(f"  Task script:      {get_task_script_path()}")
        print(f"  Startup entry:    {get_startup_entry_path()}")
        # Surface the per-probe truth so the user can see *which* signal
        # is lying when the high-level summary disagrees with reality.
        _print_deep_probes()

    if not task_installed and not startup_installed and not pids:
        print()
        print("To install:")
        print("  hermes gateway install")


def start() -> None:
    """Start the gateway using the canonical detached Windows launch path."""
    _assert_windows()
    running_pids = _gateway_pids()
    if running_pids:
        print(f"✓ Gateway already running (PID: {', '.join(map(str, running_pids))})")
        return

    task_installed = is_task_registered()
    startup_installed = is_startup_entry_installed()

    if not task_installed and not startup_installed:
        from hermes_cli.setup import prompt_yes_no

        print("✗ Gateway service is not installed")
        if not prompt_yes_no("  Install it now so the gateway starts on login?", True):
            print("  Run: hermes gateway install")
            return
        install(force=False)
        task_installed = is_task_registered()
        startup_installed = is_startup_entry_installed()
        if not task_installed and not startup_installed:
            print("⚠ Gateway install did not complete in this process.")
            print("  If a UAC prompt opened, approve it, then run: hermes gateway start")
            return

    # Manual starts use the same console-less direct spawn path as restart()
    # and install --start-now. Scheduled Task / Startup entries are only login
    # persistence mechanisms.
    pid = _spawn_detached()
    _report_gateway_start(f"direct spawn (PID {pid})")


def _drain_gateway_pid(pid: int, drain_timeout: float) -> bool:
    """Write the planned-stop marker and wait for the gateway PID to exit.

    Windows cannot deliver POSIX signals to a Python asyncio loop
    (``loop.add_signal_handler`` raises NotImplementedError), so writing
    the marker is the ONLY way to ask a running gateway to drain
    in-flight agents and persist ``resume_pending`` before exit. The
    gateway's planned-stop watcher thread (gateway/run.py) polls for
    the marker and drives the same shutdown path the SIGTERM handler
    would have on POSIX.

    Returns True if the PID exited within the timeout, False if it
    didn't (caller should escalate to schtasks /End + taskkill).
    """
    if pid <= 0:
        return False
    try:
        from gateway.status import write_planned_stop_marker, _pid_exists
    except ImportError:
        return False

    try:
        write_planned_stop_marker(pid)
    except Exception:
        # Best-effort: if the marker can't be written, we have no choice
        # but to fall through to a hard kill.  Caller decides escalation.
        pass

    deadline = time.monotonic() + max(drain_timeout, 1.0)
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.5)
    return False


def _windows_stop_drain_timeout() -> float:
    """Return a bounded Windows gateway stop grace period."""
    try:
        from hermes_cli.gateway import _get_restart_drain_timeout

        configured = float(_get_restart_drain_timeout() or 30.0)
    except Exception:
        configured = 30.0
    # Windows CLI stop must not wedge forever. Give the gateway a real
    # graceful-drain window, then escalate to the known PID.
    return max(1.0, min(configured, 30.0))


def _force_terminate_known_gateway_pids(pids: list[int]) -> int:
    """Force-kill known gateway PIDs without a broad process sweep."""
    try:
        from gateway.status import _pid_exists, terminate_pid
    except ImportError:
        return 0

    own_pid = os.getpid()
    killed = 0
    seen: set[int] = set()
    for pid in pids:
        if pid <= 0 or pid == own_pid or pid in seen:
            continue
        seen.add(pid)
        try:
            if not _pid_exists(pid):
                continue
            terminate_pid(pid, force=True)
            killed += 1
        except ProcessLookupError:
            continue
        except PermissionError:
            print(f"⚠ Permission denied to kill PID {pid}")
        except OSError as exc:
            print(f"Failed to kill PID {pid}: {exc}")
    return killed


def _collect_gateway_stop_pids(primary_pid: int | None = None) -> list[int]:
    """Collect gateway PIDs for the active profile, preserving primary first."""
    pids: list[int] = []
    if primary_pid is not None and primary_pid > 0:
        pids.append(primary_pid)
    try:
        for pid in _gateway_pids():
            if pid > 0 and pid not in pids:
                pids.append(pid)
    except Exception:
        pass
    return pids


def stop() -> None:
    """Stop the gateway.

    Writes the planned-stop marker first so the gateway can drain
    in-flight agents and persist ``resume_pending`` before exit (the
    gateway's marker-watcher thread picks this up — Windows asyncio
    can't deliver SIGTERM to the loop, so the marker is our only IPC).
    Then escalates with bounded Windows process termination against the
    known gateway PID(s).
    """
    _assert_windows()
    from gateway.status import get_running_pid

    # Phase 1: ask the running gateway (if any) to drain itself by writing
    # the planned-stop marker, then wait briefly for it to exit cleanly.
    # On clean exit, sessions land with resume_pending=True and the next
    # boot will auto-resume them.
    pid = get_running_pid()
    stop_pids = _collect_gateway_stop_pids(pid)
    drained = False
    if pid is not None:
        drained = _drain_gateway_pid(pid, _windows_stop_drain_timeout())

    stopped_any = drained
    if is_task_registered():
        code, _out, err = _exec_schtasks(["/End", "/TN", get_task_name()])
        # schtasks returns nonzero when the task isn't currently running — don't treat that as an error.
        if code == 0:
            stopped_any = True
        elif "not running" not in (err or "").lower():
            print(f"⚠ schtasks /End returned code {code}: {err.strip()}")

    # Phase 3: hard-kill any still-known gateway processes. Avoid the generic
    # process sweep here: Windows direct-spawn starts are profile-scoped, and a
    # stop command must be bounded even if the scanner or shutdown path is wedged.
    stop_pids.extend(pid for pid in _collect_gateway_stop_pids() if pid not in stop_pids)
    killed = _force_terminate_known_gateway_pids(stop_pids)
    if killed:
        stopped_any = True
        print(f"✓ Killed {killed} gateway process(es)")
    if stopped_any:
        if drained:
            print("✓ Gateway stopped (drained cleanly)")
        else:
            print("✓ Gateway stopped")
    else:
        print("✗ No gateway was running")


def _wait_for_gateway_absent(timeout_s: float = 30.0, interval_s: float = 0.5) -> bool:
    """Block until no gateway process is detectable, or the timeout elapses.

    ``stop()`` can return while the previous gateway is still draining
    in-flight agents (the drain runs up to the restart-drain timeout). Uses the
    authoritative ``get_running_pid()`` (lock + liveness + start-time +
    gateway-shape) plus the now-strict ``_gateway_pids()`` scan so a relaunch
    never races a still-alive old process.
    """
    from gateway.status import get_running_pid

    deadline = time.monotonic() + max(timeout_s, interval_s)
    while time.monotonic() < deadline:
        if get_running_pid() is None and not _gateway_pids():
            return True
        time.sleep(interval_s)
    return get_running_pid() is None and not _gateway_pids()


def restart() -> None:
    """Stop the gateway then start it again.

    Waits for the old gateway to be authoritatively gone before relaunching --
    otherwise ``start()``'s "already running" guard sees the still-draining old
    process and no-ops, and when that process later exits nothing replaces it (a
    silent outage). Fails loudly if the process can't be cleared or the relaunch
    doesn't produce a running gateway.
    """
    _assert_windows()

    stop()

    if not _wait_for_gateway_absent(timeout_s=30.0):
        print("⚠ Gateway still present after stop; forcing termination before restart...")
        _force_terminate_known_gateway_pids(_collect_gateway_stop_pids())
        if not _wait_for_gateway_absent(timeout_s=10.0):
            raise RuntimeError(
                "Gateway process still detected after force kill; refusing to "
                "start a duplicate. Investigate stray PIDs before retrying."
            )

    # Give Windows a moment to release the listening port.
    time.sleep(1.0)
    start()

    if not _wait_for_gateway_ready(timeout_s=15.0):
        raise RuntimeError(
            "Gateway restart did not produce a running gateway process. "
            "Check logs/gateway.log and run `hermes gateway status`."
        )
