"""Windows gateway service backend (Scheduled Task + Startup-folder fallback).

Mirrors the ``launchd_*`` / ``systemd_*`` contract. ``schtasks /Create ... /RL LIMITED`` runs at the
CURRENT USER's next logon without elevation. Manual starts and ``install --start-now`` use the direct
hidden-console launcher instead of ``schtasks /Run`` so start/restart behavior is consistent.
"""

from __future__ import annotations

import ctypes
import json
import locale
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree
from xml.sax.saxutils import escape

from hermes_cli._subprocess_compat import (
    _WINDOWS_GATEWAY_BREAKAWAY_ENV,
    windows_detach_flags,
    windows_detach_flags_without_breakaway,
    windows_hide_flags,
)

logger = logging.getLogger(__name__)

# Short timeouts: schtasks occasionally wedges and we don't want to hang forever.
_SCHTASKS_TIMEOUT_S = 15
# Patterns in schtasks stderr that mean "fall back to the Startup folder".
# schtasks' localized "access is denied" (en/es/cs/zh-Hans/zh-Hant/ja/ko): one vocabulary for both
# the elevated-install offer and the Startup-folder fallback.
_ACCESS_DENIED_WORDS = (
    r"access is denied|acceso denegado|přístup byl odepřen|拒绝访问|拒絕存取|アクセスが拒否されました|"
    r"액세스가 거부되었습니다"
)
_FALLBACK_PATTERNS = re.compile(
    rf"({_ACCESS_DENIED_WORDS}|schtasks timed out|schtasks produced no output)", re.IGNORECASE
)
_ACCESS_DENIED_PATTERN = re.compile(rf"({_ACCESS_DENIED_WORDS})", re.IGNORECASE)

# Set by _spawn_detached() when the breakaway spawn failed and it retried WITHOUT
# CREATE_BREAKAWAY_FROM_JOB — the child stays in the parent's Job Object and may be killed when this
# shell exits. Dict (not bare bool) so the flag is mutable without ``global``.
_LAST_SPAWN_BREAKAWAY_FALLBACK: dict = {"fallback": False}

_TASK_NAME_DEFAULT = "Hermes_Gateway"
_TASK_DESCRIPTION = "Hermes Agent Gateway - Messaging Platform Integration"
_TASK_LOGON_DELAY = "PT30S"
_TASK_RESTART_INTERVAL = "PT1M"
_TASK_RESTART_COUNT = 999

_GATEWAY_ENV = (("PYTHONIOENCODING", "utf-8"), ("HERMES_GATEWAY_DETACHED", "1"), ("HERMES_SUPERVISED_CHILD", "1"))


def _schtasks_encoding() -> str:
    """Console encoding for ``schtasks.exe`` output: localized Windows emits the OEM/ANSI code page,
    not UTF-8, and decoding with the wrong codec raised UnicodeDecodeError in subprocess' reader
    threads. Prefer the locale's preferred encoding, fall back to UTF-8."""
    try:
        return locale.getpreferredencoding(False) or "utf-8"
    except Exception:
        return "utf-8"


def _windows_console_encodings() -> list[str]:
    """Code pages a console tool such as ``schtasks.exe`` writes to a pipe, most likely first: the
    console output code page (65001 once ``configure_windows_stdio`` ran, else the OEM page), the OEM
    page, then the ANSI page. Read from kernel32, so Python's UTF-8 mode cannot disguise them."""
    kernel32 = getattr(getattr(ctypes, "windll", None), "kernel32", None)
    if kernel32 is None:
        return [_schtasks_encoding()]
    pages: list[str] = []
    for getter in ("GetConsoleOutputCP", "GetOEMCP", "GetACP"):
        try:
            code_page = int(getattr(kernel32, getter)())
        except (AttributeError, OSError, ValueError):
            continue
        if code_page > 0 and f"cp{code_page}" not in pages:
            pages.append(f"cp{code_page}")
    return pages or [_schtasks_encoding()]


def _decode_schtasks_output(data: bytes) -> str:
    """Decode captured ``schtasks.exe`` bytes. schtasks writes the console/OEM code page even when this
    process runs in UTF-8 mode (the launcher sets ``PYTHONUTF8=1``), so ``locale.getpreferredencoding``
    is the wrong codec on a non-ASCII account path: ``C:\\Users\\方舟`` came back as ``����`` and the
    Scheduled-Task drift check could never settle (#116193). Strict UTF-8 first (ASCII and genuine
    UTF-8 output pass; legacy multi-byte text fails loudly), then the native code pages, then a lossy
    fallback so a reader thread never raises."""
    for encoding in ("utf-8", *_windows_console_encodings()):
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _assert_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError("gateway_windows is Windows-only")


def _hermes_home() -> Path:
    from hermes_cli.config import get_hermes_home

    return Path(get_hermes_home())


def hermes_service_roots() -> tuple[str, ...]:
    """Directories a Hermes-owned SCM service binary lives under: the checkout (its ``venv`` included),
    the running interpreter's ``Scripts`` dir (``hermes.exe`` shim) and the ``gateway-service`` launcher dir."""
    project_root = Path(__file__).resolve().parent.parent
    return (str(project_root), str(Path(sys.executable).parent), str(_hermes_home() / "gateway-service"))


def _normalize_windows_path(value: str) -> str:
    return value.strip().lstrip('"').replace("\\", "/").rstrip("/").casefold()


def hermes_owns_windows_service(name: str, binpath: str, hermes_roots: tuple[str, ...]) -> bool:
    """Positive ownership of an SCM service: Hermes-named (``hermes*``) or its binary path starts under a
    Hermes root. Pure so it is testable off-Windows. A Scheduled-Task-launched gateway descends from
    ``svchost.exe`` hosting ``Schedule``; without this gate the updater took Task Scheduler for the
    gateway's supervisor and ``sc.exe stop Schedule`` aborted every update (#97208)."""
    normalized_name = "".join(char for char in name.casefold() if char.isalnum())
    if normalized_name.startswith("hermes"):
        return True
    candidate = _normalize_windows_path(binpath)
    return any(candidate.startswith(_normalize_windows_path(root) + "/") for root in hermes_roots if root)


def _preserve_hermes_home_path(path: str | Path) -> str:
    r"""Render Hermes-owned paths under the configured HERMES_HOME spelling.

    ``%LOCALAPPDATA%\hermes`` may be a symlink/junction to another drive; launcher files must not
    bake in the resolved target for paths under HERMES_HOME.
    """
    candidate = Path(path)
    try:
        home = _hermes_home()
        resolved_home = home.resolve()
        resolved_candidate = candidate.resolve()
        home_key = os.path.normcase(str(resolved_home))
        candidate_key = os.path.normcase(str(resolved_candidate))
        if os.path.commonpath([home_key, candidate_key]) == home_key:
            return str(home / os.path.relpath(str(resolved_candidate), str(resolved_home)))
    except Exception:
        pass
    return str(candidate)


# ── Quoting helpers. cmd.exe (.cmd body), VBScript literals and schtasks /TR are three DIFFERENT
# parsers — never reuse one helper for another. The task XML path avoids /TR quoting entirely.

def _quote_cmd_script_arg(value: str) -> str:
    """Quote one argument INSIDE a .cmd file for cmd.exe: split on spaces/tabs outside double quotes,
    embedded quotes doubled. Line breaks are refused — they'd end the logical command line."""
    if "\r" in value or "\n" in value:
        raise ValueError(f"refusing to quote value containing newline: {value!r}")
    if not value:
        return '""'
    if not re.search(r'[ \t"]', value):
        return value
    return '"' + value.replace('"', '""') + '"'


def _quote_vbs_string(value: str) -> str:
    """VBScript double-quoted literal (embedded quote doubled; newline refused)."""
    if "\r" in value or "\n" in value:
        raise ValueError(f"refusing to quote VBScript value containing newline: {value!r}")
    return '"' + value.replace('"', '""') + '"'


# ── schtasks.exe wrapper

def _exec_schtasks(args: list[str]) -> tuple[int, str, str]:
    """Run ``schtasks.exe`` with a hard timeout. Return (code, stdout, stderr); a wedge returns
    code=124 with a synthetic stderr so the fallback regex matches."""
    _assert_windows()
    schtasks = shutil.which("schtasks")
    if schtasks is None:
        return (1, "", "schtasks.exe not found on PATH")
    try:
        # Bytes, decoded by _decode_schtasks_output: a non-UTF-8 status line must never surface a
        # UnicodeDecodeError from subprocess' reader threads. CREATE_NO_WINDOW: no flashing console under a TUI.
        proc = subprocess.run(
            [schtasks, *args], capture_output=True, text=False,
            timeout=_SCHTASKS_TIMEOUT_S, creationflags=windows_hide_flags(),
        )
        return (
            proc.returncode,
            _decode_schtasks_output(proc.stdout or b""),
            _decode_schtasks_output(proc.stderr or b""),
        )
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
    """Launch an elevated gateway subcommand via UAC and return True on handoff. The child is console
    ``python.exe`` with ``SW_HIDE``: it owns one hidden console its subprocesses (schtasks, taskkill)
    inherit — no visible window and no per-descendant conhost flashes (the console-less pythonw.exe
    alternative re-created #54220/#56747 for every descendant).

    All operator decisions are already collected in the parent shell before this point. See #54220, #56747.
    """
    _assert_windows()
    args = ["-m", "hermes_cli.main", *_current_profile_cli_args(), "gateway", command, *(extra_args or [])]
    params = subprocess.list2cmdline(args)
    cwd = str(Path(__file__).resolve().parent.parent)
    try:
        result = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, cwd, 0)  # 0 = SW_HIDE
    except Exception as exc:
        print(f"⚠ Could not launch elevated gateway {command} prompt: {exc}")
        return False
    if result <= 32:
        print(f"⚠ Elevated gateway {command} prompt was not started (ShellExecuteW={result})")
        return False
    return True


def _launch_elevated_install(force: bool = False, *, start_now: bool | None = None, start_on_login: bool | None = None) -> bool:
    """Launch an elevated gateway install via UAC and return True on handoff."""
    overrides = {"HERMES_GATEWAY_ELEVATED_HANDOFF": "1"}
    extra_args = ["--elevated-handoff"]
    if force:
        extra_args.append("--force")
    for choice, env_key, flag in (
        (start_now, "HERMES_GATEWAY_INSTALL_START_NOW", "start-now"),
        (start_on_login, "HERMES_GATEWAY_INSTALL_START_ON_LOGIN", "start-on-login"),
    ):
        if choice is not None:
            overrides[env_key] = "1" if choice else "0"
            extra_args.append(f"--{flag}" if choice else f"--no-{flag}")
    saved = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        return _launch_elevated_gateway_command("install", extra_args)
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


# ── Paths: where we stash our task script and where Startup lives

def get_task_name(home: str | Path | None = None) -> str:
    """Scheduled Task name for *home* (or the active profile when omitted)."""
    _assert_windows()
    from hermes_cli.gateway import _profile_suffix  # local: avoids circular init during boot

    suffix = _profile_suffix(home)
    return f"{_TASK_NAME_DEFAULT}_{suffix}" if suffix else _TASK_NAME_DEFAULT


def _sanitize_filename(value: str) -> str:
    """Remove characters illegal in Windows filenames."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)


def get_task_script_path(home: Path | None = None) -> Path:
    """The generated ``gateway.cmd`` wrapper under ``<HERMES_HOME>/gateway-service/`` (per-profile
    installs stay self-contained); the VBS launcher lives beside it."""
    _assert_windows()
    script_dir = (home if home is not None else _hermes_home()) / "gateway-service"
    script_dir.mkdir(parents=True, exist_ok=True)
    task_name = get_task_name(home) if home is not None else get_task_name()
    return script_dir / f"{_sanitize_filename(task_name)}.cmd"


def _startup_dir() -> Path:
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    userprofile = os.environ.get("USERPROFILE", "").strip() or os.environ.get("HOME", "").strip()
    if not userprofile:
        raise RuntimeError("neither APPDATA nor USERPROFILE is set — cannot resolve Startup folder")
    return Path(userprofile).joinpath("AppData", "Roaming", "Microsoft", "Windows", "Start Menu", "Programs", "Startup")


def get_startup_entry_path(home: Path | None = None) -> Path:
    _assert_windows()
    task_name = get_task_name(home) if home is not None else get_task_name()
    return _startup_dir() / f"{_sanitize_filename(task_name)}.vbs"


def _legacy_startup_entry_path(home: Path | None = None) -> Path:
    _assert_windows()
    task_name = get_task_name(home) if home is not None else get_task_name()
    return _startup_dir() / f"{_sanitize_filename(task_name)}.cmd"


def _startup_staging_path() -> Path:
    """The Startup-folder staging file; also the debris a pre-fix failed swap left behind (#114093)."""
    return get_startup_entry_path().with_suffix(".tmp")


def _stable_gateway_working_dir(project_root: Path) -> str:
    """Stable cwd for detached/startup runs: anchor at HERMES_HOME when it exists (mirrors the POSIX
    service invariant) so a moved checkout/worktree can't fail the ``cd`` step; else the checkout."""
    from hermes_cli.config import get_hermes_home

    try:
        home = get_hermes_home()
        if home and Path(home).is_dir():
            return str(Path(home))
    except Exception:
        pass
    return str(project_root)


# ── Script rendering

def _gateway_run_argv(python_exe: str, profile_arg: str) -> list[str]:
    """``python -m hermes_cli.main [--profile X] gateway run`` — shared by every launcher renderer."""
    argv = [python_exe, "-m", "hermes_cli.main"]
    if profile_arg:
        argv.extend(profile_arg.split())
    argv.extend(["gateway", "run"])
    return argv


def _launcher_settings(home: Path | None = None) -> tuple[str, str, str, str]:
    """Return (python_path, working_dir, hermes_home, profile_arg) for generated launchers.
    ``home`` targets another profile's HERMES_HOME (per-profile cold-start, #110959)."""
    from hermes_cli.gateway import PROJECT_ROOT, _profile_arg, get_python_path  # avoid circular init

    hermes_home = str(home if home is not None else _hermes_home())
    return (
        _preserve_hermes_home_path(get_python_path()),
        _stable_gateway_working_dir(PROJECT_ROOT),
        hermes_home,
        _profile_arg(hermes_home),
    )


def _launcher_pythonpath_entries(extra_pythonpath: list[str]) -> list[str]:
    return [
        _preserve_hermes_home_path(Path(__file__).resolve().parent.parent),
        *[_preserve_hermes_home_path(entry) for entry in extra_pythonpath],
    ]


def _build_gateway_cmd_script(python_path: str, working_dir: str, hermes_home: str, profile_arg: str) -> str:
    """Build the ``gateway.cmd`` wrapper (CRLF-terminated). No PATH overrides (rewriting PATH breaks
    Homebrew/nvm-style installs), no ``start`` (extra wrapper process muddles lifecycle/status), no
    ``--replace`` (repeated /Run calls must be idempotent, not takeover loops)."""
    python_exe_path, venv_dir, extra_pythonpath = _resolve_detached_python(python_path)
    pythonpath = ";".join([*_launcher_pythonpath_entries(extra_pythonpath), "%PYTHONPATH%"])
    lines = [
        "@echo off",
        f"rem {_TASK_DESCRIPTION}",
        f"cd /d {_quote_cmd_script_arg(working_dir)}",
        f'set "HERMES_HOME={hermes_home}"',
        *[f'set "{k}={v}"' for k, v in _GATEWAY_ENV],
        # VIRTUAL_ENV lets the gateway's own python detection find the venv.
        f'set "VIRTUAL_ENV={_preserve_hermes_home_path(venv_dir)}"',
        f'set "PYTHONPATH={pythonpath}"',
        " ".join(_quote_cmd_script_arg(a) for a in _gateway_run_argv(python_exe_path, profile_arg)),
        "exit /b 0",
    ]
    return "\r\n".join(lines) + "\r\n"


def _build_gateway_vbs_script(python_path: str, working_dir: str, hermes_home: str, profile_arg: str) -> str:
    """Build the hidden-console ``gateway.vbs`` launcher (CRLF-terminated).

    Run via ``wscript.exe``, not ``cmd.exe``: at logon Windows broadcasts CTRL_CLOSE_EVENT to console
    groups, killing a cmd-hosted gateway with STATUS_CONTROL_C_EXIT, which Task Scheduler treats as a
    user cancel (``RestartOnFailure`` never fires). wscript has no console; python.exe runs with window
    style 0 so descendants inherit one hidden console instead of flashing their own (#54220/#56747).

    Why: issue #45599 root cause #1.
    ``wscript.exe`` is a GUI-subsystem executable with no console, so this launcher receives no console
    control events. It ``Run``s the console ``python.exe`` with window style 0 (hidden): the gateway owns a
    single hidden console — never shown, never CTRL_CLOSE'd at logon, and inherited by every
    console-subsystem descendant (git, gh, node, …) so none of them allocate a visible flashing conhost
    (#54220/#56747; the previous console-less pythonw.exe gateway forced exactly that per-descendant flash).
    No cmd.exe anywhere in the chain. Mirrors ``_build_gateway_cmd_script`` (same env + argv via
    ``_resolve_detached_python``).
    """
    python_exe_path, venv_dir, extra_pythonpath = _resolve_detached_python(python_path)
    # list2cmdline gives CreateProcess-correct quoting for WScript.Shell.Run.
    command_line = subprocess.list2cmdline(_gateway_run_argv(python_exe_path, profile_arg))
    static_pythonpath = os.pathsep.join(_launcher_pythonpath_entries(extra_pythonpath))
    q = _quote_vbs_string
    lines = [
        f"' {_TASK_DESCRIPTION}",
        "Option Explicit",
        "Dim sh, env, existing_pp",
        'Set sh = CreateObject("WScript.Shell")',
        'Set env = sh.Environment("PROCESS")',
        f"env.Item({q('HERMES_HOME')}) = {q(hermes_home)}",
        *[f"env.Item({q(k)}) = {q(v)}" for k, v in _GATEWAY_ENV],
        f"env.Item({q('VIRTUAL_ENV')}) = {q(_preserve_hermes_home_path(venv_dir))}",
        # Mirror the cmd wrapper's ``PYTHONPATH=<static>;%PYTHONPATH%`` at runtime.
        f"existing_pp = env.Item({q('PYTHONPATH')})",
        "If Len(existing_pp) > 0 Then",
        f"  env.Item({q('PYTHONPATH')}) = {q(static_pythonpath + os.pathsep)} & existing_pp",
        "Else",
        f"  env.Item({q('PYTHONPATH')}) = {q(static_pythonpath)}",
        "End If",
        f"sh.CurrentDirectory = {q(working_dir)}",
        # Window style 0 = hidden; bWaitOnReturn False = detached/async.
        f"sh.Run {q(command_line)}, 0, False",
    ]
    return "\r\n".join(lines) + "\r\n"


def _build_startup_launcher(script_path: Path) -> str:
    """The tiny Startup-folder .vbs that chains hidden. Quits silently if the target is gone so a
    stale entry doesn't error on every login."""
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


def _write_task_script(home: Path | None = None) -> Path:
    """Generate the gateway.cmd wrapper (kept as a compatibility artifact) and the console-less .vbs
    launcher used by the Scheduled Task and Startup fallback. Return the .cmd path."""
    _assert_windows()
    settings = _launcher_settings(home)
    script_path = get_task_script_path(home)
    _atomic_write(script_path, _build_gateway_cmd_script(*settings), script_path.with_suffix(".tmp"))
    # Also render the console-less .vbs launcher used by Scheduled Task and the Startup-folder fallback via
    # wscript.exe (issue #45599 fix A). The .cmd wrapper stays as a generated helper/compatibility artifact.
    vbs_path = script_path.with_suffix(".vbs")
    _atomic_write(vbs_path, _build_gateway_vbs_script(*settings), vbs_path.with_name(vbs_path.name + ".tmp"))
    return script_path


def _atomic_write(path: Path, content: str, tmp: Path) -> None:
    """Write ``content`` verbatim (no newline translation) via ``tmp`` then rename over ``path``.

    The staging file is removed even when the rename fails: the Startup-folder caller stages
    inside the Startup folder itself, and Windows opens every file there at login — a leftover
    ``Hermes_Gateway.tmp`` pops up in Notepad after every sign-in (#114093).
    """
    try:
        tmp.write_text(content, encoding="utf-8", newline="")
        tmp.replace(path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# ── Install / uninstall

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
    """Task Scheduler XML with safe long-running defaults. ``launcher_path`` is the console-less
    ``.vbs`` run via ``wscript.exe`` (see ``_build_gateway_vbs_script`` for why not cmd.exe).

    See #45599.
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


def _install_scheduled_task(task_name: str, script_path: Path) -> tuple[bool, str]:
    """Create or replace the Scheduled Task. Returns (success, detail). Always delete+create, never
    ``/Change``: it preserves stale repeat/restart settings that relaunch the gateway every minute."""
    delete_code, delete_out, delete_err = _exec_schtasks(["/Delete", "/F", "/TN", task_name])
    delete_detail = (delete_err or delete_out or "").strip()
    if "cannot find" in delete_detail.lower():
        delete_detail = ""
    if delete_code != 0 and delete_detail and _is_access_denied(delete_detail):
        return (False, f"schtasks /Delete failed (code {delete_code}): {delete_detail}")
    # Other /Delete failures are non-fatal: /Create /F may still replace it; keep the detail.
    user = _resolve_task_user()
    launcher_path = script_path.with_suffix(".vbs")   # the task launches the console-less .vbs
    xml_path = launcher_path.with_suffix(".task.xml")
    xml_path.write_text(_build_scheduled_task_xml(task_name, launcher_path, user), encoding="utf-16", newline="")
    # Immediate manual starts use _spawn_detached(). See #45599.
    base = ["/Create", "/F", "/TN", task_name, "/XML", str(xml_path)]
    variants = [[*base, "/RU", user, "/NP", "/IT"], base] if user else [base]
    last_code, last_err = 1, ""
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
    if delete_detail:
        last_err = f"{last_err.strip()} (delete detail: {delete_detail})"
    return (False, f"schtasks /Create failed (code {last_code}): {last_err.strip()}")


def _install_startup_entry(script_path: Path) -> Path:
    """Write the Startup-folder fallback launcher. Returns its path."""
    entry = get_startup_entry_path()
    entry.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(entry, _build_startup_launcher(script_path), _startup_staging_path())
    legacy_entry = _legacy_startup_entry_path()
    try:
        if legacy_entry.exists():
            legacy_entry.unlink()
    except OSError:
        pass
    return entry


def _resolve_detached_python(python_exe: str) -> tuple[str, Path, list[str]]:
    """Return (hidden_console_python, venv_dir, extra_pythonpath) for detached runs. ``extra_pythonpath``
    is always empty now; the tuple shape is kept so every call site stays unchanged.

    Returns the venv's **console** ``python.exe`` — deliberately NOT ``pythonw.exe``. Every detached launch
    path pairs this interpreter with a hidden-console mechanism (``CREATE_NO_WINDOW`` creationflags, or
    ``WScript.Shell.Run`` window style 0), so the daemon owns a single hidden console that all of its
    console-subsystem descendants (git, gh, cmd, node, wmic, powershell, …) inherit instead of each
    allocating a visible flashing one. A GUI-subsystem ``pythonw.exe`` daemon has NO console, which is what
    made every descendant spawn flash (#54220/#56747) and forced the endless per-call-site CREATE_NO_WINDOW
    sweep. Root cause isolated + A/B verified on Windows 11 by the desktop backend fix (commit aa2ae36c3f).
    - uv venv launcher: ``venv\\Scripts\\python.exe`` under ``CREATE_NO_WINDOW`` re-execs the base
    interpreter *windowless* — the child inherits the shim's hidden console, so no conhost flashes (the
    #52239 concern). The historical "CREATE_NO_WINDOW cannot suppress the second window" observations were
    made while ``DETACHED_PROCESS`` was in the flag bundle, where MSDN specifies CREATE_NO_WINDOW is IGNORED
    — the hide bit was dead, not ineffective. The base-interpreter + PYTHONPATH-overlay detour is therefore
    unnecessary; the venv shim resolves imports itself. - Console python restores stdout/stderr, so daemon
    logs flow normally.
    Legacy normalization: launchers and argv snapshots from pre-aa2ae36c3f installs lead with
    ``pythonw.exe``. When the sibling console ``python.exe`` exists, swap to it so respawns and regenerated
    launchers get the hidden-console design instead of resurrecting the console-less daemon (the
    #54220/#56747 flash class, plus the ``sys.stderr is None`` startup-crash class from #71671).
    """
    p = Path(python_exe)
    if p.name.lower() in ("pythonw.exe", "pythonw"):
        sibling = p.with_name("python.exe" if p.suffix else "python")
        try:
            if sibling.exists():
                p = sibling
                python_exe = str(sibling)
        except OSError:
            # Can't stat the sibling — keep the original interpreter: a console-less gateway is
            # worse than a hidden-console one, but a failed respawn is worse still.
            pass
    return (python_exe, p.parent.parent, [])


def _prepend_pythonpath(env_overlay: dict[str, str], entries: list[str]) -> None:
    clean_entries = [entry for entry in entries if entry]
    if not clean_entries:
        return
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        clean_entries.append(existing)
    env_overlay["PYTHONPATH"] = os.pathsep.join(clean_entries)


def _build_gateway_argv(home: Path | None = None) -> tuple[list[str], str, dict[str, str]]:
    """Build (argv, working_dir, env_overlay) for the gateway subprocess — the same logical command
    as gateway.cmd, assembled as a native argv so no cmd.exe layer sits in between."""
    _assert_windows()
    from hermes_cli.gateway import PROJECT_ROOT

    python_path, working_dir, hermes_home, profile_arg = _launcher_settings(home)
    python_exe, venv_dir, extra_pythonpath = _resolve_detached_python(python_path)
    env_overlay = {"HERMES_HOME": hermes_home, **dict(_GATEWAY_ENV), "VIRTUAL_ENV": _preserve_hermes_home_path(venv_dir)}
    _prepend_pythonpath(env_overlay, [_preserve_hermes_home_path(p) for p in (PROJECT_ROOT, *extra_pythonpath)])
    return _gateway_run_argv(python_exe, profile_arg), working_dir, env_overlay


def windowless_gateway_restart_spec(run_argv: list[str]) -> tuple[list[str], str, dict[str, str]]:
    """(argv, cwd, env overlay) for a hidden-console gateway respawn; arguments after the interpreter
    are preserved verbatim. Non-Windows or a non-python argv[0] → argv unchanged, empty overlay.

    The post-update restart paths build their respawn command from ``get_python_path()`` (the venv's console
    ``python.exe``). That is the right interpreter: the watcher launches it with ``CREATE_NO_WINDOW`` detach
    flags, so the respawned gateway owns a single hidden console that all of its descendants inherit —
    nothing flashes (#54220/#56747; the old pythonw.exe rewrite here produced a console-less gateway whose
    every console-subsystem child allocated a visible conhost). This helper now only normalizes the
    interpreter via ``_resolve_detached_python`` and supplies the stable cwd + env overlay (HERMES_HOME,
    VIRTUAL_ENV, PYTHONPATH) so the respawn doesn't depend on the watcher's transient working directory.
    """
    if not run_argv or sys.platform != "win32":
        return run_argv, "", {}
    from hermes_cli.gateway import PROJECT_ROOT

    try:
        hidden_console_python, venv_dir, extra_pythonpath = _resolve_detached_python(run_argv[0])
    except Exception:
        return run_argv, "", {}

    try:
        hermes_home = str(_hermes_home().resolve())
    except Exception:
        hermes_home = ""
    env_overlay: dict[str, str] = {"PYTHONIOENCODING": "utf-8", "HERMES_GATEWAY_DETACHED": "1", "VIRTUAL_ENV": str(venv_dir)}
    if hermes_home:
        env_overlay["HERMES_HOME"] = hermes_home
    _prepend_pythonpath(env_overlay, [str(PROJECT_ROOT), *extra_pythonpath])
    return [hidden_console_python, *run_argv[1:]], _stable_gateway_working_dir(PROJECT_ROOT), env_overlay


def _spawn_detached(script_path: Path | None = None, home: Path | None = None) -> int:
    """Launch the gateway as a fully detached background process (``script_path`` is ignored; kept
    for API symmetry; ``home`` spawns another profile's gateway — ``--profile`` is derived from it). Spawns python.exe directly — a cmd.exe shim inherits the parent console and
    gets reaped when the shell exits. Flags: CREATE_NEW_PROCESS_GROUP (no Ctrl+C from our group),
    CREATE_NO_WINDOW (hidden console descendants inherit, so nothing flashes — #54220/#56747; the old
    DETACHED_PROCESS made every descendant spawn flash), CREATE_BREAKAWAY_FROM_JOB
    (escape a parent Job Object — some Windows Terminal versions wrap children in one).

    With ``CREATE_NO_WINDOW`` the gateway gets its OWN hidden console instead of inheriting ours, so it
    survives our shell closing, and every console-subsystem descendant it spawns inherits that hidden
    console instead of flashing a visible one (#54220/#56747 — this is why we don't use console-less
    pythonw.exe here). Combined with CREATE_NEW_PROCESS_GROUP + DEVNULL stdin + a fresh env, the resulting
    process is independent of whichever shell started it.
    """
    _assert_windows()
    argv, working_dir, env_overlay = _build_gateway_argv(home)
    env = {**os.environ, **env_overlay}

    # Stray print()/native stderr goes to a sidecar log; real gateway logs still land in gateway.log
    # via the logging FileHandler.
    log_dir = _hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stray_log = log_dir / "gateway-stdio.log"

    def _popen(breakaway: str, flags: int):
        with open(stray_log, "ab", buffering=0) as log_fh:
            return subprocess.Popen(
                argv, cwd=working_dir, env={**env, _WINDOWS_GATEWAY_BREAKAWAY_ENV: breakaway}, creationflags=flags,
                close_fds=True, stdin=subprocess.DEVNULL, stdout=log_fh, stderr=log_fh,
            )

    try:
        proc = _popen("1", windows_detach_flags())
        _LAST_SPAWN_BREAKAWAY_FALLBACK["fallback"] = False
    except OSError as exc:
        # CREATE_BREAKAWAY_FROM_JOB fails with "access denied" when the parent's job object forbids
        # breakaway (some Windows Terminal configs). Retry without it — the hidden-console
        # CREATE_NO_WINDOW spawn is usually enough on its own.
        error_code = getattr(exc, "winerror", None)
        if error_code is None:
            error_code = exc.errno
        logger.warning("Gateway breakaway spawn failed (error=%s); retrying without CREATE_BREAKAWAY_FROM_JOB", error_code)
        proc = _popen("0", windows_detach_flags_without_breakaway())
        _LAST_SPAWN_BREAKAWAY_FALLBACK["fallback"] = True
    return proc.pid


def _stdin_is_interactive(*, isatty: bool, console_mode_ok: bool | None) -> bool:
    """A human can answer a prompt only on a real console. The Windows CRT reports isatty()==True for
    every character device — the NUL device included (`hermes gateway start < NUL`, stdin=DEVNULL) — so
    isatty must be confirmed by GetConsoleMode accepting the handle (#113977). ``console_mode_ok`` is
    None where that fact does not exist (not Windows) and isatty alone decides."""
    return isatty and console_mode_ok is not False


def _stdin_console_mode_ok() -> bool | None:
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
    return bool(kernel32.GetConsoleMode(handle, ctypes.byref(ctypes.c_ulong())))


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


def _prompt_install_choices(start_now: bool | None = None, start_on_login: bool | None = None) -> tuple[bool, bool]:
    """Return (start_now, start_on_login), asking before any UAC escalation."""
    if start_now is None:
        start_now = _install_choice_from_env("HERMES_GATEWAY_INSTALL_START_NOW")
    if start_on_login is None:
        start_on_login = _install_choice_from_env("HERMES_GATEWAY_INSTALL_START_ON_LOGIN")
    if start_now is not None and start_on_login is not None:
        return start_now, start_on_login

    from hermes_cli.setup import prompt_yes_no

    if start_now is None:
        start_now = prompt_yes_no("Start the gateway now after install?", True)
    if start_on_login is None:
        start_on_login = prompt_yes_no("Start the gateway automatically on Windows login with a Scheduled Task?", True)
    return start_now, start_on_login


def _report_already_running(running_pids: list[int]) -> None:
    print(f"✓ Gateway already running (PID: {', '.join(map(str, running_pids))})")


def _start_or_report_running(running_pids: list[int] | None = None) -> None:
    """Spawn the gateway unless one is already running for this profile."""
    if running_pids is None:
        running_pids = _gateway_pids()
    if running_pids:
        _report_already_running(running_pids)
    else:
        pid = _spawn_detached()
        _report_gateway_start("direct spawn")


def _install_startup_fallback(script_path: Path, start_now: bool, detail: str) -> None:
    """Install the Startup-folder fallback and optionally start once."""
    print(f"↻ Scheduled Task install blocked ({detail.splitlines()[0]}) — using Startup folder fallback")
    entry = _install_startup_entry(script_path)
    print(f"✓ Installed Windows login item: {entry}")
    print(f"  Task script: {script_path}")

    # Re-running install must be safe: the fallback only installs login persistence; starting is
    # controlled by the pre-UAC start_now answer so every user decision precedes elevation.
    running_pids = _gateway_pids()
    if running_pids or start_now:
        _start_or_report_running(running_pids)
    else:
        from hermes_cli.gateway import _profile_arg

        profile_arg = _profile_arg()
        start_cmd = f"hermes {profile_arg} gateway start" if profile_arg else "hermes gateway start"
        print("ℹ Startup fallback installed; gateway not started now.")
        print(f"  Start manually with: {start_cmd}")
    _print_next_steps()


def _offer_elevated_install(headline: str, force: bool, start_now: bool, start_on_login: bool) -> bool:
    """Offer the UAC prompt for a Scheduled Task install. True when handed off to an elevated child."""
    from hermes_cli.setup import prompt_yes_no

    print(headline)
    print("  UAC is Windows' admin approval prompt; it is needed to create/update the Scheduled Task.")
    if prompt_yes_no("  Open the UAC prompt now?", False):
        if _launch_elevated_install(force=force, start_now=start_now, start_on_login=start_on_login):
            print("✓ Launched elevated Hermes gateway install prompt.")
            if start_now:
                print("  Approve the Windows UAC prompt; the elevated install will start the gateway afterwards.")
            else:
                print("  Approve the Windows UAC prompt, then run: hermes gateway status")
            return True
        print("⚠ Falling back to Startup folder because elevation was unavailable or cancelled.")
    else:
        print("  Skipped elevation. Falling back to Startup folder.")
    return False


def install(
    force: bool = False, *, start_now: bool | None = None, start_on_login: bool | None = None,
    elevated_handoff: bool = False,
) -> None:
    """Install the gateway as a Windows Scheduled Task (with Startup fallback). Idempotent — we
    always reconcile; ``force`` exists for API parity with launchd/systemd."""
    _assert_windows()
    start_now, start_on_login = _prompt_install_choices(start_now, start_on_login)

    if not start_on_login:
        print("ℹ Skipped Windows login auto-start install.")
        if start_now:
            _start_or_report_running()
        else:
            print("ℹ Gateway not started and no auto-start service installed.")
            print("  Run in the foreground later with: hermes gateway run")
        return

    task_name = get_task_name()
    script_path = _write_task_script()
    # A pre-fix install that failed its Startup-folder swap left `Hermes_Gateway.tmp` there, and the
    # Scheduled Task path below never touches that folder — sweep it so a re-run clears the debris.
    try:
        _startup_staging_path().unlink(missing_ok=True)
    except OSError:
        pass
    if force:
        # Pre-suffix strays (task ``Hermes_Gateway``, Startup ``Hermes_Gateway.vbs``) are unreachable by
        # the current names, so a plain reconcile never heals them (#116157).
        from hermes_cli.gateway_windows_legacy import remove_legacy_launchers
        remove_legacy_launchers()

    # On locked-down accounts schtasks can sit for the full timeout before returning Access Denied.
    # All intent questions were asked above, so ask for UAC before touching schtasks.
    if not _is_running_as_admin() and not elevated_handoff:
        if _offer_elevated_install(
            "↻ Scheduled Task install may need administrator approval on this Windows account.",
            force, start_now, start_on_login,
        ):
            return
        _install_startup_fallback(script_path, start_now, "administrator approval was not used")
        return

    ok, detail = _install_scheduled_task(task_name, script_path)
    if ok:
        print(f"✓ {detail}")
        print(f"  Task script: {script_path}")
        print("ℹ Gateway auto-start installed for Windows login.")
        if start_now:
            _start_or_report_running()
        else:
            print("ℹ Gateway not started now.")
            print("  Start manually with: hermes gateway start")
        _print_next_steps()
        return

    # Prefer a real Scheduled Task over the Startup fallback when elevation is the only blocker.
    if _is_access_denied(detail) and not _is_running_as_admin() and _offer_elevated_install(
        f"↻ Scheduled Task install needs administrator approval ({detail.splitlines()[0]})",
        force, start_now, start_on_login,
    ):
        return

    if _should_fall_back(1, detail):
        _install_startup_fallback(script_path, start_now, detail)
        return

    raise RuntimeError(f"Windows gateway install failed: {detail}")


def _live_gateway_pids(all_profiles: bool = False, home: Path | None = None) -> list[int]:
    """Live gateway PIDs for the readiness poll. ``home`` scopes the probe to ONE profile's identity
    files (a still-running sibling must not vouch for a per-profile spawn, #110959); otherwise the
    process-table discovery for the active profile or the whole fleet."""
    if home is not None:
        from gateway.status import get_running_pid

        pid = get_running_pid(home / "gateway.pid", cleanup_stale=False)
        return [pid] if pid else []
    from hermes_cli.gateway import find_gateway_pids

    return list(find_gateway_pids(all_profiles=all_profiles))


def _confirm_gateway_stable(
    initial_pids: list[int], confirm_s: float, interval_s: float, all_profiles: bool = False, home: Path | None = None,
) -> list[int]:
    """Re-check a freshly detected gateway for ``confirm_s`` seconds: one process-table hit proves
    the child was *created*, not that it survived startup (or a parent Job Object teardown).

    A single process-table hit only proves the child was *created*, not that it survived startup — a gateway
    that crashes moments after spawn (or is reaped by the parent shell's Job Object, #91675/#84185) passes a
    first-hit poll and then dies. Require the gateway to stay visible for the whole confirmation window
    before we vouch for it. Returns the last observed PID list, or ``[]`` if the gateway vanished
    mid-window.
    """
    if confirm_s <= 0:
        return initial_pids
    pids = initial_pids
    confirm_deadline = time.monotonic() + confirm_s
    while time.monotonic() < confirm_deadline:
        time.sleep(interval_s)
        pids = _live_gateway_pids(all_profiles, home)
        if not pids:
            return []
    return pids


def _wait_for_gateway_ready(
    timeout_s: float = 6.0, interval_s: float = 0.4, confirm_s: float = 2.0, all_profiles: bool = False,
    home: Path | None = None,
) -> list[int]:
    """Poll for a live gateway for up to ``timeout_s``; a first hit is provisional until the gateway
    stays visible for ``confirm_s`` more seconds (a child that dies right after spawn earns no ✓)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pids = _live_gateway_pids(all_profiles, home)
        if pids:
            confirmed = _confirm_gateway_stable(pids, confirm_s, interval_s, all_profiles=all_profiles, home=home)
            if confirmed:
                return confirmed
            continue  # died during confirmation — keep polling until deadline
        time.sleep(interval_s)
    return []


# ---------------------------------------------------------------------------
# Start attestation — honest reporting for deaths AFTER the liveness poll
#
# The poll cannot observe a death after this CLI process exits (the parent shell's Job Object tears
# the gateway down on CLI exit). So every ✓ persists a marker recording which PIDs we vouched for;
# the NEXT gateway CLI invocation checks it: PIDs gone with no clean exit in the lifecycle ledger
# means the earlier ✓ was a lie, and we say so — once — with the schtasks recovery hint.
# ---------------------------------------------------------------------------

_START_ATTESTATION_RELATIVE = ("state", "gateway.start-attestation.json")
_UNMAPPED_START_ATTESTATION_RELATIVE = ("state", "gateway.unmapped-start-attestation.json")


def _start_attestation_path(home: Path | None = None) -> Path:
    """Marker path; ``home`` addresses another profile's marker (per-profile cold-start, #110959)."""
    return (home if home is not None else _hermes_home()).joinpath(*_START_ATTESTATION_RELATIVE)


def _write_start_attestation(pids: list[int], via: str, home: Path | None = None) -> None:
    """Persist the PIDs a ✓ vouched for. Best-effort, never raises.

    ``generation`` identifies this marker instance: the update resume token records the generation
    whose death authorized a cold-start, so execution consumes exactly that marker and never a
    newer one written by a concurrent ``hermes gateway start`` (#110020 review)."""
    try:
        path = _start_attestation_path(home)
        path.parent.mkdir(parents=True, exist_ok=True)
        from hermes_cli.process_identity import _process_create_time

        payload = {
            "pids": [int(p) for p in pids], "via": via, "ts": datetime.now(timezone.utc).isoformat(),
            "generation": uuid.uuid4().hex,
        }
        # Bind each PID to its incarnation (#110020 review): the ledger sentinel is matched by PID
        # only otherwise, so a stale marker would be re-read against whatever lifecycle wrote last.
        create_times = {str(int(p)): _process_create_time(int(p)) for p in pids}
        payload["create_times"] = {k: v for k, v in create_times.items() if v is not None}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        logger.debug("Failed to write gateway start attestation", exc_info=True)


def _write_unmapped_start_attestation(pids: list[int], via: str) -> None:
    """Persist a diagnostic without assigning an unmapped gateway a profile."""
    try:
        path = _hermes_home().joinpath(*_UNMAPPED_START_ATTESTATION_RELATIVE)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pids": [int(p) for p in pids], "via": via,
            "ts": datetime.now(timezone.utc).isoformat(), "generation": uuid.uuid4().hex,
            "identity": "unmapped",
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        logger.debug("Failed to write unmapped gateway start attestation", exc_info=True)


def _unmapped_start_attestation_path() -> Path:
    """A neutral marker for a process that could not be assigned a profile/home."""
    return _hermes_home().joinpath(*_UNMAPPED_START_ATTESTATION_RELATIVE)


def _read_unmapped_start_attestation() -> object | None:
    try:
        return json.loads(_unmapped_start_attestation_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _clear_unmapped_start_attestation() -> None:
    try:
        _unmapped_start_attestation_path().unlink(missing_ok=True)
    except OSError:
        pass


def _clear_start_attestation(home: Path | None = None) -> None:
    try:
        _start_attestation_path(home).unlink(missing_ok=True)
    except OSError:
        pass


# A start attestation older than this is no authority (#110020 review (d)): the marker is a one-shot
# meant to bridge the seconds between a ✓ and the next ``hermes gateway status``/``update``; a
# historical marker must never later override Desktop ownership into a duplicate gateway (#76129).
START_ATTESTATION_MAX_AGE_S = 24 * 3600
# Same slack process_identity uses for psutil create_time comparisons (PID reuse disambiguation).
_CREATE_TIME_TOLERANCE_S = 2.0
# A backwards clock step (NTP) between write and read must not kill a fresh marker.
_ATTESTATION_CLOCK_SLACK_S = 60.0


def _attestation_within_horizon(data: object) -> bool:
    """False for a marker whose ``ts`` is missing, unparsable or older than the horizon (fail closed)."""
    try:
        ts = datetime.fromisoformat(str(data["ts"])) if isinstance(data, dict) else None
        if ts is None:
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = time.time() - ts.timestamp()
        return -_ATTESTATION_CLOCK_SLACK_S <= age <= START_ATTESTATION_MAX_AGE_S
    except Exception:
        return False


def _attestation_generation(data: object) -> str | None:
    """The marker instance identity, or ``None`` for a marker that carries none."""
    return str(data["generation"]) if isinstance(data, dict) and data.get("generation") else None


def _consume_start_attestation(generation: str, home: Path | None = None) -> None:
    """Clear the marker only while it is still the ``generation`` that was acted on; a newer
    marker belongs to a gateway start this caller knows nothing about and keeps its own report."""
    # Best-effort read-then-unlink: a marker written in between loses one post-start report, never authority.
    if _attestation_generation(_read_start_attestation(home)) == generation:
        _clear_start_attestation(home)


def _read_start_attestation(home: Path | None = None) -> object | None:
    """Parsed attestation payload (any JSON type), or ``None`` when absent/unreadable. Never raises."""
    try:
        return json.loads(_start_attestation_path(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _attested_pids_from(data: object) -> list[int]:
    """PID list from an attestation payload; empty for anything malformed."""
    if not isinstance(data, dict):
        return []
    pids = data.get("pids")
    # Fail closed: a null/malformed marker must never authorize a cold start (or raise on iteration).
    # Exact positive ints only — ``isinstance(True, int)`` holds, and 0 / negatives are not PIDs; one
    # bad item taints the whole list because the writer never emits such values.
    if not isinstance(pids, list) or not all(type(p) is int and p > 0 for p in pids):
        return []
    return list(pids)


def _attested_create_time(data: object, pid: int) -> float | None:
    """The process create time the marker bound ``pid`` to, or ``None`` (older marker / psutil silent)."""
    times = data.get("create_times") if isinstance(data, dict) else None
    value = times.get(str(pid)) if isinstance(times, dict) else None
    return float(value) if type(value) in (int, float) else None


def _attested_pid_exited_cleanly(pid: int, create_time: float | None = None, home: Path | None = None) -> bool:
    """True when the lifecycle ledger shows a clean exit for ``pid`` — or, for a marker that bound
    ``pid`` to a ``create_time``, whenever the sentinel cannot be shown to describe THAT incarnation
    (#110020 review): a sentinel for another PID or another start time means an unrelated lifecycle
    has run since and the marker is stale; "unknown" must never read as "dead". A missing sentinel
    still reads as dead (the attested process never booted far enough to claim it)."""
    try:
        from gateway.lifecycle_ledger import get_lifecycle_sentinel_path

        sentinel = get_lifecycle_sentinel_path(home if home is not None else _hermes_home())
        data = json.loads(sentinel.read_text(encoding="utf-8"))
    except OSError:
        return False
    except Exception:
        return create_time is not None
    if not isinstance(data, dict):
        return create_time is not None
    if create_time is not None:
        if data.get("pid") != pid:
            return True
        sentinel_birth = data.get("create_time")
        # A sentinel from a gateway older than the identity stamp cannot be told apart: PID-only rule.
        if type(sentinel_birth) in (int, float) and abs(float(sentinel_birth) - create_time) > _CREATE_TIME_TOLERANCE_S:
            return True
    return data.get("phase") == "exited" and data.get("pid") == pid


def _attested_dead(
    attested: list[int], current_pids: list[int], data: object = None, home: Path | None = None
) -> bool:
    """The liveness rule shared by the consuming and read-only probes: attested PIDs are dead when
    no gateway runs now and the lifecycle ledger shows no clean exit for any of them."""
    return not current_pids and not any(
        _attested_pid_exited_cleanly(pid, _attested_create_time(data, pid), home) for pid in attested
    )


def attested_death_generation(current_pids: list[int], home: Path | None = None) -> str | None:
    """The generation of a start attestation that vouches for gateway PID(s) gone without a clean exit,
    or ``None``.

    Read-only twin of :func:`check_start_attestation` for callers that must not consume the
    one-shot marker — ``hermes update`` consults it to decide whether a Desktop-owned install
    still owes a gateway cold-start (#109538) and records the generation in its resume token so the
    execution step consumes exactly the marker it was authorized by. Callers pass the liveness they
    already established (``[]`` after their own discovery came back empty) so the process table is
    not scanned twice. ``None`` for anything undecidable (no marker, no generation, a clean ledger
    exit): "unknown" must never read as "dead". ``home`` probes another profile's marker and ledger
    (the updater evaluates every profile that is not running, #110959)."""
    data = _read_start_attestation(home)
    attested = _attested_pids_from(data)
    if not attested or not _attestation_within_horizon(data) or not _attested_dead(attested, current_pids, data, home):
        return None
    return _attestation_generation(data)


def check_start_attestation(current_pids: list[int] | None = None) -> str | None:
    """Surface (once) a gateway that died after a ✓ was printed for it. Never raises. Gateway running
    or a clean-exit ledger record: clear silently; otherwise return a warning and consume the marker."""
    data = _read_start_attestation()
    if data is None:
        return None
    attested = _attested_pids_from(data)
    if not attested:
        _clear_start_attestation()
        return None

    if current_pids is None:
        try:
            from hermes_cli.gateway import find_gateway_pids

            current_pids = list(find_gateway_pids())
        except Exception:
            return None

    _clear_start_attestation()
    if not _attested_dead(attested, current_pids, data):
        return None
    return _format_attestation_warning(attested, data)


def check_unmapped_start_attestation(current_pids: list[int] | None = None) -> str | None:
    """Surface a one-shot diagnostic for a previously verified unmapped gateway.

    This intentionally does not use a profile's ledger or task hint: an
    unmapped process has no trustworthy home and must not be represented as
    the active/default profile.  It is diagnostic-only, unlike profile
    attestations which can participate in a cold-start plan.
    """
    data = _read_unmapped_start_attestation()
    if not isinstance(data, dict) or data.get("identity") != "unmapped":
        if data is not None:
            _clear_unmapped_start_attestation()
        return None
    attested = _attested_pids_from(data)
    if not attested or not _attestation_within_horizon(data):
        _clear_unmapped_start_attestation()
        return None
    if current_pids is None:
        try:
            from hermes_cli.gateway import find_gateway_pids

            current_pids = list(find_gateway_pids(all_profiles=True))
        except Exception:
            return None
    _clear_unmapped_start_attestation()
    if current_pids:
        return None
    via = data.get("via") or "post-update recovery"
    ts = data.get("ts") or "unknown time"
    return (
        f"⚠ A previously verified unmapped gateway ({via}, {ts}) is no longer running "
        f"(PID {', '.join(map(str, attested))}). Its profile could not be determined."
    )


def _format_attestation_warning(attested: list[int], data: dict) -> str:
    via = data.get("via") or "direct spawn"
    ts = data.get("ts") or "unknown time"
    lines = [
        f"⚠ The previous gateway start ({via}, {ts}) reported success, but the "
        f"process (PID {', '.join(map(str, attested))}) died without a clean "
        "shutdown record.",
        "  This usually means the shell that ran `hermes gateway start` was inside "
        "a Windows Job Object that killed the gateway on exit (#91675).",
    ]
    hint = _task_run_hint("  Recovery: schtasks /Run /TN {}   (Task Scheduler starts the gateway outside any Job Object)")
    if hint:
        lines.append(hint)
    return "\n".join(lines)


def _task_run_hint(fmt: str) -> str | None:
    """``fmt`` with the task name filled in, when a Scheduled Task is registered. Never raises."""
    try:
        if is_task_registered():
            return fmt.format(get_task_name())
    except Exception:
        pass
    return None


def _print_task_run_hint(fmt: str) -> None:
    hint = _task_run_hint(fmt)
    if hint:
        print(hint)


def _print_start_attestation_warning() -> None:
    """Print stale profile/unmapped attestation diagnostics once. Never raises."""
    try:
        warning = check_start_attestation()
    except Exception:
        warning = None
    if warning:
        print(warning)
    try:
        warning = check_unmapped_start_attestation()
    except Exception:
        return
    if warning:
        print(warning)


def _report_gateway_start(via: str) -> None:
    pids = _wait_for_gateway_ready()
    if pids:
        print(f"✓ Gateway started via {via} (PID: {', '.join(map(str, pids))})")
        if _LAST_SPAWN_BREAKAWAY_FALLBACK.get("fallback"):
            print("⚠ The gateway could not break away from this shell's Job Object; it may be killed when this shell exits.")
            _print_task_run_hint("  If it dies, start it with: schtasks /Run /TN {}")
        _write_start_attestation(pids, via)
    else:
        print(f"✗ Gateway start via {via} FAILED — no stable gateway process detected within the verification window.")
        print("  (The process may have been created and then killed — e.g. by a parent Job Object, #91675.)")
        print(f"  Check the log for startup errors:\n    type {_hermes_home()}\\logs\\gateway.log\n    type {_hermes_home()}\\logs\\gateway-stdio.log")
        _print_task_run_hint("  Recovery: schtasks /Run /TN {}   (starts the gateway outside any Job Object)")


def _print_next_steps() -> None:
    print("\nNext steps:\n  hermes gateway status                      # Check status")
    print(f"  type {_hermes_home()}\\logs\\gateway.log       # View logs")


def uninstall() -> None:
    """Remove both the Scheduled Task and the Startup-folder fallback, if present."""
    _assert_windows()
    task_name = get_task_name()
    script_path = get_task_script_path()

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
                if _launch_elevated_gateway_command("uninstall"):
                    print("✓ Launched elevated Hermes gateway uninstall prompt.")
                    print("  Approve the Windows UAC prompt, then run: hermes gateway status")
                    return
                print("⚠ Elevated uninstall prompt was unavailable or cancelled.")
            else:
                print("  Skipped elevation. Scheduled Task was not removed.")
        else:
            print(f"⚠ schtasks /Delete returned code {code}: {detail}")

    for path, label in (
        (get_startup_entry_path(), "Windows login item"), (_legacy_startup_entry_path(), "legacy Windows login item"),
        (_startup_staging_path(), "Windows login item staging file"),
        (script_path, "Task script"), (script_path.with_suffix(".vbs"), "Task launcher"),
    ):
        try:
            path.unlink()
            print(f"✓ Removed {label}: {path}")
        except FileNotFoundError:
            pass

    from hermes_cli.gateway_windows_legacy import remove_legacy_launchers
    remove_legacy_launchers()

    if is_task_registered() and not scheduled_task_removed:
        print(f"⚠ Scheduled Task still registered: {task_name}")


# ── Status / start / stop / restart

def _task_registration_state(*, home: Path | None = None) -> str:
    """``registered`` or conservative ``query-failed`` for a task lookup.

    ``schtasks /Query`` uses the same non-zero code for a missing task and
    several transport failures.  The bool public API remains intentionally
    lossy, but update recovery must never turn an indeterminate lookup into a
    claim that no task exists.
    """
    task_name = get_task_name(home) if home is not None else get_task_name()
    code, _out, _err = _exec_schtasks(["/Query", "/TN", task_name])
    return "registered" if code == 0 else "query-failed"


def is_task_registered(*, home: Path | None = None) -> bool:
    return _task_registration_state(home=home) == "registered"


def _run_scheduled_task_once(*, home: Path | None = None) -> tuple[int, str, str]:
    """One-shot ``schtasks /Run`` for post-update Job Object recovery only.

    The Scheduled Task is login persistence; ordinary ``start()`` / ``restart()``
    stay on ``_spawn_detached``. Only the updater's post-relaunch / cold-start
    recovery may use this to start the gateway outside the parent Job Object
    (#107002 / #48820).
    """
    task_name = get_task_name(home) if home is not None else get_task_name()
    return _exec_schtasks(["/Run", "/TN", task_name])


def is_startup_entry_installed(*, home: Path | None = None) -> bool:
    if home is None:
        return get_startup_entry_path().exists() or _legacy_startup_entry_path().exists()
    return get_startup_entry_path(home).exists() or _legacy_startup_entry_path(home).exists()


def _query_scheduled_task_xml(task_name: str) -> str | None:
    """Return a registered task's XML, or ``None`` when it cannot be inspected (fail open: a
    localized ``schtasks`` failure is not evidence about an otherwise working task)."""
    code, out, err = _exec_schtasks(["/Query", "/TN", task_name, "/XML"])
    if code != 0 or not out.strip():
        logger.debug("Could not query Scheduled Task XML for %r: %s", task_name, (err or out).strip())
        return None
    return out


def _task_xml_leaf_values(xml: str) -> dict[str, str] | None:
    """Namespace-agnostic ``Task/Settings/...`` leaf-path → text map, or ``None`` for invalid XML.
    The root ``version`` attribute is exposed as ``Task@version``."""
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return None
    values: dict[str, str] = {"Task@version": root.attrib.get("version", "")}

    def visit(element: ElementTree.Element, path: tuple[str, ...]) -> None:
        current_path = (*path, element.tag.rsplit("}", 1)[-1])
        children = list(element)
        if not children:
            values["/".join(current_path)] = " ".join((element.text or "").split())
        for child in children:
            visit(child, current_path)

    visit(root, ())
    return values


# Allowlist of template leaves whose absence/mismatch on the live task means it predates the current
# template (#113670). Never a full-leaf compare: schtasks exports <UserId> as a SID while the template
# writes DOMAIN\user, so equality would flag every healthy registration.
_TASK_DRIFT_LEAVES = {
    "Task/Settings/RestartOnFailure/Interval": "RestartOnFailure",
    "Task/Triggers/LogonTrigger/Delay": "LogonTrigger Delay",
    "Task/Actions/Exec/Arguments": "launcher arguments",
}


def compare_scheduled_task_drift(registered_xml: str, template_xml: str) -> list[str]:
    """Human-readable drift fragments between a registered task export and the current template,
    over ``_TASK_DRIFT_LEAVES`` plus the Task ``version``. Empty when aligned or when either side
    does not parse (fail open)."""
    live = _task_xml_leaf_values(registered_xml)
    want = _task_xml_leaf_values(template_xml)
    if live is None or want is None:
        return []
    missing = [label for path, label in _TASK_DRIFT_LEAVES.items() if path in want and path not in live]
    differs = [label for path, label in _TASK_DRIFT_LEAVES.items() if path in want and path in live and live[path] != want[path]]
    drift = []
    if missing:
        drift.append(f"missing: {', '.join(missing)}")
    drift.extend(f"{label} differs" for label in differs)
    if live["Task@version"] != want["Task@version"]:
        drift.append(f"version {live['Task@version']} vs {want['Task@version']}")
    return drift


def _scheduled_task_template(task_name: str, home: Path | None = None) -> str:
    script_path = get_task_script_path(home) if home is not None else get_task_script_path()
    return _build_scheduled_task_xml(
        task_name, script_path.with_suffix(".vbs"), _resolve_task_user()
    )


def _task_action_is_hermes_managed(
    registered_xml: str, template_xml: str, *, task_name: str | None = None
) -> bool:
    """Only replace a task action when it is exactly the managed launcher.

    Drift repair may update Hermes' own template settings, but an action that
    points somewhere else is user-owned and must not be overwritten by update.
    """
    registered = _task_xml_leaf_values(registered_xml)
    template = _task_xml_leaf_values(template_xml)
    if registered is None or template is None:
        return False
    command = str(registered.get("Task/Actions/Exec/Command") or "").casefold()
    arguments = str(registered.get("Task/Actions/Exec/Arguments") or "")
    wanted_arguments = str(template.get("Task/Actions/Exec/Arguments") or "")
    # Only support the two launcher forms Hermes itself has emitted: the
    # older single quoted script, and the current ``//B //Nologo`` form.  A
    # quoted path elsewhere in a custom argument string (for example a
    # ``/target:`` value) is not the script that wscript executes.
    launcher_args = re.compile(
        r'\s*(?://B\s+//Nologo\s+)?"([^"]+)"\s*', re.IGNORECASE
    )
    registered_match = launcher_args.fullmatch(arguments)
    wanted_match = launcher_args.fullmatch(wanted_arguments)
    if command not in {"wscript.exe", "wscript"} or not registered_match or not wanted_match:
        return False
    # Ownership is target-scoped.  A matching task-shaped suffix in another
    # home is insufficient evidence that changing its Action is safe.
    registered_launcher = registered_match.group(1).replace("/", "\\").casefold()
    wanted_launcher = wanted_match.group(1).replace("/", "\\").casefold()
    return registered_launcher == wanted_launcher


def scheduled_task_drift(task_name: str, home: Path | None = None) -> list[str]:
    """Drift fragments between the registered task and ``_build_scheduled_task_xml``; empty when
    aligned or when the task cannot be queried."""
    registered = _query_scheduled_task_xml(task_name)
    if registered is None:
        return []
    template = _scheduled_task_template(task_name, home)
    return compare_scheduled_task_drift(registered, template)


def _print_scheduled_task_drift(task_name: str) -> None:
    """Warn when the registered task predates the current template (status is read-only; the
    repair runs from ``start()`` / ``hermes update`` via ``reconcile_scheduled_task``)."""
    drift = scheduled_task_drift(task_name)
    if drift:
        print(f"⚠ Scheduled Task registration predates the current template ({'; '.join(drift)})")
        print("  Repair: hermes gateway start  (or: hermes gateway install)")


def reconcile_scheduled_task(task_name: str, *, home: Path | None = None) -> bool:
    """Re-register the task from the current template when it drifts (#113670) — the Windows sibling
    of ``gateway.py::refresh_systemd_unit_if_needed``. Template hardening (``RestartOnFailure``, logon
    ``Delay``) otherwise only ever reaches fresh installs. False when aligned/unqueryable or when
    ``schtasks`` refused (typically Access Denied — the elevating ``hermes gateway install`` is the fallback)."""
    registered = _query_scheduled_task_xml(task_name)
    if registered is None:
        return False
    template = _scheduled_task_template(task_name, home)
    drift = compare_scheduled_task_drift(registered, template)
    if not drift:
        return False
    if not _task_action_is_hermes_managed(registered, template, task_name=task_name):
        print("⚠ Scheduled Task action is not Hermes-managed; leaving it unchanged")
        return False
    print(f"↻ Repairing outdated Scheduled Task registration ({'; '.join(drift)})")
    script_path = _write_task_script(home) if home is not None else _write_task_script()
    ok, detail = _install_scheduled_task(task_name, script_path)
    print(f"{'✓' if ok else '⚠'} {detail}")
    if not ok:
        print("  Repair manually: hermes gateway install")
    return ok


def is_installed(*, home: Path | None = None) -> bool:
    """True when either the schtasks entry or the Startup fallback is present."""
    if home is None:
        return is_task_registered() or is_startup_entry_installed()
    return is_task_registered(home=home) or is_startup_entry_installed(home=home)


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
        if key == "last result":
            info.setdefault("last run result", value)
        elif key in {"status", "last run time", "last run result"}:
            info[key] = value
    return info


def _gateway_pids() -> list[int]:
    """Reuse the cross-platform PID scanner in gateway.py."""
    from hermes_cli.gateway import find_gateway_pids

    return list(find_gateway_pids())


def _probe(index: int, ok: bool, message: str) -> None:
    print(f"  [{index}] {'PASS' if ok else 'FAIL':4s}  {message}")


def _probe_missing(index: int, path: Path, label: str) -> bool:
    if path.exists():
        return False
    _probe(index, False, f"{label} missing: {path}")
    return True


def _probe_pid_file(pid_path: Path) -> int | None:
    if _probe_missing(1, pid_path, "PID file"):
        return None
    try:
        data = json.loads(pid_path.read_text(encoding="utf-8"))
        pid_value = int(data.get("pid")) if data.get("pid") is not None else None
        _probe(1, True, f"PID file present: {pid_path} (pid={pid_value})")
        return pid_value
    except Exception as exc:
        _probe(1, False, f"PID file present but unreadable: {exc}")
        return None


def _probe_lock_file(lock_path: Path) -> None:
    if _probe_missing(2, lock_path, "Lock file"):
        return
    try:
        from gateway.status import is_gateway_runtime_lock_active

        _probe(2, is_gateway_runtime_lock_active(lock_path), f"Lock file held by a live process: {lock_path}")
    except Exception as exc:
        _probe(2, False, f"Could not probe lock: {exc}")


def _probe_running_pid() -> int | None:
    try:
        from gateway.status import get_running_pid

        running_pid = get_running_pid(cleanup_stale=False)
        _probe(3, running_pid is not None, f"get_running_pid() => {running_pid}")
        return running_pid
    except Exception as exc:
        _probe(3, False, f"get_running_pid() raised: {exc!r}")
        return None


def _probe_pid_exists(candidate_pid: int | None) -> None:
    if candidate_pid is None:
        _probe(4, False, "No candidate PID to verify")
        return
    try:
        from gateway.status import _pid_exists

        alive = bool(_pid_exists(candidate_pid))
        _probe(4, alive, f"_pid_exists({candidate_pid}) => {alive}")
    except Exception as exc:
        _probe(4, False, f"_pid_exists raised: {exc!r}")


def _probe_state_file(state_path: Path) -> None:
    if _probe_missing(5, state_path, "gateway_state.json"):
        return
    try:
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
        gateway_state = state_data.get("gateway_state")
        updated_at = state_data.get("updated_at")
        age_str = ""
        if updated_at:
            try:
                updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                age_seconds = int((datetime.now(timezone.utc) - updated_dt).total_seconds())
                age_str = f" (updated {age_seconds}s ago)"
            except Exception:
                pass
        _probe(5, gateway_state in ("running", "degraded"), f"gateway_state.json state={gateway_state!r}{age_str}")
    except Exception as exc:
        _probe(5, False, f"gateway_state.json present but unreadable: {exc}")


def _probe_exit_diag(diag_path: Path) -> None:
    if _probe_missing(6, diag_path, "exit-diag log"):
        return
    try:
        with open(diag_path, "rb") as fh:
            fh.seek(0, 2)   # last ~4KB; one event is well under 500 bytes
            size = fh.tell()
            fh.seek(max(0, size - 4096))
            tail = fh.read().decode("utf-8", errors="replace").splitlines()
        last_event = next((ln for ln in reversed(tail) if ln.strip()), "")
        if not last_event:
            _probe(6, False, f"exit-diag log empty: {diag_path}")
            return
        try:
            event = json.loads(last_event)
            tag = event.get("tag", "?")
            _probe(6, tag in ("gateway.start",), f"Last lifecycle event: tag={tag} pid={event.get('pid', '?')} ts={event.get('ts', '?')}")
        except Exception:
            _probe(6, False, f"Last lifecycle line not JSON: {last_event[:120]}")
    except Exception as exc:
        _probe(6, False, f"exit-diag log unreadable: {exc}")


def _print_deep_probes() -> None:
    """Print PASS/FAIL per individual liveness signal, so when the collapsed ✓/✗ summary disagrees
    with reality the user can see exactly which signal is wrong."""
    home = _hermes_home()
    print("\nDeep probes:")
    pid_value = _probe_pid_file(home / "gateway.pid")
    _probe_lock_file(home / "gateway.lock")
    running_pid = _probe_running_pid()
    _probe_pid_exists(running_pid if running_pid is not None else pid_value)
    _probe_state_file(home / "gateway_state.json")
    _probe_exit_diag(home / "logs" / "gateway-exit-diag.log")


def status(deep: bool = False) -> None:
    """Print a status report for the Windows gateway service."""
    _assert_windows()
    _print_start_attestation_warning()   # once: a gateway that died after a previous ✓
    task_name = get_task_name()
    task_installed = is_task_registered()
    startup_installed = is_startup_entry_installed()
    pids = _gateway_pids()

    if task_installed:
        print(f"✓ Scheduled Task registered: {task_name}")
        info = query_task_status()
        for key in ("status", "last run time", "last run result"):
            if key in info:
                print(f"  {key.title()}: {info[key]}")
        _print_scheduled_task_drift(task_name)
    elif startup_installed:
        entry = get_startup_entry_path()
        print(f"✓ Windows login item installed: {entry if entry.exists() else _legacy_startup_entry_path()}")
    else:
        print("✗ Gateway service not installed")
    from hermes_cli.gateway_windows_legacy import warn_legacy_launchers
    warn_legacy_launchers()

    print(f"✓ Gateway process running (PID: {', '.join(map(str, pids))})" if pids else "✗ No gateway process detected")

    if deep:
        print()
        print(f"  Task name:        {task_name}")
        print(f"  Task script:      {get_task_script_path()}")
        print(f"  Startup entry:    {get_startup_entry_path()}")
        _print_deep_probes()

    if not task_installed and not startup_installed and not pids:
        print("\nTo install:\n  hermes gateway install")


def start() -> None:
    """Start the gateway using the canonical detached Windows launch path."""
    _assert_windows()
    _print_start_attestation_warning()   # once: the LAST start's ✓ turned out to be false
    running_pids = _gateway_pids()
    if running_pids:
        _report_already_running(running_pids)
        return

    if not is_task_registered() and not is_startup_entry_installed():
        # Login persistence is a lasting system change: a bare ``start`` installs it only on an explicit
        # answer — the HERMES_GATEWAY_INSTALL_START_ON_LOGIN override or a real TTY prompt — never on a
        # non-TTY default (#113977). Declining still starts the gateway; the command is ``start``.
        start_on_login = _install_choice_from_env("HERMES_GATEWAY_INSTALL_START_ON_LOGIN")
        if start_on_login is None:
            from hermes_cli.setup import is_interactive_stdin, is_noninteractive, prompt_yes_no

            print("✗ Gateway service is not installed")
            if is_noninteractive() or not _stdin_is_interactive(
                isatty=is_interactive_stdin(), console_mode_ok=_stdin_console_mode_ok()
            ):
                start_on_login = False
            else:
                start_on_login = prompt_yes_no("  Install it now so the gateway starts on login?", True)
        if start_on_login:
            # install() starts the gateway itself (start_now) and reports the outcome — including a UAC
            # hand-off to an elevated child — so there is nothing left to spawn or to warn about here.
            install(force=False, start_now=True, start_on_login=True)
            return
        print("ℹ Login auto-start not installed; add it later with: hermes gateway install")
    elif is_task_registered():
        reconcile_scheduled_task(get_task_name())   # like systemd's regenerate-on-stale before a start

    # Manual starts use the same console-less direct spawn as restart() and install --start-now;
    # Scheduled Task / Startup entries are only login persistence. The sole
    # ``schtasks /Run`` path is post-update Job Object recovery in
    # ``update_cmd_windows`` (#107002); do not change ordinary start() to /Run.
    pid = _spawn_detached()
    _report_gateway_start("direct spawn")


def _drain_gateway_pid(pid: int, drain_timeout: float) -> bool:
    """Write the planned-stop marker and wait for the PID to exit. Windows can't deliver POSIX signals
    to an asyncio loop, so the marker is the ONLY way to ask the gateway to drain and persist."""
    if pid <= 0:
        return False
    try:
        from gateway.status import write_planned_stop_marker, _pid_exists
    except ImportError:
        return False

    try:
        write_planned_stop_marker(pid)
    except Exception:
        pass   # best-effort; caller escalates to a hard kill

    deadline = time.monotonic() + max(drain_timeout, 1.0)
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.5)
    return False


def _windows_stop_drain_timeout() -> float:
    """Bounded stop grace period: a real graceful-drain window, but the CLI must never wedge."""
    try:
        from hermes_cli.gateway import _get_restart_drain_timeout

        configured = float(_get_restart_drain_timeout() or 30.0)
    except Exception:
        configured = 30.0
    return max(1.0, min(configured, 30.0))


def _gateway_pid_identities(pids: list[int]) -> dict[int, int | None]:
    """``{pid: start_time}`` fingerprints, captured BEFORE any drain/wait so a later force-kill can
    detect that the PID was recycled meanwhile."""
    try:
        from gateway.status import get_process_start_time
    except ImportError:
        return {pid: None for pid in pids}
    return {pid: get_process_start_time(pid) for pid in pids}


def _force_terminate_known_gateway_pids(identities: dict[int, int | None]) -> int:
    """Force-kill known gateway PIDs without a broad process sweep. ``identities`` maps each PID to
    the start time observed when it was identified as a gateway (``_gateway_pid_identities``);
    ``terminate_pid`` refuses the kill when the live process no longer matches. Re-reading the start
    time here would compare the process with itself and taskkill whatever now owns the PID."""
    try:
        from gateway.status import _pid_exists, terminate_pid
    except ImportError:
        return 0

    own_pid = os.getpid()
    killed = 0
    for pid, expected_start_time in identities.items():
        if pid <= 0 or pid == own_pid:
            continue
        try:
            if not _pid_exists(pid):
                continue
            terminate_pid(pid, force=True, expected_start_time=expected_start_time)
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
    """Stop the gateway: planned-stop marker first so it can drain in-flight agents and persist
    ``resume_pending`` (Windows asyncio can't receive SIGTERM — the marker is our only IPC), then
    ``schtasks /End``, then a bounded hard-kill of known PIDs."""
    _assert_windows()
    from gateway.status import get_running_pid

    # A user-initiated stop is a planned death: don't later report it as a silent crash.
    _clear_start_attestation()

    pid = get_running_pid()
    stop_pids = _collect_gateway_stop_pids(pid)
    # Fingerprint before the drain: the kill below must refuse a PID recycled during the wait.
    identities = _gateway_pid_identities(stop_pids)
    drained = pid is not None and _drain_gateway_pid(pid, _windows_stop_drain_timeout())

    stopped_any = drained
    if is_task_registered():
        code, _out, err = _exec_schtasks(["/End", "/TN", get_task_name()])
        # schtasks returns nonzero when the task isn't currently running — not an error.
        if code == 0:
            stopped_any = True
        elif "not running" not in (err or "").lower():
            print(f"⚠ schtasks /End returned code {code}: {err.strip()}")

    # No generic process sweep: starts are profile-scoped and stop must stay bounded even if wedged.
    late_pids = [pid for pid in _collect_gateway_stop_pids() if pid not in identities]
    identities.update(_gateway_pid_identities(late_pids))
    killed = _force_terminate_known_gateway_pids(identities)
    if killed:
        stopped_any = True
        print(f"✓ Killed {killed} gateway process(es)")
    if stopped_any:
        print("✓ Gateway stopped (drained cleanly)" if drained else "✓ Gateway stopped")
    else:
        print("✗ No gateway was running")


def _wait_for_gateway_absent(timeout_s: float = 30.0, interval_s: float = 0.5) -> bool:
    """Block until no gateway is detectable (authoritative ``get_running_pid()`` plus the strict
    ``_gateway_pids()`` scan) or the timeout elapses, so a relaunch never races a draining process."""
    from gateway.status import get_running_pid

    def _absent() -> bool:
        return get_running_pid() is None and not _gateway_pids()

    deadline = time.monotonic() + max(timeout_s, interval_s)
    while time.monotonic() < deadline:
        if _absent():
            return True
        time.sleep(interval_s)
    return _absent()


def restart() -> None:
    """Stop then start. Waits for the old gateway to be authoritatively gone first; otherwise
    ``start()``'s "already running" guard sees the draining process and no-ops, and nothing
    replaces it when it exits (a silent outage). Fails loudly on either side."""
    _assert_windows()

    stop()

    if not _wait_for_gateway_absent(timeout_s=30.0):
        print("⚠ Gateway still present after stop; forcing termination before restart...")
        _force_terminate_known_gateway_pids(_gateway_pid_identities(_collect_gateway_stop_pids()))
        if not _wait_for_gateway_absent(timeout_s=10.0):
            raise RuntimeError(
                "Gateway process still detected after force kill; refusing to "
                "start a duplicate. Investigate stray PIDs before retrying."
            )

    from hermes_cli.gateway import _wait_for_api_server_port_free  # avoid circular init

    _wait_for_api_server_port_free()
    start()

    if not _wait_for_gateway_ready(timeout_s=15.0):
        raise RuntimeError(
            "Gateway restart did not produce a running gateway process. "
            "Check logs/gateway.log and run `hermes gateway status`."
        )
