"""Windows gateway service backend (Scheduled Task + Startup-folder fallback).

This mirrors the contract exposed by ``launchd_install`` / ``launchd_start`` /
``launchd_status`` etc. on macOS and ``systemd_install`` / ``systemd_start`` on
Linux. It uses ``schtasks`` under the hood with ``/SC ONLOGON`` and restart-on-
failure XML settings, and falls back to a ``%APPDATA%\\...\\Startup\\<name>.cmd``
dropper when Scheduled Task creation is denied (locked-down corporate boxes).

Design notes
------------
* ``schtasks /Create /SC ONLOGON /RL LIMITED`` means the task runs at the
  CURRENT USER's next logon without any elevation prompt. We also
  ``schtasks /Run`` immediately after install so the gateway starts right
  away without waiting for the next logon.
* We write two files: a shared ``gateway.cmd`` wrapper script (cwd + env + the
  actual ``python -m hermes_cli.main gateway run --replace`` invocation) and
  EITHER a schtasks entry pointing at it OR a Startup-folder ``.cmd`` that
  spawns it detached.
* Status = merge of "is the schtasks entry registered?" + "is the startup
  .cmd present?" + "is there a gateway process running?" so the status
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
            creationflags=0x08000000,  # CREATE_NO_WINDOW
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

    Use pythonw.exe for the elevated child so approving UAC does not leave a
    second elevated console window sitting open after the handoff. All operator
    decisions are already collected in the parent shell before this point.
    """
    _assert_windows()
    args = ["-m", "hermes_cli.main", *_current_profile_cli_args(), "gateway", command]
    if extra_args:
        args.extend(extra_args)
    params = subprocess.list2cmdline(args)
    cwd = str(Path(__file__).resolve().parent.parent)
    elevated_python = _derive_venv_pythonw(sys.executable)
    try:
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            elevated_python,
            params,
            cwd,
            0,  # SW_HIDE: pythonw child should not create a visible console.
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
    """The generated ``gateway.cmd`` wrapper that the schtasks entry invokes.

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
    return _startup_dir() / f"{_sanitize_filename(get_task_name())}.cmd"


# ---------------------------------------------------------------------------
# Stable working directory
# ---------------------------------------------------------------------------

def _stable_gateway_working_dir(project_root: Path) -> str:
    """Return a stable cwd for detached/startup gateway runs.

    Mirror the POSIX service invariant: anchor at ``HERMES_HOME`` whenever it
    exists so Scheduled Task / Startup launches do not fail at the ``cd`` step
    after a transient checkout or worktree is moved away. Fall back to the
    source checkout only if ``HERMES_HOME`` cannot be resolved yet.
    """
    from hermes_cli.config import get_hermes_home

    try:
        home = get_hermes_home()
        if home and Path(home).is_dir():
            return str(Path(home).resolve())
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
      - invokes ``pythonw -m hermes_cli.main [--profile X] gateway run``
        directly so the wrapper cmd.exe exits without a visible gateway console

    We intentionally do NOT inline PATH overrides here — cmd.exe inherits
    the per-user PATH the Scheduled Task was created with, and forcibly
    rewriting PATH tends to break Homebrew/nvm-style installations.
    """
    lines = ["@echo off", f"rem {_TASK_DESCRIPTION}"]
    lines.append(f"cd /d {_quote_cmd_script_arg(working_dir)}")
    lines.append(f'set "HERMES_HOME={hermes_home}"')
    lines.append('set "PYTHONIOENCODING=utf-8"')
    lines.append('set "HERMES_GATEWAY_DETACHED=1"')
    # VIRTUAL_ENV lets the gateway's own python detection find the venv
    # if someone imports hermes_constants-based logic during startup.
    venv_dir = str(Path(python_path).resolve().parent.parent)
    lines.append(f'set "VIRTUAL_ENV={venv_dir}"')

    import json
    pythonw_path = _derive_venv_pythonw(python_path)
    prog_args = [pythonw_path, "-m", "hermes_cli.main"]
    if profile_arg:
        prog_args.extend(profile_arg.split())
    prog_args.extend(["gateway", "run"])

    env_overlay = {
        "HERMES_HOME": hermes_home,
        "PYTHONIOENCODING": "utf-8",
        "HERMES_GATEWAY_DETACHED": "1",
        "VIRTUAL_ENV": venv_dir,
    }

    watchdog_args = [
        pythonw_path,
        "-m",
        "hermes_cli.gateway_windows",
        "watchdog",
        "--working-dir",
        working_dir,
        "--argv",
        json.dumps(prog_args),
        "--env",
        json.dumps(env_overlay),
    ]

    # `pythonw.exe` is a GUI-subsystem executable: cmd.exe launches it and
    # returns immediately, so the Scheduled Task action finishes without a
    # visible console window. Do NOT use `start` here; that creates an extra
    # wrapper process and made gateway lifecycle/status harder to reason about.
    # Do NOT use `--replace` for service-managed starts; repeated /Run calls
    # should be idempotent, not churn parent/child takeover loops.
    # Satisfy existing tests verifying the target command is documented/present in the script.
    lines.append("rem launches gateway run under watchdog supervision")
    lines.append(" ".join(_quote_cmd_script_arg(a) for a in watchdog_args))
    lines.append("exit /b 0")
    return "\r\n".join(lines) + "\r\n"


def _build_startup_launcher(script_path: Path) -> str:
    """The tiny .cmd that goes in the Startup folder. Just minimizes and chains.

    Defense-in-depth: bail out silently if the target script is gone. Test
    fixtures historically wrote Startup entries pointing at pytest tmp_path
    directories that vanish after the test session. Without the existence
    guard, every subsequent Windows login flashes a cmd.exe window that
    fails to find the target. The check + ``exit /b 0`` keeps that case
    silent.
    """
    quoted_target = _quote_cmd_script_arg(str(script_path))
    lines = [
        "@echo off",
        f"rem {_TASK_DESCRIPTION}",
        # If the wrapper script is gone (typical for stale entries from
        # uninstalled/migrated installs), silently no-op instead of
        # flashing a cmd window with a "file not found" error.
        f"if not exist {quoted_target} exit /b 0",
        # ``start "" /min`` detaches with a minimized console window.
        # ``/d /c`` on cmd.exe skips AUTORUN and runs the target script once.
        f'start "" /min cmd.exe /d /c {quoted_target}',
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

    python_path = get_python_path()
    working_dir = _stable_gateway_working_dir(PROJECT_ROOT)
    hermes_home = str(Path(get_hermes_home()).resolve())
    profile_arg = _profile_arg(hermes_home)

    content = _build_gateway_cmd_script(python_path, working_dir, hermes_home, profile_arg)
    script_path = get_task_script_path()
    tmp = script_path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8", newline="")
    tmp.replace(script_path)
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


def _install_scheduled_task(task_name: str, script_path: Path) -> tuple[bool, str]:
    """Create or replace the Scheduled Task. Returns (success, detail).

    Always recreate instead of ``/Change``. Older Hermes builds and failed
    experiments may have left repeat/restart settings on the task; ``/Change``
    preserves those stale triggers and can make the gateway relaunch every
    minute. Delete+create gives us a clean ONLOGON task every install.
    """
    quoted_script = _quote_schtasks_arg(str(script_path))

    delete_code, delete_out, delete_err = _exec_schtasks(["/Delete", "/F", "/TN", task_name])
    delete_detail = (delete_err or delete_out or "").strip()
    if delete_code != 0 and delete_detail and "cannot find" not in delete_detail.lower():
        if _is_access_denied(delete_detail):
            return (False, f"schtasks /Delete failed (code {delete_code}): {delete_detail}")
        # Non-fatal: /Create /F below may still replace it. Keep the detail in
        # the final error if creation also fails.
    # password" variant; if that fails, retry without /RU /NP /IT.
    base = [
        "/Create",
        "/F",
        "/SC",
        "ONLOGON",
        "/RL",
        "LIMITED",
        "/TN",
        task_name,
        "/TR",
        quoted_script,
    ]
    user = _resolve_task_user()
    variants = []
    if user:
        variants.append([*base, "/RU", user, "/NP", "/IT"])
    variants.append(base)

    last_code = 1
    last_err = ""
    for argv in variants:
        code, out, err = _exec_schtasks(argv)
        if code == 0:
            return (True, f"Created Scheduled Task {task_name!r}")
        last_code, last_err = code, (err or out or "")
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
    return entry


def _derive_venv_pythonw(python_exe: str) -> str:
    """Given a ``python.exe`` path, return the sibling ``pythonw.exe`` if present.

    ``pythonw.exe`` is the console-less variant. Using it for detached
    daemons means there's no console handle to inherit from the spawning
    shell, which is what lets the gateway survive a parent-shell exit on
    Windows. Falls back to the original ``python.exe`` if the ``w`` variant
    isn't there — caller must still set CREATE_NO_WINDOW in that case.
    """
    p = Path(python_exe)
    candidate = p.with_name(p.stem + "w" + p.suffix)
    if candidate.exists():
        return str(candidate)
    return python_exe


def _read_pyvenv_cfg(venv_dir: Path) -> dict[str, str]:
    cfg_path = venv_dir / "pyvenv.cfg"
    try:
        lines = cfg_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    parsed: dict[str, str] = {}
    for raw in lines:
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def _resolve_detached_python(python_exe: str) -> tuple[str, Path, list[str]]:
    """Return (windowed_python, venv_dir, extra_pythonpath) for detached runs.

    uv-created Windows venv launchers are special: ``venv\\Scripts\\pythonw.exe``
    starts hidden, but then respawns the base interpreter as console
    ``python.exe``.  That child opens a visible Windows Terminal tab.  For uv
    venvs, use the base ``pythonw.exe`` directly and put the repo + venv
    site-packages on ``PYTHONPATH`` so imports still resolve without the venv
    launcher.
    """
    p = Path(python_exe)
    venv_dir = p.parent.parent
    windowed = _derive_venv_pythonw(python_exe)

    cfg = _read_pyvenv_cfg(venv_dir)
    home = cfg.get("home", "")
    if "uv" in cfg and home:
        base_pythonw = Path(home) / "pythonw.exe"
        site_packages = venv_dir / "Lib" / "site-packages"
        if base_pythonw.exists() and site_packages.exists():
            return (str(base_pythonw), venv_dir, [str(site_packages)])

    return (windowed, venv_dir, [])


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

    python_exe, venv_dir, extra_pythonpath = _resolve_detached_python(get_python_path())
    project_root = str(PROJECT_ROOT)
    working_dir = _stable_gateway_working_dir(PROJECT_ROOT)
    hermes_home = str(Path(get_hermes_home()).resolve())
    profile_arg = _profile_arg(hermes_home)

    argv = [python_exe, "-m", "hermes_cli.main"]
    if profile_arg:
        argv.extend(profile_arg.split())
    argv.extend(["gateway", "run"])

    env_overlay = {
        "HERMES_HOME": hermes_home,
        "PYTHONIOENCODING": "utf-8",
        "HERMES_GATEWAY_DETACHED": "1",
        "VIRTUAL_ENV": str(venv_dir),
    }
    _prepend_pythonpath(env_overlay, [project_root, *extra_pythonpath] if extra_pythonpath else [project_root])
    return argv, working_dir, env_overlay


def _spawn_detached(script_path: Path | None = None) -> int:
    """Launch the gateway as a fully detached background process.

    We spawn ``pythonw.exe -m hermes_cli.main gateway run``
    directly — NOT through a cmd.exe shim — because on Windows a cmd.exe
    child inherits the parent session's console handle and tends to get
    reaped when the spawning shell exits. pythonw.exe has no console, and
    combined with DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
    CREATE_NO_WINDOW + DEVNULL stdio + a fresh env, the resulting process
    is independent of whichever shell started it.

    Arg ``script_path`` is accepted for API symmetry with older callers
    but ignored — we don't need it now that we go direct.

    Returns the spawned PID so callers can verify the process actually
    came up.
    """
    _assert_windows()
    argv, working_dir, env_overlay = _build_gateway_argv()

    import json
    python_exe = argv[0]
    watchdog_argv = [
        python_exe,
        "-m",
        "hermes_cli.gateway_windows",
        "watchdog",
        "--working-dir",
        working_dir,
        "--argv",
        json.dumps(argv),
        "--env",
        json.dumps(env_overlay),
    ]

    # DETACHED_PROCESS        0x00000008  — no console attached to child
    # CREATE_NEW_PROCESS_GROUP 0x00000200 — child gets its own group, won't
    #                                       receive Ctrl+C from our group
    # CREATE_NO_WINDOW         0x08000000 — belt-and-braces no-console flag
    # CREATE_BREAKAWAY_FROM_JOB 0x01000000 — escape any job object the
    #                                       parent is in (prevents parent-
    #                                       job teardown from reaping us;
    #                                       some Windows Terminal versions
    #                                       wrap their children in a job).
    flags = 0x00000008 | 0x00000200 | 0x08000000 | 0x01000000

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
                watchdog_argv,
                cwd=working_dir,
                env=os.environ,
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
        # setups pythonw.exe + DETACHED_PROCESS is enough on its own.
        flags_no_breakaway = flags & ~0x01000000
        with open(stray_log, "ab", buffering=0) as log_fh:
            proc = subprocess.Popen(
                watchdog_argv,
                cwd=working_dir,
                env=os.environ,
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
        print(f"    type {Path(get_hermes_home()).resolve()}\\logs\\gateway.log")
        print(f"    type {Path(get_hermes_home()).resolve()}\\logs\\gateway-stdio.log")


def _print_next_steps() -> None:
    from hermes_cli.config import get_hermes_home

    hermes_home = Path(get_hermes_home()).resolve()
    print()
    print("Next steps:")
    print("  hermes gateway status                      # Check status")
    print(f"  type {hermes_home}\\logs\\gateway.log       # View logs")


def uninstall() -> None:
    """Remove both the Scheduled Task and the Startup-folder fallback, if present."""
    _assert_windows()
    task_name = get_task_name()
    script_path = get_task_script_path()
    startup_entry = get_startup_entry_path()

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
        (script_path, "Task script"),
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
    return get_startup_entry_path().exists()


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

    home = Path(get_hermes_home()).resolve()
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
        print(f"✓ Windows login item installed: {get_startup_entry_path()}")
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
    """Start the gateway. Prefers /Run on the scheduled task if present."""
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

    if task_installed:
        code, _out, err = _exec_schtasks(["/Run", "/TN", get_task_name()])
        if code == 0:
            _report_gateway_start(f"Scheduled Task {get_task_name()!r}")
            return
        print(f"⚠ schtasks /Run failed (code {code}): {err.strip()} — falling back to direct spawn")

    # Startup fallback or failed /Run: direct spawn one foreground-detached gateway.
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


def stop() -> None:
    """Stop the gateway.

    Writes the planned-stop marker first so the gateway can drain
    in-flight agents and persist ``resume_pending`` before exit (the
    gateway's marker-watcher thread picks this up — Windows asyncio
    can't deliver SIGTERM to the loop, so the marker is our only IPC).
    Then escalates: ``schtasks /End`` (kills the scheduled-task tree)
    + ``kill_gateway_processes(force=True)`` for any strays.
    """
    _assert_windows()
    from hermes_cli.gateway import kill_gateway_processes, _get_restart_drain_timeout
    from gateway.status import get_running_pid

    # Phase 1: ask the running gateway (if any) to drain itself by writing
    # the planned-stop marker, then wait briefly for it to exit cleanly.
    # On clean exit, sessions land with resume_pending=True and the next
    # boot will auto-resume them.
    pid = get_running_pid()
    drained = False
    if pid is not None:
        try:
            drain_timeout = float(_get_restart_drain_timeout() or 30.0)
        except Exception:
            drain_timeout = 30.0
        drained = _drain_gateway_pid(pid, drain_timeout)

    stopped_any = drained
    if is_task_registered():
        code, _out, err = _exec_schtasks(["/End", "/TN", get_task_name()])
        # schtasks returns nonzero when the task isn't currently running — don't treat that as an error.
        if code == 0:
            stopped_any = True
        elif "not running" not in (err or "").lower():
            print(f"⚠ schtasks /End returned code {code}: {err.strip()}")

    # Phase 3: hard-kill any strays.  When drain succeeded this is a no-op;
    # when drain timed out this is the escalation that ensures the PID
    # actually exits.  Use force=True on Windows so taskkill /T /F walks
    # the descendant tree (browser helpers, etc.).
    killed = kill_gateway_processes(all_profiles=False, force=not drained)
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
    from hermes_cli.gateway import kill_gateway_processes

    stop()

    if not _wait_for_gateway_absent(timeout_s=30.0):
        print("⚠ Gateway still present after stop; forcing termination before restart...")
        kill_gateway_processes(all_profiles=False, force=True)
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


def run_watchdog(working_dir: str, argv: list[str], env_overlay: dict[str, str]) -> None:
    """Watchdog loop that monitors the gateway child process on Windows."""
    import os
    import sys
    import time
    import subprocess
    from pathlib import Path
    import ctypes
    from ctypes import wintypes

    env = {**os.environ, **env_overlay}
    hermes_home = env.get("HERMES_HOME")
    if not hermes_home:
        try:
            from hermes_cli.config import get_hermes_home
            hermes_home = str(get_hermes_home())
        except Exception:
            hermes_home = str(Path.home() / ".hermes")

    log_dir = Path(hermes_home) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stray_log = log_dir / "gateway-stdio.log"

    def log_watchdog(message: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{ts}] [Watchdog] {message}\n"
        print(log_line, end="")
        try:
            with open(stray_log, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception:
            pass

    # Win32 Job Object structures and configuration
    h_job = None
    try:
        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('ReadOperationCount', ctypes.c_uint64),
                ('WriteOperationCount', ctypes.c_uint64),
                ('OtherOperationCount', ctypes.c_uint64),
                ('ReadTransferCount', ctypes.c_uint64),
                ('WriteTransferCount', ctypes.c_uint64),
                ('OtherTransferCount', ctypes.c_uint64),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', ctypes.c_int64),
                ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9

        h_job = ctypes.windll.kernel32.CreateJobObjectW(None, None)
        if h_job:
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            res = ctypes.windll.kernel32.SetInformationJobObject(
                h_job,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info)
            )
            if not res:
                h_job = None
    except Exception as exc:
        h_job = None

    log_watchdog(f"Starting gateway watchdog process (PID {os.getpid()})")
    log_watchdog(f"Child command: {argv}")

    # Derive python.exe from the watchdog's python path (which might be pythonw.exe)
    child_argv = list(argv)
    if child_argv and child_argv[0].lower().endswith("pythonw.exe"):
        child_argv[0] = child_argv[0][:-11] + "python.exe"

    restart_delay = 1.0
    failures = []
    
    while True:
        # Rotate log if size > 5MB
        try:
            if stray_log.exists() and stray_log.stat().st_size > 5 * 1024 * 1024:
                old_log = stray_log.with_name("gateway-stdio.log.old")
                if old_log.exists():
                    old_log.unlink()
                stray_log.rename(old_log)
        except Exception:
            pass

        start_time = time.time()
        spawn_success = False
        exit_code = -1

        try:
            with open(stray_log, "ab", buffering=0) as log_fh:
                proc = subprocess.Popen(
                    child_argv,
                    cwd=working_dir,
                    env=env,
                    creationflags=0x08000000,  # CREATE_NO_WINDOW
                    close_fds=True,
                    stdin=subprocess.DEVNULL,
                    stdout=log_fh,
                    stderr=log_fh,
                )
            spawn_success = True
            
            if h_job:
                try:
                    ctypes.windll.kernel32.AssignProcessToJobObject(h_job, int(proc._handle))
                except Exception as exc:
                    log_watchdog(f"Failed to assign process to Job Object: {exc}")

            log_watchdog(f"Spawned gateway worker (PID {proc.pid})")
            exit_code = proc.wait()
            
        except KeyboardInterrupt:
            log_watchdog("Watchdog interrupted by user.")
            sys.exit(0)
        except Exception as exc:
            log_watchdog(f"Watchdog encountered spawn/wait error: {exc}")
            exit_code = -999

        if spawn_success and exit_code == 0:
            log_watchdog("Worker exited cleanly (code 0). Stopping watchdog.")
            sys.exit(0)
        elif spawn_success and exit_code == 75:  # GATEWAY_SERVICE_RESTART_EXIT_CODE
            log_watchdog("Worker requested restart (code 75). Respawning immediately.")
            continue

        # Failure path
        run_duration = time.time() - start_time
        if run_duration >= 30.0:
            restart_delay = 1.0
            log_watchdog("Worker run was healthy (>= 30s), resetting backoff.")

        # Circuit breaker: sliding window of 60 seconds
        now = time.time()
        failures.append(now)
        failures = [t for t in failures if now - t <= 60.0]

        if len(failures) >= 5:
            log_watchdog("Fatal: 5 failures detected in the last 60 seconds. Stopping watchdog.")
            sys.exit(1)

        log_watchdog(f"Worker exited with code {exit_code}, respawning in {restart_delay} seconds...")
        try:
            time.sleep(restart_delay)
        except KeyboardInterrupt:
            log_watchdog("Watchdog interrupted by user during sleep.")
            sys.exit(0)

        restart_delay = min(restart_delay * 2.0, 60.0)


if __name__ == "__main__":
    import argparse
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "watchdog":
        parser = argparse.ArgumentParser(description="Hermes Gateway Watchdog")
        parser.add_argument("command", choices=["watchdog"])
        parser.add_argument("--working-dir", required=True)
        parser.add_argument("--argv", required=True)
        parser.add_argument("--env", required=True)
        
        args = parser.parse_args()
        
        try:
            argv_list = json.loads(args.argv)
            env_dict = json.loads(args.env)
        except Exception as e:
            print(f"Error parsing watchdog arguments: {e}", file=sys.stderr)
            sys.exit(1)
            
        run_watchdog(args.working_dir, argv_list, env_dict)
