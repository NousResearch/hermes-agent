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
    r"(access is denied|acceso denegado|schtasks timed out|schtasks produced no output)",
    re.IGNORECASE,
)
_ACCESS_DENIED_PATTERN = re.compile(r"(access is denied|acceso denegado)", re.IGNORECASE)

_TASK_NAME_DEFAULT = "Hermes_Gateway"
_TASK_DESCRIPTION = "Hermes Agent Gateway - Messaging Platform Integration"


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
      - cd's into the project directory
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

    pythonw_path = _derive_venv_pythonw(python_path)
    prog_args = [pythonw_path, "-m", "hermes_cli.main"]
    if profile_arg:
        prog_args.extend(profile_arg.split())
    prog_args.extend(["gateway", "run"])
    # `pythonw.exe` is a GUI-subsystem executable: cmd.exe launches it and
    # returns immediately, so the Scheduled Task action finishes without a
    # visible console window. Do NOT use `start` here; that creates an extra
    # wrapper process and made gateway lifecycle/status harder to reason about.
    # Do NOT use `--replace` for service-managed starts; repeated /Run calls
    # should be idempotent, not churn parent/child takeover loops.
    lines.append(" ".join(_quote_cmd_script_arg(a) for a in prog_args))
    lines.append("exit /b 0")
    return "\r\n".join(lines) + "\r\n"


def _build_startup_launcher(script_path: Path) -> str:
    """The tiny .cmd that goes in the Startup folder. Just minimizes and chains."""
    lines = [
        "@echo off",
        f"rem {_TASK_DESCRIPTION}",
        # ``start "" /min`` detaches with a minimized console window.
        # ``/d /c`` on cmd.exe skips AUTORUN and runs the target script once.
        f'start "" /min cmd.exe /d /c {_quote_cmd_script_arg(str(script_path))}',
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
    working_dir = str(PROJECT_ROOT)
    hermes_home = str(Path(get_hermes_home()).resolve())
    profile_arg = _profile_arg(hermes_home)

    content = _build_gateway_cmd_script(python_path, working_dir, hermes_home, profile_arg)
    script_path = get_task_script_path()
    script_path.write_text(content, encoding="utf-8", newline="")
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
    entry.write_text(_build_startup_launcher(script_path), encoding="utf-8", newline="")
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

    python_exe = _derive_venv_pythonw(get_python_path())
    working_dir = str(PROJECT_ROOT)
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
        "VIRTUAL_ENV": str(Path(python_exe).resolve().parent.parent),
    }
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

    # Inherit PATH etc. from the current env, overlay our required vars.
    env = {**os.environ, **env_overlay}

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
        # setups pythonw.exe + DETACHED_PROCESS is enough on its own.
        flags_no_breakaway = flags & ~0x01000000
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

    for path, label in [(startup_entry, "Windows login item"), (script_path, "Task script")]:
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
        print(f"  Task name:     {task_name}")
        print(f"  Task script:   {get_task_script_path()}")
        print(f"  Startup entry: {get_startup_entry_path()}")

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


def stop() -> None:
    """Stop the gateway. Tries /End on the scheduled task, then kills any stragglers."""
    _assert_windows()
    from hermes_cli.gateway import kill_gateway_processes

    stopped_any = False
    if is_task_registered():
        code, _out, err = _exec_schtasks(["/End", "/TN", get_task_name()])
        # schtasks returns nonzero when the task isn't currently running — don't treat that as an error.
        if code == 0:
            stopped_any = True
        elif "not running" not in (err or "").lower():
            print(f"⚠ schtasks /End returned code {code}: {err.strip()}")

    killed = kill_gateway_processes(all_profiles=False)
    if killed:
        stopped_any = True
        print(f"✓ Killed {killed} gateway process(es)")
    if stopped_any:
        print("✓ Gateway stopped")
    else:
        print("✗ No gateway was running")


def restart() -> None:
    """Stop the gateway then start it again."""
    _assert_windows()
    stop()
    # Give Windows a moment to release the listening port.
    time.sleep(1.0)
    start()
