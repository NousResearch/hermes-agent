"""
Gateway runtime status helpers.

Provides PID-file based detection of whether the gateway daemon is running,
used by send_message's check_fn to gate availability in the CLI.

The PID file lives at ``{HERMES_HOME}/gateway.pid``.  HERMES_HOME defaults to
``~/.hermes`` but can be overridden via the environment variable.  This means
separate HERMES_HOME directories naturally get separate PID files — a property
that will be useful when we add named profiles (multiple agents running
concurrently under distinct configurations).
"""

import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Any, Optional
from utils import atomic_json_write

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_GATEWAY_KIND = "hermes-gateway"
_RUNTIME_STATUS_FILE = "gateway_state.json"
_LOCKS_DIRNAME = "gateway-locks"
_IS_WINDOWS = sys.platform == "win32"
_UNSET = object()
_GATEWAY_LOCK_FILENAME = "gateway.lock"
_gateway_lock_handle = None
# Windows byte-range locks are mandatory for other readers. Lock a byte well
# past the JSON payload so runtime status / PID readers can still read the file
# while another process holds the mutual-exclusion lock.
_WINDOWS_LOCK_OFFSET = 1024 * 1024


def _get_pid_path() -> Path:
    """Return the path to the gateway PID file, respecting HERMES_HOME."""
    home = get_hermes_home()
    return home / "gateway.pid"


def _get_gateway_lock_path(pid_path: Optional[Path] = None) -> Path:
    """Return the path to the runtime gateway lock file."""
    if pid_path is not None:
        return pid_path.with_name(_GATEWAY_LOCK_FILENAME)
    home = get_hermes_home()
    return home / _GATEWAY_LOCK_FILENAME


def _get_runtime_status_path() -> Path:
    """Return the persisted runtime health/status file path."""
    return _get_pid_path().with_name(_RUNTIME_STATUS_FILE)


def _get_lock_dir() -> Path:
    """Return the machine-local directory for token-scoped gateway locks."""
    override = os.getenv("HERMES_GATEWAY_LOCK_DIR")
    if override:
        return Path(override)
    state_home = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_home / "hermes" / _LOCKS_DIRNAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def terminate_pid(pid: int, *, force: bool = False) -> None:
    """Terminate a PID with platform-appropriate force semantics.

    POSIX uses SIGTERM/SIGKILL. Windows uses taskkill /T /F for true force-kill
    because os.kill(..., SIGTERM) is not equivalent to a tree-killing hard stop.
    """
    if force and _IS_WINDOWS:
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            os.kill(pid, signal.SIGTERM)
            return

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise OSError(details or f"taskkill failed for PID {pid}")
        return

    sig = signal.SIGTERM if not force else getattr(signal, "SIGKILL", signal.SIGTERM)
    os.kill(pid, sig)


def _scope_hash(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _get_scope_lock_path(scope: str, identity: str) -> Path:
    return _get_lock_dir() / f"{scope}-{_scope_hash(identity)}.lock"


def _get_process_start_time(pid: int) -> Optional[int]:
    """Return a stable per-process start-time fingerprint, or None.

    Used as a PID-reuse guard: a ``(pid, start_time)`` pair uniquely identifies
    a process, so a recycled PID (same number, different process) yields a
    different value and is never mistaken for the original.

    On Linux this is field 22 of ``/proc/<pid>/stat`` (start time in clock
    ticks since boot, an int).  On platforms without ``/proc`` (macOS, Windows)
    we fall back to ``psutil.Process(pid).create_time()`` — a float epoch
    timestamp — quantized to an int (centiseconds) for stable equality.

    The two sources are never mixed on a single platform: ``/proc`` always
    succeeds first on Linux, and always fails on macOS/Windows so psutil is
    always used there.  Because the guard only compares the value recorded at
    spawn against the live value *on the same host*, the differing units across
    platforms are irrelevant — only same-source equality matters.
    """
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        # Field 22 in /proc/<pid>/stat is process start time (clock ticks).
        return int(stat_path.read_text(encoding="utf-8").split()[21])
    except (FileNotFoundError, IndexError, PermissionError, ValueError, OSError):
        pass

    # No /proc (macOS / Windows): psutil is a hard dependency and exposes a
    # cross-platform creation time.  Quantize to centiseconds so repeated reads
    # of the same process compare equal without float-precision fragility.
    try:
        import psutil  # type: ignore
        return int(round(psutil.Process(pid).create_time() * 100))
    except Exception:
        return None


def get_process_start_time(pid: int) -> Optional[int]:
    """Public wrapper for retrieving a process start time when available."""
    return _get_process_start_time(pid)


def _read_process_cmdline(pid: int) -> Optional[str]:
    """Return the process command line as a space-separated string.

    On Linux, reads /proc/<pid>/cmdline directly.  On macOS and other
    platforms without /proc, falls back to ``ps -p <pid> -o command=``.
    On Windows (no /proc, no ps), uses psutil.
    """
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = cmdline_path.read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        pass
    else:
        if raw:
            return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()

    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass

    # Windows fallback: psutil (already used by _pid_exists)
    try:
        import psutil  # type: ignore
        proc = psutil.Process(pid)
        cmdline_parts = proc.cmdline()
        if cmdline_parts:
            return " ".join(cmdline_parts)
    except Exception:
        pass

    return None


def _gateway_command_subcommand(command: str | None) -> str | None:
    """Return the Hermes gateway lifecycle subcommand from a command line.

    Lifecycle decisions (is the gateway up? did restart relaunch it?) must not
    fire on loose substring matches.  The previous ``"... gateway" in cmdline``
    test also matched ``hermes_cli.main gateway status`` and even unrelated
    processes like ``python -m tui_gateway`` -- which made ``restart()`` race
    against a still-draining old process and ``status``/``start`` report false
    positives.  This requires the actual ``gateway`` subcommand followed by
    ``run`` (or one of the gateway-dedicated entrypoints), excluding the other
    ``gateway`` management subcommands and any process that merely contains the
    word "gateway".

    Tokenizes quote-aware (``shlex``) so quoted Windows paths with spaces
    (``"C:\\Program Files\\...\\hermes-gateway.exe"``) survive, and strips
    ``--profile``/``-p`` selectors from anywhere in argv -- Hermes's
    ``_apply_profile_override`` removes them before argparse, so the profile
    flag (and a profile literally named ``gateway``) can legally appear on
    either side of the ``gateway`` subcommand.
    """
    if not command:
        return None

    try:
        raw_tokens = shlex.split(command, posix=False)
    except ValueError:
        raw_tokens = command.split()
    # Strip surrounding quotes, normalize slashes + case per token.
    tokens = [t.strip("\"'").replace("\\", "/").lower() for t in raw_tokens]
    if not tokens:
        return None

    # Gateway-dedicated entrypoints carry no subcommand to inspect.
    for token in tokens:
        if token == "gateway/run.py" or token.endswith("/gateway/run.py"):
            return "run"
        basename = token.rsplit("/", 1)[-1]
        if basename in ("hermes-gateway", "hermes-gateway.exe"):
            return "run"

    joined = " ".join(tokens)
    has_gateway_entry = (
        "hermes_cli.main" in joined
        or "hermes_cli/main.py" in joined
        or any(t.rsplit("/", 1)[-1] in ("hermes", "hermes.exe") for t in tokens)
    )
    if not has_gateway_entry:
        return None

    # Drop profile selectors anywhere: --profile X / -p X / --profile=X / -p=X.
    # This consumes a profile VALUE of "gateway" too, so the real subcommand
    # token is the one we land on below.
    filtered: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in ("--profile", "-p"):
            skip_next = True
            continue
        if token.startswith("--profile=") or token.startswith("-p="):
            continue
        filtered.append(token)

    for i, token in enumerate(filtered):
        if token != "gateway":
            continue
        if i + 1 >= len(filtered):
            return "run"  # bare `hermes gateway` defaults to `run`
        return filtered[i + 1]
    return None


def looks_like_gateway_command_line(command: str | None) -> bool:
    """Return True only for a real ``gateway run`` process command line."""
    return _gateway_command_subcommand(command) == "run"


def looks_like_gateway_runtime_command_line(command: str | None) -> bool:
    """Return True for command lines that can host the gateway runtime.

    ``gateway restart`` is normally a management command, not the gateway
    runtime. On hosts without a service manager, though, the manual restart
    fallback executes ``run_gateway()`` in that same process, so its argv stays
    as ``gateway restart`` while it owns the webhook port and writes runtime
    state. Keep the public ``looks_like_gateway_command_line()`` strict, and
    use this broader matcher only when validating Hermes-owned runtime records
    or no-supervisor cleanup scans.
    """
    return _gateway_command_subcommand(command) in {"run", "restart"}


def _looks_like_gateway_process(pid: int) -> bool:
    """Return True when the live PID still looks like the Hermes gateway."""
    cmdline = _read_process_cmdline(pid)
    if not cmdline:
        return False
    return looks_like_gateway_command_line(cmdline)


def _record_looks_like_gateway(record: dict[str, Any]) -> bool:
    """Validate gateway identity from PID-file metadata when cmdline is unavailable."""
    if record.get("kind") != _GATEWAY_KIND:
        return False

    argv = record.get("argv")
    if not isinstance(argv, list) or not argv:
        return False

    cmdline = " ".join(str(part) for part in argv)
    return looks_like_gateway_runtime_command_line(cmdline)


def _profile_name_for_home(profile_home: Path) -> Optional[str]:
    """Return the profile id a HERMES_HOME directory represents, or None.

    A named profile's home is ``<root>/profiles/<name>`` (immediate parent is
    ``profiles``).  The root/default home (``~/.hermes`` or ``$HERMES_HOME``)
    has no such parent, so it maps to the default profile (``None`` here, which
    callers treat as "the bare, flag-less gateway").
    """
    if profile_home.parent.name == "profiles":
        return profile_home.name
    return None


def _command_line_belongs_to_profile(command: str, profile_home: Path) -> bool:
    """Return True when a gateway command line belongs to ``profile_home``.

    Mirrors ``hermes_cli.gateway._matches_current_profile`` so the dashboard's
    cross-profile liveness fallback scopes a live PID to the *right* profile.
    In a per-profile container, one profile's stale ``gateway_state.json`` can
    record a PID that the OS has since recycled onto a DIFFERENT profile's live
    gateway.  That recycled PID's command line still ``looks_like_gateway`` —
    so without a profile check the dead profile is reported running.  A named
    profile gateway carries ``-p <name>``/``--profile <name>`` (or, rarely, an
    explicit ``HERMES_HOME=<path>``) on its argv; the default/root gateway runs
    bare with no profile flag.
    """
    command_lc = command.lower()
    profile_name = _profile_name_for_home(profile_home)
    home_lc = str(profile_home).lower()

    if profile_name is not None and profile_name != "default":
        profile_lc = profile_name.lower()
        return (
            f"--profile {profile_lc}" in command_lc
            or f"-p {profile_lc}" in command_lc
            or f"hermes_home={home_lc}" in command_lc
        )

    # Default/root profile: the gateway runs with no profile flag. Accept unless
    # the command advertises *some other* profile (an explicit -p/--profile) or
    # a non-matching explicit HERMES_HOME= on the argv. HERMES_HOME is usually
    # passed via the environment (not visible on the command line), so its mere
    # absence is not disqualifying — only a conflicting explicit value is.
    if "--profile " in command_lc or " -p " in command_lc:
        return False
    if "hermes_home=" in command_lc and f"hermes_home={home_lc}" not in command_lc:
        return False
    return True


def _record_matches_live_gateway_pid(
    record: dict[str, Any],
    pid: int,
    *,
    expected_home: Optional[Path] = None,
) -> bool:
    """Return True when a live PID still identifies as this gateway record.

    Prefer the live command line whenever it is readable. Runtime status files
    can outlive the gateway process they describe; if PID reuse leaves the same
    PID occupied by an s6 supervisor/log process, the stale record's argv should
    not make that unrelated process count as a running gateway.

    When ``expected_home`` is provided (the dashboard enumerating a specific
    profile's state file), the readable live command line must additionally
    belong to *that* profile — otherwise a PID recycled onto a different
    profile's live gateway would make the dead profile look alive.  When the
    live command line cannot be read (Windows/permission), fall back to the
    persisted record so cross-platform behavior is preserved.
    """
    live_cmdline = _read_process_cmdline(pid)
    if live_cmdline:
        if not looks_like_gateway_runtime_command_line(live_cmdline):
            return False
        if expected_home is not None and not _command_line_belongs_to_profile(
            live_cmdline, expected_home
        ):
            return False
        return True
    return _record_looks_like_gateway(record)


def _build_pid_record() -> dict:
    return {
        "pid": os.getpid(),
        "kind": _GATEWAY_KIND,
        "argv": list(sys.argv),
        "start_time": _get_process_start_time(os.getpid()),
    }


def _build_runtime_status_record() -> dict[str, Any]:
    payload = _build_pid_record()
    payload.update({
        "gateway_state": "starting",
        "exit_reason": None,
        "restart_requested": False,
        "active_agents": 0,
        "active_session_keys": [],
        "platforms": {},
        "updated_at": _utc_now_iso(),
    })
    return payload


def _read_json_file(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        # OSError: file vanished or permission flipped between exists() and
        # read. UnicodeDecodeError: file holds non-UTF-8 / binary garbage
        # (a truncated or clobbered status file). Either way it's unusable.
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    atomic_json_write(path, payload, indent=None, separators=(",", ":"))


def _read_pid_record(pid_path: Optional[Path] = None) -> Optional[dict]:
    pid_path = pid_path or _get_pid_path()
    if not pid_path.exists():
        return None

    try:
        raw = pid_path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        # File was deleted between exists() and read_text(), permission
        # flipped, or it holds non-UTF-8 / binary garbage.
        return None
    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            return {"pid": int(raw)}
        except ValueError:
            return None

    if isinstance(payload, int):
        return {"pid": payload}
    if isinstance(payload, dict):
        return payload
    return None


def _read_gateway_lock_record(lock_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    return _read_pid_record(lock_path or _get_gateway_lock_path())


def _pid_from_record(record: Optional[dict[str, Any]]) -> Optional[int]:
    if not record:
        return None
    try:
        return int(record["pid"])
    except (KeyError, TypeError, ValueError):
        return None


def _cleanup_invalid_pid_path(pid_path: Path, *, cleanup_stale: bool) -> None:
    """Delete a stale gateway PID file (and its sibling lock metadata).

    Called from ``get_running_pid()`` after the runtime lock has already been
    confirmed inactive, so the on-disk metadata is known to belong to a dead
    process.  Unlike ``remove_pid_file()`` (which defensively refuses to delete
    a PID file whose ``pid`` field differs from ``os.getpid()`` to protect
    ``--replace`` handoffs), this path force-unlinks both files so the next
    startup sees a clean slate.
    """
    if not cleanup_stale:
        return
    try:
        pid_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        _get_gateway_lock_path(pid_path).unlink(missing_ok=True)
    except Exception:
        pass


def _write_gateway_lock_record(handle) -> None:
    handle.seek(0)
    handle.truncate()
    json.dump(_build_pid_record(), handle)
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass


def _try_acquire_file_lock(handle) -> bool:
    try:
        if _IS_WINDOWS:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("\n")
                handle.flush()
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _pid_exists(pid: int) -> bool:
    """Cross-platform "is this PID alive" check that does NOT kill the target.

    CRITICAL on Windows: Python's ``os.kill(pid, 0)`` is NOT a no-op like it
    is on POSIX. CPython's Windows implementation
    (``Modules/posixmodule.c::os_kill_impl``) treats ``sig=0`` as
    ``CTRL_C_EVENT`` because the two values collide at the C level, and
    routes it through ``GenerateConsoleCtrlEvent(0, pid)`` — which sends
    a Ctrl+C to the entire console process group containing the target
    PID, not just the PID itself. Any caller that wanted to "check if
    this PID is alive" via ``os.kill(pid, 0)`` on Windows was silently
    killing that process (and often unrelated processes in the same
    console group). Long-standing Python quirk; see bpo-14484.

    Implementation: prefer :mod:`psutil` (hard dependency — the canonical
    cross-platform answer, maintained by Giampaolo Rodolà, uses
    ``OpenProcess + GetExitCodeProcess`` on Windows internally). Fall back
    to a hand-rolled ctypes ``OpenProcess`` / ``WaitForSingleObject`` pair
    on Windows + ``os.kill(pid, 0)`` on POSIX if psutil is somehow
    unavailable — e.g. stripped-down install or import error during the
    scaffold phase before ``psutil`` is pip-installed.
    """
    try:
        import psutil  # type: ignore
        return bool(psutil.pid_exists(int(pid)))
    except ImportError:
        pass  # Fall through to stdlib fallback.

    if _IS_WINDOWS:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            # Pin return types — default ctypes restype is c_int (signed),
            # which mangles WAIT_* DWORD return codes into negative numbers.
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.restype = ctypes.c_uint
            kernel32.GetLastError.restype = ctypes.c_uint
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            SYNCHRONIZE = 0x100000  # required for WaitForSingleObject
            WAIT_TIMEOUT = 0x00000102
            ERROR_INVALID_PARAMETER = 87
            ERROR_ACCESS_DENIED = 5
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, int(pid)
            )
            if not handle:
                err = kernel32.GetLastError()
                if err == ERROR_INVALID_PARAMETER:
                    return False  # PID definitely gone
                if err == ERROR_ACCESS_DENIED:
                    return True   # Exists but owned by another user/session
                return False      # Conservative default for unknown errors
            try:
                wait_result = kernel32.WaitForSingleObject(handle, 0)
                # WAIT_TIMEOUT = still running; anything else (WAIT_OBJECT_0
                # via exit, WAIT_FAILED via handle issue) = treat as gone.
                return wait_result == WAIT_TIMEOUT
            finally:
                kernel32.CloseHandle(handle)
        except (OSError, AttributeError):
            return False
    else:
        try:
            os.kill(int(pid), 0)  # windows-footgun: ok — POSIX-only branch (the whole point of _pid_exists)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we can't signal it — still alive.
            return True
        except OSError:
            return False



def _release_file_lock(handle) -> None:
    try:
        if _IS_WINDOWS:
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def acquire_gateway_runtime_lock() -> bool:
    """Claim the cross-process runtime lock for the gateway.

    Unlike the PID file, the lock is owned by the live process itself. If the
    process dies abruptly, the OS releases the lock automatically.
    """
    global _gateway_lock_handle
    if _gateway_lock_handle is not None:
        return True

    path = _get_gateway_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+", encoding="utf-8")
    if not _try_acquire_file_lock(handle):
        handle.close()
        return False
    _write_gateway_lock_record(handle)
    _gateway_lock_handle = handle
    return True


def release_gateway_runtime_lock() -> None:
    """Release the gateway runtime lock when owned by this process."""
    global _gateway_lock_handle
    handle = _gateway_lock_handle
    if handle is None:
        return
    _gateway_lock_handle = None
    _release_file_lock(handle)
    try:
        handle.close()
    except OSError:
        pass


def is_gateway_runtime_lock_active(lock_path: Optional[Path] = None) -> bool:
    """Return True when some process currently owns the gateway runtime lock."""
    global _gateway_lock_handle
    resolved_lock_path = lock_path or _get_gateway_lock_path()
    if _gateway_lock_handle is not None and resolved_lock_path == _get_gateway_lock_path():
        return True

    if not resolved_lock_path.exists():
        return False

    handle = open(resolved_lock_path, "a+", encoding="utf-8")
    try:
        if _try_acquire_file_lock(handle):
            _release_file_lock(handle)
            return False
        return True
    finally:
        try:
            handle.close()
        except OSError:
            pass


def write_pid_file() -> None:
    """Write the current process PID and metadata to the gateway PID file.

    Uses atomic O_CREAT | O_EXCL creation so that concurrent --replace
    invocations race: exactly one process wins and the rest get
    FileExistsError.
    """
    path = _get_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps(_build_pid_record())
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise  # Let caller decide: another gateway is racing us
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(record)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_runtime_status(
    *,
    gateway_state: Any = _UNSET,
    exit_reason: Any = _UNSET,
    restart_requested: Any = _UNSET,
    active_agents: Any = _UNSET,
    active_session_keys: Any = _UNSET,
    platform: Any = _UNSET,
    platform_state: Any = _UNSET,
    error_code: Any = _UNSET,
    error_message: Any = _UNSET,
    served_profiles: Any = _UNSET,
) -> None:
    """Persist gateway runtime health information for diagnostics/status."""
    path = _get_runtime_status_path()
    payload = _read_json_file(path) or _build_runtime_status_record()
    current_record = _build_pid_record()
    payload.setdefault("platforms", {})
    payload["kind"] = current_record["kind"]
    payload["pid"] = current_record["pid"]
    payload["argv"] = current_record["argv"]
    payload["start_time"] = current_record["start_time"]
    payload["updated_at"] = _utc_now_iso()

    if gateway_state is not _UNSET:
        payload["gateway_state"] = gateway_state
    if exit_reason is not _UNSET:
        payload["exit_reason"] = exit_reason
    if restart_requested is not _UNSET:
        payload["restart_requested"] = bool(restart_requested)
    if active_agents is not _UNSET:
        payload["active_agents"] = parse_active_agents(active_agents)
    if active_session_keys is not _UNSET:
        payload["active_session_keys"] = parse_active_session_keys(active_session_keys)
    if served_profiles is not _UNSET:
        # Profiles this gateway multiplexes (multi-profile mode). Absent/empty
        # for a single-profile gateway. Lets `hermes status` show per-profile
        # coverage without a second probe.
        payload["served_profiles"] = list(served_profiles or [])

    if platform is not _UNSET:
        platform_payload = payload["platforms"].get(platform, {})
        if platform_state is not _UNSET:
            platform_payload["state"] = platform_state
        if error_code is not _UNSET:
            platform_payload["error_code"] = error_code
        if error_message is not _UNSET:
            platform_payload["error_message"] = error_message
        platform_payload["updated_at"] = _utc_now_iso()
        payload["platforms"][platform] = platform_payload

    _write_json_file(path, payload)


def read_runtime_status(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Read the persisted gateway runtime health/status information.

    ``path`` is optional so callers that need to inspect a *different*
    profile's state file (e.g. the dashboard enumerating every profile)
    can do so without mutating ``HERMES_HOME`` in-process.  Defaults to
    the active profile's ``gateway_state.json``.
    """
    return _read_json_file(path or _get_runtime_status_path())


def parse_active_agents(raw: Any) -> int:
    """Coerce a persisted ``active_agents`` value to a clamped non-negative int.

    The shared coercion for the in-flight gateway-turn count. Used on the WRITE
    side (``write_runtime_status``) and by both HTTP read surfaces
    (``/api/status`` and ``/health/detailed``) so the count is clamped to a
    single contract — never negative, never raising on a manually-edited or
    otherwise non-numeric value (degrades to ``0``).
    """
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def parse_active_session_keys(raw: Any) -> list[str]:
    """Return sanitized active gateway session keys.

    These keys contain routing/session identifiers, not message bodies or
    secrets. They are persisted so crash recovery can resume the exact
    in-flight sessions instead of guessing from a stale ``updated_at`` window.
    """
    if raw is None or not isinstance(raw, (list, tuple, set)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        key = item.strip()
        if not key or len(key) > 512 or key in seen:
            continue
        result.append(key)
        seen.add(key)
    return result


# States in which the gateway is alive and could be asked to drain.  Anything
# else (draining already, stopping, stopped, startup_failed, None) is NOT a
# valid begin-drain target.
_DRAINABLE_GATEWAY_STATES = frozenset({"running"})


def derive_gateway_busy(
    *, gateway_running: bool, gateway_state: Any, active_agents: Any
) -> bool:
    """Whether the gateway is actively processing in-flight turns.

    The contract NAS gates lifecycle actions on.  Busy iff the gateway is live
    (``gateway_running``), in the ``running`` state, AND at least one agent is
    mid-turn (``active_agents > 0``).  Degrades to ``False`` whenever liveness
    is unknown, the state is anything but ``running``, or the count is
    absent/unparseable — i.e. a down or file-absent gateway reads "not busy",
    never a spurious "busy".

    NOTE: liveness keys off ``gateway_running`` (a live PID / health probe),
    NEVER ``updated_at`` — a healthy idle gateway never advances that timestamp.
    """
    if not gateway_running:
        return False
    if gateway_state not in _DRAINABLE_GATEWAY_STATES:
        return False
    try:
        return int(active_agents) > 0
    except (TypeError, ValueError):
        return False


def derive_gateway_drainable(*, gateway_running: bool, gateway_state: Any) -> bool:
    """Whether the gateway can accept a begin-drain request right now.

    True iff the gateway is live and in the ``running`` state — i.e. not already
    draining/stopping/stopped and not in a failed-start state.  This is
    independent of ``active_agents``: an idle running gateway is drainable (the
    drain just completes immediately).  Degrades to ``False`` for a down or
    non-running gateway.
    """
    return bool(gateway_running) and gateway_state in _DRAINABLE_GATEWAY_STATES


def get_runtime_status_running_pid(
    runtime: Optional[dict[str, Any]] = None,
    *,
    expected_home: Optional[Path] = None,
) -> Optional[int]:
    """Return a live gateway PID from the runtime status record, if valid.

    ``get_running_pid()`` is the primary liveness source because it verifies the
    runtime lock and PID file.  Launch-service managers can still leave us with
    a live process and a fresh ``gateway_state.json`` but no ``gateway.pid``; use
    this as a conservative fallback by checking both the persisted state and the
    OS process identity.

    ``expected_home`` scopes the OS-identity check to a specific profile's
    HERMES_HOME.  Pass it when validating *another* profile's state file (the
    dashboard enumerating every profile): a stale record whose PID the OS has
    recycled onto a different profile's live gateway must not be reported
    running for the dead profile.  Omit it (the default) for the active
    profile, where any live gateway command line is acceptable.
    """
    payload = runtime if runtime is not None else read_runtime_status()
    if not isinstance(payload, dict):
        return None
    if payload.get("gateway_state") in {None, "stopped", "startup_failed"}:
        return None

    pid = _pid_from_record(payload)
    if pid is None or not _pid_exists(pid):
        return None

    recorded_start = payload.get("start_time")
    current_start = _get_process_start_time(pid)
    if (
        recorded_start is not None
        and current_start is not None
        and current_start != recorded_start
    ):
        return None

    if _record_matches_live_gateway_pid(payload, pid, expected_home=expected_home):
        return pid
    return None


def remove_pid_file() -> None:
    """Remove the gateway PID file, but only if it belongs to this process.

    During --replace handoffs, the old process's atexit handler can fire AFTER
    the new process has written its own PID file.  Blindly removing the file
    would delete the new process's record, leaving the gateway running with no
    PID file (invisible to ``get_running_pid()``).
    """
    try:
        path = _get_pid_path()
        record = _read_json_file(path)
        if record is not None:
            try:
                file_pid = int(record["pid"])
            except (KeyError, TypeError, ValueError):
                file_pid = None
            if file_pid is not None and file_pid != os.getpid():
                # PID file belongs to a different process — leave it alone.
                return
        path.unlink(missing_ok=True)
    except Exception:
        pass


def acquire_scoped_lock(scope: str, identity: str, metadata: Optional[dict[str, Any]] = None) -> tuple[bool, Optional[dict[str, Any]]]:
    """Acquire a machine-local lock keyed by scope + identity.

    Used to prevent multiple local gateways from using the same external identity
    at once (e.g. the same Telegram bot token across different HERMES_HOME dirs).
    """
    lock_path = _get_scope_lock_path(scope, identity)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **_build_pid_record(),
        "scope": scope,
        "identity_hash": _scope_hash(identity),
        "metadata": metadata or {},
        "updated_at": _utc_now_iso(),
    }

    existing = _read_json_file(lock_path)
    if existing is None and lock_path.exists():
        # Lock file exists but is empty or contains invalid JSON — treat as
        # stale.  This happens when a previous process was killed between
        # O_CREAT|O_EXCL and the subsequent json.dump() (e.g. DNS failure
        # during rapid Slack reconnect retries).
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
    if existing:
        try:
            existing_pid = int(existing["pid"])
        except (KeyError, TypeError, ValueError):
            existing_pid = None

        if existing_pid == os.getpid() and existing.get("start_time") == record.get("start_time"):
            _write_json_file(lock_path, record)
            return True, existing

        stale = existing_pid is None
        if not stale:
            if not _pid_exists(existing_pid):
                stale = True
            else:
                current_start = _get_process_start_time(existing_pid)
                if (
                    existing.get("start_time") is not None
                    and current_start is not None
                    and current_start != existing.get("start_time")
                ):
                    stale = True
                # When start_time comparison is unavailable (macOS / Windows
                # have no /proc, so both sides are None), fall back to
                # checking the live process command line.  When cmdline is
                # also unreadable (Windows has no ps), consult the lock
                # record's own argv — the gateway writes it at startup and
                # it's the only identity signal on platforms without ps.
                # Both oracles must indicate "not a gateway" to mark stale.
                if (
                    not stale
                    and existing.get("start_time") is None
                    and current_start is None
                    and not _looks_like_gateway_process(existing_pid)
                ):
                    live_cmdline = _read_process_cmdline(existing_pid)
                    if live_cmdline is not None or not _record_looks_like_gateway(existing):
                        stale = True
                # Secondary defence against boot-time PID+start_time collisions:
                # systemd spawns core services deterministically, so an unrelated
                # process (e.g. cron) can land on the exact same PID and jiffy
                # count as a previous gateway. If both start_times are known and
                # match but the live process is not a gateway, and we can confirm
                # that by reading its cmdline, the lock is stale.
                if (
                    not stale
                    and existing.get("start_time") is not None
                    and current_start is not None
                    and not _looks_like_gateway_process(existing_pid)
                ):
                    live_cmdline = _read_process_cmdline(existing_pid)
                    if live_cmdline is not None:
                        stale = True
                # Check if process is stopped (Ctrl+Z / SIGTSTP) — stopped
                # processes still appear alive to _pid_exists but are not
                # actually running. Treat them as stale so --replace works.
                if not stale:
                    try:
                        _proc_status = Path(f"/proc/{existing_pid}/status")
                        if _proc_status.exists():
                            for _line in _proc_status.read_text(encoding="utf-8").splitlines():
                                if _line.startswith("State:"):
                                    _state = _line.split()[1]
                                    if _state in {"T", "t"}:  # stopped or tracing stop
                                        stale = True
                                    break
                    except (OSError, PermissionError):
                        pass
        if stale:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
        else:
            return False, existing

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False, _read_json_file(lock_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
    except Exception:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return True, None


def release_scoped_lock(scope: str, identity: str) -> None:
    """Release a previously-acquired scope lock when owned by this process."""
    lock_path = _get_scope_lock_path(scope, identity)
    existing = _read_json_file(lock_path)
    if not existing:
        return
    if existing.get("pid") != os.getpid():
        return
    if existing.get("start_time") != _get_process_start_time(os.getpid()):
        return
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def release_all_scoped_locks(
    *,
    owner_pid: Optional[int] = None,
    owner_start_time: Optional[int] = None,
) -> int:
    """Remove scoped lock files in the lock directory.

    Called during --replace to clean up stale locks left by stopped/killed
    gateway processes that did not release their locks gracefully. When an
    ``owner_pid`` is provided, only lock records belonging to that gateway
    process are removed. ``owner_start_time`` further narrows the match to
    protect against PID reuse.

    When no owner is provided, preserves the legacy behavior and removes every
    scoped lock file in the directory.

    Returns the number of lock files removed.
    """
    lock_dir = _get_lock_dir()
    removed = 0
    if lock_dir.exists():
        for lock_file in lock_dir.glob("*.lock"):
            if owner_pid is not None:
                record = _read_json_file(lock_file)
                if not isinstance(record, dict):
                    continue
                try:
                    record_pid = int(record.get("pid"))
                except (TypeError, ValueError):
                    continue
                if record_pid != owner_pid:
                    continue
                if (
                    owner_start_time is not None
                    and record.get("start_time") != owner_start_time
                ):
                    continue
            try:
                lock_file.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
    return removed


# ── --replace takeover marker ─────────────────────────────────────────
#
# When a new gateway starts with ``--replace``, it SIGTERMs the existing
# gateway so it can take over the bot token. PR #5646 made SIGTERM exit
# the gateway with code 1 so ``Restart=on-failure`` can revive it after
# unexpected kills — but that also means a --replace takeover target
# exits 1, which tricks systemd into reviving it 30 seconds later,
# starting a flap loop against the replacer when both services are
# enabled in the user's systemd (e.g. ``hermes.service`` + ``hermes-
# gateway.service``).
#
# The takeover marker breaks the loop: the replacer writes a short-lived
# file naming the target PID + start_time BEFORE sending SIGTERM.
# The target's shutdown handler reads the marker and, if it names
# this process, treats the SIGTERM as a planned takeover and exits 0.
# The marker is unlinked after the target has consumed it, so a stale
# marker left by a crashed replacer can grief at most one future
# shutdown on the same PID — and only within _TAKEOVER_MARKER_TTL_S.

_TAKEOVER_MARKER_FILENAME = ".gateway-takeover.json"
_TAKEOVER_MARKER_TTL_S = 60  # Marker older than this is treated as stale
_PLANNED_STOP_MARKER_FILENAME = ".gateway-planned-stop.json"
_PLANNED_STOP_MARKER_TTL_S = 60


def _get_takeover_marker_path() -> Path:
    """Return the path to the --replace takeover marker file."""
    home = get_hermes_home()
    return home / _TAKEOVER_MARKER_FILENAME


def _get_planned_stop_marker_path() -> Path:
    """Return the path to the intentional gateway stop marker file."""
    home = get_hermes_home()
    return home / _PLANNED_STOP_MARKER_FILENAME


def _marker_is_stale(written_at: str, ttl_s: int) -> bool:
    try:
        written_dt = datetime.fromisoformat(written_at)
        age = (datetime.now(timezone.utc) - written_dt).total_seconds()
        return age > ttl_s
    except (TypeError, ValueError):
        return True


def _consume_pid_marker_for_self(
    path: Path,
    *,
    pid_field: str,
    start_time_field: str,
    ttl_s: int,
) -> bool:
    record = _read_json_file(path)
    if not record:
        return False

    try:
        target_pid = int(record[pid_field])
        target_start_time = record.get(start_time_field)
        written_at = record.get("written_at") or ""
    except (KeyError, TypeError, ValueError):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return False

    if _marker_is_stale(written_at, ttl_s):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return False

    our_pid = os.getpid()
    our_start_time = _get_process_start_time(our_pid)
    # Start-time is a PID-reuse guard. It is only meaningful when both
    # sides actually have it: ``_get_process_start_time`` returns None on
    # platforms without ``/proc`` (macOS, native Windows — the very
    # platform the planned-stop watcher exists for). Requiring a non-None
    # match there would make every consume return False, so a legitimate
    # ``hermes gateway stop`` on Windows would be misclassified as an
    # unexpected ``UNKNOWN`` exit (exit 1) and revived by the service
    # manager. So: when both start_times are known they must match; when
    # either is unknown, fall back to PID equality alone (bounded by the
    # marker's short TTL). This mirrors ``planned_stop_marker_targets_self``
    # so the watcher's non-destructive probe and this authoritative
    # consume agree on every platform (issue #34597).
    if target_pid != our_pid:
        matches = False
    elif target_start_time is not None and our_start_time is not None:
        matches = target_start_time == our_start_time
    else:
        matches = True

    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass

    return matches


def write_takeover_marker(target_pid: int) -> bool:
    """Record that ``target_pid`` is being replaced by the current process.

    Captures the target's ``start_time`` so that PID reuse after the
    target exits cannot later match the marker. Also records the
    replacer's PID and a UTC timestamp for TTL-based staleness checks.

    Returns True on successful write, False on any failure. The caller
    should proceed with the SIGTERM even if the write fails (the marker
    is a best-effort signal, not a correctness requirement).
    """
    try:
        target_start_time = _get_process_start_time(target_pid)
        record = {
            "target_pid": target_pid,
            "target_start_time": target_start_time,
            "replacer_pid": os.getpid(),
            "written_at": _utc_now_iso(),
        }
        _write_json_file(_get_takeover_marker_path(), record)
        return True
    except (OSError, PermissionError):
        return False


def consume_takeover_marker_for_self() -> bool:
    """Check & unlink the takeover marker if it names the current process.

    Returns True only when a valid (non-stale) marker names this PID +
    start_time. A returning True indicates the current SIGTERM is a
    planned --replace takeover; the caller should exit 0 instead of
    signalling ``_signal_initiated_shutdown``.

    Always unlinks the marker on match (and on detected staleness) so
    subsequent unrelated signals don't re-trigger.
    """
    return _consume_pid_marker_for_self(
        _get_takeover_marker_path(),
        pid_field="target_pid",
        start_time_field="target_start_time",
        ttl_s=_TAKEOVER_MARKER_TTL_S,
    )


def clear_takeover_marker() -> None:
    """Remove the takeover marker unconditionally. Safe to call repeatedly."""
    try:
        _get_takeover_marker_path().unlink(missing_ok=True)
    except OSError:
        pass


def write_planned_stop_marker(target_pid: int) -> bool:
    """Record that ``target_pid`` is being stopped intentionally.

    The gateway exits non-zero for unexpected SIGTERM so service managers can
    revive it. Service stop commands send the same SIGTERM, so the CLI writes
    this short-lived marker first to let the target process exit cleanly.
    """
    try:
        target_start_time = _get_process_start_time(target_pid)
        record = {
            "target_pid": target_pid,
            "target_start_time": target_start_time,
            "stopper_pid": os.getpid(),
            "written_at": _utc_now_iso(),
        }
        _write_json_file(_get_planned_stop_marker_path(), record)
        return True
    except (OSError, PermissionError):
        return False


def consume_planned_stop_marker_for_self() -> bool:
    """Return True when the current process is being intentionally stopped."""
    return _consume_pid_marker_for_self(
        _get_planned_stop_marker_path(),
        pid_field="target_pid",
        start_time_field="target_start_time",
        ttl_s=_PLANNED_STOP_MARKER_TTL_S,
    )


def planned_stop_marker_targets_self() -> bool:
    """Return True only when a live planned-stop marker names the current process.

    This is a **non-destructive** probe used by the watcher thread
    (``gateway/run.py:_run_planned_stop_watcher``) to decide whether to
    trigger shutdown. Unlike :func:`consume_planned_stop_marker_for_self`,
    it never unlinks a marker that matches us — the shutdown handler does
    the authoritative consume on its own thread.

    It *does* clean up markers that can never apply to this process:
    malformed markers and markers older than the TTL are unlinked so a
    stale file left behind by a previous gateway instance cannot wedge
    the new one. Markers naming a different PID/start_time are left in
    place (they may still be consumed legitimately by the process they
    name) but report False here.

    Returns False (without raising) on any read/parse error.
    """
    path = _get_planned_stop_marker_path()
    record = _read_json_file(path)
    if not record:
        return False

    try:
        target_pid = int(record["target_pid"])
        target_start_time = record.get("target_start_time")
        written_at = record.get("written_at") or ""
    except (KeyError, TypeError, ValueError):
        # Malformed marker can never match anyone — drop it.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return False

    if _marker_is_stale(written_at, _PLANNED_STOP_MARKER_TTL_S):
        # A marker this old is past its useful life regardless of target —
        # clean it up so it cannot crash-loop a freshly booted gateway.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return False

    our_pid = os.getpid()
    if target_pid != our_pid:
        return False

    # Start-time is a PID-reuse guard. It is only meaningful when both
    # sides actually have it: ``_get_process_start_time`` returns None on
    # platforms without ``/proc`` (macOS, native Windows — the very
    # platform this watcher exists for). Requiring a non-None match there
    # would make the watcher never fire and re-break the #33778 Windows
    # session-resume path. So: when both start_times are known they must
    # match; when either is unknown, fall back to PID equality alone
    # (the marker is short-lived under a 60s TTL, bounding reuse risk).
    our_start_time = _get_process_start_time(our_pid)
    if target_start_time is not None and our_start_time is not None:
        return target_start_time == our_start_time
    return True


def clear_planned_stop_marker() -> None:
    """Remove the planned-stop marker unconditionally."""
    try:
        _get_planned_stop_marker_path().unlink(missing_ok=True)
    except OSError:
        pass


def get_running_pid(
    pid_path: Optional[Path] = None,
    *,
    cleanup_stale: bool = True,
) -> Optional[int]:
    """Return the PID of a running gateway instance, or ``None``.

    Checks the PID file and verifies the process is actually alive.
    Cleans up stale PID files automatically.
    """
    resolved_pid_path = pid_path or _get_pid_path()
    resolved_lock_path = _get_gateway_lock_path(resolved_pid_path)
    lock_active = is_gateway_runtime_lock_active(resolved_lock_path)
    if not lock_active:
        if pid_path is None:
            runtime_pid = get_runtime_status_running_pid()
            if runtime_pid is not None:
                return runtime_pid
        _cleanup_invalid_pid_path(resolved_pid_path, cleanup_stale=cleanup_stale)
        return None

    primary_record = _read_pid_record(resolved_pid_path)
    fallback_record = _read_gateway_lock_record(resolved_lock_path)

    for record in (primary_record, fallback_record):
        pid = _pid_from_record(record)
        if pid is None:
            continue

        if not _pid_exists(pid):
            continue

        recorded_start = record.get("start_time")
        current_start = _get_process_start_time(pid)
        if recorded_start is not None and current_start is not None and current_start != recorded_start:
            continue

        if _record_matches_live_gateway_pid(record, pid):
            return pid

    _cleanup_invalid_pid_path(resolved_pid_path, cleanup_stale=cleanup_stale)
    if pid_path is None:
        runtime_pid = get_runtime_status_running_pid()
        if runtime_pid is not None:
            return runtime_pid
    return None


def is_gateway_running(
    pid_path: Optional[Path] = None,
    *,
    cleanup_stale: bool = True,
) -> bool:
    """Check if the gateway daemon is currently running."""
    return get_running_pid(pid_path, cleanup_stale=cleanup_stale) is not None
