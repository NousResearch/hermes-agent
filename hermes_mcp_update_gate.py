"""Stdlib-only update marker reader used before the MCP bridge loads the venv.

The shared marker names a PID and an integer timestamp (claim time or floored
kernel creation time), optionally followed by ``handoff-bridge``.

Ownership rules, identical in ``apps/desktop/electron/update-marker.ts``,
``hermes_cli/update_lock.py`` and ``scripts/desktop-update/windows.ps1``:

* an owner proven to be the claimant (``matching``) never expires -- a slow
  update must never be raced by a second one;
* an owner that is alive but unprovable (``unknown`` -- possibly just a
  recycled PID) expires at :data:`UPDATE_MARKER_MAX_AGE_SECONDS`, as upstream
  main does today. Without that, one recycled PID disables the MCP bridge and
  every update path permanently;
* a body that is empty, malformed or future-dated names nobody at all. It is
  not a claim, so it never gates. Mutating owners (the desktop and the CLI
  lock) reclaim it after a dwell; this module only reads;
* a ``<marker>.cas-release-<pid>-<uuid>`` tombstone whose PID is alive is a
  release still in flight, so the claim behind it is still real and it gates.
  An abandoned one (dead PID) does not; the mutating readers recover it.

Readers here do not mutate the marker; updater lock owners handle reclamation.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from hermes_constants import get_process_hermes_home

MARKER_NAME = ".hermes-update-in-progress"
MCP_MAIN_MODULE = "agent.transports.hermes_tools_mcp_server"

# Keep in sync with UPDATE_MARKER_MAX_AGE_MS in apps/desktop/electron/update-marker.ts.
# This module is the canonical Python definition; hermes_cli.update_lock imports it.
UPDATE_MARKER_MAX_AGE_SECONDS = 20 * 60
# A claimant stamps whole seconds; its kernel creation time may carry fractions.
_OWNER_START_TOLERANCE_SECONDS = 1.0

_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_INVALID_PARAMETER = 87
_FILETIME_EPOCH_OFFSET_SECONDS = 11_644_473_600.0


def _windows_open_process_error_is_definitive_exit(error: int) -> bool:
    """Return whether ``OpenProcess`` proved that a PID does not exist.

    ``ERROR_INVALID_PARAMETER`` is the documented result for an invalid or
    already-exited process identifier. Access denial and every unfamiliar API
    failure are ownership-unknown, so callers treat those PIDs as live.
    """
    return int(error) == _ERROR_INVALID_PARAMETER


def pid_alive(pid: int) -> bool:
    """Return whether *pid* is live, treating access denial as live.

    Never ``os.kill(pid, 0)`` on Windows: CPython routes signal 0 to
    ``GenerateConsoleCtrlEvent``, which Ctrl+C's the target's console group.
    """
    if pid <= 0:
        return False
    if os.name == "nt" and pid > 0xFFFFFFFF:
        return True  # Not representable by OpenProcess; never wrap to another PID.
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, OverflowError):
            return False
        except OSError:
            return True
        return True

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
        open_process.restype = ctypes.c_void_p
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_bool
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        get_exit_code.restype = ctypes.c_bool

        process = open_process(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not process:
            return not _windows_open_process_error_is_definitive_exit(ctypes.get_last_error())
        try:
            exit_code = ctypes.c_ulong()
            if not get_exit_code(process, ctypes.byref(exit_code)):
                return True
            return exit_code.value == _STILL_ACTIVE
        finally:
            close_handle(process)
    except Exception:
        # A platform/API failure is not a definitive process exit.
        return True


def _windows_pid_create_time(pid: int) -> float | None:
    if pid > 0xFFFFFFFF:
        return None
    class _FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
        open_process.restype = ctypes.c_void_p
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_bool
        get_process_times = kernel32.GetProcessTimes
        get_process_times.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(_FileTime)] * 4
        get_process_times.restype = ctypes.c_bool

        process = open_process(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not process:
            return None
        try:
            created, exited, kernel, user = _FileTime(), _FileTime(), _FileTime(), _FileTime()
            if not get_process_times(
                process, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)
            ):
                return None
            ticks = (int(created.high) << 32) | int(created.low)
            # FILETIME is 100-nanosecond ticks since 1601-01-01 UTC.
            return (ticks / 10_000_000.0) - _FILETIME_EPOCH_OFFSET_SECONDS
        finally:
            close_handle(process)
    except Exception:
        return None


# Linux reports process start time in USER_HZ ticks since boot. USER_HZ is
# fixed at 100 for userspace on every mainstream Linux ABI regardless of
# CONFIG_HZ, so no sysconf binding (and no ctypes libc load) is required.
_LINUX_USER_HZ = 100


def _linux_pid_create_time(pid: int) -> float | None:
    """Process creation epoch from /proc alone.

    ``/proc/<pid>/stat`` field 22 is the start time in USER_HZ ticks since
    boot; ``/proc/stat``'s ``btime`` is the boot wall clock. Field 2 (comm) is
    parenthesised and may contain spaces and ``)``, so split after the last one.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as handle:
            stat = handle.read()
        after_comm = stat[stat.rindex(")") + 1:].split()
        ticks = float(after_comm[19])  # after_comm[0] is field 3
        btime = 0.0
        with open("/proc/stat", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("btime "):
                    btime = float(line.split()[1])
                    break
    except (OSError, ValueError, IndexError):
        return None
    if ticks < 0 or btime <= 0:
        return None
    created = btime + ticks / _LINUX_USER_HZ
    return created if math.isfinite(created) and created > 0 else None


def _darwin_pid_create_time(pid: int) -> float | None:
    """Process creation epoch from ``ps -o lstart=`` (stdlib only, no psutil)."""
    try:
        completed = subprocess.run(
            ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = completed.stdout.strip()
    if completed.returncode != 0 or not raw:
        return None
    # ps prints the local-time ctime form, e.g. "Sat Sep  6 12:34:56 2026".
    for fmt in ("%a %b %d %H:%M:%S %Y", "%a %b  %d %H:%M:%S %Y"):
        try:
            return time.mktime(time.strptime(raw, fmt))
        except ValueError:
            continue
    return None


def pid_create_time(pid: int) -> float | None:
    """Return the process creation epoch of *pid*, or ``None`` when unprovable.

    Every platform asks the OS directly: no third-party package may be
    imported here, because this runs before the MCP bridge has decided whether
    the venv it would import from is safe to touch. ``None`` still classifies
    as ``unknown`` -- kept as defence in depth for platforms not covered here.
    """
    if pid <= 0:
        return None
    if os.name == "nt":
        return _windows_pid_create_time(pid)
    if sys.platform == "darwin":
        return _darwin_pid_create_time(pid)
    if sys.platform.startswith("linux"):
        return _linux_pid_create_time(pid)
    return None


def marker_path(hermes_home: str | Path | None = None) -> Path:
    """Path of the update marker: ``<process HERMES_HOME>/.hermes-update-in-progress``.

    Same resolution as ``hermes_cli.update_lock.update_marker_path`` (the
    process home, never a context-local profile override), so the bridge
    reads exactly the file the updater writes.
    """
    root = get_process_hermes_home() if hermes_home is None else Path(hermes_home).expanduser()
    return root / MARKER_NAME


def is_exact_mcp_module_argv(argv: Sequence[str]) -> bool:
    """True only for an argument-free ``-m <MCP_MAIN_MODULE>`` launch."""
    parts = [str(part) for part in argv]
    indexes = [index for index, token in enumerate(parts) if token == "-m"]
    if not (
        parts
        and len(indexes) == 1
        and indexes[0] + 2 == len(parts)
        and parts[indexes[0] + 1] == MCP_MAIN_MODULE
    ):
        return False

    # ``-m`` must be the interpreter's operative action, not text after a
    # script name.  Accept Python's ordinary pre-module runtime switches, but
    # reject ``-c``, ``--``, unknown switches, and non-option operands.  This
    # is intentionally smaller than the complete Python CLI grammar: failure
    # to recognize an exotic launch only makes an updater refuse to kill it.
    before_module = parts[1 : indexes[0]]
    argument_options = {"-W", "-X", "--check-hash-based-pycs"}
    long_flags = {
        "--bytes-warning",
        "--debug",
        "--dont-write-bytecode",
        "--help-env",
        "--help-xoptions",
        "--ignore-environment",
        "--inspect",
        "--isolated",
        "--no-site",
        "--no-user-site",
        "--safe-path",
        "--unbuffered",
        "--verbose",
    }
    short_flag_chars = frozenset("bBdEIiOPqRsStuUvx")
    index = 0
    while index < len(before_module):
        token = before_module[index]
        if token in argument_options:
            index += 1
            if index >= len(before_module) or before_module[index].startswith("-"):
                return False
        elif token.startswith("-W") and token != "-W":
            pass
        elif token.startswith("-X") and token != "-X":
            pass
        elif token.startswith("--check-hash-based-pycs="):
            pass
        elif token in long_flags:
            pass
        elif token.startswith("-") and not token.startswith("--"):
            if not token[1:] or any(char not in short_flag_chars for char in token[1:]):
                return False
        else:
            return False
        index += 1
    return True


@dataclass(frozen=True)
class UpdateMarkerOwner:
    """A live or unprovable owner currently holding the marker."""

    pid: int
    started_at: int
    age_seconds: float
    overdue: bool
    identity: str = "unknown"
    bridge: bool = False


def parse_update_marker(raw: str) -> tuple[int, int] | None:
    """Return ``(pid, ts)`` for a well-formed two-line marker body, else ``None``."""
    lines = raw.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) == 3 and lines[2] == "handoff-bridge":
        lines.pop()
    if len(lines) != 2:
        return None
    try:
        pid = int(lines[0].strip())
        started_at = int(lines[1].strip())
    except ValueError:
        return None
    if (pid <= 0 or started_at <= 0 or max(pid, started_at) > 2**53 - 1
            or any(not line.strip().isascii() or not line.strip().isdigit() for line in lines)):
        return None
    return pid, started_at


def pid_identity_status(
    pid: int,
    started_at: int,
    *,
    pid_alive: Callable[[int], bool] = pid_alive,
    pid_create_time: Callable[[int], float | None] = pid_create_time,
) -> str:
    """Classify the marker's ``<pid>`` against ``<ts>``: ``matching``, ``stale`` or ``unknown``."""
    _alive, _create_time = pid_alive, pid_create_time
    if pid <= 0 or started_at <= 0:
        return "stale"
    try:
        if not _alive(pid):
            return "stale"
    except Exception:
        return "unknown"
    try:
        created = _create_time(pid)
    except Exception:
        return "unknown"
    if created is None or not math.isfinite(created) or created <= 0:
        return "unknown"
    return "matching" if created < started_at + _OWNER_START_TOLERANCE_SECONDS else "stale"


class UpdateMarkerError(RuntimeError):
    """The marker exists but cannot safely authorize installation access."""


class UpdateMarkerUnhealthyError(UpdateMarkerError):
    """The marker names nobody: empty, malformed, future-dated, mid-cleanup.

    Distinct from its base class because the two demand opposite responses. A
    marker we cannot READ (permissions, I/O error) is a genuine unknown and
    callers must fail closed. A marker we read perfectly well and found to name
    nobody proves no update is running, so it must not gate anything -- it is
    the state a torn write leaves behind, and treating it as "an update is in
    progress" is what disabled the MCP bridge until a human deleted the file.
    """


def _live_release_tombstone_owner(
    marker: Path,
    artifacts: list[Path],
    pid_alive: Callable[[int], bool],
) -> int | None:
    """PID of a release still in flight, or ``None``.

    ``_remove_exact_marker`` renames the marker to
    ``<marker>.cas-release-<pid>-<uuid>`` before reading and unlinking it. While
    that PID is alive the transaction is unfinished and the claim is still real.
    An abandoned release (dead PID) is recovered by the mutating readers; this
    module only reads, so it just declines to call that one blocking.
    """
    prefix = marker.name + ".cas-release-"
    for artifact in artifacts:
        if not artifact.name.startswith(prefix):
            continue
        owner = artifact.name[len(prefix):].split("-", 1)[0]
        if owner.isdecimal() and int(owner) > 0 and pid_alive(int(owner)):
            return int(owner)
    return None


def live_update_marker_owner(
    marker: str | Path | None = None,
    *,
    now: float | None = None,
    max_age_seconds: float = UPDATE_MARKER_MAX_AGE_SECONDS,
    pid_alive: Callable[[int], bool] = pid_alive,
    pid_create_time: Callable[[int], float | None] = pid_create_time,
) -> UpdateMarkerOwner | None:
    """Read ownership without deleting any file; unknown claims remain blocking."""
    path = Path(marker) if marker is not None else marker_path()
    try:
        artifacts = list(path.parent.glob(path.name + ".cas-*"))
        if artifacts:
            releaser = _live_release_tombstone_owner(path, artifacts, pid_alive)
            if releaser is not None:
                # A live releaser holds the marker inside its rename/read/unlink
                # transaction, so an update IS running and the venv is not safe
                # to touch. update-marker.ts answers 'cleanup-race' here and
                # update_lock.py raises the blocking error; this reader used to
                # be the only one that let the bridge walk straight through.
                raise UpdateMarkerError(
                    f"Update marker cleanup is in progress (releaser pid {releaser}): {artifacts[0]}"
                )
            raise UpdateMarkerUnhealthyError(
                f"Update marker cleanup is still unresolved: {artifacts[0]}"
            )
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise UpdateMarkerError(f"Cannot read the update marker safely: {path}") from exc
    parsed = parse_update_marker(raw)
    if parsed is None:
        raise UpdateMarkerUnhealthyError(
            f"Update marker is empty or malformed and names no owner. "
            f"If no update is running, delete this file: {path}"
        )
    pid, started_at = parsed
    age = (time.time() if now is None else float(now)) - started_at
    if not math.isfinite(age) or age < -5:
        raise UpdateMarkerUnhealthyError(
            f"Update marker timestamp is in the future. "
            f"If no update is running, delete this file: {path}"
        )
    bridge = raw.replace("\r\n", "\n").rstrip("\n").endswith("\nhandoff-bridge")
    status = pid_identity_status(pid, started_at, pid_alive=pid_alive, pid_create_time=pid_create_time)
    overdue = age > max_age_seconds
    # `stale` is proven dead or proven recycled. `unknown` is alive but
    # unattributable: it holds the gate until the ceiling and is then released,
    # which is the only thing that keeps a marker recoverable at all.
    if (bridge and age > 30) or (not bridge and (status == "stale" or (status == "unknown" and overdue))):
        return None
    return UpdateMarkerOwner(pid, started_at, age, overdue, status, bridge)


def should_quiesce_mcp_bridge(
    *,
    argv: Sequence[str] | None = None,
    marker: str | Path | None = None,
    now: float | None = None,
    pid_alive: Callable[[int], bool] = pid_alive,
    pid_create_time: Callable[[int], float | None] = pid_create_time,
) -> bool:
    """Gate the exact MCP module launch while the marker prevents installation access.

    Never raises. Three things gate: a live owner, a release transaction still
    in flight under a live PID, and a marker we could not READ at all
    (permissions/IO) -- the last stays fail-closed because it might be hiding a
    real update. Everything else, including a marker that names nobody and any
    unexpected internal failure, does NOT gate: turning every exception into
    "quiesce" is how a single torn write made the bridge exit 0 at import on
    every launch, on every platform, until a human intervened.
    """
    try:
        original_argv = getattr(sys, "orig_argv", sys.argv) if argv is None else argv
        if not is_exact_mcp_module_argv(original_argv):
            return False
    except Exception:
        return False
    try:
        return (
            live_update_marker_owner(
                marker, now=now, pid_alive=pid_alive, pid_create_time=pid_create_time
            )
            is not None
        )
    except UpdateMarkerUnhealthyError:
        return False
    except UpdateMarkerError:
        return True
    except Exception:
        return False
