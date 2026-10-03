"""Cross-process mutual exclusion for in-flight Hermes updates.

The marker file the Tauri updater writes (``UpdateMarkerGuard`` in
``apps/bootstrap-installer/src-tauri/src/update.rs``) and the Electron desktop reads
(``electron/update-marker.ts``) is the single lock for **all** update entrypoints.
Format and location are byte-compatible with both readers.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep in sync with UPDATE_MARKER_MAX_AGE_MS in apps/desktop/electron/update-marker.ts and
# UPDATE_MARKER_MAX_AGE_SECS in apps/bootstrap-installer/src-tauri/src/update.rs.
# Past this age a live pid is only honored once its identity is verified: the ceiling exists
# because the OS can recycle a dead updater's pid onto an unrelated process, not because an
# update is short (a Windows update with a desktop rebuild can take 40+ minutes, #109795).
UPDATE_MARKER_MAX_AGE_SECONDS = 20 * 60

# A marker's owner is the process that wrote it (or was spawned just before the desktop wrote
# it), so the owner was created no later than the marker file's last write. A pid whose process
# was created AFTER that write is a recycled pid. The slack absorbs one-second ps resolution and
# clock granularity; keep in sync with the Rust and Electron readers.
PID_REUSE_TOLERANCE_SECONDS = 5

MARKER_NAME = ".hermes-update-in-progress"

# Set by an orchestrating updater (Tauri `hermes-setup --update`) to its own pid before
# spawning `hermes update` as a child stage; the parent holds the marker for its whole run,
# so without this the child would refuse its own parent's lock. Keep in sync with
# update_child_env in apps/bootstrap-installer/src-tauri/src/update.rs.
HANDOFF_PID_ENV = "HERMES_UPDATE_HANDOFF_PID"

# Bound on the parent chain walked by _is_ancestor_pid. Real ancestries are a
# handful of links (init -> desktop -> staged updater -> shim -> us); the cap
# only exists so an unexpected chain can never spin the walk.
_MAX_ANCESTRY_DEPTH = 128

# Exit code meaning "another updater/instance owns this install right now" — the same
# contract as the Windows shim / venv-holder guards in _cmd_update_impl, matched by the
# Tauri updater (UPDATE_EXIT_CONCURRENT in update.rs) to show "Hermes is still running".
UPDATE_EXIT_CONCURRENT = 2


def install_home(home: Path) -> Path:
    """The install root owning *home*: ``<root>/profiles/<name>`` maps to ``<root>``.

    Same lexical rule as the remote marker probes (``remoteInstallRoot`` in
    ``electron/remote-lifecycle.ts``, ``managed-ssh-update.ts``) and the Rust/Electron local readers.
    """
    parent = home.parent.name
    return home.parent.parent if (parent.lower() if os.name == "nt" else parent) == "profiles" else home


def update_marker_path() -> Path:
    """Path of the shared update marker.

    One lock per install, not per profile: it lives in the install root of the *process*
    Hermes home (never the context-local profile override). A profile gateway runs with
    ``HERMES_HOME=<root>/profiles/<name>``; a per-profile marker let its launch-time source
    completion run concurrently with ``hermes update`` (#123376). The Rust updater and the
    desktop resolve the same root.
    """
    from hermes_constants import get_process_hermes_home
    return install_home(get_process_hermes_home()) / MARKER_NAME


def _legacy_marker_paths() -> list[Path]:
    """Where an older updater run from a profile shell claimed the lock (the profile home)."""
    from hermes_constants import get_process_hermes_home
    legacy = get_process_hermes_home() / MARKER_NAME
    return [] if legacy == update_marker_path() else [legacy]


def _pid_alive(pid: int) -> bool:
    """Use the dependency-free, Windows-safe probe before PM is available."""
    if pid <= 0:
        return False
    try:
        from hermes_cli._early_recovery import _pid_is_running
        return _pid_is_running(pid)
    except Exception as exc:
        logger.debug("Could not probe pid %s: %s", pid, exc)
        return False


def _handoff_pid() -> int | None:
    """Pid of the orchestrating updater that spawned us (:data:`HANDOFF_PID_ENV`); malformed
    values count as absent so a broken handoff falls back to the normal refusal."""
    try:
        pid = int(os.environ.get(HANDOFF_PID_ENV, "").strip())
    except ValueError:
        return None
    return pid if pid > 0 else None


def _windows_creation_filetime(pid: int) -> int | None:
    """Creation time of ``pid`` as a Windows FILETIME (100ns ticks since 1601), or ``None``."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        kernel32.CloseHandle(handle)


def _parse_ps_etime(text: str) -> int | None:
    """Seconds from ``ps -o etime=`` (``[[dd-]hh:]mm:ss``, the format macOS and procps share)."""
    days, _, clock = text.strip().rpartition("-")
    parts = clock.split(":")
    if not 2 <= len(parts) <= 3 or not all(p.isdigit() for p in parts) or (days and not days.isdigit()):
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds + int(days or 0) * 86400


def _process_started_at(pid: int) -> float | None:
    """Wall-clock creation time (unix seconds) of ``pid``, or ``None`` when unverifiable.

    psutil first; the ``-I -S -B`` takeover child has no site-packages (and psutil can be
    denied another user's process), so fall back to
    GetProcessTimes on Windows and ``ps -o etime=`` elsewhere (the same probe the Rust and
    Electron readers use).
    """
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception:  # ImportError, or AccessDenied for another user's process on macOS
        pass
    if sys.platform == "win32":
        try:
            filetime = _windows_creation_filetime(pid)
        except (OSError, AttributeError, ValueError):
            return None
        return None if filetime is None else (filetime - 116444736000000000) / 10_000_000
    try:
        out = subprocess.run(
            ["ps", "-o", "etime=", "-p", str(pid)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    elapsed = _parse_ps_etime(out)
    return None if elapsed is None else time.time() - elapsed


def marker_owner_is_live(
    *, pid_alive: bool, age_seconds: float, owner_started_at: float | None, marker_written_at: float | None,
) -> bool:
    """The liveness decision shared (by contract) with the Rust and Electron readers.

    A dead pid is never live. A live pid inside the age ceiling is live. Past the ceiling a live
    pid stays live only when its identity is verified — its process predates the marker's last
    write — so a slow update is never handed to a second updater, while a recycled pid (created
    after the write) or an unverifiable one still expires on age.
    """
    if not pid_alive:
        return False
    if age_seconds <= UPDATE_MARKER_MAX_AGE_SECONDS:
        return True
    if owner_started_at is None or marker_written_at is None:
        return False
    return owner_started_at <= marker_written_at + PID_REUSE_TOLERANCE_SECONDS


def _windows_parent_pid(pid: int) -> int | None:
    """The parent of ``pid`` from a Toolhelp32 process snapshot (stdlib ctypes).

    Windows keeps a dead parent's pid in the snapshot and reuses pids, so, like
    psutil, a "parent" created after the child is a recycled pid, not our parent.
    """
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    for walk in (kernel32.Process32FirstW, kernel32.Process32NextW):
        walk.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        walk.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    created = _windows_creation_filetime

    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return None
    parent = None
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        found = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while found:
            if entry.th32ProcessID == pid:
                parent = int(entry.th32ParentProcessID)
                break
            found = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    if not parent:
        return None
    parent_created, child_created = created(parent), created(pid)
    if parent_created is not None and child_created is not None and parent_created > child_created:
        return None
    return parent


def _stdlib_parent_pid(pid: int) -> int | None:
    """The parent of ``pid`` without psutil, or ``None`` when unresolvable.

    The update-takeover child is spawned ``-I -S -B`` (hermes_cli/_old_updater.py) so
    psutil cannot import there — and that grandchild is exactly the process that most
    needs the two-hop ancestry walk to adopt the orchestrator's marker. /proc serves
    Linux; macOS keeps /proc absent, so shell out to ps once per hop; Windows has
    neither, so ask the Toolhelp32 snapshot.
    """
    if sys.platform == "win32":
        try:
            return _windows_parent_pid(pid)
        except (OSError, AttributeError, ValueError):
            return None
    try:
        if os.path.isdir("/proc"):
            with open(f"/proc/{pid}/stat", "rb") as fh:
                stat = fh.read()
        else:
            out = subprocess.run(
                ["ps", "-o", "ppid=", "-p", str(pid)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, timeout=5,
            ).stdout
            value = int(out.strip() or -1)
            return value if value > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    # Field 4 (1-indexed) is ppid, but comm may contain spaces/parens: split
    # after the closing paren of comm instead of on whitespace.
    try:
        return int(stat[stat.rindex(b")") + 2:].split()[1])
    except (ValueError, IndexError):
        return None


def _is_ancestor_pid(pid: int) -> bool:
    """True when ``pid`` is a live ancestor of this process.

    The orchestrating updater spawns ``hermes update`` as a (grand)child, so a live marker
    owned by one of our ancestors can only be the claim we are already running under — an
    unrelated concurrent updater is never in our parent chain. This heals the fleet of staged
    ``hermes-setup`` binaries that predate the HANDOFF_PID_ENV export and can never send it.

    The chain is walked one link at a time and each ancestor is tested as it is
    discovered. ``psutil.Process.parents()`` cannot be used here: it builds the
    whole chain up to the lowest pid *before* returning, and its per-link
    ``parent()`` tolerates only ``NoSuchProcess``. So any process we may not
    inspect anywhere above us raises ``AccessDenied`` and discards the
    ancestors already collected — including the orchestrator one link down.
    That is not exotic: under firejail with ``ptrace_scope=1``, and in hardened
    containers, ``/proc/1`` is unreadable, so the GUI update deadlocked against
    its own parent on every attempt. Walking incrementally means a failure
    *above* the match can no longer hide it.

    Never includes our own pid, and any failure encountered before a match
    counts as "not an ancestor": an unprovable ancestry must fall back to the
    normal refusal.
    """
    if pid <= 0:
        return False
    if pid == os.getppid():
        return True
    try:
        import psutil

        proc = psutil.Process()
        seen = {proc.pid}
        for _ in range(_MAX_ANCESTRY_DEPTH):
            parent = proc.parent()
            if parent is None:
                return False
            if parent.pid == pid:
                return True
            if parent.pid in seen:
                # Defensive only: psutil's create_time check already rejects a
                # reused ppid, so a true cycle should be unreachable.
                return False
            seen.add(parent.pid)
            proc = parent
        logger.debug(
            "Gave up walking process ancestry for pid %s after %s links",
            pid,
            _MAX_ANCESTRY_DEPTH,
        )
        return False
    except ImportError:
        # -I -S -B takeover child: walk the same chain with stdlib probes.
        child = os.getpid()
        for _ in range(32):
            parent = _stdlib_parent_pid(child)
            if parent is None:
                return False
            if parent == pid:
                return True
            if parent == child:  # pid 1 re-parenting or a kernel loop guard
                return False
            child = parent
        return False
    except Exception as exc:
        logger.debug("Could not walk process ancestry for pid %s: %s", pid, exc)
        return False


@dataclass(frozen=True)
class UpdateHolder:
    """A confirmed-live update currently holding the lock."""

    pid: int
    age_seconds: float


def read_live_update(*, path: Path | None = None) -> UpdateHolder | None:
    """Return the live update holding the lock, or ``None``.

    Mirrors ``readLiveUpdateMarker`` in ``electron/update-marker.ts``: absent, unreadable,
    malformed, dead-pid, and past-the-ceiling-without-verified-identity all mean "no live
    update" (see :func:`marker_owner_is_live`), and a stale marker file is deleted so it can't
    strand future runs. Without *path*, a legacy per-profile marker is honored too. Never raises.
    """
    if path is None:
        for legacy in _legacy_marker_paths():
            holder = read_live_update(path=legacy)
            if holder is not None:
                return holder
    marker = path or update_marker_path()
    try:
        lines = marker.read_text(encoding="utf-8-sig").splitlines()
        written_at: float | None = marker.stat().st_mtime
    except OSError:
        return None
    try:
        pid = int(lines[0].strip())
    except (IndexError, ValueError):
        pid = -1
    try:
        started_at = float(lines[1].strip())
    except (IndexError, ValueError):
        started_at = float("-inf")

    age = time.time() - started_at
    alive = _pid_alive(pid)
    # Only a well-formed marker past the ceiling pays for the identity probe.
    probe = alive and age > UPDATE_MARKER_MAX_AGE_SECONDS and started_at != float("-inf")
    owner_started_at = _process_started_at(pid) if probe else None
    if not marker_owner_is_live(
        pid_alive=alive, age_seconds=age, owner_started_at=owner_started_at, marker_written_at=written_at,
    ):
        with suppress(OSError):
            marker.unlink()
        return None
    return UpdateHolder(pid=pid, age_seconds=age)


def describe_holder(holder: UpdateHolder | None) -> str:
    """One-line, user-facing explanation of who holds the update lock."""
    minutes, seconds = divmod(int(max(0 if holder is None else holder.age_seconds, 0)), 60)
    elapsed = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    who = f", process {holder.pid}" if holder else ""
    return (
        f"✗ Another Hermes update is already running (started {elapsed} ago{who}).\n"
        "\n"
        "  Running two at once would corrupt the install. Wait for it to finish\n"
        "  (watch `hermes logs`), or close the Desktop/dashboard window that\n"
        "  started it, then run `hermes update` again. If it is stuck, stop\n"
        "  that process first."
    )


class UpdateLock:
    """Context manager owning the shared update marker for this process.

    ``acquired`` is False when another live update holds it; callers decide between hard
    refusal (CLI/dashboard) and waiting. Release only removes the marker when *we* still own
    it, so a marker rewritten by a handoff partner (the Tauri updater writes its own pid) is
    never deleted from under its new owner.
    """

    def __init__(self, *, path: Path | None = None) -> None:
        self.path = path or update_marker_path()
        self._legacy_paths = [] if path else _legacy_marker_paths()
        self.acquired = False
        self.holder: UpdateHolder | None = None

    def acquire(self) -> bool:
        """Claim the lock. Returns False (and sets ``holder``) if it's taken.

        A live holder whose pid matches :data:`HANDOFF_PID_ENV` — or is an ancestor of ours —
        is our own orchestrating parent: run under ITS claim and leave its marker untouched on
        release. The ancestry path covers staged updaters older than the env-var export.
        """
        existing = next(
            (h for p in (*self._legacy_paths, self.path) if (h := read_live_update(path=p)) is not None), None,
        )
        # A live claim naming our own pid is a killed update's marker whose pid this retry
        # inherited (containers restart pid numbering): no other live process has our pid, and
        # nothing pre-writes a marker for `hermes update` (it always runs under a parent's claim).
        # It is a new attempt, so it is claimed fresh like a dead holder's. Keeping the old
        # started_at would let the ceiling expire mid-run and admit a second updater.
        if existing is not None and existing.pid != os.getpid():
            if existing.pid == _handoff_pid() or _is_ancestor_pid(existing.pid):
                return True
            self.holder = existing
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8")
        except OSError as exc:
            # Best-effort, like the Rust guard: an unwritable marker must not block the
            # update itself (worse than the race it prevents). Degrade to pre-lock behavior.
            logger.debug("Could not write update marker %s: %s", self.path, exc)
            return True
        self.acquired = True
        return True

    def release(self) -> None:
        """Drop the marker if this process still owns it. Never raises."""
        if not self.acquired:
            return
        self.acquired = False
        try:
            owner = int(self.path.read_text(encoding="utf-8-sig").splitlines()[0].strip())
        except (OSError, IndexError, ValueError):
            return
        if owner != os.getpid():
            return  # a handoff partner took ownership — still a live update
        with suppress(OSError):
            self.path.unlink()

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
