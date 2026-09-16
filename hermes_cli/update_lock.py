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
import uuid
from dataclasses import dataclass
from pathlib import Path

from hermes_mcp_update_gate import (
    MARKER_NAME,
    UPDATE_MARKER_MAX_AGE_SECONDS,
    UpdateMarkerError,
    UpdateMarkerUnhealthyError,
    live_update_marker_owner,
    pid_alive,
    pid_create_time as _stdlib_pid_create_time,
)

logger = logging.getLogger(__name__)

__all__ = ["MARKER_NAME", "UPDATE_MARKER_MAX_AGE_SECONDS", "UpdateMarkerError", "UpdateLock",
           "UpdateHolder", "UPDATE_EXIT_CONCURRENT", "HANDOFF_PID_ENV", "describe_holder",
           "read_live_update", "update_marker_path"]

# A body that names nobody may be a rewrite caught in flight. Observe the exact
# same bytes across this dwell before reclaiming, so a real writer always wins.
UPDATE_MARKER_DWELL_SECONDS = 2.0

# Set by an orchestrating updater (Tauri `hermes-setup --update`) to its own pid before
# spawning `hermes update` as a child stage; the parent holds the marker for its whole run,
# so without this the child would refuse its own parent's lock. The writer is
# `update_child_env` in apps/bootstrap-installer/src-tauri/src/update.rs (with its own test,
# `update_child_env_names_our_pid_for_the_lock_handoff`) -- this is a live production
# channel, not a vestigial one, and removing the adoption path here dead-ends every GUI
# update on exit 2. Adoption still requires the named pid to be the marker's `matching`
# owner, so naming a pid grants nothing on its own.
HANDOFF_PID_ENV = "HERMES_UPDATE_HANDOFF_PID"

# Bound on the parent chain walked by _is_ancestor_pid. Real ancestries are a
# handful of links (init -> desktop -> staged updater -> shim -> us); the cap
# only exists so an unexpected chain can never spin the walk.
_MAX_ANCESTRY_DEPTH = 128

# Exit code meaning "another updater/instance owns this install right now" — the same
# contract as the Windows shim / venv-holder guards in _cmd_update_impl, matched by the
# Tauri updater (UPDATE_EXIT_CONCURRENT in update.rs) to show "Hermes is still running".
UPDATE_EXIT_CONCURRENT = 2


def update_marker_path() -> Path:
    """Path of the shared update marker.

    Uses the *process* Hermes home (never the context-local profile override): the Rust
    updater resolves ``$HERMES_HOME`` or the platform default and the desktop pins that same
    value into the updater's env, so a profile-scoped path would be one the other owners never look at.
    """
    from hermes_constants import get_process_hermes_home
    return get_process_hermes_home() / MARKER_NAME


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
def _pid_create_time(pid: int) -> float | None:
    if os.name == "nt":
        return _stdlib_pid_create_time(pid)
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _handoff_pid() -> int | None:
    """Pid of the orchestrating updater that spawned us (:data:`HANDOFF_PID_ENV`); malformed
    values count as absent so a broken handoff falls back to the normal refusal."""
    try:
        pid = int(os.environ.get(HANDOFF_PID_ENV, "").strip())
    except ValueError:
        return None
    return pid if pid > 0 else None


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
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    def created(target: int) -> int | None:
        handle = kernel32.OpenProcess(0x1000, False, target)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return None
            return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        finally:
            kernel32.CloseHandle(handle)

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
    """A live or unprovable owner currently holding the lock."""

    pid: int
    age_seconds: float
    identity: str = "unknown"


def _restore_marker(marker: Path, isolated: Path) -> None:
    """Restore without overwriting a successor; unresolved artifacts stay blocking."""
    try:
        os.link(isolated, marker)
    except FileExistsError:
        pass
    except OSError as exc:
        raise UpdateMarkerError("Cannot restore the isolated update marker") from exc
    isolated.unlink()


def _remove_exact_marker(marker: Path, expected: str) -> None:
    isolated = marker.with_name(f"{marker.name}.cas-release-{os.getpid()}-{uuid.uuid4()}")
    try:
        marker.rename(isolated)
    except FileNotFoundError:
        return
    try:
        if isolated.read_text(encoding="utf-8") != expected:
            _restore_marker(marker, isolated)
            return
        isolated.unlink()
    except (OSError, UnicodeError) as exc:
        if isolated.exists():
            _restore_marker(marker, isolated)
        raise UpdateMarkerError("Cannot reclaim the update marker safely") from exc


def read_live_update(
    *,
    path: Path | None = None,
    dwell_seconds: float = UPDATE_MARKER_DWELL_SECONDS,
    sleep=None,
) -> UpdateHolder | None:
    """Read the shared claim, reclaiming a proven stale or unowned exact marker.

    A marker we cannot READ still raises UpdateMarkerError and callers must
    stop. A marker we read fine and found to name nobody -- empty, malformed,
    future-dated, the residue of a torn write -- is reclaimed after a dwell
    instead of blocking every update from then on (upstream main unlinks these
    outright at :125; the PR removed that and left no way back).
    """
    marker = path or update_marker_path()
    _sleep = sleep or time.sleep
    for _ in range(3):
        try:
            artifacts = list(marker.parent.glob(marker.name + ".cas-*"))
            if artifacts:
                if len(artifacts) == 1 and artifacts[0].name.startswith(marker.name + ".cas-release-") and not marker.exists():
                    releaser = artifacts[0].name[len(marker.name + ".cas-release-"):].split("-", 1)[0]
                    # A live releaser owns the rename/read/unlink transaction.
                    # Recover only an abandoned release, never resurrect one
                    # that another process is still removing.
                    if releaser.isdecimal() and int(releaser) > 0 and not pid_alive(int(releaser)):
                        _restore_marker(marker, artifacts[0])
                        continue
                raise UpdateMarkerError(
                    f"Update marker cleanup is unresolved; retry after the updater finishes ({artifacts[0]})"
                )
            raw = marker.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError) as exc:
            raise UpdateMarkerError(f"Cannot read the update marker safely: {marker}") from exc
        try:
            owner = live_update_marker_owner(marker, pid_alive=pid_alive, pid_create_time=_pid_create_time)
        except UpdateMarkerUnhealthyError:
            # Dwell, then re-read. _remove_exact_marker is already an exact
            # content CAS, so an owner that finished writing inside the dwell
            # keeps its claim and we simply loop again.
            _sleep(dwell_seconds)
            try:
                if marker.read_text(encoding="utf-8") != raw:
                    continue
            except FileNotFoundError:
                return None
            except (OSError, UnicodeError) as exc:
                raise UpdateMarkerError(f"Cannot read the update marker safely: {marker}") from exc
            logger.warning("Reclaiming an update marker that names no owner: %s", marker)
            _remove_exact_marker(marker, raw)
            continue
        if owner is not None:
            return UpdateHolder(owner.pid, owner.age_seconds, owner.identity)
        _remove_exact_marker(marker, raw)
    raise UpdateMarkerError(f"Update marker changed during reclamation; retry ({marker})")


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
        "  started it, then run `hermes update` again."
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
        self.acquired = False
        self.holder: UpdateHolder | None = None

    def acquire(self) -> bool:
        """Claim the lock. Returns False (and sets ``holder``) if it's taken.

        A live holder whose pid matches :data:`HANDOFF_PID_ENV` — or is an ancestor of ours —
        is our own orchestrating parent: run under ITS claim and leave its marker untouched on
        release. The ancestry path covers staged updaters older than the env-var export.
        """
        existing = read_live_update(path=self.path)
        # A killed update's marker naming the pid this retry inherited (containers restart pid
        # numbering) is not special-cased here: its started_at predates our process creation, so
        # read_live_update classifies it `stale` by identity and reclaims it before we get here.
        # Adoption requires the `matching` owner identity, so naming a pid grants nothing alone.
        if existing is not None:
            if existing.identity == "matching" and (existing.pid == _handoff_pid() or _is_ancestor_pid(existing.pid)):
                return True
            self.holder = existing
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise UpdateMarkerError("Cannot create the update marker directory") from exc
        try:
            with self.path.open("x", encoding="utf-8") as claim:
                self._claim = f"{os.getpid()}\n{int(time.time())}\n"
                claim.write(self._claim)
                claim.flush()
                os.fsync(claim.fileno())
        except FileExistsError:
            self.holder = read_live_update(path=self.path)
            if self.holder is None:
                raise UpdateMarkerError("Update marker changed during acquisition; retry")
            return False
        except OSError as exc:
            raise UpdateMarkerError("Cannot claim the update marker; installation was not changed") from exc
        if list(self.path.parent.glob(self.path.name + ".cas-*")):
            _remove_exact_marker(self.path, self._claim)
            raise UpdateMarkerError("Update marker cleanup started during acquisition; retry")
        self.acquired = True
        return True

    def release(self) -> None:
        """Drop the marker if this process still owns it. Never raises."""
        if not self.acquired:
            return
        self.acquired = False
        try:
            _remove_exact_marker(self.path, self._claim)
        except (OSError, UpdateMarkerError):
            logger.warning("Could not release update marker %s", self.path)

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
