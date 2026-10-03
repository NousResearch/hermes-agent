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
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep in sync with UPDATE_MARKER_MAX_AGE_MS in apps/desktop/electron/update-marker.ts:
# a shorter ceiling here would let Python steal a lock Electron still considers live.
# A full update (git pull + uv sync + desktop rebuild) is minutes.
UPDATE_MARKER_MAX_AGE_SECONDS = 20 * 60

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
    """A confirmed-live update currently holding the lock."""

    pid: int
    age_seconds: float


def read_live_update(*, path: Path | None = None) -> UpdateHolder | None:
    """Return the live update holding the lock, or ``None``.

    Mirrors ``readLiveUpdateMarker`` in ``electron/update-marker.ts``: absent, unreadable,
    malformed, dead-pid, and past-the-ceiling all mean "no live update", and a stale marker
    file is deleted so it can't strand future runs. Never raises.
    """
    marker = path or update_marker_path()
    try:
        lines = marker.read_text(encoding="utf-8-sig").splitlines()
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
    if not _pid_alive(pid) or age > UPDATE_MARKER_MAX_AGE_SECONDS:
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
        "  started it, then run `hermes update` again."
    )


class UpdateLock:
    """Context manager owning the shared update marker for this process.

    ``acquired`` is False when another live update holds it; callers decide between hard
    refusal (CLI/dashboard) and waiting. Release only removes the marker when *we* still own
    it, so a marker rewritten by a handoff partner (the Tauri updater writes its own pid) is
    never deleted from under its new owner.
    """

    def __init__(
        self, *, path: Path | None = None,
        refresh_interval_seconds: float = UPDATE_MARKER_MAX_AGE_SECONDS / 4,
    ) -> None:
        self.path = path or update_marker_path()
        self.acquired = False
        self.holder: UpdateHolder | None = None
        self._fingerprint = uuid.uuid4().hex
        self._refresh_interval_seconds = refresh_interval_seconds
        self._stop_refresh = threading.Event()
        self._refresh_thread: threading.Thread | None = None

    def _payload(self) -> str:
        return f"{os.getpid()}\n{int(time.time())}\n{self._fingerprint}\n"

    def _ownership(self) -> str:
        """``"ours"``, ``"missing"`` (nobody holds the marker), ``"other"`` (a successor or handoff
        partner rewrote it; it is theirs now) or ``"unknown"`` (unreadable right now)."""
        try:
            lines = self.path.read_text(encoding="utf-8-sig").splitlines()
        except FileNotFoundError:
            return "missing"
        except OSError:
            return "unknown"
        if len(lines) >= 3 and lines[0].strip() == str(os.getpid()) and lines[2].strip() == self._fingerprint:
            return "ours"
        return "other"

    def _still_owns(self) -> bool:
        return self._ownership() == "ours"

    def _publish(self, *, replace: bool) -> None:
        """Write a complete payload beside the marker, fsync it, then publish it in one step.

        ``replace=False`` is the claim: ``link()`` succeeds only while no marker exists
        (``FileExistsError`` otherwise), so two contenders cannot both win. ``replace=True`` is
        the renewal: ``os.replace`` swaps the whole file, so a concurrent reader sees the old or
        the new payload and never the empty file an in-place ``O_TRUNC`` write exposes -- which
        every reader (here, ``update-marker.ts``, ``update.rs``) treats as dead and deletes.
        """
        staged = self.path.with_name(f".{self.path.name}.{self._fingerprint}.{'renew' if replace else 'claim'}")
        fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as marker:
                marker.write(self._payload())
                marker.flush()
                os.fsync(marker.fileno())
            if replace:
                os.replace(staged, self.path)
            else:
                os.link(staged, self.path)
        finally:
            with suppress(OSError):
                staged.unlink()

    def _refresh_once(self) -> bool:
        """One renewal tick. ``False`` once the marker belongs to someone else (stop renewing)."""
        state = self._ownership()
        if state == "other":
            return False  # a handoff partner or successor owns it now; never write over them
        if state == "unknown":
            return True  # e.g. a Windows sharing violation; look again next tick
        try:
            if state == "missing":
                # The marker vanished while we still run (deleted by hand, or by an older reader).
                # Claim it again, atomically: if someone else got there first, link() fails and
                # the next tick sees "other".
                self._publish(replace=False)
            else:
                self._publish(replace=True)
        except FileExistsError:
            pass
        except OSError as exc:
            # Transient (e.g. a Windows reader holding the file open): retry next tick; the
            # ceiling allows several misses before the lease goes stale.
            logger.debug("Could not refresh update marker %s: %s", self.path, exc)
        return True

    def _refresh_loop(self) -> None:
        while not self._stop_refresh.wait(self._refresh_interval_seconds):
            if not self._refresh_once():
                return

    def _start_refresh(self) -> None:
        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, name="hermes-update-lease", daemon=True,
        )
        self._refresh_thread.start()

    def acquire(self) -> bool:
        """Claim the lock. Returns False (and sets ``holder``) if it's taken.

        A live holder whose pid matches :data:`HANDOFF_PID_ENV` — or is an ancestor of ours —
        is our own orchestrating parent: run under ITS claim and leave its marker untouched on
        release. The ancestry path covers staged updaters older than the env-var export.
        """
        existing = read_live_update(path=self.path)
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
        if existing is not None:
            # PID reuse can leave a prior run's claim naming us. It cannot belong to
            # another live process, so remove it before the atomic fresh claim.
            with suppress(OSError):
                self.path.unlink()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # link() publishes a complete claim iff the marker does not exist. Unlike
            # check-then-write, two contenders cannot both win or expose partial bytes.
            self._publish(replace=False)
        except FileExistsError:
            if not self.path.parent.is_dir():
                logger.debug("Could not create update marker directory %s", self.path.parent)
                return True
            existing = read_live_update(path=self.path)
            if existing is None:
                return self.acquire()
            if existing.pid == _handoff_pid() or _is_ancestor_pid(existing.pid):
                return True
            self.holder = existing
            return False
        except OSError as exc:
            # Best-effort, like the Rust guard: an unwritable marker must not block the
            # update itself (worse than the race it prevents). Degrade to pre-lock behavior.
            logger.debug("Could not write update marker %s: %s", self.path, exc)
            return True
        self.acquired = True
        self._start_refresh()
        return True

    def release(self) -> None:
        """Drop the marker if this process still owns it. Never raises."""
        if not self.acquired:
            return
        self.acquired = False
        self._stop_refresh.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=max(1.0, self._refresh_interval_seconds + 0.1))
            self._refresh_thread = None
        if not self._still_owns():
            return  # a handoff partner took ownership — still a live update
        with suppress(OSError):
            self.path.unlink()

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
