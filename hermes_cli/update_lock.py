"""Cross-process mutual exclusion for in-flight Hermes updates.

Two artifacts, one authority each:

* The update marker ``<root hermes home>/.hermes-update-in-progress`` (contract C1, format v2)
  is shared with the Tauri updater (``UpdateMarkerGuard`` in
  ``apps/bootstrap-installer/src-tauri/src/update.rs``), the Electron gate
  (``electron/update-marker.ts``) and the Desktop hand-off scripts. Body::

      <pid>\\n<started_at>\\nct:<owner creation time, 3 decimals>\\n[delegate:<pid> ct:<ct>\\n]

  An owner is live while its pid is alive and its creation time still matches: never by age
  (the 20-minute ceiling only ages out v1 markers, which carry no creation time).
* The checkout lock ``<install root>/.hermes-update.lock`` — a kernel lock (flock / msvcrt)
  that ``hermes update`` holds for its whole process tree, so two updates of one checkout
  started from different homes exclude each other and a killed updater whose completion
  child still runs keeps the checkout locked until that child exits.
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

# Applies to v1 markers only (no creation-time line): their pid may have been reused, and a
# ceiling is the only way such a marker self-heals. A v2 owner is live for as long as it runs.
UPDATE_MARKER_MAX_AGE_SECONDS = 20 * 60

# Clock skew allowed between a recorded and a probed process creation time (C1 rule 3).
CREATE_TIME_TOLERANCE_SECONDS = 2.0

MARKER_NAME = ".hermes-update-in-progress"
CHECKOUT_LOCK_NAME = ".hermes-update.lock"

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

# msvcrt locks a byte range; lock one byte far past the holder record so other processes can
# still read who holds it (a locked range is unreadable to them on Windows).
_WINDOWS_LOCK_OFFSET = 1 << 20

_FILETIME_UNIX_EPOCH = 116444736000000000


def update_marker_path() -> Path:
    """Path of the shared update marker: always the profile-tree ROOT home.

    A sticky or ``-p`` profile re-homes ``HERMES_HOME`` to ``<root>/profiles/<p>``; the Desktop,
    the hand-off scripts and the Tauri updater all look at the root, so a profile-scoped marker
    would be one the other owners never see.
    """
    try:
        from hermes_constants import get_default_hermes_root
    except ImportError:  # a partial tree (an -I -S completion child of a stubbed checkout)
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
        root = home.parent.parent if home.parent.name == "profiles" else home
        return root / MARKER_NAME
    return get_default_hermes_root() / MARKER_NAME


def _default_install_root() -> Path:
    return Path(__file__).resolve().parents[1]


def checkout_lock_path(install_root: Path | str | None = None) -> Path:
    return Path(install_root or _default_install_root()) / CHECKOUT_LOCK_NAME


def _pid_alive(pid: int) -> bool:
    """Use the dependency-free, Windows-safe, zombie-aware probe before PM is available."""
    if pid <= 0:
        return False
    try:
        from hermes_cli._early_recovery import _pid_is_running
        return _pid_is_running(pid)
    except Exception as exc:
        logger.debug("Could not probe pid %s: %s", pid, exc)
        return False


# --- process creation time ---------------------------------------------------------------


def _windows_kernel32():
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _windows_create_filetime(pid: int) -> int | None:
    """GetProcessTimes creation FILETIME (100 ns ticks since 1601) or ``None``."""
    import ctypes
    from ctypes import wintypes

    kernel32 = _windows_kernel32()
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


def _stdlib_create_time(pid: int) -> float | None:
    """Creation time in unix seconds without psutil (``-I -S`` children, early recovery).

    Same clock psutil reports: Linux ``starttime / CLK_TCK + btime``, macOS the kernel start
    time (``ps -o lstart=``, second resolution — inside the 2 s tolerance), Windows the
    ``GetProcessTimes`` creation FILETIME.
    """
    try:
        if sys.platform == "win32":
            ticks = _windows_create_filetime(pid)
            return None if ticks is None else (ticks - _FILETIME_UNIX_EPOCH) / 1e7
        if os.path.isdir("/proc"):
            with open(f"/proc/{pid}/stat", "rb") as fh:
                stat = fh.read()
            start_ticks = int(stat[stat.rindex(b")") + 2:].split()[19])
            with open("/proc/stat", "rb") as fh:
                btime = next(int(line.split()[1]) for line in fh if line.startswith(b"btime "))
            return btime + start_ticks / os.sysconf("SC_CLK_TCK")
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5, stdin=subprocess.DEVNULL,
            env={**os.environ, "LC_ALL": "C"},
        ).stdout.strip()
        if not out:
            return None
        return time.mktime(time.strptime(" ".join(out.split()), "%a %b %d %H:%M:%S %Y"))
    except (OSError, ValueError, IndexError, StopIteration, subprocess.SubprocessError, AttributeError):
        return None


def process_create_time(pid: int | None = None) -> float | None:
    """Creation time of ``pid`` (default: this process) in unix seconds, or ``None``.

    psutil when importable — the value ``process_identity._process_create_time`` records —
    else the stdlib probe of the same kernel clock, so a marker written by either is
    comparable by the other within :data:`CREATE_TIME_TOLERANCE_SECONDS`.
    """
    target = os.getpid() if pid is None else pid
    try:
        import psutil
    except ImportError:
        return _stdlib_create_time(target)
    try:
        return float(psutil.Process(target).create_time())
    except Exception:
        return _stdlib_create_time(target)


def _identity_live(pid: int, create_time: float | None) -> bool:
    """C1 rule 3 for one (pid, ct) identity: alive, not a zombie, and not a reused pid."""
    if not _pid_alive(pid):
        return False
    if create_time is None:
        return True
    actual = process_create_time(pid)
    return actual is None or abs(actual - create_time) <= CREATE_TIME_TOLERANCE_SECONDS


def _identity_line(pid: int | None = None) -> str:
    ct = process_create_time(pid)
    return "" if ct is None else f"ct:{ct:.3f}"


# --- ancestry ----------------------------------------------------------------------------


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

    kernel32 = _windows_kernel32()
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    for walk in (kernel32.Process32FirstW, kernel32.Process32NextW):
        walk.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        walk.restype = wintypes.BOOL

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
    parent_created, child_created = _windows_create_filetime(parent), _windows_create_filetime(pid)
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
                stdin=subprocess.DEVNULL,
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


# --- the marker --------------------------------------------------------------------------


@dataclass(frozen=True)
class UpdateHolder:
    """A confirmed-live update holding the lock, or the reason a claim was refused."""

    pid: int
    age_seconds: float
    reason: str | None = None


@dataclass(frozen=True)
class _Marker:
    raw: bytes
    pid: int
    started_at: float | None
    create_time: float | None
    delegate_pid: int | None
    delegate_create_time: float | None

    @property
    def base(self) -> bytes:
        """Lines 1–3 exactly as written (what a delegate keeps byte-identical)."""
        return b"".join(self.raw.splitlines(keepends=True)[:3])

    def age(self) -> float:
        return time.time() - self.started_at if self.started_at is not None else float("inf")

    def owner_live(self) -> bool:
        if self.create_time is None and self.age() > UPDATE_MARKER_MAX_AGE_SECONDS:
            return False  # v1 marker: pid-only, so only a ceiling can expose a reused pid
        return _identity_live(self.pid, self.create_time)

    def delegate_live(self) -> bool:
        return self.delegate_pid is not None and _identity_live(self.delegate_pid, self.delegate_create_time)

    def live_pid(self) -> int | None:
        if self.owner_live():
            return self.pid
        return self.delegate_pid if self.delegate_live() else None


def _parse_ct(text: str) -> float | None:
    text = text.strip()
    if not text.startswith("ct:"):
        return None
    try:
        return float(text[3:])
    except ValueError:
        return None


def _parse_marker(raw: bytes) -> _Marker:
    lines = raw.decode("utf-8-sig", errors="replace").splitlines()

    def field(index: int, cast):
        try:
            return cast(lines[index].strip())
        except (IndexError, ValueError):
            return None

    delegate_pid = delegate_ct = None
    if len(lines) > 3 and lines[3].startswith("delegate:"):
        head, _, tail = lines[3][len("delegate:"):].partition(" ")
        with suppress(ValueError):
            delegate_pid = int(head)
        delegate_ct = _parse_ct(tail)
    pid = field(0, int)
    return _Marker(
        raw=raw, pid=pid if pid is not None else -1, started_at=field(1, float),
        create_time=_parse_ct(lines[2]) if len(lines) > 2 else None,
        delegate_pid=delegate_pid, delegate_create_time=delegate_ct,
    )


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _compare_and_delete(path: Path, expected: bytes) -> bool:
    """Delete ``path`` only while it still holds exactly ``expected`` (C1 rule 5)."""
    if _read_bytes(path) != expected:
        return False
    with suppress(FileNotFoundError):
        path.unlink()
    return True


def _compare_and_swap(path: Path, expected: bytes, new: bytes) -> bool:
    """Atomically replace ``path`` (tmp + ``os.replace``) only while it still holds ``expected``."""
    if _read_bytes(path) != expected:
        return False
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(new)
            fh.flush()
            os.fsync(fh.fileno())
        if _read_bytes(path) != expected:
            return False
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.debug("Could not rewrite update marker %s: %s", path, exc)
        return False
    finally:
        with suppress(OSError):
            tmp.unlink()


def _live_partners(marker: _Marker) -> list[int]:
    """Live identities behind a marker: the owner, then the delegate.

    A v1 marker naming our own pid is a killed update's claim whose pid this run inherited
    (containers restart pid numbering), not ours: without a creation time it cannot be told
    apart, and nothing pre-writes a v1 marker for this process.
    """
    partners = []
    if marker.owner_live() and not (marker.pid == os.getpid() and marker.create_time is None):
        partners.append(marker.pid)
    if marker.delegate_live():
        partners.append(marker.delegate_pid)
    return partners


def read_live_update(*, path: Path | None = None) -> UpdateHolder | None:
    """Return the live update holding the marker, or ``None``.

    Mirrors ``readLiveUpdateMarker`` in ``electron/update-marker.ts``: absent, unreadable,
    malformed and dead-owner all mean "no live update", and a dead marker is removed with
    compare-and-delete so it can't strand future runs. Never raises.
    """
    marker = path or update_marker_path()
    raw = _read_bytes(marker)
    if raw is None:
        return None
    parsed = _parse_marker(raw)
    live = parsed.live_pid()
    if live is None:
        _compare_and_delete(marker, raw)
        return None
    return UpdateHolder(pid=live, age_seconds=parsed.age() if parsed.started_at is not None else 0.0)


def describe_holder(holder: UpdateHolder | None) -> str:
    """One-line, user-facing explanation of who holds the update lock."""
    if holder is not None and holder.reason:
        return (
            f"✗ Cannot lock this install for the update: {holder.reason}.\n"
            "\n"
            "  Updating without the lock could let two updates corrupt the install.\n"
            "  Run `hermes update` as the user that owns the install."
        )
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


# --- the checkout lock -------------------------------------------------------------------

# This process's hold on the checkout lock: {"path", "fd", "owned", "depth"}. "owned" means
# we opened and locked it; otherwise the fd was inherited from the `hermes update` that holds
# it (pass_fds) and belongs to the whole tree — it is never unlocked or closed here.
_HELD: dict | None = None
_JOBS: list = []


def _try_lock(fd: int) -> bool:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        with suppress(OSError):
            os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def _inherited_lock_fd(path: Path) -> int | None:
    """An fd this process inherited that holds the lock on ``path`` (POSIX ``pass_fds``)."""
    if sys.platform == "win32":
        return None
    try:
        target = os.stat(path)
        fd_dir = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
        candidates = [int(name) for name in os.listdir(fd_dir) if name.isdigit()]
    except OSError:
        return None
    import fcntl

    for fd in candidates:
        try:
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) != (target.st_dev, target.st_ino):
                continue
            # Succeeds only when this open file description already holds the lock (or the
            # lock is free, in which case taking it on an fd we hold is still correct).
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            continue
    return None


def _lock_holder(fd_or_path) -> UpdateHolder:
    raw = _read_bytes(fd_or_path) or b""
    parsed = _parse_marker(raw)
    return UpdateHolder(pid=max(parsed.pid, 0), age_seconds=parsed.age() if parsed.started_at else 0.0)


def _acquire_checkout(install_root: Path) -> UpdateHolder | None:
    """Take (or join, when inherited) the checkout lock; the refusal holder, else ``None``."""
    global _HELD
    path = checkout_lock_path(install_root)
    if _HELD is not None and _HELD["path"] == str(path):
        _HELD["depth"] += 1
        return None
    inherited = _inherited_lock_fd(path)
    if inherited is not None:
        _HELD = {"path": str(path), "fd": inherited, "owned": False, "depth": 1}
        return None
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    except OSError as exc:
        return UpdateHolder(pid=0, age_seconds=0.0,
                            reason=f"{path} is not writable ({exc.strerror or exc})")
    try:
        if not _try_lock(fd):
            os.close(fd)
            return _lock_holder(path)
        record = f"{os.getpid()}\n{int(time.time())}\n{_identity_line()}\n".encode()
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, record)
    except OSError as exc:
        _unlock(fd)
        os.close(fd)
        return UpdateHolder(pid=0, age_seconds=0.0, reason=f"{path} could not be locked ({exc})")
    _HELD = {"path": str(path), "fd": fd, "owned": True, "depth": 1}
    return None


def _release_checkout() -> None:
    global _HELD
    if _HELD is None:
        return
    _HELD["depth"] -= 1
    if _HELD["depth"] > 0:
        return
    held, _HELD = _HELD, None
    if held["owned"]:
        # Close, never LOCK_UN: flock belongs to the open file description, which completion
        # children share through pass_fds. A survivor keeps the checkout locked until it exits.
        _unlock(held["fd"])
        with suppress(OSError):
            os.close(held["fd"])


def checkout_lock_fds(install_root: Path | str | None = None) -> tuple[int, ...]:
    """Fds a child of the update tree must inherit (``subprocess`` ``pass_fds``) so the
    checkout stays locked while it runs, even after its parent is killed. POSIX only."""
    if sys.platform == "win32":
        return ()
    if _HELD is not None:
        return (_HELD["fd"],)
    fd = _inherited_lock_fd(checkout_lock_path(install_root))
    return () if fd is None else (fd,)


def bind_child_to_update_tree(proc: subprocess.Popen) -> None:
    """Windows: put an update-tree child in a kill-on-close job owned by this process, so the
    child (and everything it spawns) dies when the lock owner dies and frees the lock.

    POSIX children inherit the lock fd instead (:func:`checkout_lock_fds`).
    """
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class _Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _Basic), ("IoInfo", ctypes.c_ulonglong * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    job = kernel32.CreateJobObjectW(None, None)  # unnamed, non-inheritable: only we hold it
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = _Extended()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)) \
            or not kernel32.AssignProcessToJobObject(job, int(proc._handle)):
        raise ctypes.WinError(ctypes.get_last_error())
    _JOBS.append(job)  # never closed: the handle closes when this process dies, killing the tree


def update_in_progress(install_root: Path | str | None = None) -> bool:
    """True while an update owns this install: a LIVE marker or a held checkout lock."""
    if read_live_update() is not None:
        return True
    path = checkout_lock_path(install_root)
    if _HELD is not None and _HELD["path"] == str(path):
        return True
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError:
        return False
    try:
        if not _try_lock(fd):
            return True
        _unlock(fd)
        return False
    except OSError:
        return False
    finally:
        os.close(fd)


# --- the lock object ---------------------------------------------------------------------


class UpdateLock:
    """Context manager owning the shared update marker (and, with ``install_root``, the
    checkout lock) for this process.

    ``acquired`` is True when we wrote the marker; adopting a live partner's claim (hand-off
    pid, ancestor, or an outer claim of this same process) succeeds with ``acquired`` False.
    ``acquire`` returns False (and sets ``holder``) when another live update owns either lock
    or the lock cannot be created at all — never "proceed unlocked".
    """

    def __init__(self, *, path: Path | None = None, install_root: Path | str | None = None) -> None:
        self.path = path or update_marker_path()
        self.install_root = None if install_root is None else Path(install_root)
        self.acquired = False
        self.holder: UpdateHolder | None = None
        self._written: bytes | None = None
        self._delegate_base: bytes | None = None
        self._checkout = False

    def acquire(self) -> bool:
        if self.install_root is not None:
            refused = _acquire_checkout(self.install_root)
            if refused is not None:
                self.holder = refused
                return False
            self._checkout = True
        try:
            ok = self._claim_marker()
        except BaseException:
            self._drop_checkout()
            raise
        if not ok:
            self._drop_checkout()
        return ok

    def _claim_marker(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.holder = UpdateHolder(pid=0, age_seconds=0.0, reason=f"{self.path.parent} is not writable ({exc})")
            return False
        body = f"{os.getpid()}\n{int(time.time())}\n{_identity_line()}\n".encode()
        for attempt in range(2):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
            except FileExistsError:
                raw = _read_bytes(self.path)
                if raw is None:
                    continue  # vanished between create and read: retry the create
                existing = _parse_marker(raw)
                if _live_partners(existing):
                    return self._adopt_or_refuse(existing)
                if attempt == 0 and _compare_and_delete(self.path, raw):
                    continue
                self.holder = UpdateHolder(pid=max(existing.pid, 0), age_seconds=0.0)
                return False
            except OSError as exc:
                self.holder = UpdateHolder(pid=0, age_seconds=0.0, reason=f"{self.path} is not writable ({exc})")
                return False
            try:
                os.write(fd, body)
                os.fsync(fd)
            finally:
                os.close(fd)
            self._written = body
            self.acquired = True
            return True
        self.holder = read_live_update(path=self.path) or UpdateHolder(pid=0, age_seconds=0.0)
        return False

    def _adopt_or_refuse(self, existing: _Marker) -> bool:
        """C1 rule 4: a LIVE claim by us, an ancestor or the hand-off partner is run under."""
        partners = _live_partners(existing)
        if os.getpid() not in partners and not any(
                p == _handoff_pid() or _is_ancestor_pid(p) for p in partners):
            self.holder = UpdateHolder(pid=partners[0], age_seconds=existing.age() if existing.started_at else 0.0)
            return False
        if os.getpid() not in partners and existing.delegate_pid not in partners \
                and existing.create_time is not None:
            # Rule 6: name ourselves as the delegate so the claim stays visible if the partner
            # (a hand-off script, the Tauri updater) dies while this update still runs.
            delegated = existing.base + f"delegate:{os.getpid()} {_identity_line()}\n".encode()
            if _compare_and_swap(self.path, existing.raw, delegated):
                self._written, self._delegate_base = delegated, existing.base
        return True

    def _drop_checkout(self) -> None:
        if self._checkout:
            self._checkout = False
            _release_checkout()

    def release(self) -> None:
        """Give back what we wrote (compare-and-delete / compare-and-swap). Never raises."""
        try:
            if self._written is not None:
                current = _read_bytes(self.path)
                if current is None:
                    pass
                elif self.acquired:
                    if current == self._written:
                        _compare_and_delete(self.path, current)
                    elif current.startswith(self._written) \
                            and not _parse_marker(current).delegate_live():
                        _compare_and_delete(self.path, current)  # our claim + a dead delegate
                elif current == self._written and self._delegate_base is not None:
                    if _parse_marker(self._delegate_base).owner_live():
                        _compare_and_swap(self.path, current, self._delegate_base)
                    else:
                        _compare_and_delete(self.path, current)
        except OSError as exc:
            logger.debug("Could not release update marker %s: %s", self.path, exc)
        finally:
            self.acquired = False
            self._written = self._delegate_base = None
            self._drop_checkout()

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
