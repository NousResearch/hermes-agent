"""Cross-process mutual exclusion for in-flight Hermes updates.

Two artifacts, one authority each:

* The update marker ``<root hermes home>/.hermes-update-in-progress`` (contract C1, format v2)
  is shared with the Tauri updater (``UpdateMarkerGuard`` in
  ``apps/bootstrap-installer/src-tauri/src/update.rs``), the Electron gate
  (``electron/update-marker.ts``) and the Desktop hand-off scripts. Body::

      <pid>\\n<started_at>\\nct:<owner creation time, 3 decimals>\\n[delegate:<pid> ct:<ct>\\n]

  An owner is live while its pid is alive and its creation time still matches: never by age
  (the 20-minute ceiling only ages out v1 markers, which carry no creation time).
* The checkout lock ``<git common dir>/hermes-update.lock`` (``<install root>/.hermes-update.lock``
  for a ZIP install with no ``.git``) — a kernel lock (flock / msvcrt) that ``hermes update``
  holds for its whole process tree, so two updates of one checkout started from different
  homes exclude each other and a killed updater whose completion child still runs keeps the
  checkout locked until that child exits. In the git dir it is never a worktree file: no
  ``git status``/autostash sees it, whatever the checked-out tree's ``.gitignore`` says.
"""

from __future__ import annotations

import calendar
import logging
import os
import re
import secrets
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

# A claim published by create-then-write (filesystems without hard links) is briefly empty; an
# empty marker this young is a claim in flight, not a dead one (contract A3).
EMPTY_MARKER_GRACE_SECONDS = 5.0

MARKER_NAME = ".hermes-update-in-progress"
CHECKOUT_LOCK_NAME = ".hermes-update.lock"
GIT_CHECKOUT_LOCK_NAME = "hermes-update.lock"

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


def _git_common_dir(root: Path) -> Path | None:
    """The repository's common git dir, read from disk (no git process: -I -S children).

    ``.git`` is the dir itself, or a ``gitdir: <path>`` file (linked worktree, submodule) whose
    target may name the shared dir in ``commondir``. ``None`` when ``root`` is no checkout.
    """
    dot = root / ".git"
    try:
        if dot.is_dir():
            gitdir = dot
        elif dot.is_file():
            text = dot.read_text(encoding="utf-8-sig").strip()
            if not text.startswith("gitdir:"):
                return None
            gitdir = root / text[len("gitdir:"):].strip()  # an absolute target replaces root
        else:
            return None
        common = gitdir / "commondir"
        if common.is_file():
            gitdir = gitdir / common.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None
    return Path(os.path.normpath(gitdir))


def checkout_lock_path(install_root: Path | str | None = None) -> Path:
    root = Path(install_root or _default_install_root())
    common = _git_common_dir(root)
    return root / CHECKOUT_LOCK_NAME if common is None else common / GIT_CHECKOUT_LOCK_NAME


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
    time (``ps -o lstart=`` in UTC, second resolution — inside the 2 s tolerance), Windows the
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
        # UTC wall clock: a local-time lstart is ambiguous in the repeated DST hour.
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5, stdin=subprocess.DEVNULL,
            env={"PATH": os.environ.get("PATH") or "/bin:/usr/bin", "LC_ALL": "C", "TZ": "UTC0"},
        ).stdout.strip()
        if not out:
            return None
        return float(calendar.timegm(time.strptime(" ".join(out.split()), "%a %b %d %H:%M:%S %Y")))
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


def _identity_live(pid: int, create_time: float | None, age: float) -> bool:
    """C1 rule 3 + A1 for one (pid, ct) identity of a marker ``age`` seconds old.

    Alive (not a zombie) and, when a creation time was recorded, the same process: a matching
    creation time is live however old the marker is. Without that proof — a v1 marker, or a
    creation time we cannot read (Windows denies it for elevated/other-user pids) — the pid may
    be a reused one, so only the legacy age ceiling bounds it.
    """
    if not _pid_alive(pid):
        return False
    actual = None if create_time is None else process_create_time(pid)
    if actual is None:
        return age <= UPDATE_MARKER_MAX_AGE_SECONDS
    return abs(actual - create_time) <= CREATE_TIME_TOLERANCE_SECONDS


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
    started_at: int | None
    create_time: float | None
    delegate_pid: int | None
    delegate_create_time: float | None
    in_flight: bool = False  # an empty marker younger than EMPTY_MARKER_GRACE_SECONDS

    @property
    def base(self) -> bytes:
        """Lines 1–3 exactly as written (what a delegate keeps byte-identical)."""
        return b"".join(self.raw.splitlines(keepends=True)[:3])

    def age(self) -> float:
        return time.time() - self.started_at if self.started_at is not None else float("inf")

    def owner_live(self) -> bool:
        return self.started_at is not None and _identity_live(self.pid, self.create_time, self.age())

    def delegate_live(self) -> bool:
        return self.delegate_pid is not None and self.started_at is not None \
            and _identity_live(self.delegate_pid, self.delegate_create_time, self.age())

    def live_pid(self) -> int | None:
        if self.in_flight:
            return 0
        if self.owner_live():
            return self.pid
        return self.delegate_pid if self.delegate_live() else None


_INT_LINE = re.compile(r"[0-9]+", re.ASCII)
_CT_LINE = re.compile(r"ct:([0-9]+(?:\.[0-9]+)?)", re.ASCII)
_DELEGATE_LINE = re.compile(r"delegate:([0-9]+) ct:([0-9]+(?:\.[0-9]+)?)", re.ASCII)


def _parse_marker(raw: bytes, *, mtime: float | None = None) -> _Marker:
    """Contract A2, positional and identical in every reader (Rust ``marker.rs``, Electron,
    the hand-off scripts): BOM and CRLF tolerated; line 1 pid and line 2 started_at are
    integers or the marker is MALFORMED (dead: ``started_at`` None); a bad line 3 makes it v1;
    a bad line 4 is ignored."""
    text = raw.decode("utf-8", errors="replace").removeprefix("\ufeff")
    lines = [line.removesuffix("\r").strip(" \t") for line in text.split("\n")]
    lines += [""] * (4 - len(lines))
    pid = int(lines[0]) if _INT_LINE.fullmatch(lines[0]) else -1
    started_at = int(lines[1]) if pid >= 0 and _INT_LINE.fullmatch(lines[1]) else None
    ct = _CT_LINE.fullmatch(lines[2])
    delegate = _DELEGATE_LINE.fullmatch(lines[3])
    in_flight = not raw and mtime is not None and time.time() - mtime < EMPTY_MARKER_GRACE_SECONDS
    return _Marker(
        raw=raw, pid=pid, started_at=started_at, create_time=float(ct.group(1)) if ct else None,
        delegate_pid=int(delegate.group(1)) if delegate else None,
        delegate_create_time=float(delegate.group(2)) if delegate else None, in_flight=in_flight,
    )


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _read_marker(path: Path) -> _Marker | None:
    raw = _read_bytes(path)
    if raw is None:
        return None
    mtime = None
    if not raw:
        with suppress(OSError):
            mtime = path.stat().st_mtime
    return _parse_marker(raw, mtime=mtime)


def _tmp_sibling(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")


def _sweep_dead_tmp_siblings(path: Path) -> None:
    """Reclaim ``<marker>.<pid>[.<token>].tmp`` files whose writer died between write and
    publish (contract m10): the pid is the first component after the marker name."""
    prefix = f"{path.name}."
    with suppress(OSError):
        for entry in path.parent.iterdir():
            name = entry.name
            if not (name.startswith(prefix) and name.endswith(".tmp")):
                continue
            owner = name[len(prefix):].split(".", 1)[0]
            if owner.isdigit() and int(owner) != os.getpid() and not _pid_alive(int(owner)):
                with suppress(OSError):
                    entry.unlink()


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
    tmp = _tmp_sibling(path)
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
    """Live identities behind a marker: the owner, then the delegate (``[0]`` for a claim
    still being written).

    A v1 marker naming our own pid is a killed update's claim whose pid this run inherited
    (containers restart pid numbering), not ours: without a creation time it cannot be told
    apart, and nothing pre-writes a v1 marker for this process.
    """
    if marker.in_flight:
        return [0]
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
    parsed = _read_marker(marker)
    if parsed is None:
        return None
    live = parsed.live_pid()
    if live is None:
        _compare_and_delete(marker, parsed.raw)
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
    who = f", process {holder.pid}" if holder and holder.pid else ""
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


def _open_lock_file(path: Path) -> tuple[int | None, object]:
    """``(fd, True)`` read-write; ``(fd, False)`` read-only for an existing lock file we may not
    write (left root-owned by a ``sudo hermes update``: the kernel lock works on a read-only fd,
    contract A5); ``(None, reason)`` when neither opens."""
    binary = getattr(os, "O_BINARY", 0)
    try:
        return os.open(path, os.O_RDWR | os.O_CREAT | binary, 0o644), True
    except PermissionError as exc:
        denied = exc
    except OSError as exc:
        return None, exc.strerror or exc
    try:
        return os.open(path, os.O_RDONLY | binary), False
    except OSError:
        return None, denied.strerror or denied


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
    fd, writable = _open_lock_file(path)
    if fd is None:
        return UpdateHolder(pid=0, age_seconds=0.0, reason=f"{path} is not writable ({writable})")
    try:
        if not _try_lock(fd):
            os.close(fd)
            return _lock_holder(path)
        if writable is True:
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


def bind_child_to_update_tree(proc: subprocess.Popen) -> bool:
    """Windows: put an update-tree child in a kill-on-close job owned by this process, so the
    child (and everything it spawns) dies when the lock owner dies and frees the lock.

    The job allows breakaway: a process the tree starts with ``CREATE_BREAKAWAY_FROM_JOB`` (the
    gateways an update restarts or resumes, ``gateway_windows._spawn_detached``) leaves it and
    outlives the update; every other descendant stays bound. Not ``SILENT_BREAKAWAY_OK``, which
    would let every descendant escape.

    POSIX children inherit the lock fd instead (:func:`checkout_lock_fds`). Returns False (and
    logs) when the job cannot be set up: the caller runs post-commit work, which must not fail
    over a weaker lock.
    """
    if sys.platform != "win32":
        return True
    try:
        _bind_to_kill_on_close_job(proc)
    except OSError as exc:
        logger.warning("Could not bind update child %s to the update's job: %s", proc.pid, exc)
        return False
    return True


def _bind_to_kill_on_close_job(proc: subprocess.Popen) -> None:
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
    # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK
    limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x0800
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


def _publish_exclusive(path: Path, body: bytes) -> bool:
    """Contract A3: publish ``body`` at ``path`` only if nothing is there, never as an empty
    file a reader could judge dead. Write a private tmp sibling, then hard-link it into place
    (fails if the marker exists); a filesystem without hard links falls back to an exclusive
    create (readers grant a young empty marker EMPTY_MARKER_GRACE_SECONDS). False = taken."""
    tmp = _tmp_sibling(path)
    try:
        with open(tmp, "xb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
            return True
        except FileExistsError:
            return False
        except OSError as exc:
            logger.debug("No hard link for the update marker (%s); exclusive create instead", exc)
    finally:
        with suppress(OSError):
            tmp.unlink()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


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
        _sweep_dead_tmp_siblings(self.path)
        body = f"{os.getpid()}\n{int(time.time())}\n{_identity_line()}\n".encode()
        for attempt in range(2):
            try:
                published = _publish_exclusive(self.path, body)
            except OSError as exc:
                self.holder = UpdateHolder(pid=0, age_seconds=0.0, reason=f"{self.path} is not writable ({exc})")
                return False
            if published:
                self._written = body
                self.acquired = True
                return True
            existing = _read_marker(self.path)
            if existing is None:
                continue  # vanished between publish and read: retry
            if _live_partners(existing):
                return self._adopt_or_refuse(existing)
            if attempt == 0 and _compare_and_delete(self.path, existing.raw):
                continue
            self.holder = UpdateHolder(pid=max(existing.pid, 0), age_seconds=0.0)
            return False
        self.holder = read_live_update(path=self.path) or UpdateHolder(pid=0, age_seconds=0.0)
        return False

    def _adopt_or_refuse(self, existing: _Marker) -> bool:
        """C1 rule 4: a LIVE claim by us, an ancestor or the hand-off partner is run under."""
        partners = _live_partners(existing)
        if os.getpid() not in partners and not any(
                p and (p == _handoff_pid() or _is_ancestor_pid(p)) for p in partners):
            self.holder = UpdateHolder(pid=partners[0], age_seconds=existing.age() if existing.started_at else 0.0)
            return False
        own = _identity_line()
        if os.getpid() not in partners and existing.delegate_pid not in partners \
                and existing.create_time is not None and own:
            # Rule 6: name ourselves as the delegate so the claim stays visible if the partner
            # (a hand-off script, the Tauri updater) dies while this update still runs.
            base = existing.base if existing.base.endswith(b"\n") else existing.base + b"\n"
            delegated = base + f"delegate:{os.getpid()} {own}\n".encode()
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
