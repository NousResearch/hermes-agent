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
import errno
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG

logger = logging.getLogger(__name__)

# Applies to v1 markers only (no creation-time line): their pid may have been reused, and a
# ceiling is the only way such a marker self-heals. A v2 owner is live for as long as it runs.
UPDATE_MARKER_MAX_AGE_SECONDS = 20 * 60

# Clock skew allowed between a recorded and a probed process creation time (C1 rule 3).
CREATE_TIME_TOLERANCE_SECONDS = 2.0
# Our own creation time re-probed by the same clock: only the marker's 3-decimal rounding differs,
# while two processes are at least one scheduler tick (10 ms) apart.
_OWN_CREATE_TIME_EPSILON = 0.005

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
# R5b: the lease bytes just past it. A Windows process that joins its ancestor's lock (it cannot
# inherit it) also locks one lease byte of its own: a completion child the job refused runs
# outside the kill-on-close job and outlives a killed owner, and its lease keeps the checkout
# busy until it exits. Takers and probes treat any held lease as a held lock.
_LEASE_SLOTS = 16

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


def _own_create_time() -> float | None:
    """This process's creation time (cached per pid: a forked child is another process)."""
    pid = os.getpid()
    if _OWN_CT.get("pid") != pid:
        _OWN_CT.update(pid=pid, ct=process_create_time(pid))
    return _OWN_CT.get("ct")


_OWN_CT: dict = {}


def _as_ct(recorded) -> float | None:
    """A recorded creation time as a float: a number, ``"ct:<s>"`` / ``"<s>"`` text, or None."""
    if recorded is None or isinstance(recorded, (int, float)):
        return None if recorded is None else float(recorded)
    match = _CT_VALUE.fullmatch(str(recorded).strip().removeprefix("ct:"))
    return float(match.group(0)) if match else None


def incarnation_live(pid: int, recorded_ct=None) -> bool | None:
    """THE one incarnation rule (A7 rule 4) for every Python identity reader (marker, pause record).

    An identity is (pid, creation time). ``True``: that process is running. ``False``: no such
    process, a different incarnation of the pid, or a claim naming OUR pid that is not us — our
    own pid is ours only within :data:`_OWN_CREATE_TIME_EPSILON` of our creation time, and a
    claim without a creation time naming our pid is a previous incarnation (a fresh pid
    namespace hands a killed update's pid to the next launch). ``None``: alive but unprovable —
    no creation time recorded, or the live one unreadable (Windows denies elevated/other-user
    pids); each caller applies its own bound (the marker: the v1 age ceiling).

    ``recorded_ct`` is a float or the marker spelling ``"ct:<seconds>"``.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    recorded = _as_ct(recorded_ct)
    if pid == os.getpid():
        own = _own_create_time()
        if own is None:
            return None  # degraded: we cannot tell our incarnations apart
        return recorded is not None and abs(own - recorded) <= _OWN_CREATE_TIME_EPSILON
    if not _pid_alive(pid):
        return False
    if recorded is None:
        return None
    actual = process_create_time(pid)
    return None if actual is None else abs(actual - recorded) <= CREATE_TIME_TOLERANCE_SECONDS


@dataclass(frozen=True)
class _World:
    """What a marker verdict depends on besides the bytes: us, the clock, the process table.
    Production uses :func:`_real_world`; the shared corpus injects a table."""

    pid: int
    ct: float | None
    now: float
    alive: object  # Callable[[int], bool]
    ct_of: object  # Callable[[int], float | None]


def _real_world() -> _World:
    return _World(os.getpid(), _own_create_time(), time.time(), _pid_alive, process_create_time)


def _identity_live(pid: int, create_time: float | None, age: float, world: _World | None = None) -> bool:
    """C1 rule 3 + A1 + A7 rule 4 for one (pid, ct) identity of a marker ``age`` seconds old.

    Alive (not a zombie) and, when a creation time was recorded, the same process: a matching
    creation time is live however old the marker is. Without that proof — a v1 marker, or a
    creation time we cannot read — the pid may be a reused one, so only the legacy age ceiling
    bounds it. Our own pid follows :func:`incarnation_live` (exact creation time, no-ct = dead).
    """
    w = world or _real_world()
    if pid <= 0:
        return False
    if pid == w.pid:  # we are alive by definition: only the incarnation is in question
        if w.ct is None:
            # Degraded: our own creation time is unreadable, so our claims are v1 too.
            return create_time is None
        return create_time is not None and abs(w.ct - create_time) <= _OWN_CREATE_TIME_EPSILON
    if not w.alive(pid):
        return False
    actual = None if create_time is None else w.ct_of(pid)
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
    """A confirmed-live update holding the lock, or the reason a claim was refused.

    ``held``: the marker's identities are dead but the checkout kernel lock is still held (a
    killed updater's completion/build child or git still runs) — the update is NOT over; the
    hand-off scripts' ``held`` verdict."""

    pid: int
    age_seconds: float
    reason: str | None = None
    held: bool = False


_U32_MAX = 0xFFFFFFFF
_U64_MAX = 0xFFFFFFFFFFFFFFFF
_INT_LINE = re.compile(r"[0-9]+", re.ASCII)
_CT_VALUE = re.compile(r"[0-9]+(?:\.[0-9]+)?", re.ASCII)
_CT_LINE = re.compile(r"ct:([0-9]+(?:\.[0-9]+)?)", re.ASCII)
_DELEGATE_LINE = re.compile(r"delegate:([0-9]+) ct:([0-9]+(?:\.[0-9]+)?)", re.ASCII)
_RUN_LINE = re.compile(r"run:([A-Za-z0-9._-]{1,128})", re.ASCII)


@dataclass(frozen=True)
class _Marker:
    raw: bytes
    pid: int
    started_at: int | None
    create_time: float | None
    delegate_pid: int | None
    delegate_create_time: float | None
    in_flight: bool = False  # an empty marker younger than EMPTY_MARKER_GRACE_SECONDS
    ct_text: str | None = None
    delegate_ct_text: str | None = None
    runs: tuple[str, ...] = ()

    @property
    def base(self) -> bytes:
        """Lines 1–3 exactly as written (what a delegate keeps byte-identical)."""
        return b"".join(self.raw.splitlines(keepends=True)[:3])

    @property
    def run(self) -> str | None:
        return self.runs[0] if self.runs else None

    def age(self, world: _World | None = None) -> float:
        now = time.time() if world is None else world.now
        return now - self.started_at if self.started_at is not None else float("inf")

    def owner_live(self, world: _World | None = None) -> bool:
        return self.started_at is not None \
            and _identity_live(self.pid, self.create_time, self.age(world), world)

    def delegate_live(self, world: _World | None = None) -> bool:
        return self.delegate_pid is not None and self.started_at is not None \
            and _identity_live(self.delegate_pid, self.delegate_create_time, self.age(world), world)

    def live_pid(self, world: _World | None = None) -> int | None:
        if self.in_flight:
            return 0
        if self.owner_live(world):
            return self.pid
        return self.delegate_pid if self.delegate_live(world) else None

    def canonical(self, pid: int, ct_text: str | None) -> bytes:
        """A rewritten claim (A7 rule 5): ``pid`` owns it from ``started_at`` on; run lines kept."""
        body = f"{pid}\n{self.started_at}\n" + (f"ct:{ct_text}\n" if ct_text else "")
        return (body + "".join(f"run:{run}\n" for run in self.runs)).encode()


def _bounded_int(text: str, limit: int) -> int | None:
    """ASCII digits whose value fits ``limit`` (u32 pid, u64 started_at — Rust's ``parse``), else
    None. Leading zeros are fine; the significant digits are counted BEFORE ``int()``, which
    refuses past 4300 digits — an oversized field is malformed, never an exception."""
    if not _INT_LINE.fullmatch(text):
        return None
    significant = text.lstrip("0") or "0"
    if len(significant) > len(str(limit)):
        return None
    value = int(significant)
    return value if value <= limit else None


def _parse_marker(raw: bytes, *, mtime: float | None = None) -> _Marker:
    """Contract A2 + A7 rule 7, identical in every reader (Rust ``marker.rs``, Electron, the
    hand-off scripts; ``tests/fixtures/update_marker_corpus.json``): BOM, CRLF and surrounding
    spaces/tabs tolerated; line 1 pid (u32) and line 2 started_at (u64) are integers or the marker is
    MALFORMED (dead: ``started_at`` None); a bad line 3 makes it v1; lines 4+ are tagged — the
    first well-formed ``delegate:<pid> ct:<ct>`` and every ``run:<id>`` — anything else ignored."""
    text = raw.decode("utf-8", errors="replace").removeprefix("\ufeff")
    lines = [line.removesuffix("\r").strip(" \t") for line in text.split("\n")]
    lines += [""] * (3 - len(lines))
    pid = _bounded_int(lines[0], _U32_MAX)
    started_at = _bounded_int(lines[1], _U64_MAX) if pid is not None else None
    pid = -1 if pid is None else pid
    ct = _CT_LINE.fullmatch(lines[2])
    delegates = ((m, _bounded_int(m.group(1), _U32_MAX)) for m in map(_DELEGATE_LINE.fullmatch, lines[3:]) if m)
    delegate, delegate_pid = next(((m, d) for m, d in delegates if d is not None), (None, None))
    runs = tuple(m.group(1) for m in map(_RUN_LINE.fullmatch, lines[3:]) if m)
    in_flight = not raw and mtime is not None and time.time() - mtime < EMPTY_MARKER_GRACE_SECONDS
    return _Marker(
        raw=raw, pid=pid, started_at=started_at, create_time=float(ct.group(1)) if ct else None,
        delegate_pid=delegate_pid,
        delegate_create_time=float(delegate.group(2)) if delegate else None, in_flight=in_flight,
        ct_text=ct.group(1) if ct else None, delegate_ct_text=delegate.group(2) if delegate else None,
        runs=runs,
    )


def judge_marker(raw: bytes, world: _World | None = None) -> tuple[str, int | None, str | None]:
    """``(verdict, owner, run)`` of marker bytes — the shared corpus' judge contract.

    verdict: ``malformed`` | ``dead`` | ``ours`` (a live identity is this incarnation) | ``live``;
    owner: the owner identity if live, else the delegate if live, else None.
    """
    w = world or _real_world()
    marker = _parse_marker(raw)
    if marker.started_at is None:
        return "malformed", None, None
    owner = marker.live_pid(w)
    if owner is None:
        return "dead", None, marker.run
    ours = (marker.pid == w.pid and marker.owner_live(w)) \
        or (marker.delegate_pid == w.pid and marker.delegate_live(w))
    return ("ours" if ours else "live"), owner, marker.run


def _release_decision(raw: bytes, world: _World | None = None) -> tuple[str, bytes | None]:
    """A7 rule 5: what releasing ``world``'s claim does to marker bytes ``raw``.

    ``("delete", None)`` | ``("rewrite", new bytes)`` | ``("keep", None)``. The owner (line 1–3
    identity is our incarnation) deletes — regardless of delegate lines — unless a live delegate
    other than us exists: then that delegate becomes the owner. A delegate (us) drops its line
    while the owner lives, else deletes. Anything not naming us is kept.
    """
    w = world or _real_world()
    marker = _parse_marker(raw)
    if marker.started_at is None:
        return "keep", None
    if marker.pid == w.pid and marker.owner_live(w):
        if marker.delegate_pid not in (None, w.pid) and marker.delegate_live(w):
            return "rewrite", marker.canonical(marker.delegate_pid, marker.delegate_ct_text)
        return "delete", None
    if marker.delegate_pid == w.pid and marker.delegate_live(w):
        if marker.owner_live(w):
            return "rewrite", marker.canonical(marker.pid, marker.ct_text)
        return "delete", None
    return "keep", None


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


# Every tmp write of ours publishes or is unlinked well inside this: an older tmp under our own
# pid number is a previous holder's (containers reuse pids every boot), never ours in flight.
OWN_PID_TMP_STALE_SECONDS = 60.0


def _tmp_writer_gone(owner: int | None, entry: Path) -> bool:
    if owner is None:
        return True  # ASCII digits past u32: no process has that pid
    if owner != os.getpid():
        return not _pid_alive(owner)
    return time.time() - entry.stat().st_mtime > OWN_PID_TMP_STALE_SECONDS


def _sweep_dead_tmp_siblings(path: Path) -> None:
    """Reclaim ``<marker>.<pid>[.<token>].tmp`` files whose writer died between write and
    publish (contract m10): the pid is the first component after the marker name."""
    prefix = f"{path.name}."
    with suppress(OSError):
        for entry in path.parent.iterdir():
            name = entry.name
            owner = name[len(prefix):].split(".", 1)[0]
            if not (name.startswith(prefix) and name.endswith(".tmp") and _INT_LINE.fullmatch(owner)):
                continue
            with suppress(OSError):
                if _tmp_writer_gone(_bounded_int(owner, _U32_MAX), entry):
                    entry.unlink()


# --- the marker mutex (A7 rule 1) -------------------------------------------------------------

MUTEX_SUFFIX = ".lock"
# Bounded wait for the sidecar: holders never run git/builds/network inside it.
MUTEX_WAIT_SECONDS = 10.0


class MarkerBusy(OSError):
    """The marker mutex stayed held past :data:`MUTEX_WAIT_SECONDS`: report busy."""


_MUTEX_HELD = threading.local()


def marker_mutex_path(path: Path) -> Path:
    """``<marker>.lock``: the sidecar every marker MUTATION holds a kernel lock on. Never deleted
    (deleting it would split the lock across two inodes)."""
    return path.with_name(path.name + MUTEX_SUFFIX)


def _windows_open_exclusive(path: Path):
    """The Windows sidecar mutex: the file opened with NO sharing (``FileShare.None`` in the
    PowerShell scripts, ``share_mode(0)`` in marker.rs). ``None`` = someone else has it open."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    invalid = ctypes.c_void_p(-1).value
    for access, disposition in ((0xC0000000, 4), (0x80000000, 3)):  # RW OPEN_ALWAYS; R OPEN_EXISTING
        handle = kernel32.CreateFileW(str(path), access, 0, None, disposition, 0x80, None)
        if handle and handle != invalid:
            return handle
        error = ctypes.get_last_error()
        if error == 32:  # ERROR_SHARING_VIOLATION: held
            return None
        if error != 5:  # anything but access denied: no read-only retry helps
            break
    raise ctypes.WinError(error)


def _windows_close(handle) -> None:
    import ctypes

    ctypes.WinDLL("kernel32").CloseHandle(handle)


def _mutex_acquire(path: Path, wait: float):
    deadline = time.monotonic() + wait
    if sys.platform == "win32":
        while True:
            handle = _windows_open_exclusive(path)
            if handle is not None:
                return handle
            if time.monotonic() >= deadline:
                raise MarkerBusy(f"{path} is held by another update")
            time.sleep(0.02)
    import fcntl

    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except PermissionError:
        fd = os.open(path, os.O_RDONLY)  # flock works on a read-only fd (root-owned sidecar)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise MarkerBusy(f"{path} is held by another update") from None
                time.sleep(0.02)
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def marker_mutex(path: Path, *, wait: float = MUTEX_WAIT_SECONDS):
    """Hold the kernel lock on ``<marker>.lock`` (A7 rule 1): read → judge → mutate of the marker
    at ``path`` happens entirely inside. Re-entrant per thread; the kernel releases it if we die.
    Raises :class:`MarkerBusy` after ``wait`` seconds, ``OSError`` when it cannot be opened."""
    key = str(marker_mutex_path(path))
    held = getattr(_MUTEX_HELD, "keys", None)
    if held is None:
        held = _MUTEX_HELD.keys = {}
    if held.get(key):
        held[key] += 1
        try:
            yield
        finally:
            held[key] -= 1
        return
    handle = _mutex_acquire(Path(key), wait)
    held[key] = 1
    try:
        yield
    finally:
        held.pop(key, None)
        if sys.platform == "win32":
            _windows_close(handle)
        else:
            os.close(handle)  # closing the only fd of this open file description drops the flock


def _compare_and_delete(path: Path, expected: bytes) -> bool:
    """Delete ``path`` only while it still holds exactly ``expected`` — compared and unlinked under
    the marker mutex, so no claim can be published in between (A7; C1 rule 5 was not CAS)."""
    with marker_mutex(path):
        if _read_bytes(path) != expected:
            return False
        with suppress(FileNotFoundError):
            path.unlink()
        return True


def _compare_and_swap(path: Path, expected: bytes, new: bytes) -> bool:
    """Atomically replace ``path`` (tmp + ``os.replace``) only while it still holds ``expected``,
    under the marker mutex."""
    with marker_mutex(path):
        if _read_bytes(path) != expected:
            return False
        tmp = _tmp_sibling(path)
        try:
            with open(tmp, "wb") as fh:
                fh.write(new)
                fh.flush()
                os.fsync(fh.fileno())
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
    still being written). A claim naming our own pid counts only at our exact incarnation."""
    if marker.in_flight:
        return [0]
    partners = []
    if marker.owner_live():
        partners.append(marker.pid)
    if marker.delegate_live():
        partners.append(marker.delegate_pid)
    return partners


def read_live_update(*, path: Path | None = None, install_root: Path | str | None = None) -> UpdateHolder | None:
    """Return the live update holding the marker, or ``None``.

    Absent, unreadable, malformed and dead-owner all mean "no live update" — unless the
    checkout kernel lock of ``install_root`` (default: this checkout) is held: then a killed
    updater's tree (its completion, build or git) still runs, the marker is kept, and the
    answer is a ``held`` holder (R6; ``marker.sh``/``marker.ps1`` answer ``held`` too). Otherwise
    a dead marker is reclaimed — judged again and removed inside the marker mutex, so a claim
    published after our first read is never the one deleted. Never raises.
    """
    marker = path or update_marker_path()
    try:
        parsed = _read_marker(marker)
        if parsed is None:
            return None
        live = parsed.live_pid()
        if live is not None:
            return UpdateHolder(pid=live, age_seconds=parsed.age() if parsed.started_at is not None else 0.0)
        if _reclaim_dead(marker, install_root) == "held":
            return UpdateHolder(pid=0, age_seconds=max(parsed.age(), 0.0) if parsed.started_at is not None else 0.0,
                                held=True)
    except Exception as exc:  # "never raises": an unreadable state is no live update
        logger.debug("Could not judge update marker %s: %s", marker, exc)
    return None


def _reclaim_dead(path: Path, install_root: Path | str | None = None) -> str:
    """Remove a dead marker under the mutex: ``reclaimed`` | ``held`` (dead, but the checkout lock
    is held, so it is kept) | ``live`` (a claim appeared) | ``absent`` | ``busy``."""
    try:
        with marker_mutex(path):
            current = _read_marker(path)
            if current is None:
                return "absent"
            if current.live_pid() is not None:
                return "live"
            if checkout_lock_held(install_root):
                return "held"
            with suppress(FileNotFoundError):
                path.unlink()
            return "reclaimed"
    except OSError as exc:  # MarkerBusy included: someone else is deciding right now
        logger.debug("Left update marker %s for its current mutator: %s", path, exc)
        return "busy"


def legacy_profile_claims(root: Path | None = None) -> list[UpdateHolder]:
    """Live claims in pre-root-marker ``<root>/profiles/<p>/.hermes-update-in-progress`` files (R11).

    Before the marker moved to the profile-tree root, an update under a named profile claimed
    its PROFILE home and never took the checkout lock. During the transition such an update can
    still be running: honor it (v1 rules: pid alive, within the legacy age ceiling, never our own
    previous incarnation). Read-only: an old updater reclaims its own dead markers.
    """
    root = update_marker_path().parent if root is None else root
    claims = []
    with suppress(OSError):
        for profile in (root / "profiles").iterdir():
            marker = _read_marker(profile / MARKER_NAME)
            if marker is None or marker.in_flight:
                continue
            live = marker.live_pid()
            if live:
                claims.append(UpdateHolder(pid=live, age_seconds=marker.age()))
    return claims


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
    if holder is not None and holder.held:
        who = "; its owner exited but a process it started still holds the checkout"
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
        if not _lock_bytes(fd, _WINDOWS_LOCK_OFFSET):
            return False
        if not _lock_bytes(fd, _WINDOWS_LOCK_OFFSET + 1, _LEASE_SLOTS):  # a dead owner's leased child
            _unlock(fd)
            return False
        _unlock(fd, _WINDOWS_LOCK_OFFSET + 1, _LEASE_SLOTS)
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _lock_bytes(fd: int, offset: int, count: int = 1) -> bool:
    """Windows: lock ``count`` bytes at ``offset`` without waiting (fails if any is held)."""
    import msvcrt

    os.lseek(fd, offset, os.SEEK_SET)
    try:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, count)
    except OSError:
        return False
    return True


def _unlock(fd: int, offset: int = _WINDOWS_LOCK_OFFSET, count: int = 1) -> None:
    if sys.platform == "win32":
        import msvcrt

        with suppress(OSError):
            os.lseek(fd, offset, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, count)


def _take_lease(fd: int) -> int | None:
    """Windows: lock a free lease byte (R5b) on ``fd``; its offset, or None when all are held
    (the tree's other joiners hold them: this one still runs inside their custody)."""
    for offset in range(_WINDOWS_LOCK_OFFSET + 1, _WINDOWS_LOCK_OFFSET + 1 + _LEASE_SLOTS):
        if _lock_bytes(fd, offset):
            return offset
    return None


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
            # pass_fds clears CLOEXEC in the exec'd child. Local opens (including the
            # concurrent status probe) retain it and must never donate their lifetime:
            # their owner may close the fd while our update still holds custody.
            if not os.get_inheritable(fd):
                continue
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
    """Who holds the checkout lock, from its holder record. A recorded updater that is gone (its
    incarnation is dead) while the lock stays held means a process it started holds it: that is
    the ``held`` holder, never the dead pid (the marker reader's wording, R6)."""
    raw = _read_bytes(fd_or_path) or b""
    parsed = _parse_marker(raw)
    age = parsed.age() if parsed.started_at else 0.0
    if parsed.pid > 0 and incarnation_live(parsed.pid, parsed.create_time) is False:
        return UpdateHolder(pid=0, age_seconds=age, held=True)
    return UpdateHolder(pid=max(parsed.pid, 0), age_seconds=age)


def _open_lock_file(path: Path) -> tuple[int | None, object]:
    """``(fd, True)`` read-write; ``(fd, False)`` read-only for an existing lock file we may not
    write (left root-owned by a ``sudo hermes update``: the kernel lock works on a read-only fd,
    contract A5; POSIX only — a Windows owner needs its record); ``(None, reason)`` when neither opens."""
    # Never follow a link planted at the name: the holder record is written into this file.
    binary = getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
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
    found = os.fstat(fd)
    if not S_ISREG(found.st_mode) or found.st_nlink != 1:
        # A hard link (or special file) at the name: truncating it would rewrite data outside
        # the install. Refuse; never unlink/recreate it (waiters need one stable inode).
        os.close(fd)
        return UpdateHolder(pid=0, age_seconds=0.0, reason=f"{path} is not a regular single-link file; delete it")
    try:
        if not _lock_with_contention_wait(fd, path):
            if _held_by_our_windows_ancestor(path):
                # Windows has no fd inheritance: the update tree's children run in the lock
                # owner's kill-on-close job (bind_child_to_update_tree) and die with it. One the
                # job refused does not, so every joiner also holds a lease byte (R5b), taken
                # while the owner is seen alive: it never covers a writer after a free lock.
                lease = _take_lease(fd)
                if _held_by_our_windows_ancestor(path):
                    _HELD = {"path": str(path), "fd": None, "owned": False, "depth": 1,
                             "lease": None if lease is None else (fd, lease)}
                    if lease is None:
                        os.close(fd)
                    return None
                if lease is not None:
                    _unlock(fd, lease)
            os.close(fd)
            return _lock_holder(path)
        if writable is not True and sys.platform == "win32":
            # No fd inheritance on Windows: the update tree's children find their owner by
            # this record (_held_by_our_windows_ancestor), so an owner that cannot write it
            # would admit itself and then have its own children refused.
            _unlock(fd)
            os.close(fd)
            return UpdateHolder(pid=0, age_seconds=0.0,
                                reason=f"{path} is read-only; delete it or fix its permissions")
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


# A "held?" probe (marker.sh/marker.ps1 checkout_lock_held, Python checkout_lock_held) takes the
# lock for microseconds: an updater that collides with one waits this long before "busy" (D17:
# 257 of 3283 tight-loop acquires failed spuriously with no real holder).
CHECKOUT_CONTENTION_WAIT_SECONDS = 3.0


def _lock_with_contention_wait(fd: int, path: Path) -> bool:
    if _try_lock(fd):
        return True
    if _held_by_our_windows_ancestor(path):
        return False  # our own update tree holds it and never frees it for us: answer at once
    deadline = time.monotonic() + CHECKOUT_CONTENTION_WAIT_SECONDS
    while time.monotonic() < deadline:
        time.sleep(0.02)
        if _try_lock(fd):
            return True
    return False


def _held_by_our_windows_ancestor(path: Path) -> bool:
    """Windows: the checkout lock's holder record names a LIVE incarnation of one of our
    ancestors (the ``hermes update`` whose completion/build children we are)."""
    if sys.platform != "win32":
        return False
    record = _parse_marker(_read_bytes(path) or b"")
    if record.started_at is None or record.pid == os.getpid():
        return False
    return incarnation_live(record.pid, record.create_time) is True and _is_ancestor_pid(record.pid)


def _release_checkout() -> None:
    global _HELD
    if _HELD is None:
        return
    _HELD["depth"] -= 1
    if _HELD["depth"] > 0:
        return
    held, _HELD = _HELD, None
    if held.get("lease"):
        lease_fd, offset = held["lease"]
        _unlock(lease_fd, offset)
        with suppress(OSError):
            os.close(lease_fd)
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
        return () if _HELD["fd"] is None else (_HELD["fd"],)
    fd = _inherited_lock_fd(checkout_lock_path(install_root))
    return () if fd is None else (fd,)


def custody_spawn_kwargs() -> dict:
    """``subprocess`` kwargs that keep a checkout MUTATOR (git, the Node source build) in the
    update tree's custody (R2): POSIX children inherit the held lock fd, so the checkout stays
    locked until the last of them exits even if this process is killed. ``{}`` when this process
    holds no checkout lock, and on Windows (bind the Popen with bind_child_to_update_tree)."""
    if sys.platform == "win32" or _HELD is None or _HELD["fd"] is None:
        return {}
    return {"pass_fds": (_HELD["fd"],)}


CREATE_SUSPENDED = 0x00000004


def bind_child_to_update_tree(proc: subprocess.Popen) -> OSError | None:
    """Windows: put an update-tree child in a kill-on-close job owned by this process, so the
    child (and everything it spawns) dies when the lock owner dies and frees the lock.

    Create ``proc`` with :data:`CREATE_SUSPENDED` and resume it after the bind
    (:func:`resume_suspended_child`): it runs no instruction before, so nothing it starts can
    escape the job. The job allows breakaway: a process the tree starts
    with ``CREATE_BREAKAWAY_FROM_JOB`` (the gateways an update restarts or resumes,
    ``gateway_windows._spawn_detached``) leaves it and outlives the update; every other
    descendant stays bound. Not ``SILENT_BREAKAWAY_OK``, which would let every descendant escape.

    POSIX children inherit the lock fd instead (:func:`checkout_lock_fds`). Returns ``None`` when
    bound, else the refusal (logged): the caller runs post-commit work, which must not fail over
    a weaker lock, so it records the refusal and runs the child, which joins the lock holding its
    own lease byte (R5b): the checkout stays busy until it exits, even after the owner's death.
    """
    if sys.platform != "win32":
        return None
    try:
        _bind_to_kill_on_close_job(proc)
    except OSError as exc:
        logger.warning("Could not bind update child %s to the update's job: %s", proc.pid, exc)
        return exc
    return None


def resume_suspended_child(proc: subprocess.Popen) -> None:
    """Windows: resume a :data:`CREATE_SUSPENDED` child (a no-op for a running one). A failed
    resume kills it and raises ``OSError``."""
    import ctypes

    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    if ntdll.NtResumeProcess(int(proc._handle)) != 0:
        proc.kill()
        raise OSError(f"could not resume update child {proc.pid}")


def _bind_to_kill_on_close_job(proc: subprocess.Popen) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    if not kernel32.AssignProcessToJobObject(update_tree_job(), int(proc._handle)):
        raise ctypes.WinError(ctypes.get_last_error())


def update_tree_job() -> int:
    """Windows: this process's kill-on-close job for the update tree (created once, never
    closed: the handle closes when this process dies, killing every process still in it)."""
    if _JOBS:
        return _JOBS[0]
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
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel32.CreateJobObjectW(None, None)  # unnamed, non-inheritable: only we hold it
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = _Extended()
    # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK
    limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x0800
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise ctypes.WinError(error)
    _JOBS.append(job)  # never closed: the handle closes when this process dies, killing the tree
    return job


def update_in_progress(install_root: Path | str | None = None) -> bool:
    """True while an update owns this install: a LIVE marker or a held checkout lock."""
    return read_live_update(install_root=install_root) is not None or checkout_lock_held(install_root)


def checkout_lock_held(install_root: Path | str | None = None) -> bool:
    """True while some process (this one included) holds the checkout kernel lock.

    A probe takes the lock for the microseconds of one try and drops it (closing the fd), the
    way ``marker.sh::checkout_lock_held`` does; an updater acquiring at that instant waits
    :data:`CHECKOUT_CONTENTION_WAIT_SECONDS` instead of failing (R6/D17)."""
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
    made = os.fstat(fd)
    try:
        if os.write(fd, body) != len(body):
            raise OSError(errno.ENOSPC, f"short write publishing {path}")
        os.fsync(fd)
    except BaseException:
        # A torn claim names our live pid, so it would block every updater for as long as we
        # run without our ever having acquired: withdraw the inode WE created — compared under
        # the marker mutex, so a claimant that reclaimed it and published its own keeps that.
        os.close(fd)
        with suppress(OSError), marker_mutex(path):
            now = os.stat(path)
            if (now.st_dev, now.st_ino) == (made.st_dev, made.st_ino):
                path.unlink()
        raise
    os.close(fd)
    return True


# Per-process claim depth for each marker path: nested UpdateLocks (an update's completion run
# in-process, venv_sync finishing a tail) share one claim, released by the outermost.
_CLAIMS: dict[str, int] = {}


class UpdateLock:
    """Context manager owning the shared update marker (and, with ``install_root``, the
    checkout lock) for this process.

    ``acquired`` is True when we wrote the marker; adopting a live partner's claim (hand-off
    pid, ancestor, or an outer claim of this same process) succeeds with ``acquired`` False.
    ``acquire`` returns False (and sets ``holder``) when another live update owns either lock
    or the lock cannot be created at all — never "proceed unlocked".

    ``install_root`` names the checkout whose kernel lock guards the claim (default: this
    checkout). With ``checkout_first`` (the default, when ``install_root`` is given) ``acquire``
    takes that lock before the marker. ``checkout_first=False`` claims the marker alone — a
    launch that may run under a live update's claim (venv_sync) and takes the checkout lock later,
    only for its own mutation. Either way a dead marker over a checkout lock held by another
    process tree is refused as ``held`` and kept, never reclaimed (R6).
    """

    def __init__(self, *, path: Path | None = None, install_root: Path | str | None = None,
                 checkout_first: bool = True) -> None:
        self.path = path or update_marker_path()
        self.install_root = None if install_root is None else Path(install_root)
        self.checkout_first = checkout_first
        self.acquired = False
        self.holder: UpdateHolder | None = None
        self._claimed = False
        self._checkout = False

    def acquire(self) -> bool:
        if self.install_root is not None and self.checkout_first \
                and not self.acquire_checkout(self.install_root):
            return False
        try:
            ok = self._claim_marker()
        except BaseException:
            self._drop_checkout()
            raise
        if not ok:
            self._drop_checkout()
            return False
        key = str(self.path)
        _CLAIMS[key] = _CLAIMS.get(key, 0) + 1
        self._claimed = True
        return True

    def acquire_checkout(self, install_root: Path | str) -> bool:
        """Take (or join) the checkout lock — ACQUIRE custody, never sample it (R2): a caller
        that will mutate the checkout under a marker it adopted or claimed calls this first."""
        if self._checkout:
            return True
        refused = _acquire_checkout(Path(install_root))
        if refused is not None:
            self.holder = refused
            return False
        self._checkout = True
        return True

    def _claim_marker(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.holder = UpdateHolder(pid=0, age_seconds=0.0, reason=f"{self.path.parent} is not writable ({exc})")
            return False
        _sweep_dead_tmp_siblings(self.path)
        if _CLAIMS.get(str(self.path)) is None:
            legacy = [claim for claim in legacy_profile_claims(self.path.parent)
                      if not self._is_partner(claim.pid)]
            if legacy:
                self.holder = legacy[0]
                return False
        body = f"{os.getpid()}\n{int(time.time())}\n{_identity_line()}\n".encode()
        try:
            for _attempt in range(3):
                # A7 rule 2: an exclusive create needs no mutex — it only succeeds on no marker.
                if _publish_exclusive(self.path, body):
                    self.acquired = True
                    return True
                with marker_mutex(self.path):
                    existing = _read_marker(self.path)
                    if existing is None:
                        continue  # vanished between publish and read: retry
                    if _live_partners(existing):
                        return self._adopt_or_refuse(existing)
                    if self._checkout_held_elsewhere():
                        # R6: the marker's owner is dead but a process it started (completion,
                        # build, git) still holds the checkout. The update is not over: keep
                        # its marker and refuse as ``held``.
                        self.holder = UpdateHolder(
                            pid=0, held=True,
                            age_seconds=max(existing.age(), 0.0) if existing.started_at is not None else 0.0)
                        return False
                    # Dead or malformed, judged under the mutex: reclaim and claim in one hold.
                    with suppress(FileNotFoundError):
                        self.path.unlink()
                    if _publish_exclusive(self.path, body):
                        self.acquired = True
                        return True
        except MarkerBusy:
            self.holder = UpdateHolder(pid=0, age_seconds=0.0)
            return False
        except OSError as exc:
            self.holder = UpdateHolder(pid=0, age_seconds=0.0, reason=f"{self.path} is not writable ({exc})")
            return False
        self.holder = read_live_update(path=self.path, install_root=self.install_root) \
            or UpdateHolder(pid=0, age_seconds=0.0)
        return False

    def _checkout_held_elsewhere(self) -> bool:
        """The checkout lock is held, and not by this process's own update tree (our hold, or
        a lock fd inherited from the update that spawned us — that tree may reclaim)."""
        if self._checkout:
            return False
        path = checkout_lock_path(self.install_root)
        if _HELD is not None and _HELD["path"] == str(path):
            return False
        # Only when held: a free lock is not taken by the inherited-fd lookup below.
        return checkout_lock_held(self.install_root) and _inherited_lock_fd(path) is None

    @staticmethod
    def _is_partner(pid: int) -> bool:
        return pid == os.getpid() or (pid and (pid == _handoff_pid() or _is_ancestor_pid(pid)))

    def _adopt_or_refuse(self, existing: _Marker) -> bool:
        """C1 rule 4: a LIVE claim by us, an ancestor or the hand-off partner is run under.
        Called inside the marker mutex."""
        partners = _live_partners(existing)
        if not any(self._is_partner(p) for p in partners):
            self.holder = UpdateHolder(pid=partners[0], age_seconds=existing.age() if existing.started_at else 0.0)
            return False
        own = _identity_line()
        if os.getpid() not in partners and existing.delegate_pid not in partners \
                and existing.create_time is not None and own:
            # Rule 6: name ourselves as the delegate (line 4) so the claim stays visible if the
            # partner (a hand-off script, the Tauri updater) dies while this update still runs.
            base = existing.base if existing.base.endswith(b"\n") else existing.base + b"\n"
            delegated = base + f"delegate:{os.getpid()} {own}\n".encode() \
                + "".join(f"run:{run}\n" for run in existing.runs).encode()
            _compare_and_swap(self.path, existing.raw, delegated)
        return True

    def _drop_checkout(self) -> None:
        if self._checkout:
            self._checkout = False
            _release_checkout()

    def release(self) -> None:
        """Give back our part of the claim (A7 rule 5), decided under the marker mutex by
        identity, not by the bytes we once wrote: whatever names this incarnation — as owner, or
        as a delegate someone else wrote in (the hand-off script names its child) — is released;
        a live delegate inherits an owner's claim. Only the outermost claim of a process
        releases. Never raises."""
        try:
            if self._claimed:
                key = str(self.path)
                _CLAIMS[key] = _CLAIMS.get(key, 1) - 1
                if _CLAIMS[key] <= 0:
                    _CLAIMS.pop(key, None)
                    _release_marker(self.path)
        except OSError as exc:
            logger.debug("Could not release update marker %s: %s", self.path, exc)
        finally:
            self.acquired = self._claimed = False
            self._drop_checkout()

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


def _release_marker(path: Path) -> None:
    with marker_mutex(path):
        raw = _read_bytes(path)
        if raw is None:
            return
        action, new = _release_decision(raw)
        if action == "delete":
            with suppress(FileNotFoundError):
                path.unlink()
        elif action == "rewrite":
            _compare_and_swap(path, raw, new)
