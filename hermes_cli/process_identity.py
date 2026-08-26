"""Process identity: spawn tags, the machine-wide spawn ledger, and the
Windows job-object self-attach.

Three layers that make every long-lived Hermes process positively
identifiable, so reapers (``hermes update``, Desktop startup sweeps) never
have to guess lineage from PPID archaeology or cmdline pattern-matching:

1. **Spawn tag** (``HERMES_SPAWN`` env var): every spawner stamps its children
   with ``v1:<install_id>:<purpose>:<spawner_pid>:<spawner_create>``. A scanner
   that can read the child's environment classifies it instantly: which
   install, what it is, who spawned it, and when.

2. **Spawn ledger** (``spawn-ledger.json`` at the machine Hermes root): every
   long-lived process (serve/dashboard backend, gateway) self-registers
   ``pid + create_time + purpose + spawner`` at startup. ``pid`` alone is
   forgeable by reuse; the ``(pid, create_time)`` pair is not. Reapers
   cross-check live processes against the ledger for positive identification
   even when environment reads are denied (Windows frequently denies
   ``Process.environ()`` cross-session).

3. **Job object self-attach** (Windows): a backend places itself in a job with
   ``KILL_ON_JOB_CLOSE`` so its whole child tree dies atomically with it —
   no launcher→worker two-hop chains left holding ``.pyd`` locks after the
   visible root is killed. ``BREAKAWAY_OK`` is set so the existing
   ``CREATE_BREAKAWAY_FROM_JOB`` spawns (gateway relaunch during update,
   watchers that must outlive their spawner) keep working unchanged.

All of it is best-effort and fail-safe: identity failures degrade to the
legacy heuristics, they never block startup or updates. The ledger tolerates
corruption the same way ``backend-ownership.json`` does post-#89298:
an unreadable ledger is quarantined aside (``.corrupt``), never rewritten
blind.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SPAWN_ENV_VAR = "HERMES_SPAWN"
_TAG_VERSION = "v1"
LEDGER_FILENAME = "spawn-ledger.json"

#: Purposes a reaper may treat as "safe to kill when the owner is gone".
#: Interactive processes (chat, REPLs) are deliberately NOT in this set.
REAPABLE_PURPOSES = frozenset({"serve", "dashboard", "gateway"})

_IS_WINDOWS = platform.system() == "Windows"

# Module-global job handle: must live exactly as long as this process so the
# kernel closes it (and kills the job) when we die. Never close it manually.
_JOB_HANDLE = None
_LEDGER_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Install identity
# ---------------------------------------------------------------------------

def install_id(project_root: Optional[Path] = None) -> str:
    """Stable 12-hex identifier for THIS install (derived from its path).

    Lets a reaper reject processes from a different Hermes install on the
    same machine without path comparisons at scan time.
    """
    if project_root is None:
        try:
            from hermes_constants import PROJECT_ROOT as _root

            project_root = Path(_root)
        except Exception:
            project_root = Path(__file__).resolve().parent.parent
    try:
        canonical = str(Path(project_root).resolve()).lower()
    except OSError:
        canonical = str(project_root).lower()
    return hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest()[:12]


def _own_create_time() -> Optional[float]:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).create_time())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Layer 1 — spawn tags
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpawnTag:
    install: str
    purpose: str
    spawner_pid: int
    spawner_create: Optional[float]


def build_spawn_tag(purpose: str, *, project_root: Optional[Path] = None) -> str:
    """Value for the child's ``HERMES_SPAWN`` env var, stamped by the spawner."""
    create = _own_create_time()
    create_part = f"{create:.3f}" if create is not None else "-"
    return ":".join(
        (_TAG_VERSION, install_id(project_root), purpose, str(os.getpid()), create_part)
    )


def spawn_env(purpose: str, *, project_root: Optional[Path] = None) -> dict[str, str]:
    """Env fragment a spawner merges into a child's environment."""
    return {SPAWN_ENV_VAR: build_spawn_tag(purpose, project_root=project_root)}


def parse_spawn_tag(raw: object) -> Optional[SpawnTag]:
    """Parse a ``HERMES_SPAWN`` value; ``None`` for anything malformed."""
    if not isinstance(raw, str):
        return None
    parts = raw.split(":")
    if len(parts) != 5 or parts[0] != _TAG_VERSION:
        return None
    _, install, purpose, pid_s, create_s = parts
    if not install or not purpose:
        return None
    try:
        pid = int(pid_s)
    except ValueError:
        return None
    if pid <= 0:
        return None
    create: Optional[float] = None
    if create_s != "-":
        try:
            create = float(create_s)
        except ValueError:
            return None
    return SpawnTag(install=install, purpose=purpose, spawner_pid=pid, spawner_create=create)


# ---------------------------------------------------------------------------
# Layer 2 — spawn ledger
# ---------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    pid: int
    create_time: Optional[float]
    purpose: str
    install: str
    spawner_pid: Optional[int]
    spawner_create: Optional[float]
    registered_at: float
    argv: str
    # Structured launch identity (#63206): what a relauncher needs to bring
    # this runtime back after an update, without parsing argv. Empty for
    # purposes that don't supply it; readers must use .get() — older ledger
    # files on disk predate these keys.
    host: str = ""
    port: Optional[int] = None
    profile: str = ""


def _ledger_path() -> Path:
    """Machine-root ledger path (shared by every profile of this install)."""
    try:
        from hermes_constants import get_default_hermes_root

        return Path(get_default_hermes_root()) / LEDGER_FILENAME
    except Exception:
        from hermes_cli.config import get_hermes_home

        return Path(get_hermes_home()) / LEDGER_FILENAME


def _read_ledger(path: Path) -> Optional[list[dict]]:
    """Entries list, ``[]`` for empty/missing, ``None`` for CORRUPT.

    Mirrors the #89298 contract: corrupt is a distinct state that must never
    be silently treated as an empty roster.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        return None
    if not text.strip():
        return []
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    return [e for e in parsed if isinstance(e, dict)]


def _quarantine_ledger(path: Path) -> None:
    parked = path.with_suffix(path.suffix + ".corrupt")
    try:
        os.replace(path, parked)
        logger.warning("spawn ledger was unreadable; moved to %s", parked)
    except OSError:
        pass


def _pid_alive_matches(pid: int, create_time: Optional[float]) -> Optional[bool]:
    """True/False when provable; ``None`` when psutil can't say."""
    try:
        import psutil
    except Exception:
        return None
    try:
        proc = psutil.Process(int(pid))
        if create_time is None:
            return True
        return abs(float(proc.create_time()) - float(create_time)) < 2.0
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return None


def register_self(
    purpose: str,
    *,
    project_root: Optional[Path] = None,
    detail: Optional[dict] = None,
) -> bool:
    """Record this process in the machine spawn ledger. Best-effort.

    Called at the top of every long-lived entry point (serve/dashboard
    backend, gateway run loop). Dead entries — ``(pid, create_time)`` no
    longer live — are pruned on every write so the ledger tracks reality
    instead of growing forever.

    ``detail`` optionally carries structured launch identity (#63206) —
    ``host``/``port``/``profile`` — so the update pipeline can relaunch a
    manually-started serve with its real bind address instead of guessing
    from argv.
    """
    tag = parse_spawn_tag(os.environ.get(SPAWN_ENV_VAR))
    spawner_pid: Optional[int] = tag.spawner_pid if tag else None
    spawner_create: Optional[float] = tag.spawner_create if tag else None
    if spawner_pid is None:
        # Desktop compatibility: the Electron app already stamps children with
        # HERMES_PARENT_PID (+ optional `winms:<ms>` start marker) for its
        # parent-death watchdog. Reuse it as spawner identity so ledger
        # lineage works with every Desktop version, no TS change needed.
        try:
            raw = int(os.environ.get("HERMES_PARENT_PID", ""))
            if raw > 0:
                spawner_pid = raw
        except (TypeError, ValueError):
            pass
        marker = os.environ.get("HERMES_PARENT_START_MARKER", "")
        if spawner_pid is not None and marker.startswith("winms:"):
            try:
                spawner_create = float(marker.split(":", 1)[1]) / 1000.0
            except (ValueError, IndexError):
                spawner_create = None
    entry = LedgerEntry(
        pid=os.getpid(),
        create_time=_own_create_time(),
        purpose=purpose,
        install=install_id(project_root),
        spawner_pid=spawner_pid,
        spawner_create=spawner_create,
        registered_at=time.time(),
        argv="",
    )
    if detail:
        try:
            entry.host = str(detail.get("host") or "")
            port = detail.get("port")
            entry.port = int(port) if port is not None else None
            entry.profile = str(detail.get("profile") or "")
        except (TypeError, ValueError):
            pass
    try:
        import sys as _sys

        # 10 tokens (was 6): enough for `hermes serve --host X --port N
        # --profile P` — the relaunch shapes #63206 needs — while still
        # bounding pathological argv. Structured detail above is the
        # canonical identity; argv is the human-readable fallback.
        entry.argv = " ".join(_sys.argv[:10])
    except Exception:
        pass

    path = _ledger_path()
    with _LEDGER_LOCK:
        entries = _read_ledger(path)
        if entries is None:
            _quarantine_ledger(path)
            entries = []
        pruned: list[dict] = []
        for e in entries:
            pid = e.get("pid")
            if not isinstance(pid, int) or pid == entry.pid:
                continue
            alive = _pid_alive_matches(pid, e.get("create_time"))
            if alive is False:
                continue  # provably dead → prune
            pruned.append(e)
        pruned.append(asdict(entry))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(pruned, indent=2), encoding="utf-8")
            os.replace(tmp, path)
            return True
        except OSError:
            logger.debug("spawn ledger write failed", exc_info=True)
            return False


def ledger_entries(*, project_root: Optional[Path] = None) -> list[dict]:
    """Live-verified ledger entries for THIS install.

    Entries whose ``(pid, create_time)`` no longer matches a live process are
    excluded (PID reuse reads as dead, thanks to the create-time pair). A
    corrupt ledger is quarantined and read as empty — identical philosophy to
    the backend-ownership fix (#89298): never let corruption erase or fake
    a roster; never let it block the caller either.
    """
    want_install = install_id(project_root)
    path = _ledger_path()
    with _LEDGER_LOCK:
        entries = _read_ledger(path)
        if entries is None:
            _quarantine_ledger(path)
            return []
    out: list[dict] = []
    for e in entries:
        if e.get("install") != want_install:
            continue
        pid = e.get("pid")
        if not isinstance(pid, int):
            continue
        if _pid_alive_matches(pid, e.get("create_time")) is False:
            continue
        out.append(e)
    return out


def spawner_is_dead(entry: dict) -> Optional[bool]:
    """Is the recorded spawner of this entry provably gone?

    ``True`` → owner gone (orphaned by identity, not by PPID guessing).
    ``False`` → owner still alive. ``None`` → no spawner recorded / unprovable.
    """
    spawner_pid = entry.get("spawner_pid")
    if not isinstance(spawner_pid, int) or spawner_pid <= 0:
        return None
    alive = _pid_alive_matches(spawner_pid, entry.get("spawner_create"))
    if alive is None:
        return None
    return not alive


# ---------------------------------------------------------------------------
# Layer 3 — Windows job-object self-attach
# ---------------------------------------------------------------------------

def attach_self_to_kill_on_close_job() -> bool:
    """Place this process in a job that dies (whole tree) when we die.

    Windows-only, best-effort, idempotent. ``BREAKAWAY_OK`` is included so
    children spawned with ``CREATE_BREAKAWAY_FROM_JOB`` (gateway relaunch
    during update, detached watchers) keep escaping exactly as they do today.
    Nested jobs are supported since Windows 8, so being inside another job
    (Terminal, CI runners) does not prevent the attach on any supported OS.
    """
    global _JOB_HANDLE
    if not _IS_WINDOWS or _JOB_HANDLE is not None:
        return _JOB_HANDLE is not None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x0800
        JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x1000
        JobObjectExtendedLimitInformation = 9

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(wintypes.ULONG)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return False
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_BREAKAWAY_OK
            | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
        )
        ok = kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        )
        if not ok:
            kernel32.CloseHandle(job)
            return False
        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            kernel32.CloseHandle(job)
            return False
        _JOB_HANDLE = job  # keep alive for the life of the process — never close
        logger.debug("attached to kill-on-close job object")
        return True
    except Exception:
        logger.debug("job object self-attach failed", exc_info=True)
        return False
