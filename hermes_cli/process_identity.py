"""Process identity: spawn tags, the machine-wide spawn ledger, and the Windows job-object self-attach.

Three layers make every long-lived Hermes process positively identifiable, so reapers (``hermes
update``, Desktop startup sweeps) never guess lineage from PPID archaeology or cmdline matching:
1. spawn tags (``HERMES_SPAWN`` env stamped by the spawner); 2. a ``(pid, create_time)`` ledger;
3. Windows job-object self-attach with ``KILL_ON_JOB_CLOSE`` so the whole child tree dies with the
root — no launcher→worker chains left holding ``.pyd`` locks.
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
REAPABLE_PURPOSES = frozenset({"serve", "dashboard", "gateway", "mcp-helper"})

_IS_WINDOWS = platform.system() == "Windows"

# Module-global job handle: must live exactly as long as this process so the
# kernel closes it (and kills the job) when we die. Never close it manually.
_JOB_HANDLE = None
_LEDGER_LOCK = threading.Lock()


def install_id(project_root: Optional[Path] = None) -> str:
    """Stable 12-hex identifier for THIS install (derived from its path)."""
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


def _process_create_time(pid: Optional[int] = None) -> Optional[float]:
    """``psutil`` create time for ``pid`` (default: this process); ``None`` when psutil can't say."""
    try:
        import psutil

        return float(psutil.Process(os.getpid() if pid is None else pid).create_time())
    except Exception:
        return None


# Layer 1 — spawn tags


@dataclass(frozen=True)
class SpawnTag:
    install: str
    purpose: str
    spawner_pid: int
    spawner_create: Optional[float]


def build_spawn_tag(purpose: str, *, project_root: Optional[Path] = None) -> str:
    """Value for the child's ``HERMES_SPAWN`` env var, stamped by the spawner."""
    create = _process_create_time()
    create_part = f"{create:.3f}" if create is not None else "-"
    return ":".join((_TAG_VERSION, install_id(project_root), purpose, str(os.getpid()), create_part))


def spawn_env(purpose: str, *, project_root: Optional[Path] = None) -> dict[str, str]:
    """Env fragment a spawner merges into a child's environment."""
    return {SPAWN_ENV_VAR: build_spawn_tag(purpose, project_root=project_root)}


def parse_spawn_tag(raw: object) -> Optional[SpawnTag]:
    """Parse a ``HERMES_SPAWN`` value; ``None`` for anything malformed."""
    parts = raw.split(":") if isinstance(raw, str) else []
    if len(parts) != 5 or parts[0] != _TAG_VERSION:
        return None
    _, install, purpose, pid_s, create_s = parts
    if not install or not purpose:
        return None
    try:
        pid = int(pid_s)
        create = None if create_s == "-" else float(create_s)
    except ValueError:
        return None
    return SpawnTag(install, purpose, pid, create) if pid > 0 else None


# Layer 2 — spawn ledger


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
    # Structured launch identity a relauncher needs after an update, without parsing argv. Empty
    # for purposes that don't supply it; readers must use .get() — older ledger files predate these.
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
    """Entries list, ``[]`` for empty/missing, ``None`` for CORRUPT (never silently an empty roster).

    Mirrors the #89298 contract: corrupt is a distinct state that must never be silently treated as an empty
    roster.
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
    return [e for e in parsed if isinstance(e, dict)] if isinstance(parsed, list) else None


def _read_ledger_or_quarantine(path: Path) -> Optional[list[dict]]:
    """Ledger entries; ``None`` after parking a corrupt file. Caller holds ``_LEDGER_LOCK``."""
    entries = _read_ledger(path)
    if entries is None:
        parked = path.with_suffix(path.suffix + ".corrupt")
        try:
            os.replace(path, parked)
            logger.warning("spawn ledger was unreadable; moved to %s", parked)
        except OSError:
            pass
    return entries


def _same_incarnation(proc, create_time: Optional[float]) -> bool:
    """Does the live ``proc`` match a recorded ``create_time`` (2 s tolerance; ``None`` matches)?"""
    return create_time is None or abs(float(proc.create_time()) - float(create_time)) < 2.0


def _pid_alive_matches(pid: int, create_time: Optional[float]) -> Optional[bool]:
    """True/False when provable; ``None`` when psutil can't say."""
    try:
        import psutil
    except Exception:
        return None
    try:
        return _same_incarnation(psutil.Process(int(pid)), create_time)
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return None


def register_self(purpose: str, *, project_root: Optional[Path] = None, detail: Optional[dict] = None) -> bool:
    """Record this process in the machine spawn ledger. Best-effort.

    Called at the top of every long-lived entry point; dead ``(pid, create_time)`` entries are
    pruned on every write. ``detail`` may carry ``host``/``port``/``profile`` so the update
    pipeline can relaunch a manually-started serve with its real bind address.
    """
    tag = parse_spawn_tag(os.environ.get(SPAWN_ENV_VAR))
    spawner_pid, spawner_create = (tag.spawner_pid, tag.spawner_create) if tag else _desktop_spawner_identity()
    entry = _new_entry(os.getpid(), _process_create_time(), purpose, project_root, spawner_pid, spawner_create)
    if detail:
        try:
            entry.host = str(detail.get("host") or "")
            entry.port = int(detail["port"]) if detail.get("port") is not None else None
            entry.profile = str(detail.get("profile") or "")
        except (TypeError, ValueError):
            pass
    try:
        import sys as _sys

        # 10 tokens: enough for `hermes serve --host X --port N --profile P` while bounding
        # pathological argv. Structured detail is canonical; argv is the human-readable fallback.
        entry.argv = " ".join(_sys.argv[:10])
    except Exception:
        pass
    return _append_entry(entry)


def _desktop_spawner_identity() -> tuple[Optional[int], Optional[float]]:
    """Spawner ``(pid, create_time)`` from the Electron app's HERMES_PARENT_PID (+ optional
    ``winms:<ms>`` start marker) parent-death watchdog vars, so ledger lineage works with every
    Desktop version without a TS change. ``(None, None)`` when absent/malformed."""
    try:
        spawner_pid = int(os.environ.get("HERMES_PARENT_PID", ""))
    except (TypeError, ValueError):
        spawner_pid = 0
    if spawner_pid <= 0:
        return None, None
    marker = os.environ.get("HERMES_PARENT_START_MARKER", "")
    if not marker.startswith("winms:"):
        return spawner_pid, None
    try:
        return spawner_pid, float(marker.split(":", 1)[1]) / 1000.0
    except (ValueError, IndexError):
        return spawner_pid, None


def _new_entry(
    pid: int, create_time: Optional[float], purpose: str, project_root: Optional[Path],
    spawner_pid: Optional[int], spawner_create: Optional[float],
) -> LedgerEntry:
    return LedgerEntry(
        pid, create_time, purpose, install_id(project_root), spawner_pid, spawner_create, time.time(), argv=""
    )


def _append_entry(entry: LedgerEntry) -> bool:
    """Prune dead entries and append ``entry`` — the ONLY ledger write path.

    Serialized under ``_LEDGER_LOCK`` with an atomic tmp+replace; no writer touches the file
    outside this function.

    See #91660.
    """
    path = _ledger_path()
    with _LEDGER_LOCK:
        entries = _read_ledger_or_quarantine(path) or []
        # Drop malformed entries, our own stale entry, and provably dead pids.
        pruned = [
            e for e in entries
            if isinstance(e.get("pid"), int)
            and e["pid"] != entry.pid
            and _pid_alive_matches(e["pid"], e.get("create_time")) is not False
        ]
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


def register_child(pid: int, purpose: str, *, project_root: Optional[Path] = None) -> bool:
    """Record a CHILD process this process just spawned. Best-effort.

    Mirror of :func:`register_self` for children that cannot register themselves (stdio MCP
    helpers: arbitrary ``npx``/binary servers never import Hermes code). Records the child's
    ``(pid, create_time)`` with THIS process as spawner, so a helper whose spawner is provably gone
    is a reapable orphan and one whose spawner is alive is never reaped.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    child_create = _process_create_time(pid) if pid > 0 else None
    if child_create is None:
        return False
    entry = _new_entry(pid, child_create, purpose, project_root, os.getpid(), _process_create_time())
    try:
        import psutil

        entry.argv = " ".join(psutil.Process(pid).cmdline()[:10])
    except Exception:
        pass
    return _append_entry(entry)


def ledger_entries(*, project_root: Optional[Path] = None) -> list[dict]:
    """Live-verified ledger entries for THIS install (a corrupt ledger is quarantined, read as empty).

    Entries whose ``(pid, create_time)`` no longer matches a live process are excluded (PID reuse reads as
    dead, thanks to the create-time pair). A corrupt ledger is quarantined and read as empty — identical
    philosophy to the backend-ownership fix (#89298): never let corruption erase or fake a roster; never let
    it block the caller either.
    """
    want_install = install_id(project_root)
    with _LEDGER_LOCK:
        entries = _read_ledger_or_quarantine(_ledger_path())
    if entries is None:
        return []
    return [
        e for e in entries
        if e.get("install") == want_install
        and isinstance(e.get("pid"), int)
        and _pid_alive_matches(e["pid"], e.get("create_time")) is not False
    ]


def spawner_is_dead(entry: dict) -> Optional[bool]:
    """Is the recorded spawner of this entry provably gone? ``None`` when unrecorded/unprovable."""
    spawner_pid = entry.get("spawner_pid")
    if not isinstance(spawner_pid, int) or spawner_pid <= 0:
        return None
    alive = _pid_alive_matches(spawner_pid, entry.get("spawner_create"))
    return None if alive is None else not alive


def reap_orphaned_mcp_helpers(*, project_root: Optional[Path] = None, kill_fn=None) -> list[int]:
    """Kill ledger-registered stdio MCP helpers whose spawner is provably dead.

    Ledger-driven startup-sweep rung (not cmdline-heuristic): a helper is reaped ONLY when it has a
    live ``mcp-helper`` entry for THIS install AND ``spawner_is_dead`` is ``True`` — never
    ``None``/unprovable, never a live spawner.
    """
    reaped: list[int] = []
    try:
        entries = ledger_entries(project_root=project_root)
    except Exception:
        return reaped
    own_pid = os.getpid()
    for entry in entries:
        try:
            pid = entry.get("pid")
            if entry.get("purpose") != "mcp-helper" or not isinstance(pid, int) or pid <= 0 or pid == own_pid:
                continue
            if spawner_is_dead(entry) is not True:
                continue  # live or unprovable spawner → never touch
            if kill_fn is not None:
                kill_fn(pid)
            else:
                import psutil

                proc = psutil.Process(pid)
                if not _same_incarnation(proc, entry.get("create_time")):
                    continue  # PID reused since registration
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except psutil.TimeoutExpired:
                    proc.kill()
            reaped.append(pid)
        except Exception:
            logger.debug("mcp-helper orphan reap failed for %s", entry, exc_info=True)
    if reaped:
        logger.info("reaped %d orphaned stdio MCP helper(s): %s", len(reaped), reaped)
    return reaped


# Layer 3 — Windows job-object self-attach


def attach_self_to_kill_on_close_job() -> bool:
    """Place this process in a job that dies (whole tree) when we die. Windows-only, idempotent.

    ``BREAKAWAY_OK`` keeps children spawned with ``CREATE_BREAKAWAY_FROM_JOB`` (gateway relaunch
    during update, detached watchers) escaping exactly as before.
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
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER), ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(wintypes.ULONG)), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                *((n, ctypes.c_size_t) for n in (
                    "ProcessMemoryLimit", "JobMemoryLimit", "PeakProcessMemoryUsed", "PeakJobMemoryUsed")),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return False
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
        )
        ok = kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info))
        if not ok or not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            kernel32.CloseHandle(job)
            return False
        _JOB_HANDLE = job  # keep alive for the life of the process — never close
        logger.debug("attached to kill-on-close job object")
        return True
    except Exception:
        logger.debug("job object self-attach failed", exc_info=True)
        return False
