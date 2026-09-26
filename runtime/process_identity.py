"""Shared process identity and incarnation primitives.

This module is deliberately CLI-independent. It owns the stable process fingerprints and spawn
identity values used by Gateway, tools, and the remaining CLI orchestration code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from utils import atomic_json_write

SPAWN_ENV_VAR = "HERMES_SPAWN"
_TAG_VERSION = "v1"
LEDGER_FILENAME = "spawn-ledger.json"
REAPABLE_PURPOSES = frozenset({"serve", "dashboard", "gateway", "mcp-helper"})
_LEDGER_LOCK = threading.Lock()
logger = logging.getLogger(__name__)

# Same-host start-time readings can drift by ~1 s between the claim-time and a later liveness read
# (macOS ``kern.boottime`` adjustment, #117505). Both fingerprint scales are ×100 (Linux /proc ticks,
# psutil centiseconds), so 200 means 2 s on either platform — a recycled PID is essentially never
# that close to the original's start time.
START_TIME_DRIFT_TOLERANCE = 200
IS_WINDOWS = sys.platform == "win32"


def install_id(project_root: Optional[Path] = None) -> str:
    """Stable 12-hex identifier for this Hermes install, derived from its path."""
    if project_root is None:
        try:
            from hermes_constants import PROJECT_ROOT as root

            project_root = Path(root)
        except Exception:
            project_root = Path(__file__).resolve().parent.parent
    try:
        canonical = str(Path(project_root).resolve()).lower()
    except OSError:
        canonical = str(project_root).lower()
    return hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest()[:12]


def _process_create_time(pid: Optional[int] = None) -> Optional[float]:
    """``psutil`` create time for ``pid`` (default: this process), or ``None`` when unknown."""
    try:
        import psutil

        return float(psutil.Process(os.getpid() if pid is None else pid).create_time())
    except Exception:
        return None


def get_process_start_time(pid: int) -> Optional[int]:
    """Stable same-host process-start fingerprint used as a PID-reuse guard.

    Linux uses ``/proc/<pid>/stat`` field 22. Other hosts use psutil ``create_time()`` in
    centiseconds. Units differ across platforms; fingerprints are only compared on the same host.
    """
    try:
        return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])
    except (IndexError, ValueError, OSError):
        pass
    try:
        import psutil

        return int(round(psutil.Process(pid).create_time() * 100))
    except Exception:
        return None


def _process_start_time(pid: int) -> int | None:
    """The repository's stable process-start fingerprint, if available."""
    try:
        return get_process_start_time(pid)
    except Exception:
        return None


def _text_names_hermes(text: str) -> bool:
    r"""True when *text* names Hermes at a path-segment / token boundary.

    A bare ``"hermes" in text`` substring test would also match unrelated processes whose paths
    merely contain the letters (``...\\shermesa\\...``) - the false-positive class this prevents.
    """
    return any(
        token.startswith(("hermes", ".hermes"))
        for token in re.split(r"[\\/\s=,;\"']+", text.lower())
    )


def _process_command_is_hermes(pid: int) -> bool:
    """Best-effort check that *pid* currently runs Hermes code."""
    try:
        import psutil

        process = psutil.Process(pid)
        command = " ".join(process.cmdline() or [])
        executable = process.exe() or ""
        return _text_names_hermes(f"{command} {executable}")
    except Exception:
        return False


def pid_is_hermes(pid: int, *, expected_start_time: int | None = None) -> bool:
    """Whether it is safe to use destructive process-tree termination for *pid*.

    The PID must be valid, currently exist, and identify a Hermes process on Windows. When the
    caller captured a start-time fingerprint before the destructive action, the live process must
    still have the same ``(pid, start_time)`` identity. Any ambiguity fails closed.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if not IS_WINDOWS:
        if expected_start_time is None:
            return True
        try:
            return _process_start_time(pid) == expected_start_time
        except Exception:
            return False
    try:
        current_start_time = _process_start_time(pid)
    except Exception:
        return False
    if current_start_time is None:
        return False
    if expected_start_time is not None and current_start_time != expected_start_time:
        return False
    try:
        return _process_command_is_hermes(pid)
    except Exception:
        return False


def start_time_fingerprints_match(
    recorded: Any,
    current: Any,
    tolerance: int = START_TIME_DRIFT_TOLERANCE,
) -> bool:
    """Whether two same-host start-time fingerprints identify the same process incarnation."""
    return abs(int(current) - int(recorded)) <= tolerance


def _same_incarnation(proc, create_time: Optional[float]) -> bool:
    """Whether ``proc`` matches a recorded psutil create time (2 s tolerance; ``None`` matches)."""
    return create_time is None or abs(float(proc.create_time()) - float(create_time)) < 2.0


def _pid_alive_matches(pid: int, create_time: Optional[float]) -> Optional[bool]:
    """True/False when PID incarnation liveness is provable; ``None`` when it is not."""
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


@dataclass(frozen=True)
class SpawnTag:
    install: str
    purpose: str
    spawner_pid: int
    spawner_create: Optional[float]


def build_spawn_tag(purpose: str, *, project_root: Optional[Path] = None) -> str:
    """Value for a child's ``HERMES_SPAWN`` environment variable."""
    create = _process_create_time()
    create_part = f"{create:.3f}" if create is not None else "-"
    return ":".join((_TAG_VERSION, install_id(project_root), purpose, str(os.getpid()), create_part))


def spawn_env(purpose: str, *, project_root: Optional[Path] = None) -> dict[str, str]:
    """Environment fragment a spawner merges into a child process."""
    return {SPAWN_ENV_VAR: build_spawn_tag(purpose, project_root=project_root)}


def parse_spawn_tag(raw: object) -> Optional[SpawnTag]:
    """Parse a ``HERMES_SPAWN`` value; return ``None`` when malformed."""
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
    host: str = ""
    port: Optional[int] = None
    profile: str = ""


def _ledger_path() -> Path:
    """Machine-root ledger path shared by every profile of this install."""
    try:
        from hermes_constants import get_default_hermes_root

        return Path(get_default_hermes_root()) / LEDGER_FILENAME
    except Exception:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / LEDGER_FILENAME


def _read_ledger(path: Path) -> Optional[list[dict]]:
    """Entries list, [] for empty/missing, None for corrupt or unreadable."""
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
    return [entry for entry in parsed if isinstance(entry, dict)] if isinstance(parsed, list) else None


def _read_ledger_or_quarantine(path: Path) -> Optional[list[dict]]:
    """Ledger entries; None after parking a corrupt file. Caller holds the ledger lock."""
    entries = _read_ledger(path)
    if entries is None:
        parked = path.with_suffix(path.suffix + ".corrupt")
        try:
            os.replace(path, parked)
            logger.warning("spawn ledger was unreadable; moved to %s", parked)
        except OSError:
            pass
    return entries


def _desktop_spawner_identity() -> tuple[Optional[int], Optional[float]]:
    """Spawner identity from Desktop parent-death watchdog environment."""
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
    pid: int,
    create_time: Optional[float],
    purpose: str,
    project_root: Optional[Path],
    spawner_pid: Optional[int],
    spawner_create: Optional[float],
) -> LedgerEntry:
    return LedgerEntry(
        pid,
        create_time,
        purpose,
        install_id(project_root),
        spawner_pid,
        spawner_create,
        time.time(),
        argv="",
    )


def _append_entry(entry: LedgerEntry) -> bool:
    """Prune dead entries and append entry through the single atomic ledger write path."""
    path = _ledger_path()
    with _LEDGER_LOCK:
        entries = _read_ledger_or_quarantine(path) or []
        pruned = [
            existing
            for existing in entries
            if isinstance(existing.get("pid"), int)
            and existing["pid"] != entry.pid
            and _pid_alive_matches(existing["pid"], existing.get("create_time")) is not False
        ]
        pruned.append(asdict(entry))
        try:
            from hermes_constants import mkdir_under_hermes_home

            mkdir_under_hermes_home(path.parent)
            atomic_json_write(path, pruned, mode=0o600, ensure_ascii=True)
            return True
        except OSError:
            logger.debug("spawn ledger write failed", exc_info=True)
            return False


def register_self(
    purpose: str,
    *,
    project_root: Optional[Path] = None,
    detail: Optional[dict] = None,
) -> bool:
    """Record this process in the machine spawn ledger and prune provably dead entries."""
    tag = parse_spawn_tag(os.environ.get(SPAWN_ENV_VAR))
    spawner_pid, spawner_create = (
        (tag.spawner_pid, tag.spawner_create) if tag else _desktop_spawner_identity()
    )
    entry = _new_entry(
        os.getpid(),
        _process_create_time(),
        purpose,
        project_root,
        spawner_pid,
        spawner_create,
    )
    if detail:
        try:
            entry.host = str(detail.get("host") or "")
            entry.port = int(detail["port"]) if detail.get("port") is not None else None
            entry.profile = str(detail.get("profile") or "")
        except (TypeError, ValueError):
            pass
    try:
        import sys as _sys

        entry.argv = " ".join(_sys.argv[:10])
    except Exception:
        pass
    return _append_entry(entry)


def register_child(pid: int, purpose: str, *, project_root: Optional[Path] = None) -> bool:
    """Record a child that cannot register itself, preserving its spawner identity."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    child_create = _process_create_time(pid) if pid > 0 else None
    if child_create is None:
        return False
    entry = _new_entry(
        pid,
        child_create,
        purpose,
        project_root,
        os.getpid(),
        _process_create_time(),
    )
    try:
        import psutil

        entry.argv = " ".join(psutil.Process(pid).cmdline()[:10])
    except Exception:
        pass
    return _append_entry(entry)


def ledger_entries(*, project_root: Optional[Path] = None) -> list[dict]:
    """Live-verified ledger entries for this install."""
    want_install = install_id(project_root)
    with _LEDGER_LOCK:
        entries = _read_ledger_or_quarantine(_ledger_path())
    if entries is None:
        return []
    return [
        entry
        for entry in entries
        if entry.get("install") == want_install
        and isinstance(entry.get("pid"), int)
        and _pid_alive_matches(entry["pid"], entry.get("create_time")) is not False
    ]


def spawner_is_dead(entry: dict) -> Optional[bool]:
    """Whether the recorded spawner is provably gone; None when unprovable or unrecorded."""
    spawner_pid = entry.get("spawner_pid")
    if not isinstance(spawner_pid, int) or spawner_pid <= 0:
        return None
    alive = _pid_alive_matches(spawner_pid, entry.get("spawner_create"))
    return None if alive is None else not alive


def reap_orphaned_mcp_helpers(
    *,
    project_root: Optional[Path] = None,
    kill_fn=None,
) -> list[int]:
    """Kill ledger-registered stdio MCP helpers whose recorded spawner is provably dead."""
    reaped: list[int] = []
    try:
        entries = ledger_entries(project_root=project_root)
    except Exception:
        return reaped
    own_pid = os.getpid()
    for entry in entries:
        try:
            pid = entry.get("pid")
            if (
                entry.get("purpose") != "mcp-helper"
                or not isinstance(pid, int)
                or pid <= 0
                or pid == own_pid
            ):
                continue
            if spawner_is_dead(entry) is not True:
                continue
            if kill_fn is not None:
                kill_fn(pid)
            else:
                import psutil

                proc = psutil.Process(pid)
                if not _same_incarnation(proc, entry.get("create_time")):
                    continue
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


__all__ = [
    "LEDGER_FILENAME",
    "LedgerEntry",
    "REAPABLE_PURPOSES",
    "SPAWN_ENV_VAR",
    "IS_WINDOWS",
    "START_TIME_DRIFT_TOLERANCE",
    "SpawnTag",
    "_append_entry",
    "_ledger_path",
    "_pid_alive_matches",
    "_process_command_is_hermes",
    "_process_create_time",
    "_process_start_time",
    "_same_incarnation",
    "_text_names_hermes",
    "build_spawn_tag",
    "get_process_start_time",
    "install_id",
    "ledger_entries",
    "parse_spawn_tag",
    "pid_is_hermes",
    "reap_orphaned_mcp_helpers",
    "register_child",
    "register_self",
    "spawn_env",
    "spawner_is_dead",
    "start_time_fingerprints_match",
]
