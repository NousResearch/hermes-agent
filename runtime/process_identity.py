"""Shared process identity and incarnation primitives.

This runtime-owned module preserves Hermes' spawn tags, machine-wide spawn ledger, PID/create-time
identity checks, and stale-PID guards without depending on CLI implementation modules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SPAWN_ENV_VAR = "HERMES_SPAWN"
_TAG_VERSION = "v1"
LEDGER_FILENAME = "spawn-ledger.json"

#: Purposes a reaper may treat as "safe to kill when the owner is gone".
#: Interactive processes (chat, REPLs) are deliberately NOT in this set.
REAPABLE_PURPOSES = frozenset({"serve", "dashboard", "gateway", "mcp-helper"})

_IS_WINDOWS = platform.system() == "Windows"
IS_WINDOWS = _IS_WINDOWS

_LEDGER_LOCK = threading.Lock()

# Same-host start-time readings can drift by ~1 s between claim-time and a later liveness read.
# Both fingerprint scales are ×100, so 200 means 2 seconds on either platform.
START_TIME_DRIFT_TOLERANCE = 200


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
    r"""True when *text* names Hermes at a path-segment / token boundary."""
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
    """Whether destructive process-tree termination is safe for *pid*.

    Windows requires both a live incarnation and a Hermes command identity. On other
    hosts, an optional start-time fingerprint still guards against PID reuse.
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


def posix_is_zombie(pid: int) -> bool:
    """Whether *pid* is a zombie, using /proc or a bounded ps fallback."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            fields = fh.read().split()
        return len(fields) > 2 and fields[2] == "Z"
    except FileNotFoundError:
        try:
            import subprocess

            result = subprocess.run(
                ["ps", "-o", "state=", "-p", str(pid)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            return result.returncode == 0 and result.stdout.strip().startswith("Z")
        except Exception:
            return False
    except (IndexError, PermissionError, OSError):
        return False


def win32_pid_exists(pid: int) -> bool:
    """psutil-free Windows liveness probe via OpenProcess/WaitForSingleObject."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.restype = ctypes.c_uint
        kernel32.GetLastError.restype = ctypes.c_uint
        query_limited, synchronize = 0x1000, 0x100000
        wait_timeout, access_denied = 0x00000102, 5
        handle = kernel32.OpenProcess(query_limited | synchronize, False, pid)
        if not handle:
            return kernel32.GetLastError() == access_denied
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError):
        return False


def pid_exists_stdlib(pid: int) -> bool:
    """Stdlib-only process liveness check; zombies report dead."""
    pid = int(pid)
    if IS_WINDOWS:
        return win32_pid_exists(pid)
    if posix_is_zombie(pid):
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def start_time_fingerprints_match(
    recorded: Any,
    current: Any,
    tolerance: int = START_TIME_DRIFT_TOLERANCE,
) -> bool:
    """Whether two same-host start-time fingerprints identify the same process incarnation."""
    return abs(int(current) - int(recorded)) <= tolerance


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
    hermes_home: str = ""
    # `serve --isolated`: opted out of the host singleton (Desktop's SSH backend for another
    # machine). Attach-first readers must never adopt it; argv is truncated, so this is canonical.
    isolated: bool = False


def _ledger_path() -> Path:
    """Machine-root ledger path (shared by every profile of this install)."""
    try:
        from hermes_constants import get_default_hermes_root

        return Path(get_default_hermes_root()) / LEDGER_FILENAME
    except Exception:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / LEDGER_FILENAME


def _read_ledger(path: Path) -> Optional[list[dict]]:
    """Entries list, ``[]`` for empty/missing, ``None`` for CORRUPT (never silently an empty roster).

    Mirrors the #89298 contract: corrupt is a distinct state that must never be silently treated as an empty
    roster.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError):
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


def _pid_alive_matches(pid: int, create_time: Optional[float], *, strict: bool = False) -> Optional[bool]:
    """True/False when provable; ``None`` when psutil can't say."""
    try:
        import psutil
    except Exception:
        return None
    try:
        proc = psutil.Process(int(pid))
        if strict:
            return (create_time is not None and proc.create_time() == create_time
                    and proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE)
        # A zombie keeps its create_time until its parent reaps it, but it is already dead: the
        # dashboard stop check (``gateway.status._pid_exists``) books it stopped, so the ledger must
        # not report it as a live pre-update survivor. Windows has no zombies (and status() is slow there).
        return _same_incarnation(proc, create_time) and (
            os.name == "nt" or proc.status() != getattr(psutil, "STATUS_ZOMBIE", "zombie"))
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return None


def register_self(purpose: str, *, project_root: Optional[Path] = None, detail: Optional[dict] = None) -> bool:
    """Record this process in the machine spawn ledger. Best-effort.

    Called at the top of every long-lived entry point; dead ``(pid, create_time)`` entries are
    pruned on every write. ``detail`` may carry ``host``/``port``/``profile`` so the update
    pipeline can relaunch a manually-started serve with its real bind address, and ``isolated``
    so attach-first discovery skips a backend that opted out of the host singleton.
    """
    from hermes_constants import hermes_home_key

    tag = parse_spawn_tag(os.environ.get(SPAWN_ENV_VAR))
    spawner_pid, spawner_create = (tag.spawner_pid, tag.spawner_create) if tag else _desktop_spawner_identity()
    entry = _new_entry(os.getpid(), _process_create_time(), purpose, project_root, spawner_pid, spawner_create)
    entry.hermes_home = hermes_home_key()
    if detail:
        try:
            entry.host = str(detail.get("host") or "")
            entry.port = int(detail["port"]) if detail.get("port") is not None else None
            entry.profile = str(detail.get("profile") or "")
            entry.isolated = bool(detail.get("isolated"))
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
            from hermes_constants import mkdir_under_hermes_home
            from utils import atomic_json_write

            mkdir_under_hermes_home(path.parent)
            # argv may carry surrogate-escaped bytes (non-UTF-8 paths); ensure_ascii keeps the
            # utf-8 text handle from raising UnicodeEncodeError (a ValueError, not an OSError).
            atomic_json_write(path, pruned, mode=0o600, ensure_ascii=True)
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


def ledger_entries(
    *, project_root: Optional[Path] = None, all_installs: bool = False, verified_only: bool = False,
) -> list[dict]:
    """Ledger entries for this install, or all installs when explicitly requested.

    ``verified_only`` requires an exact live PID/create-time pair. The default preserves
    unknown processes for reapers, which must not mistake missing proof for a dead process.

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
        if (all_installs or e.get("install") == want_install)
        and isinstance(e.get("pid"), int)
        and (_pid_alive_matches(e["pid"], e.get("create_time"), strict=True) is True
             if verified_only else _pid_alive_matches(e["pid"], e.get("create_time")) is not False)
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
                # Windows: descendants (npx.cmd → node.exe) have no pgid to group-kill and
                # reparent with ParentId=null when the direct child exits first (#61059), so
                # reap the whole tree. On POSIX the killpg-based sweep already reaches them.
                if _IS_WINDOWS:
                    _kill_process_tree_windows(proc)
                else:
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


def _kill_process_tree_windows(proc) -> None:
    """Terminate *proc* and every still-alive descendant (npx.cmd → node.exe), Windows-only
    (#61059): without a pgid there is no group-kill, and grandchildren reparent to nothing
    (ParentId=null) once the direct child exits, so they must be reached through the tree."""
    import psutil

    try:
        descendants = proc.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        descendants = []
    for child in descendants:
        try:
            child.terminate()
        except Exception:  # noqa: BLE001 - raced away or refused; keep going
            pass
    try:
        proc.terminate()
    except Exception:  # noqa: BLE001
        pass
    try:
        _, alive = psutil.wait_procs(descendants + [proc], timeout=2.0)
    except Exception:  # noqa: BLE001 - broken fake/raced process; nothing more to force-kill
        return
    for survivor in alive:
        try:
            survivor.kill()
        except Exception:  # noqa: BLE001
            pass
