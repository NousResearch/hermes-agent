"""
Orphan process cleanup for the Hermes gateway.

Detects and terminates orphaned hermes CLI processes (e.g. ``hermes --resume --yolo``)
that survive gateway restarts.  Also sets ``PR_SET_CHILD_SUBREAPER`` on startup so
that any grandchildren spawned by the gateway are automatically reparented to it
instead of becoming init-system orphans.

Public API
----------
- ``set_subreaper()``          – make this process a subreaper (Linux only, no-op elsewhere)
- ``scan_and_cleanup_orphans()`` – find & kill orphaned hermes processes
- ``register_cli_session()`` / ``unregister_cli_session()`` – track live CLI sessions
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LINUX = platform.system() == "Linux"

_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37

_HERMES_PATTERNS = (
    "hermes --resume",
    "hermes_cli.main --resume",
    "hermes --yolo",
)

# How old (seconds) a tracked CLI session must be before we consider it stale.
# 300 s (5 min) gives time for normal handshakes without keeping zombies forever.
_DEFAULT_STALE_TTL = 300

_registry_path: Optional[Path] = None


def _get_registry_path() -> Path:
    """Return the path to the CLI session registry file."""
    global _registry_path
    if _registry_path is None:
        home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
        _registry_path = home / "cli_sessions.json"
    return _registry_path


# ---------------------------------------------------------------------------
# Subreaper
# ---------------------------------------------------------------------------

def set_subreaper() -> bool:
    """
    Set ``PR_SET_CHILD_SUBREAPER`` on this process (Linux only).

    When the gateway is a subreaper, any descendant that becomes orphaned
    (its parent exits) is reparented to the gateway rather than to PID 1.
    This lets the gateway see and manage those processes cleanly.

    Returns True on success (or no-op on non-Linux), False if the prctl call
    failed.
    """
    if not _LINUX:
        logger.debug("PR_SET_CHILD_SUBREAPER not available on %s", platform.system())
        return True  # not an error, just not applicable

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        ret = libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
        if ret == 0:
            logger.info("Gateway set as child subreaper")
            return True
        else:
            logger.warning("prctl(PR_SET_CHILD_SUBREAPER) returned %d", ret)
            return False
    except (OSError, AttributeError) as exc:
        logger.warning("Failed to set subreaper: %s", exc)
        return False


def is_subreaper() -> bool:
    """Return True if this process is currently a child subreaper (Linux only)."""
    if not _LINUX:
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        flag = ctypes.c_int()
        ret = libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(flag), 0, 0, 0)
        return ret == 0 and flag.value != 0
    except (OSError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# CLI session registry (for tracking --resume sessions we spawn)
# ---------------------------------------------------------------------------

def _read_registry() -> Dict[str, Any]:
    """Read the CLI session registry from disk."""
    path = _get_registry_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_registry(data: Dict[str, Any]) -> None:
    """Write the CLI session registry to disk."""
    path = _get_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, indent=2))
    except OSError as exc:
        logger.warning("Failed to write CLI session registry: %s", exc)


def register_cli_session(session_id: str, pid: Optional[int] = None) -> None:
    """
    Register a CLI ``--resume`` session so the gateway can track it.

    Called when a ``hermes --resume`` process starts.  The gateway will clean
    up stale registrations on next startup.
    """
    pid = pid or os.getpid()
    data = _read_registry()
    data[session_id] = {
        "pid": pid,
        "started_at": time.time(),
        "cmdline": " ".join(os.sys.argv),
    }
    _write_registry(data)
    logger.debug("Registered CLI session %s (PID %d)", session_id, pid)


def unregister_cli_session(session_id: str) -> None:
    """Remove a CLI session from the registry (called on normal exit)."""
    data = _read_registry()
    if session_id in data:
        del data[session_id]
        _write_registry(data)
        logger.debug("Unregistered CLI session %s", session_id)


# ---------------------------------------------------------------------------
# Process inspection helpers
# ---------------------------------------------------------------------------

def _is_pid_alive(pid: int) -> bool:
    """Check if a PID exists and is a running process (not a zombie)."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False

    # A zombie process (state Z) is technically signable but effectively dead
    if _LINUX:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            # State is field 3 (0-indexed: 2), surrounded by parentheses
            state = stat.split(")")[1].strip().split()[0]
            return state != "Z"  # Z = zombie
        except (FileNotFoundError, IndexError, OSError):
            pass

    return True


def _reap_zombie(pid: int) -> None:
    """Try to reap a zombie process via waitpid (non-blocking)."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass  # Not our child or already reaped


def _read_cmdline(pid: int) -> Optional[str]:
    """Read the command line of a process from /proc (Linux) or ps (fallback)."""
    # Try /proc first (fast, no subprocess)
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        if raw:
            return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except (FileNotFoundError, PermissionError, OSError):
        pass

    # Fallback to ps
    try:
        import subprocess
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    return None


def _looks_like_hermes_process(cmdline: str) -> bool:
    """Return True if the command line matches a hermes CLI process."""
    if not cmdline:
        return False
    lower = cmdline.lower()
    return any(pat in lower for pat in _HERMES_PATTERNS)


def _get_process_parent(pid: int) -> Optional[int]:
    """Get the parent PID of a process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # Parent PID is field 4 (0-indexed: 3)
        return int(stat.split()[3])
    except (FileNotFoundError, IndexError, ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# Orphan scanning & cleanup
# ---------------------------------------------------------------------------

def scan_for_orphaned_hermes(
    gateway_pid: Optional[int] = None,
    stale_ttl: float = _DEFAULT_STALE_TTL,
) -> List[Dict[str, Any]]:
    """
    Scan running processes for orphaned hermes CLI sessions.

    A process is considered orphaned if:
    1. Its command line matches a hermes pattern (--resume, --yolo)
    2. Its parent PID is 1 (init) or does not match the current gateway
    3. It has been running longer than *stale_ttl* seconds

    Returns a list of dicts with keys: pid, cmdline, parent_pid, age_seconds.
    """
    gateway_pid = gateway_pid or os.getpid()
    orphans: List[Dict[str, Any]] = []

    # Check the registry first
    registry = _read_registry()
    now = time.time()
    for session_id, info in list(registry.items()):
        rpid = info.get("pid")
        started_at = info.get("started_at", 0)
        age = now - started_at

        if rpid is None:
            continue

        if not _is_pid_alive(rpid):
            # Process is dead, clean registry entry
            del registry[session_id]
            continue

        if age < stale_ttl:
            # Not stale yet
            continue

        cmdline = _read_cmdline(rpid) or info.get("cmdline", "")
        orphans.append({
            "pid": rpid,
            "cmdline": cmdline,
            "parent_pid": _get_process_parent(rpid),
            "age_seconds": age,
            "session_id": session_id,
            "source": "registry",
        })

    # Save cleaned registry
    _write_registry(registry)

    # Also scan /proc for hermes processes not in the registry
    if _LINUX:
        proc_dir = Path("/proc")
        try:
            for entry in proc_dir.iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    pid = int(entry.name)
                except ValueError:
                    continue

                if pid == gateway_pid:
                    continue
                if pid == 1:
                    continue
                if pid == os.getpid():
                    continue

                cmdline = _read_cmdline(pid)
                if not cmdline or not _looks_like_hermes_process(cmdline):
                    continue

                parent_pid = _get_process_parent(pid)
                # Skip if this is a direct child of the current gateway
                if parent_pid == gateway_pid:
                    continue

                # Check if already in orphans list (from registry)
                if any(o["pid"] == pid for o in orphans):
                    continue

                # Get age from /proc
                try:
                    stat = Path(f"/proc/{pid}/stat").read_text().split()
                    starttime = int(stat[21])
                    # Convert clock ticks to seconds (usually 100 Hz)
                    clk_tck = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", 100))
                    uptime = float(Path("/proc/uptime").read_text().split()[0])
                    age = uptime - (starttime / clk_tck)
                except (FileNotFoundError, IndexError, ValueError, OSError):
                    age = stale_ttl + 1  # assume stale if we can't determine

                if age < stale_ttl:
                    continue

                orphans.append({
                    "pid": pid,
                    "cmdline": cmdline,
                    "parent_pid": parent_pid,
                    "age_seconds": age,
                    "session_id": None,
                    "source": "proc_scan",
                })

        except (PermissionError, OSError) as exc:
            logger.warning("Proc scan error: %s", exc)

    return orphans


def terminate_orphans(
    orphans: List[Dict[str, Any]],
    grace_period: float = 5.0,
) -> Dict[str, Any]:
    """
    Terminate orphaned processes, trying SIGTERM first then SIGKILL.

    Returns a summary dict with keys: terminated, killed, failed, skipped.
    """
    result = {"terminated": [], "killed": [], "failed": [], "skipped": []}

    for orphan in orphans:
        pid = orphan["pid"]
        cmdline = orphan.get("cmdline", "")

        if not _is_pid_alive(pid):
            result["skipped"].append({"pid": pid, "reason": "already_dead"})
            continue

        logger.info("Terminating orphaned hermes process PID %d: %s", pid, cmdline[:120])

        # Try graceful termination first
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError) as exc:
            result["failed"].append({"pid": pid, "reason": str(exc)})
            continue

        # Wait for graceful exit
        deadline = time.time() + grace_period
        while time.time() < deadline:
            if not _is_pid_alive(pid):
                _reap_zombie(pid)
                result["terminated"].append(pid)
                break
            time.sleep(0.5)

        # Force kill if still alive
        if _is_pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
                _reap_zombie(pid)
                if not _is_pid_alive(pid):
                    result["killed"].append(pid)
                else:
                    result["failed"].append({"pid": pid, "reason": "survived_sigkill"})
            except (ProcessLookupError, PermissionError) as exc:
                result["failed"].append({"pid": pid, "reason": str(exc)})

    return result


def scan_and_cleanup_orphans(
    gateway_pid: Optional[int] = None,
    stale_ttl: float = _DEFAULT_STALE_TTL,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Full orphan cleanup pipeline: scan → filter → terminate.

    Called on gateway startup to clean up leftover ``hermes --resume`` sessions.

    Returns a summary dict with keys:
      - orphans_found: number of orphaned processes detected
      - cleanup_result: dict from terminate_orphans()
      - dry_run: whether this was a dry run

    Set *dry_run* to True to only report what would be cleaned up without
    actually killing anything.
    """
    orphans = scan_for_orphaned_hermes(gateway_pid=gateway_pid, stale_ttl=stale_ttl)

    if not orphans:
        logger.info("No orphaned hermes processes found")
        return {
            "orphans_found": 0,
            "cleanup_result": None,
            "dry_run": dry_run,
        }

    logger.info("Found %d orphaned hermes process(es)", len(orphans))
    for o in orphans:
        logger.info("  PID %d (age %.0fs): %s", o["pid"], o["age_seconds"], o["cmdline"][:120])

    if dry_run:
        return {
            "orphans_found": len(orphans),
            "cleanup_result": None,
            "dry_run": True,
            "orphans": orphans,
        }

    cleanup_result = terminate_orphans(orphans)
    logger.info(
        "Orphan cleanup complete: %d terminated, %d killed, %d failed",
        len(cleanup_result["terminated"]),
        len(cleanup_result["killed"]),
        len(cleanup_result["failed"]),
    )

    return {
        "orphans_found": len(orphans),
        "cleanup_result": cleanup_result,
        "dry_run": False,
    }
