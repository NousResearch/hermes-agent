"""Restart markers for dashboard/serve processes stopped by an update."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

# EX_TEMPFAIL from sysexits.h, matching the gateway restart contract: a
# non-zero exit asks Restart=on-failure supervisors to bring the service back.
RESTART_EXIT_CODE = 75

_MARKER_FILENAME = "serve_restart.json"


def _marker_path() -> Path:
    return get_hermes_home() / "runtime" / _MARKER_FILENAME


def _remove_marker(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("Failed to remove serve restart marker %s: %s", path, exc)


def write_restart_markers(pids: list[int]) -> None:
    """Record processes that should exit non-zero after update shutdown."""
    normalized = sorted({pid for pid in pids if pid > 0})
    if not normalized:
        return
    try:
        atomic_json_write(
            _marker_path(),
            {"pids": normalized, "written_at": time.time()},
            sort_keys=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to write serve restart marker: %s", exc)


def consume_restart_marker(pid: int, max_age_seconds: float = 600) -> bool:
    """Consume a fresh update marker for *pid*, cleaning stale residue."""
    path = _marker_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, json.JSONDecodeError):
        _remove_marker(path)
        return False

    try:
        written_at = float(data["written_at"])
        marked_pids = [int(value) for value in data["pids"]]
        age = time.time() - written_at
    except (KeyError, TypeError, ValueError):
        _remove_marker(path)
        return False

    if age < 0 or age > max_age_seconds or pid not in marked_pids:
        _remove_marker(path)
        return False

    remaining = [marked_pid for marked_pid in marked_pids if marked_pid != pid]
    if not remaining:
        _remove_marker(path)
    else:
        try:
            atomic_json_write(
                path,
                {"pids": remaining, "written_at": written_at},
                sort_keys=True,
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("Failed to update serve restart marker %s: %s", path, exc)
    return True
