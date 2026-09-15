"""Minimal systemd ExecStop helper for planned gateway shutdowns.

This module intentionally imports only Python's standard library. It runs from
systemd while the application may be partially torn down, so it must not load
``gateway`` (whose package initializer imports the full application stack) or
any optional provider dependency.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

_MARKER_FILENAME = ".gateway-planned-stop.json"


def _get_process_hermes_home() -> Path:
    """Return the process-level HERMES_HOME used by the gateway."""
    value = os.environ.get("HERMES_HOME", "").strip()
    if value:
        return Path(value)
    return Path.home() / ".hermes"


def _get_process_start_time(pid: int) -> int | None:
    """Return the Linux process start-time fingerprint, when available."""
    try:
        # Match gateway.status: field 22 in /proc/<pid>/stat.
        return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])
    except (FileNotFoundError, IndexError, PermissionError, ValueError, OSError):
        return None


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    """Atomically write a compact, owner-readable JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def write_planned_stop_marker(target_pid: int) -> bool:
    """Write the marker consumed by the gateway's SIGTERM handler."""
    try:
        record: dict[str, object] = {
            "target_pid": target_pid,
            "target_start_time": _get_process_start_time(target_pid),
            "stopper_pid": os.getpid(),
            # The systemd ExecStop marker belongs to the real SIGTERM handler;
            # the polling watcher must not consume it first.
            "trigger_watcher": False,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_file(_get_process_hermes_home() / _MARKER_FILENAME, record)
        return True
    except OSError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("systemd planned-stop helper requires exactly one PID", file=sys.stderr)
        return 2

    try:
        pid = int(args[0])
    except (TypeError, ValueError):
        print("systemd planned-stop helper received an invalid PID", file=sys.stderr)
        return 2
    if pid <= 0:
        print("systemd planned-stop helper requires a positive PID", file=sys.stderr)
        return 2

    if write_planned_stop_marker(pid):
        return 0
    print("systemd planned-stop helper could not write the marker", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
