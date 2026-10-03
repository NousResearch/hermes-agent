#!/usr/bin/env python3
"""Reconcile durable coordination wakes without running Hermes or an LLM.

The normal CLI/TUI/gateway notification consumers remain the only delivery
workers.  This watchdog repairs board state (expired holders and missing
outbox wakes); those existing consumers lease any resulting event.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Any, Mapping

_WINDOWS = os.name == "nt"
if _WINDOWS:
    import msvcrt
else:
    import fcntl


def _default_board_script() -> Path:
    configured = os.environ.get("HERMES_COORD_BOARD_SCRIPT", "").strip()
    if configured:
        return Path(configured).expanduser()
    installed = Path.home() / ".hermes" / "scripts" / "session_coord.py"
    if installed.is_file():
        return installed
    return Path(__file__).with_name("session_coord.py")


def _default_lock_file() -> Path:
    return Path.home() / ".hermes" / "runtime" / "coord_resume_watchdog.lock"


def _acquire_lock(lock_handle) -> bool:
    if _WINDOWS:
        lock_handle.seek(0, os.SEEK_END)
        if lock_handle.tell() == 0:
            lock_handle.write(b"\0")
            lock_handle.flush()
        lock_handle.seek(0)
        try:
            msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        return True

    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _release_lock(lock_handle) -> None:
    if _WINDOWS:
        lock_handle.seek(0)
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)


def _has_reconciled_work(payload: Mapping[str, Any]) -> bool:
    """Recognize board action fields while ignoring status/diagnostic metadata."""
    action_keys = {"enqueued", "stale_events_canceled", "unknown_leases"}
    for key in action_keys:
        value = payload.get(key)
        if isinstance(value, bool):
            if value:
                return True
        elif isinstance(value, (int, float)):
            if value != 0:
                return True
        elif isinstance(value, (list, tuple, dict, set, str)) and len(value) > 0:
            return True
    return False


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile coordination waits; native Hermes consumers perform delivery.")
    parser.add_argument("--board-script", type=Path, default=_default_board_script())
    parser.add_argument("--lock-file", type=Path, default=_default_lock_file())
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--json", action="store_true", help="print the board JSON even when no work changed")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    board = args.board_script.expanduser()
    # Installation is intentionally optional. A preinstalled cron entry stays
    # inert until the framework-agnostic board CLI exists.
    if not board.is_file():
        return 0

    lock_path = args.lock_file.expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            acquired = _acquire_lock(lock_handle)
        except OSError as exc:
            print(f"coord_resume_watchdog: lock failed: {exc}", file=sys.stderr)
            return 1
        if not acquired:
            return 0

        completed = None
        execution_error = None
        try:
            completed = subprocess.run(  # nosec B603
                [sys.executable, str(board), "wake-reconcile", "--json"],
                text=True,
                capture_output=True,
                timeout=max(0.1, float(args.timeout)),
                check=False,
            )
        except subprocess.TimeoutExpired:
            execution_error = "coord_resume_watchdog: wake-reconcile timed out"
        except OSError as exc:
            execution_error = f"coord_resume_watchdog: wake-reconcile failed: {exc}"
        try:
            _release_lock(lock_handle)
        except OSError as exc:
            print(f"coord_resume_watchdog: lock release failed: {exc}", file=sys.stderr)
            return 1

    if execution_error is not None:
        print(execution_error, file=sys.stderr)
        return 1
    if completed is None:
        print("coord_resume_watchdog: no process result", file=sys.stderr)
        return 1

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "board command failed").strip()
        print(f"coord_resume_watchdog: {detail}", file=sys.stderr)
        return 1
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        print(f"coord_resume_watchdog: invalid board JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(payload, dict):
        print("coord_resume_watchdog: board result must be a JSON object", file=sys.stderr)
        return 1

    if args.json or _has_reconciled_work(payload):
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


def main(argv: list[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
