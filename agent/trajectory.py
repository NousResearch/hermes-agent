"""Trajectory saving utilities and static helpers.

_convert_to_trajectory_format stays as an AIAgent method (batch_runner.py
calls agent._convert_to_trajectory_format). Only the static helpers and
the file-write logic live here.
"""

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import fcntl
except ImportError:  # pragma: no cover - platform-dependent
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - platform-dependent
    msvcrt = None

logger = logging.getLogger(__name__)

# Bounded wait for the advisory lock before a save is skipped (fail-closed).
TRAJECTORY_LOCK_TIMEOUT_SECONDS = 10.0


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Check if content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    if not content:
        return False
    return "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


@contextmanager
def _trajectory_lock(filename: str):
    """Cross-process advisory lock around one JSONL append.

    Same shape as the auth.json lock in ``hermes_cli/auth.py``: a sidecar
    ``<filename>.lock`` file held via ``fcntl.flock`` (POSIX) or
    ``msvcrt.locking`` (Windows, which requires the lock file to be
    populated with the pointer at offset 0). Raises ``TimeoutError``
    fail-closed when the lock cannot be acquired — an unserialized append
    can interleave two writers' JSON lines and corrupt the file, so a
    dropped sample is strictly better than a torn write.
    """
    if fcntl is None and msvcrt is None:  # pragma: no cover - rare platform
        yield
        return

    lock_path = Path(str(filename) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # On Windows, msvcrt.locking needs the file to have content and the
    # file pointer at position 0. Ensure the lock file has at least 1 byte.
    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")

    with lock_path.open("r+" if msvcrt else "a+", encoding="utf-8") as lock_file:
        deadline = time.monotonic() + TRAJECTORY_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                if fcntl:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except (BlockingIOError, OSError, PermissionError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"trajectory lock not acquired within "
                        f"{TRAJECTORY_LOCK_TIMEOUT_SECONDS}s: {lock_path}"
                    )
                time.sleep(0.05)
        try:
            yield
        finally:
            if fcntl:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
            elif msvcrt:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass


def save_trajectory(trajectory: List[Dict[str, Any]], model: str,
                    completed: bool, filename: str = None):
    """Append a trajectory entry to a JSONL file.

    Appends are serialized across processes with an advisory sidecar lock;
    if the lock cannot be acquired within
    :data:`TRAJECTORY_LOCK_TIMEOUT_SECONDS` the save is skipped (logged)
    rather than written unserialized.

    Args:
        trajectory: The ShareGPT-format conversation list.
        model: Model name for metadata.
        completed: Whether the conversation completed successfully.
        filename: Override output filename. Defaults to trajectory_samples.jsonl
                  or failed_trajectories.jsonl based on ``completed``.
    """
    if filename is None:
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"

    entry = {
        "conversations": trajectory,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "completed": completed,
    }

    try:
        with _trajectory_lock(filename):
            with open(filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Trajectory saved to %s", filename)
    except TimeoutError as e:
        logger.warning("Trajectory not saved (lock contention): %s", e)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)
