"""Trajectory saving + scratchpad helpers (``_convert_to_trajectory_format`` stays an AIAgent method — batch_runner.py calls it)."""

import json
import gzip
import io
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Whether content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    return bool(content) and "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


def _lock_append_handle(f, acquire: bool) -> None:
    """Exclusive whole-file lock on an append handle: ``flock`` on POSIX, a 1-byte
    ``msvcrt.locking`` range at offset 0 on Windows (append position is restored by the OS)."""
    if os.name == "nt":
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if acquire else fcntl.LOCK_UN)


def _build_gzip_member(line: str) -> bytes:
    """Return one complete gzip member without touching the destination file."""
    member = io.BytesIO()
    with gzip.GzipFile(fileobj=member, mode="wb") as compressed:
        compressed.write(line.encode("utf-8"))
    return member.getvalue()


def save_trajectory(trajectory: List[Dict[str, Any]], model: str, completed: bool, filename: str = None):
    """Append a ShareGPT-format entry, gzip-compressed by default."""
    if filename is None:
        filename = "trajectory_samples.jsonl.gz" if completed else "failed_trajectories.jsonl.gz"
    entry = {"conversations": trajectory, "timestamp": datetime.now().isoformat(), "model": model, "completed": completed}
    try:
        line = json.dumps(entry, ensure_ascii=False) + "\n"  # serialize before taking the lock
        is_gzip = str(filename).endswith(".gz")
        if is_gzip:
            # Finalize the complete gzip member away from the destination. A
            # killed worker can then leave no half-written member for the next
            # append to consume.
            payload = _build_gzip_member(line)
            with open(filename, "ab") as raw:
                locked = False
                try:
                    # Lock the stable raw descriptor for the one-shot append;
                    # Windows locking must never use a moving gzip wrapper.
                    _lock_append_handle(raw, True)
                    locked = True
                    raw.write(payload)
                    raw.flush()
                    locked = False
                finally:
                    if locked:
                        try:
                            _lock_append_handle(raw, False)
                        except (OSError, ValueError):
                            pass
        else:
            with open(filename, "a", encoding="utf-8") as text_file:
                locked = False
                try:
                    _lock_append_handle(text_file, True)
                    locked = True
                    text_file.write(line)
                    text_file.flush()
                    locked = False
                finally:
                    if locked:
                        try:
                            _lock_append_handle(text_file, False)
                        except (OSError, ValueError):
                            pass
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)
