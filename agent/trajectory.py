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


def save_trajectory(trajectory: List[Dict[str, Any]], model: str, completed: bool, filename: str = None):
    """Append a ShareGPT-format entry, gzip-compressed by default."""
    if filename is None:
        filename = "trajectory_samples.jsonl.gz" if completed else "failed_trajectories.jsonl.gz"
    entry = {"conversations": trajectory, "timestamp": datetime.now().isoformat(), "model": model, "completed": completed}
    try:
        line = json.dumps(entry, ensure_ascii=False) + "\n"  # serialize before taking the lock
        is_gzip = str(filename).endswith(".gz")
        raw = open(filename, "ab") if is_gzip else None
        f = io.TextIOWrapper(gzip.GzipFile(fileobj=raw, mode="ab"), encoding="utf-8") if raw is not None else open(filename, "a", encoding="utf-8")
        lock_handle = raw or f
        locked = False
        try:
            # Gateway sessions and batch workers append to the SAME default file; without an
            # exclusive lock around write+flush, entries larger than one write() interleave and the
            # JSONL stops parsing (#12684). Gzip close writes the member trailer, so the handle must
            # remain locked through close/finalization as well.
            # Lock the stable raw descriptor for gzip members. GzipFile may
            # flush a header and move the logical wrapper position before the
            # first write; Windows locking must never be based on that moving
            # wrapper offset.
            _lock_append_handle(lock_handle, True)
            locked = True
            f.write(line)
            f.flush()
            f.close()
            if raw is not None:
                raw.flush()
            locked = False  # close releases the OS handle lock after gzip finalization
        finally:
            if locked:
                try:
                    _lock_append_handle(lock_handle, False)
                except (OSError, ValueError):
                    pass
            if not f.closed:
                f.close()
            if raw is not None and not raw.closed:
                raw.close()
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)
